from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3.refine_schedule import make_tail_schedule
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    MODE_VIDEO_AUDIO,
    VIDEO_AUDIO_SEED_NAMESPACE,
    derive_target_refine_seed,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass import (
    SecondPassContractError,
    update_second_pass_geometry,
)
from ComfyUI_H3_Continuum_Join.v3 import targeted_refine_sampling
from ComfyUI_H3_Continuum_Join.v3.targeted_refine_sampling import (
    _prepare_video_audio_noise_and_mask,
    sample_video_audio_refine_chunk,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import (
    run_targeted_second_pass_groups,
)


class _Nested:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors

    def to(self, *_args, **_kwargs):
        return self


def _sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def test_video_audio_noise_is_random_and_mask_one_for_both_streams():
    video = torch.zeros((1, 24, 3, 4, 4))
    audio = torch.zeros((1, 32, 2, 12))
    calls = []

    def prepare_noise(samples, seed, batch_inds):
        calls.append((samples, seed, batch_inds))
        fill = 0.25 if samples.ndim == 5 else 0.75
        return torch.full_like(samples, fill, device="cpu")

    noise, noise_mask = _prepare_video_audio_noise_and_mask(
        video,
        audio,
        seed=321,
        batch_inds=[0],
        prepare_noise_fn=prepare_noise,
        nested_builder=_Nested,
    )

    video_noise, audio_noise = noise.unbind()
    video_mask, audio_mask = noise_mask.unbind()
    assert calls == [(video, 321, [0]), (audio, 321, [0])]
    assert torch.equal(video_noise, torch.full_like(video, 0.25))
    assert torch.equal(audio_noise, torch.full_like(audio, 0.75))
    assert tuple(video_mask.shape) == (1, 1, 1, 1, 1)
    assert tuple(audio_mask.shape) == (1, 1, 1, 1)
    assert torch.all(video_mask == 1)
    assert torch.all(audio_mask == 1)


def test_video_audio_sampler_passes_both_random_streams_and_exact_sigmas(monkeypatch):
    video = torch.randn((1, 24, 3, 4, 4))
    audio = torch.randn((1, 32, 2, 12))
    latent_samples = _Nested((video, audio))
    sigmas = torch.tensor([0.35, 0.2, 0.0])
    captured = {"prepare_noise": []}

    class _Guider:
        def sample(self, noise, latent_image, sampler, supplied_sigmas, **kwargs):
            captured.update(
                noise=noise,
                latent_image=latent_image,
                sampler=sampler,
                sigmas=supplied_sigmas,
                **kwargs,
            )
            return _Nested((video + 3.0, audio + 5.0))

    def prepare_noise(samples, seed, batch_inds):
        captured["prepare_noise"].append((samples, seed, batch_inds))
        fill = 0.2 if samples.ndim == 5 else 0.8
        return torch.full_like(samples, fill, device="cpu")

    fake_runtime = (
        SimpleNamespace(intermediate_device=lambda: torch.device("cpu")),
        SimpleNamespace(NestedTensor=_Nested),
        SimpleNamespace(
            fix_empty_latent_channels=lambda _model, samples, _spatial, _temporal: samples,
            prepare_noise=prepare_noise,
        ),
        SimpleNamespace(PROGRESS_BAR_ENABLED=False),
        SimpleNamespace(prepare_callback=lambda *_args: None),
    )
    monkeypatch.setattr(
        targeted_refine_sampling,
        "_load_runtime_modules",
        lambda: fake_runtime,
    )
    monkeypatch.setattr(
        targeted_refine_sampling,
        "_make_basic_guider",
        lambda *_args: _Guider(),
    )

    sampler = object()
    output = sample_video_audio_refine_chunk(
        model=object(),
        conditioning=["conditioning"],
        latent={"samples": latent_samples},
        sampler=sampler,
        sigmas=sigmas,
        seed=654,
        enable_preview=False,
    )

    assert captured["prepare_noise"] == [(video, 654, None), (audio, 654, None)]
    assert captured["latent_image"] is latent_samples
    assert captured["sigmas"] is sigmas
    assert captured["sampler"] is sampler
    video_noise, audio_noise = captured["noise"].unbind()
    video_mask, audio_mask = captured["denoise_mask"].unbind()
    assert torch.equal(video_noise, torch.full_like(video_noise, 0.2))
    assert torch.equal(audio_noise, torch.full_like(audio_noise, 0.8))
    assert torch.all(video_mask == 1)
    assert torch.all(audio_mask == 1)
    sampled_video, sampled_audio = output["samples"].unbind()
    assert torch.equal(sampled_video, video + 3.0)
    assert torch.equal(sampled_audio, audio + 5.0)


def _group(group_id: int, *, temporal: int, audio_t: int, terminal: bool) -> dict:
    return {
        "group_id": group_id,
        "logical_chunks": [1] if group_id == 0 else [2, 3],
        "physical_prompt": f"group {group_id + 1}",
        "prompt_policy": "paired_timeline_v1" if terminal else "single",
        "physical_frames": 260 if terminal else 124,
        "trim_prefix_frames": 22 if terminal else 0,
        "terminal_merged": terminal,
        "source_width": 128,
        "source_height": 128,
        "source_batch": 1,
        "latent_channels": 24,
        "source_latent_t": temporal,
        "source_latent_h": 8,
        "source_latent_w": 8,
        "source_audio_shape": [1, 32, 2, audio_t],
    }


def _fixture():
    groups = [
        _group(0, temporal=31, audio_t=207, terminal=False),
        _group(1, temporal=65, audio_t=433, terminal=True),
    ]
    videos = [
        {
            "samples": torch.arange(
                24 * group["source_latent_t"] * 8 * 8,
                dtype=torch.float32,
            ).reshape(1, 24, group["source_latent_t"], 8, 8)
        }
        for group in groups
    ]
    audios = [
        {
            "samples": torch.arange(
                32 * 2 * group["source_audio_shape"][-1],
                dtype=torch.float32,
            ).reshape(group["source_audio_shape"])
        }
        for group in groups
    ]
    plan = {
        "width": 128,
        "height": 128,
        "existing_top_level": {"preserved": True},
        "second_pass_contract": {
            "version": 1,
            "existing_contract_field": "preserved",
            "physical_groups": groups,
        },
    }
    return groups, videos, audios, plan


def _run_video_audio(sample_fn):
    groups, videos, audios, plan = _fixture()
    source_sigmas = torch.tensor([0.6, 0.3, 0.0], dtype=torch.float32)
    schedule = make_tail_schedule(source_sigmas, evaluation_count=1)
    result = run_targeted_second_pass_groups(
        model="model",
        clip="clip",
        sampler="sampler",
        sigmas=source_sigmas,
        video_latents=videos,
        audio_latents=audios,
        assembly_plan=plan,
        refine_seed=91,
        refine_target=MODE_VIDEO_AUDIO,
        encode_prompt_fn=lambda _clip, prompt, **_kwargs: [prompt],
        latent_builder=lambda video, audio: {"video": video, "audio": audio},
        sample_fn=sample_fn,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=lambda source, **kwargs: (source, kwargs),
        refine_schedule=schedule,
        enable_preview=False,
    )
    return groups, videos, audios, plan, source_sigmas, schedule, result


def test_video_audio_group_gate_adopts_both_streams_once_per_physical_group():
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return {
            "video": kwargs["latent"]["video"] + float(len(calls)),
            "audio": kwargs["latent"]["audio"] + 100.0 + float(len(calls)),
        }

    groups, videos, audios, plan, _sigmas, schedule, result = _run_video_audio(
        sample
    )
    plan_before = copy.deepcopy(plan)
    output_videos, output_audios, updated_plan, status = result

    assert plan == plan_before
    assert len(calls) == len(groups) == 2
    assert groups[1]["logical_chunks"] == [2, 3]
    assert groups[1]["terminal_merged"] is True
    assert all(call["sigmas"] is schedule.sigmas for call in calls)
    assert [call["seed"] for call in calls] == [
        derive_target_refine_seed(91, 0, MODE_VIDEO_AUDIO),
        derive_target_refine_seed(91, 1, MODE_VIDEO_AUDIO),
    ]
    assert all(
        output is not source
        for output, source in zip(output_videos, videos, strict=True)
    )
    assert all(
        output is not source
        for output, source in zip(output_audios, audios, strict=True)
    )
    assert [_sha(item["samples"]) for item in output_videos] != [
        _sha(item["samples"]) for item in videos
    ]
    assert [_sha(item["samples"]) for item in output_audios] != [
        _sha(item["samples"]) for item in audios
    ]
    assert [tuple(item["samples"].shape) for item in output_videos] == [
        tuple(item["samples"].shape) for item in videos
    ]
    assert [tuple(item["samples"].shape) for item in output_audios] == [
        tuple(item["samples"].shape) for item in audios
    ]
    assert "GPU Experimental PASS / Production HOLD" in status
    assert status.count("sampled_video_adopted=true") == 2

    contract = updated_plan["second_pass_contract"]
    assert contract["version"] == 1
    assert contract["refine_target_contract"]["mode"] == MODE_VIDEO_AUDIO
    assert (
        contract["refine_target_contract"]["seed_namespace"]
        == VIDEO_AUDIO_SEED_NAMESPACE
    )
    assert contract["video_output"] == "sampled"
    assert contract["video_sampling"] == "seeded_random_mask_1"
    assert contract["audio_output"] == "sampled"
    assert contract["audio_sampling"] == "seeded_random_mask_1"
    assert contract["refine_group_seeds"] == [call["seed"] for call in calls]
    assert contract["refine_schedule"]["schedule_hash"] == schedule.contract[
        "schedule_hash"
    ]
    assert contract["refine_execution_contract"]["target"] == contract[
        "refine_target_contract"
    ]

    expected_plan = update_second_pass_geometry(plan, output_videos)
    normalized = copy.deepcopy(updated_plan)
    normalized_contract = normalized["second_pass_contract"]
    for key in (
        "execution",
        "conditioning_sources",
        "refine_seed_base",
        "refine_group_seeds",
        "video_output",
        "video_sampling",
        "audio_output",
        "audio_sampling",
        "refine_schedule",
        "refine_schedule_identity",
        "refine_target_contract",
        "refine_execution_contract",
    ):
        normalized_contract.pop(key)
    assert normalized == expected_plan


@pytest.mark.parametrize(
    "defect,match",
    [
        ("video_shape", "video group 0 changed"),
        ("audio_shape", "audio group 0 changed"),
        ("video_nan", "video group 0 contains NaN or Inf"),
        ("audio_nan", "audio group 0 contains NaN or Inf"),
    ],
)
def test_video_audio_rejects_invalid_sampled_streams(defect, match):
    def sample(**kwargs):
        video = kwargs["latent"]["video"].clone()
        audio = kwargs["latent"]["audio"].clone()
        if defect == "video_shape":
            video = video[..., :-1]
        elif defect == "audio_shape":
            audio = audio[..., :-1]
        elif defect == "video_nan":
            video.reshape(-1)[0] = float("nan")
        elif defect == "audio_nan":
            audio.reshape(-1)[0] = float("inf")
        return {"video": video, "audio": audio}

    with pytest.raises(SecondPassContractError, match=match):
        _run_video_audio(sample)
