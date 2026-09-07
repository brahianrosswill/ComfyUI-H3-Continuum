from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3 import targeted_refine_sampling
from ComfyUI_H3_Continuum_Join.v3.refine_schedule import make_tail_schedule
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    MODE_AUDIO_ONLY,
    derive_target_refine_seed,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass import update_second_pass_geometry
from ComfyUI_H3_Continuum_Join.v3.targeted_refine_sampling import (
    _prepare_audio_only_noise_and_mask,
    sample_audio_only_refine_chunk,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import (
    run_targeted_second_pass_groups,
)


class _Nested:
    is_nested = True

    def __init__(self, tensors):
        self.tensors = list(tensors)

    def unbind(self):
        return self.tensors

    def to(self, *_args, **_kwargs):
        return self


def _sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def test_audio_only_noise_is_random_for_audio_zero_for_video_with_minimal_masks():
    video = torch.ones((1, 24, 3, 4, 5), dtype=torch.float16)
    audio = torch.zeros((1, 32, 2, 12), dtype=torch.float16)
    calls = []

    def prepare_noise(samples, seed, batch_inds):
        calls.append((samples, seed, batch_inds))
        return torch.full_like(samples, 0.625, device="cpu")

    noise, mask = _prepare_audio_only_noise_and_mask(
        video,
        audio,
        seed=321,
        batch_inds=[0],
        prepare_noise_fn=prepare_noise,
        nested_builder=_Nested,
    )

    video_noise, audio_noise = noise.unbind()
    video_mask, audio_mask = mask.unbind()
    assert calls == [(audio, 321, [0])]
    assert torch.count_nonzero(video_noise) == 0
    assert video_noise.dtype == video.dtype
    assert torch.equal(audio_noise, torch.full_like(audio_noise, 0.625))
    assert tuple(video_mask.shape) == (1, 1, 1, 1, 1)
    assert tuple(audio_mask.shape) == (1, 1, 1, 1)
    assert torch.all(video_mask == 0)
    assert torch.all(audio_mask == 1)


def test_audio_only_sampler_passes_target_policy_and_exact_sigmas_to_core(monkeypatch):
    video = torch.randn((1, 24, 3, 4, 4))
    audio = torch.randn((1, 32, 2, 12))
    latent_samples = _Nested((video, audio))
    sigmas = torch.tensor([0.35, 0.2, 0.0])
    captured = {}

    class _Guider:
        def sample(self, noise, latent_image, sampler, supplied_sigmas, **kwargs):
            captured.update(
                noise=noise,
                latent_image=latent_image,
                sampler=sampler,
                sigmas=supplied_sigmas,
                **kwargs,
            )
            return _Nested((video + 500.0, audio + 2.0))

    def prepare_noise(samples, seed, batch_inds):
        captured["prepare_noise"] = (samples, seed, batch_inds)
        return torch.full_like(samples, 0.75, device="cpu")

    fake_sample = SimpleNamespace(
        fix_empty_latent_channels=lambda _model, samples, _spatial, _temporal: samples,
        prepare_noise=prepare_noise,
    )
    fake_runtime = (
        SimpleNamespace(intermediate_device=lambda: torch.device("cpu")),
        SimpleNamespace(NestedTensor=_Nested),
        fake_sample,
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
    output = sample_audio_only_refine_chunk(
        model=object(),
        conditioning=["conditioning"],
        latent={"samples": latent_samples},
        sampler=sampler,
        sigmas=sigmas,
        seed=654,
        enable_preview=False,
    )

    assert captured["prepare_noise"] == (audio, 654, None)
    assert captured["latent_image"] is latent_samples
    assert captured["sigmas"] is sigmas
    assert captured["sampler"] is sampler
    video_noise, audio_noise = captured["noise"].unbind()
    video_mask, audio_mask = captured["denoise_mask"].unbind()
    assert torch.count_nonzero(video_noise) == 0
    assert torch.equal(audio_noise, torch.full_like(audio_noise, 0.75))
    assert torch.all(video_mask == 0)
    assert torch.all(audio_mask == 1)
    sampled_video, sampled_audio = output["samples"].unbind()
    assert torch.equal(sampled_video, video + 500.0)
    assert torch.equal(sampled_audio, audio + 2.0)


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


def test_audio_only_group_gate_preserves_video_and_samples_each_physical_group_once():
    groups, videos, audios, plan = _fixture()
    plan_before = copy.deepcopy(plan)
    video_hashes = [_sha(item["samples"]) for item in videos]
    source_sigmas = torch.tensor([0.6, 0.3, 0.0], dtype=torch.float32)
    schedule = make_tail_schedule(source_sigmas, evaluation_count=1)
    schedule_before = copy.deepcopy(dict(schedule.contract))
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return {
            "video": kwargs["latent"]["video"] + 1000.0,
            "audio": kwargs["latent"]["audio"] + float(len(calls)),
        }

    result = run_targeted_second_pass_groups(
        model="model",
        clip="clip",
        sampler="sampler",
        sigmas=source_sigmas,
        video_latents=videos,
        audio_latents=audios,
        assembly_plan=plan,
        refine_seed=91,
        refine_target=MODE_AUDIO_ONLY,
        encode_prompt_fn=lambda _clip, prompt, **_kwargs: [prompt],
        latent_builder=lambda video, audio: {"video": video, "audio": audio},
        sample_fn=sample,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=lambda source, **kwargs: (source, kwargs),
        refine_schedule=schedule,
        enable_preview=False,
    )
    output_videos, output_audios, updated_plan, status = result

    assert plan == plan_before
    assert dict(schedule.contract) == schedule_before
    assert all(
        output is source
        for output, source in zip(output_videos, videos, strict=True)
    )
    assert [_sha(item["samples"]) for item in output_videos] == video_hashes
    assert [tuple(item["samples"].shape) for item in output_audios] == [
        tuple(item["samples"].shape) for item in audios
    ]
    assert all(
        output is not source
        for output, source in zip(output_audios, audios, strict=True)
    )
    assert [call["seed"] for call in calls] == [
        derive_target_refine_seed(91, 0, MODE_AUDIO_ONLY),
        derive_target_refine_seed(91, 1, MODE_AUDIO_ONLY),
    ]
    assert all(call["sigmas"] is schedule.sigmas for call in calls)
    assert [call["conditioning"] for call in calls] == [["group 1"], ["group 2"]]
    assert len(calls) == len(groups) == 2
    assert groups[1]["logical_chunks"] == [2, 3]
    assert groups[1]["terminal_merged"] is True
    assert "Production HOLD" in status

    contract = updated_plan["second_pass_contract"]
    assert contract["version"] == 1
    assert contract["refine_target_contract"]["mode"] == MODE_AUDIO_ONLY
    assert contract["video_output"] == "bit_exact_first_pass_passthrough"
    assert contract["video_sampling"] == "zero_noise_mask_locked"
    assert contract["audio_output"] == "sampled"
    assert contract["audio_sampling"] == "seeded_random_mask_1"
    assert contract["refine_group_seeds"] == [call["seed"] for call in calls]
    assert contract["refine_schedule"]["schedule_hash"] == schedule.contract["schedule_hash"]
    assert contract["refine_execution_contract"]["sigma_range"]["sigma_hash"] == (
        schedule.contract["sigma_hash"]
    )

    expected_plan = update_second_pass_geometry(plan, videos)
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
