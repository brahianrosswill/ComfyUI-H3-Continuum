from __future__ import annotations

import copy
import hashlib

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3.refine_schedule import (
    AUDIO_POLICY_LOCKED_PASSTHROUGH,
    NOISE_MODE_VIDEO_RANDOM_AUDIO_ZERO,
    make_tail_schedule,
)
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    AUDIO_ONLY_SEED_NAMESPACE,
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
    OUTPUT_INPUT_PASSTHROUGH,
    OUTPUT_SAMPLED,
    VIDEO_AUDIO_SEED_NAMESPACE,
    VIDEO_ONLY_SEED_NAMESPACE,
    RefineTargetError,
    build_target_execution_contract,
    derive_target_refine_seed,
    make_refine_target,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass import (
    derive_refine_seed,
    run_second_pass_groups,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass_nodes import (
    H3ContinuumSecondPassV35,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_refine_sampling import (
    sample_targeted_refine_chunk,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import (
    run_targeted_second_pass_groups,
)


def _sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize(
    "mode,video_policy,audio_policy,namespace",
    [
        (
            MODE_VIDEO_ONLY,
            ("seeded_random", 1, OUTPUT_SAMPLED),
            ("zero", 0, OUTPUT_INPUT_PASSTHROUGH),
            VIDEO_ONLY_SEED_NAMESPACE,
        ),
        (
            MODE_AUDIO_ONLY,
            ("zero", 0, OUTPUT_INPUT_PASSTHROUGH),
            ("seeded_random", 1, OUTPUT_SAMPLED),
            AUDIO_ONLY_SEED_NAMESPACE,
        ),
        (
            MODE_VIDEO_AUDIO,
            ("seeded_random", 1, OUTPUT_SAMPLED),
            ("seeded_random", 1, OUTPUT_SAMPLED),
            VIDEO_AUDIO_SEED_NAMESPACE,
        ),
    ],
)
def test_refine_target_v1_freezes_all_three_stream_policies(
    mode,
    video_policy,
    audio_policy,
    namespace,
):
    target = make_refine_target(mode)
    assert target.contract["magic"] == "H3_CONTINUUM_REFINE_TARGET"
    assert target.contract["schema_version"] == 1
    assert target.contract["mode"] == mode
    assert (
        target.contract["video"]["noise"],
        target.contract["video"]["mask"],
        target.contract["video"]["output"],
    ) == video_policy
    assert (
        target.contract["audio"]["noise"],
        target.contract["audio"]["mask"],
        target.contract["audio"]["output"],
    ) == audio_policy
    assert target.contract["seed_namespace"] == namespace
    assert len(target.contract["target_hash"]) == 64


def test_target_seed_namespaces_are_stable_and_video_only_is_legacy_exact():
    assert derive_target_refine_seed(42, 0, MODE_VIDEO_ONLY) == derive_refine_seed(42, 0)
    seeds = {
        derive_target_refine_seed(42, 0, mode)
        for mode in (MODE_VIDEO_ONLY, MODE_AUDIO_ONLY, MODE_VIDEO_AUDIO)
    }
    assert len(seeds) == 3
    assert derive_target_refine_seed(42, 1, MODE_VIDEO_ONLY) != derive_refine_seed(42, 0)


def test_target_execution_uses_sigma_range_without_mutating_schedule_v1():
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0], dtype=torch.float32)
    schedule = make_tail_schedule(sigmas, evaluation_count=2)
    before = copy.deepcopy(dict(schedule.contract))
    execution = build_target_execution_contract(
        make_refine_target(MODE_VIDEO_ONLY),
        schedule.contract,
    )

    assert dict(schedule.contract) == before
    assert schedule.contract["noise_mode"] == NOISE_MODE_VIDEO_RANDOM_AUDIO_ZERO
    assert schedule.contract["audio_policy"] == AUDIO_POLICY_LOCKED_PASSTHROUGH
    assert execution["policy_authority"] == "refine_target_contract_v1"
    assert execution["sigma_range"]["sigma_hash"] == schedule.contract["sigma_hash"]
    assert execution["sigma_range"]["source_schedule_hash"] == schedule.contract["schedule_hash"]
    assert "noise_mode" not in execution["sigma_range"]
    assert "audio_policy" not in execution["sigma_range"]


def test_invalid_target_contracts_fail_before_sampling():
    with pytest.raises(RefineTargetError, match="unsupported"):
        make_refine_target("unknown")

    called = False

    def forbidden(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("future target must stop before Sampling")

    with pytest.raises(RefineTargetError, match="unsupported"):
        sample_targeted_refine_chunk(
            model=object(),
            conditioning=[],
            latent={"samples": object()},
            sampler=object(),
            sigmas=torch.tensor([0.2, 0.0]),
            seed=1,
            refine_target="unknown",
            legacy_sample_fn=forbidden,
        )
    assert called is False


def test_targeted_video_only_sampler_delegates_every_argument_unchanged():
    captured = {}
    sentinel = {"samples": object()}
    inputs = {
        "model": object(),
        "conditioning": [[torch.zeros((1, 1, 1)), {}]],
        "latent": {"samples": object()},
        "sampler": object(),
        "sigmas": torch.tensor([0.4, 0.0]),
        "seed": 99,
        "enable_preview": False,
    }

    def delegate(**kwargs):
        captured.update(kwargs)
        return sentinel

    result = sample_targeted_refine_chunk(
        **inputs,
        refine_target=MODE_VIDEO_ONLY,
        legacy_sample_fn=delegate,
    )
    assert result is sentinel
    assert captured == inputs


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
        {"samples": torch.arange(24 * group["source_latent_t"] * 8 * 8, dtype=torch.float32).reshape(1, 24, group["source_latent_t"], 8, 8)}
        for group in groups
    ]
    audios = [
        {"samples": torch.arange(32 * 2 * group["source_audio_shape"][-1], dtype=torch.float32).reshape(group["source_audio_shape"])}
        for group in groups
    ]
    plan = {
        "width": 128,
        "height": 128,
        "second_pass_contract": {"version": 1, "physical_groups": groups},
    }
    return groups, videos, audios, plan


def _run(*, targeted: bool):
    groups, videos, audios, plan = _fixture()
    calls = []

    def sample(**kwargs):
        calls.append(kwargs)
        return {
            "video": kwargs["latent"]["video"] + float(len(calls)),
            "audio": kwargs["latent"]["audio"] + 100.0,
        }

    common = dict(
        model="model",
        clip="clip",
        sampler="sampler",
        sigmas=torch.tensor([0.6, 0.3, 0.0], dtype=torch.float32),
        video_latents=videos,
        audio_latents=audios,
        assembly_plan=plan,
        refine_seed=73,
        encode_prompt_fn=lambda _clip, prompt, **_kwargs: [prompt],
        latent_builder=lambda video, audio: {"video": video, "audio": audio},
        sample_fn=sample,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=lambda source, **kwargs: (source, kwargs),
        enable_preview=False,
    )
    if targeted:
        result = run_targeted_second_pass_groups(
            **common,
            refine_target=MODE_VIDEO_ONLY,
        )
    else:
        result = run_second_pass_groups(**common)
    return groups, videos, audios, common["sigmas"], calls, result


def test_targeted_video_only_is_exact_legacy_second_pass_parity():
    legacy = _run(targeted=False)
    targeted = _run(targeted=True)
    legacy_groups, _legacy_inputs, legacy_audio_inputs, legacy_sigmas, legacy_calls, legacy_result = legacy
    targeted_groups, _targeted_inputs, targeted_audio_inputs, targeted_sigmas, targeted_calls, targeted_result = targeted
    legacy_videos, legacy_audios, legacy_plan, legacy_status = legacy_result
    targeted_videos, targeted_audios, targeted_plan, targeted_status = targeted_result

    assert [_sha(item["samples"]) for item in targeted_videos] == [
        _sha(item["samples"]) for item in legacy_videos
    ]
    assert all(
        output is source
        for output, source in zip(targeted_audios, targeted_audio_inputs, strict=True)
    )
    assert all(
        output is source
        for output, source in zip(legacy_audios, legacy_audio_inputs, strict=True)
    )
    assert [call["seed"] for call in targeted_calls] == [
        call["seed"] for call in legacy_calls
    ] == [derive_refine_seed(73, 0), derive_refine_seed(73, 1)]
    assert all(call["sigmas"] is targeted_sigmas for call in targeted_calls)
    assert all(call["sigmas"] is legacy_sigmas for call in legacy_calls)
    assert [call["conditioning"] for call in targeted_calls] == [
        call["conditioning"] for call in legacy_calls
    ] == [["group 1"], ["group 2"]]
    assert len(targeted_calls) == len(legacy_calls) == len(legacy_groups) == len(targeted_groups) == 2
    assert legacy_status == targeted_status

    normalized_targeted = copy.deepcopy(targeted_plan)
    target_contract = normalized_targeted["second_pass_contract"].pop(
        "refine_target_contract"
    )
    execution_contract = normalized_targeted["second_pass_contract"].pop(
        "refine_execution_contract"
    )
    assert normalized_targeted == legacy_plan
    assert target_contract["mode"] == MODE_VIDEO_ONLY
    assert execution_contract["target"] == target_contract
    assert execution_contract["sigma_range"]["sigma_hash"] == (
        legacy_plan["second_pass_contract"]["refine_schedule"]["sigma_hash"]
    )


def test_existing_v35_public_schema_remains_frozen():
    schema = H3ContinuumSecondPassV35.INPUT_TYPES()
    assert list(schema["required"]) == [
        "model",
        "clip",
        "sampler",
        "sigmas",
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "refine_seed",
    ]
    assert list(schema["optional"]) == ["refine_context", "video_vae"]
    assert "refine_target" not in schema["required"]
    assert "refine_target" not in schema["optional"]
