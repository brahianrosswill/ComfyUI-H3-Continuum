from __future__ import annotations

import copy

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3 import second_pass, second_pass_nodes
from ComfyUI_H3_Continuum_Join.v3 import targeted_second_pass
from ComfyUI_H3_Continuum_Join.v3.refine_scope import (
    MAGIC,
    MODE_ALL,
    MODE_LOGICAL_CHUNK,
    MODE_PHYSICAL_GROUP,
    RefineScopeError,
    resolve_refine_scope,
    serializable_scope_contract,
)
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
    derive_target_refine_seed,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass_nodes import (
    H3ContinuumSelectiveSecondPassExperimental,
    REFINE_SCOPE_ALL,
    REFINE_SCOPE_LOGICAL_CHUNK,
    REFINE_SCOPE_PHYSICAL_GROUP,
    REFINE_TARGET_VIDEO_ONLY,
)
from ComfyUI_H3_Continuum_Join.v3.targeted_second_pass import (
    run_targeted_second_pass_groups,
)


def _group(
    group_id: int,
    logical_chunks: list[int],
    *,
    temporal: int,
    audio_t: int,
    terminal: bool = False,
) -> dict:
    return {
        "group_id": group_id,
        "logical_chunks": logical_chunks,
        "physical_prompt": f"group {group_id + 1}",
        "prompt_policy": "paired_timeline_v1" if terminal else "single",
        "physical_frames": 260 if terminal else 124,
        "trim_prefix_frames": 22 if group_id else 0,
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


def _fixture(*, terminal: bool = False):
    if terminal:
        groups = [
            _group(0, [1], temporal=31, audio_t=207),
            _group(1, [2, 3], temporal=65, audio_t=433, terminal=True),
        ]
    else:
        groups = [
            _group(0, [1], temporal=31, audio_t=207),
            _group(1, [2], temporal=36, audio_t=235),
            _group(2, [3], temporal=36, audio_t=235),
        ]
    videos = [
        {
            "samples": torch.full(
                (1, 24, group["source_latent_t"], 8, 8),
                float(index + 1),
            )
        }
        for index, group in enumerate(groups)
    ]
    audios = [
        {
            "samples": torch.full(
                tuple(group["source_audio_shape"]),
                float(index + 11),
            )
        }
        for index, group in enumerate(groups)
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


def _run(
    target: str,
    *,
    scope: str = MODE_ALL,
    scope_index: int = 1,
    terminal: bool = False,
    observers: dict | None = None,
):
    groups, videos, audios, plan = _fixture(terminal=terminal)
    calls = observers if observers is not None else {}
    calls.setdefault("encode", [])
    calls.setdefault("clone", [])
    calls.setdefault("sample", [])
    sigmas = torch.tensor([0.6, 0.3, 0.0], dtype=torch.float32)

    def encode(_clip, prompt, **_kwargs):
        calls["encode"].append(prompt)
        return [prompt]

    def clone(model, **kwargs):
        calls["clone"].append(kwargs)
        return model

    def sample(**kwargs):
        calls["sample"].append(kwargs)
        return {
            "video": kwargs["latent"]["video"] + 100.0,
            "audio": kwargs["latent"]["audio"] + 200.0,
        }

    result = run_targeted_second_pass_groups(
        model="model",
        clip="clip",
        sampler="sampler",
        sigmas=sigmas,
        video_latents=videos,
        audio_latents=audios,
        assembly_plan=plan,
        refine_seed=91,
        refine_target=target,
        refine_scope=scope,
        refine_scope_index=scope_index,
        encode_prompt_fn=encode,
        latent_builder=lambda video, audio: {"video": video, "audio": audio},
        sample_fn=sample,
        stream_extractor=lambda value: (value["video"], value["audio"]),
        clone_model_fn=clone,
        enable_preview=False,
    )
    return groups, videos, audios, plan, sigmas, calls, result


def test_scope_resolver_is_pure_and_terminal_logical_selection_is_atomic():
    groups, _videos, _audios, plan = _fixture(terminal=True)
    plan_before = copy.deepcopy(plan)

    all_scope = resolve_refine_scope(MODE_ALL, 99, plan)
    physical = resolve_refine_scope(MODE_PHYSICAL_GROUP, 2, plan)
    logical = resolve_refine_scope(MODE_LOGICAL_CHUNK, 3, plan)

    assert plan == plan_before
    assert all_scope.selected_group_indices == (0, 1)
    assert all_scope.contract["requested_index"] is None
    assert physical.selected_group_indices == (1,)
    assert physical.contract["selected_logical_chunks"] == (2, 3)
    assert physical.contract["terminal_expanded"] is False
    assert logical.selected_group_indices == (1,)
    assert logical.contract["selected_logical_chunks"] == (2, 3)
    assert logical.contract["terminal_expanded"] is True
    assert logical.contract["magic"] == MAGIC
    serialized = serializable_scope_contract(logical)
    serialized["selected_logical_chunks"].append(99)
    assert logical.contract["selected_logical_chunks"] == (2, 3)
    assert groups[1]["logical_chunks"] == [2, 3]


@pytest.mark.parametrize(
    ("mode", "index", "match"),
    (
        ("unknown", 1, "unsupported refine scope"),
        (1, 1, "unsupported refine scope"),
        (MODE_PHYSICAL_GROUP, 0, "positive integer"),
        (MODE_PHYSICAL_GROUP, 1.5, "positive integer"),
        (MODE_PHYSICAL_GROUP, 3, "outside"),
        (MODE_LOGICAL_CHUNK, 4, "not present"),
    ),
)
def test_invalid_scope_fails_before_any_execution(mode, index, match):
    _groups, _videos, _audios, plan = _fixture(terminal=True)
    with pytest.raises(RefineScopeError, match=match):
        resolve_refine_scope(mode, index, plan)


@pytest.mark.parametrize(
    ("target", "video_changed", "audio_changed"),
    (
        (MODE_VIDEO_ONLY, True, False),
        (MODE_AUDIO_ONLY, False, True),
        (MODE_VIDEO_AUDIO, True, True),
    ),
)
def test_physical_group_scope_does_no_work_for_unselected_groups(
    monkeypatch,
    target,
    video_changed,
    audio_changed,
):
    derived_indices = []
    if target == MODE_VIDEO_ONLY:
        original = second_pass.derive_refine_seed

        def derive(seed, index):
            derived_indices.append(index)
            return original(seed, index)

        monkeypatch.setattr(second_pass, "derive_refine_seed", derive)
    else:
        original = targeted_second_pass.derive_target_refine_seed

        def derive(seed, index, resolved_target):
            derived_indices.append(index)
            return original(seed, index, resolved_target)

        monkeypatch.setattr(targeted_second_pass, "derive_target_refine_seed", derive)

    groups, videos, audios, plan, sigmas, calls, result = _run(
        target,
        scope=MODE_PHYSICAL_GROUP,
        scope_index=2,
    )
    output_videos, output_audios, updated_plan, status = result

    assert len(groups) == 3
    assert calls["encode"] == ["group 2"]
    assert len(calls["clone"]) == len(calls["sample"]) == 1
    assert derived_indices == [1]
    assert calls["sample"][0]["sigmas"] is sigmas
    assert calls["sample"][0]["seed"] == derive_target_refine_seed(91, 1, target)

    for index in (0, 2):
        assert output_videos[index] is videos[index]
        assert output_audios[index] is audios[index]
    assert (output_videos[1] is not videos[1]) is video_changed
    assert (output_audios[1] is not audios[1]) is audio_changed

    assert plan["existing_top_level"] == {"preserved": True}
    assert plan["second_pass_contract"]["existing_contract_field"] == "preserved"
    contract = updated_plan["second_pass_contract"]
    assert contract["version"] == 1
    assert contract["existing_contract_field"] == "preserved"
    assert contract["physical_groups"][:3] != []
    assert contract["refine_group_seeds"] == [calls["sample"][0]["seed"]]
    assert "refine_scope_contract" not in contract
    assert "no conditioning, MODEL clone, Sampling, or seed derivation" in status


def test_non_selected_refine_context_conditioning_is_not_adapted_or_cloned():
    groups, videos, _audios, plan = _fixture()
    calls = {"adapt": [], "clone": [], "consume": []}

    def validate(_context, *, assembly_plan):
        assert assembly_plan is plan
        return {"complete": True, "groups": groups}

    def adapt(group, **_kwargs):
        calls["adapt"].append(group["group_id"])
        return [[f"conditioning-{group['group_id']}", {}]], None

    def clone(model, **kwargs):
        calls["clone"].append(kwargs["chunk_index"])
        return model

    prepared = second_pass.prepare_physical_refine_groups(
        model="model",
        clip="clip",
        video_latents=videos,
        assembly_plan=plan,
        refine_context=object(),
        validate_refine_context_fn=validate,
        adapt_group_conditioning_fn=adapt,
        clone_model_fn=clone,
        group_consumer_fn=lambda group_index, *_args: calls["consume"].append(
            group_index
        ),
        retain_group_outputs=False,
        selected_group_indices=(1,),
    )

    assert calls == {"adapt": [1], "clone": [2], "consume": [1]}
    assert prepared["selected_group_indices"] == [1]
    assert prepared["selected_group_count"] == 1


def test_selected_group_seed_matches_all_mode_and_all_mode_has_a3c_plan_contract():
    *_prefix, all_calls, all_result = _run(MODE_VIDEO_AUDIO)
    *_scoped_prefix, scoped_calls, scoped_result = _run(
        MODE_VIDEO_AUDIO,
        scope=MODE_PHYSICAL_GROUP,
        scope_index=2,
    )
    all_plan = all_result[2]
    scoped_plan = scoped_result[2]

    assert scoped_calls["sample"][0]["seed"] == all_calls["sample"][1]["seed"]
    assert "refine_scope_contract" not in all_plan["second_pass_contract"]
    assert set(scoped_plan["second_pass_contract"]) == set(
        all_plan["second_pass_contract"]
    )
    assert "Refine Scope:" not in all_result[3]
    assert scoped_plan["second_pass_contract"]["version"] == 1


@pytest.mark.parametrize("logical_chunk", (2, 3))
def test_terminal_logical_chunk_scope_expands_to_one_atomic_physical_pair(
    logical_chunk,
):
    groups, videos, audios, _plan, _sigmas, calls, result = _run(
        MODE_VIDEO_AUDIO,
        scope=MODE_LOGICAL_CHUNK,
        scope_index=logical_chunk,
        terminal=True,
    )
    output_videos, output_audios, updated_plan, status = result

    assert groups[1]["logical_chunks"] == [2, 3]
    assert groups[1]["terminal_merged"] is True
    assert len(calls["encode"]) == len(calls["clone"]) == len(calls["sample"]) == 1
    assert output_videos[0] is videos[0]
    assert output_audios[0] is audios[0]
    assert output_videos[1] is not videos[1]
    assert output_audios[1] is not audios[1]
    assert "refine_scope_contract" not in updated_plan["second_pass_contract"]
    assert "selected_physical_groups=[2]" in status
    assert "selected_logical_chunks=[2, 3]" in status
    assert "terminal_expanded=true" in status


def test_public_scope_labels_forward_without_frontend_or_silent_fallback(monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return kwargs["video_latents"], kwargs["audio_latents"], kwargs[
            "assembly_plan"
        ], "ok"

    monkeypatch.setattr(second_pass_nodes, "run_targeted_second_pass_groups", fake_run)
    node = H3ContinuumSelectiveSecondPassExperimental()
    common = {
        "model": [object()],
        "clip": [object()],
        "sampler": [object()],
        "sigmas": [object()],
        "video_latents": [{"samples": object()}],
        "audio_latents": [{"samples": object()}],
        "assembly_plan": [{"second_pass_contract": {"version": 1}}],
        "refine_seed": [7],
        "refine_target": [REFINE_TARGET_VIDEO_ONLY],
    }
    node.refine(
        **common,
        refine_scope=[REFINE_SCOPE_PHYSICAL_GROUP],
        refine_scope_index=[2],
    )
    node.refine(
        **common,
        refine_scope=[REFINE_SCOPE_LOGICAL_CHUNK],
        refine_scope_index=[3],
    )
    node.refine(**common)

    assert [(call["refine_scope"], call["refine_scope_index"]) for call in calls] == [
        (MODE_PHYSICAL_GROUP, 2),
        (MODE_LOGICAL_CHUNK, 3),
        (MODE_ALL, 1),
    ]

    with pytest.raises(ValueError, match="unsupported selective refine scope"):
        node.refine(
            **common,
            refine_scope=["unknown"],
            refine_scope_index=[1],
        )
    assert len(calls) == 3


def test_invalid_public_scope_fails_before_any_execution():
    calls = {"encode": [], "clone": [], "sample": []}
    with pytest.raises(RefineScopeError, match="not present"):
        _run(
            MODE_VIDEO_AUDIO,
            scope=MODE_LOGICAL_CHUNK,
            scope_index=99,
            observers=calls,
        )
    assert calls == {"encode": [], "clone": [], "sample": []}


def test_terminal_logical_halves_resolve_to_the_same_group_seed():
    *_, chunk_2_calls, _chunk_2_result = _run(
        MODE_VIDEO_AUDIO,
        scope=MODE_LOGICAL_CHUNK,
        scope_index=2,
        terminal=True,
    )
    *_, chunk_3_calls, _chunk_3_result = _run(
        MODE_VIDEO_AUDIO,
        scope=MODE_LOGICAL_CHUNK,
        scope_index=3,
        terminal=True,
    )
    assert len(chunk_2_calls["sample"]) == len(chunk_3_calls["sample"]) == 1
    assert chunk_2_calls["sample"][0]["seed"] == chunk_3_calls["sample"][0][
        "seed"
    ]


def test_scope_does_not_extend_assembly_plan_or_run_storage_contract():
    *_all_prefix, all_result = _run(MODE_VIDEO_AUDIO)
    *_scope_prefix, scoped_result = _run(
        MODE_VIDEO_AUDIO,
        scope=MODE_LOGICAL_CHUNK,
        scope_index=2,
    )
    all_contract = all_result[2]["second_pass_contract"]
    scoped_contract = scoped_result[2]["second_pass_contract"]

    assert all_result[2].keys() == scoped_result[2].keys()
    assert all_contract.keys() == scoped_contract.keys()
    assert "refine_scope_contract" not in scoped_contract
    assert "run_storage" not in targeted_second_pass.__dict__
