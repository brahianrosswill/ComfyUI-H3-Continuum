from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3 import second_pass_nodes
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV38
from ComfyUI_H3_Continuum_Join.v3.refine_target import (
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass_nodes import (
    H3ContinuumSecondPassV35,
    H3ContinuumSelectiveSecondPassExperimental,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    REFINE_TARGET_AUDIO_ONLY,
    REFINE_TARGET_OPTIONS,
    REFINE_TARGET_VIDEO_AUDIO,
    REFINE_TARGET_VIDEO_ONLY,
    REFINE_SCOPE_ALL,
    REFINE_SCOPE_LOGICAL_CHUNK,
    REFINE_SCOPE_OPTIONS,
    REFINE_SCOPE_PHYSICAL_GROUP,
    REFINE_SCOPE_TIME_WINDOW,
)


_V35_SCHEMA_BEFORE_SELECTIVE_NODE = copy.deepcopy(
    H3ContinuumSecondPassV35.INPUT_TYPES()
)
_V38_SAMPLER_SCHEMA_BEFORE_SELECTIVE_NODE = copy.deepcopy(
    H3ContinuumSamplerV38.INPUT_TYPES()
)


def _node_arguments(target: str, *, video_latents=None, audio_latents=None, plan=None):
    return {
        "model": [object()],
        "clip": [object()],
        "sampler": [object()],
        "sigmas": [object()],
        "video_latents": video_latents or [{"samples": object()}],
        "audio_latents": audio_latents or [{"samples": object()}],
        "assembly_plan": [plan or {"second_pass_contract": {"version": 1}}],
        "refine_seed": [73],
        "refine_target": [target],
    }


def test_selective_node_public_schema_registration_and_display_contract():
    node_id = "H3ContinuumSelectiveSecondPassExperimental"
    schema = H3ContinuumSelectiveSecondPassExperimental.INPUT_TYPES()

    assert NODE_CLASS_MAPPINGS[node_id] is H3ContinuumSelectiveSecondPassExperimental
    assert node_id not in root_nodes.NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS[node_id] == (
        "H3 Continuum Selective Second Pass (Experimental)"
    )
    assert node_id not in root_nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert H3ContinuumSelectiveSecondPassExperimental.DEPRECATED is False
    assert H3ContinuumSelectiveSecondPassExperimental.CATEGORY == (
        "MiniMax H3/Continuum/Advanced"
    )
    assert tuple(schema["required"]) == (
        "model",
        "clip",
        "sampler",
        "sigmas",
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "refine_seed",
        "refine_target",
        "refine_scope",
        "refine_scope_index",
        "window_start_sec",
        "window_end_sec",
    )
    assert tuple(schema["optional"]) == ("refine_context", "video_vae")
    assert schema["required"]["refine_target"] == (
        REFINE_TARGET_OPTIONS,
        {"default": REFINE_TARGET_VIDEO_ONLY},
    )
    assert REFINE_TARGET_OPTIONS == (
        "Video Only",
        "Audio Only",
        "Video + Audio",
    )
    assert schema["required"]["refine_scope"] == (
        REFINE_SCOPE_OPTIONS,
        {"default": REFINE_SCOPE_ALL},
    )
    assert REFINE_SCOPE_OPTIONS == (
        "All",
        "Physical Group",
        "Logical Chunk",
        REFINE_SCOPE_TIME_WINDOW,
    )
    assert schema["required"]["refine_scope_index"][0] == "INT"
    assert schema["required"]["refine_scope_index"][1]["default"] == 1
    assert schema["required"]["window_start_sec"][1]["default"] == 0.0
    assert schema["required"]["window_end_sec"][1]["default"] == 5.0
    assert H3ContinuumSelectiveSecondPassExperimental.INPUT_IS_LIST is True
    assert H3ContinuumSelectiveSecondPassExperimental.RETURN_TYPES == (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
    )
    assert H3ContinuumSelectiveSecondPassExperimental.RETURN_NAMES == (
        "refined_video_latents",
        "audio_latents",
        "updated_assembly_plan",
        "status",
    )
    assert H3ContinuumSelectiveSecondPassExperimental.OUTPUT_IS_LIST == (
        True,
        True,
        False,
        False,
    )
    assert "Experimental" in H3ContinuumSelectiveSecondPassExperimental.DESCRIPTION
    assert "denoise-mask" in H3ContinuumSelectiveSecondPassExperimental.DESCRIPTION


def test_a3c_saved_workflow_loads_missing_a4_tail_widgets_as_all_defaults():
    fixture_path = (
        Path(__file__).parent
        / "fixtures"
        / "v38_a3c_selective_refine_saved_workflow.json"
    )
    workflow = json.loads(fixture_path.read_text(encoding="utf-8"))
    saved = workflow["nodes"][0]
    assert saved["type"] == "H3ContinuumSelectiveSecondPassExperimental"
    assert saved["widgets_values"] == [73, REFINE_TARGET_VIDEO_AUDIO]

    schema = H3ContinuumSelectiveSecondPassExperimental.INPUT_TYPES()["required"]
    widget_names = (
        "refine_seed",
        "refine_target",
        "refine_scope",
        "refine_scope_index",
        "window_start_sec",
        "window_end_sec",
    )
    restored = {}
    for index, name in enumerate(widget_names):
        if index < len(saved["widgets_values"]):
            restored[name] = saved["widgets_values"][index]
        else:
            restored[name] = schema[name][1]["default"]

    assert restored == {
        "refine_seed": 73,
        "refine_target": REFINE_TARGET_VIDEO_AUDIO,
        "refine_scope": REFINE_SCOPE_ALL,
        "refine_scope_index": 1,
        "window_start_sec": 0.0,
        "window_end_sec": 5.0,
    }


def test_selective_schema_does_not_mutate_v35_or_v38_sampler_schema():
    H3ContinuumSelectiveSecondPassExperimental.INPUT_TYPES()
    assert H3ContinuumSecondPassV35.INPUT_TYPES() == _V35_SCHEMA_BEFORE_SELECTIVE_NODE
    assert H3ContinuumSamplerV38.INPUT_TYPES() == (
        _V38_SAMPLER_SCHEMA_BEFORE_SELECTIVE_NODE
    )


@pytest.mark.parametrize(
    ("label", "expected_mode", "status_fragment"),
    (
        (REFINE_TARGET_VIDEO_ONLY, MODE_VIDEO_ONLY, "Audio=bit-exact passthrough"),
        (
            REFINE_TARGET_AUDIO_ONLY,
            MODE_AUDIO_ONLY,
            "Experimental / Production HOLD",
        ),
        (
            REFINE_TARGET_VIDEO_AUDIO,
            MODE_VIDEO_AUDIO,
            "GPU Experimental PASS / Production HOLD",
        ),
    ),
)
def test_selective_node_resolves_each_ui_label_to_canonical_target(
    monkeypatch,
    label,
    expected_mode,
    status_fragment,
):
    captured = {}
    output_videos = [{"video": label}]
    output_audios = [{"audio": label}]
    output_plan = {"plan": label}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return output_videos, output_audios, output_plan, "internal status"

    monkeypatch.setattr(second_pass_nodes, "run_targeted_second_pass_groups", fake_run)
    result = H3ContinuumSelectiveSecondPassExperimental().refine(
        **_node_arguments(label)
    )

    assert captured["refine_target"] == expected_mode
    assert result[:3] == (output_videos, output_audios, output_plan)
    assert result[0] is output_videos
    assert result[1] is output_audios
    assert result[2] is output_plan
    assert result[3].startswith(f"Selective Refine: {label};")
    assert status_fragment in result[3]
    assert result[3].endswith("internal status")


def test_selective_node_rejects_unknown_target_without_silent_fallback(monkeypatch):
    called = False

    def fake_run(**_kwargs):
        nonlocal called
        called = True
        raise AssertionError("targeted runner must not be called")

    monkeypatch.setattr(second_pass_nodes, "run_targeted_second_pass_groups", fake_run)
    with pytest.raises(ValueError, match="unsupported selective refine target"):
        H3ContinuumSelectiveSecondPassExperimental().refine(
            **_node_arguments("unknown")
        )
    assert called is False


def test_video_only_adapter_matches_v35_inputs_and_preserves_output_objects(monkeypatch):
    video_output = [{"samples": object()}]
    audio_output = [{"samples": object()}]
    plan_output = {"second_pass_contract": {"version": 1}}
    legacy_calls = []
    targeted_calls = []

    def result_for(kwargs, calls):
        calls.append(kwargs)
        return video_output, audio_output, plan_output, "legacy status"

    monkeypatch.setattr(
        second_pass_nodes,
        "run_second_pass_groups",
        lambda **kwargs: result_for(kwargs, legacy_calls),
    )
    monkeypatch.setattr(
        second_pass_nodes,
        "run_targeted_second_pass_groups",
        lambda **kwargs: result_for(kwargs, targeted_calls),
    )
    shared = _node_arguments(REFINE_TARGET_VIDEO_ONLY)
    legacy_result = H3ContinuumSecondPassV35().refine(
        **{key: value for key, value in shared.items() if key != "refine_target"}
    )
    selective_result = H3ContinuumSelectiveSecondPassExperimental().refine(**shared)

    assert legacy_result[:3] == selective_result[:3]
    assert all(
        selective_result[index] is legacy_result[index] for index in range(3)
    )
    assert audio_output[0] is selective_result[1][0]
    assert targeted_calls[0]["refine_target"] == MODE_VIDEO_ONLY
    for name in (
        "model",
        "clip",
        "sampler",
        "sigmas",
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "refine_seed",
        "refine_context",
        "video_vae",
        "refine_schedule",
    ):
        assert targeted_calls[0][name] is legacy_calls[0][name] or (
            targeted_calls[0][name] == legacy_calls[0][name]
        )


@pytest.mark.parametrize(
    ("label", "mode", "video_is_source", "audio_is_source"),
    (
        (REFINE_TARGET_AUDIO_ONLY, MODE_AUDIO_ONLY, True, False),
        (REFINE_TARGET_VIDEO_AUDIO, MODE_VIDEO_AUDIO, False, False),
    ),
)
def test_experimental_target_outputs_are_forwarded_without_adapter_copies(
    monkeypatch,
    label,
    mode,
    video_is_source,
    audio_is_source,
):
    source_videos = [{"samples": object()}]
    source_audios = [{"samples": object()}]
    output_videos = source_videos if video_is_source else [{"samples": object()}]
    output_audios = source_audios if audio_is_source else [{"samples": object()}]
    output_plan = {"second_pass_contract": {"version": 1}}

    def fake_run(**kwargs):
        assert kwargs["refine_target"] == mode
        return output_videos, output_audios, output_plan, "target status"

    monkeypatch.setattr(second_pass_nodes, "run_targeted_second_pass_groups", fake_run)
    result = H3ContinuumSelectiveSecondPassExperimental().refine(
        **_node_arguments(
            label,
            video_latents=source_videos,
            audio_latents=source_audios,
        )
    )

    assert result[0] is output_videos
    assert result[1] is output_audios
    assert result[2] is output_plan
    assert (result[0] is source_videos) is video_is_source
    assert (result[1] is source_audios) is audio_is_source


def test_terminal_physical_group_lists_remain_atomic_through_public_adapter(
    monkeypatch,
):
    videos = [{"group": 1}, {"group": "terminal-2-3"}]
    audios = [{"group": 1}, {"group": "terminal-2-3"}]
    plan = {
        "second_pass_contract": {
            "version": 1,
            "physical_groups": [
                {"logical_chunks": [1], "terminal_merged": False},
                {"logical_chunks": [2, 3], "terminal_merged": True},
            ],
        }
    }

    def fake_run(**kwargs):
        assert kwargs["video_latents"] is videos
        assert kwargs["audio_latents"] is audios
        assert kwargs["assembly_plan"] is plan
        assert len(kwargs["video_latents"]) == 2
        assert plan["second_pass_contract"]["physical_groups"][1][
            "logical_chunks"
        ] == [2, 3]
        return videos, audios, plan, "terminal atomic"

    monkeypatch.setattr(second_pass_nodes, "run_targeted_second_pass_groups", fake_run)
    result = H3ContinuumSelectiveSecondPassExperimental().refine(
        **_node_arguments(
            REFINE_TARGET_VIDEO_AUDIO,
            video_latents=videos,
            audio_latents=audios,
            plan=plan,
        )
    )

    assert result[0] is videos
    assert result[1] is audios
    assert result[2] is plan
    assert "physical groups=2" in result[3]
