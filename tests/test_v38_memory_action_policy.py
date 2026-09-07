from __future__ import annotations

import ast
from pathlib import Path

import torch

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.reference import (
    REFERENCE_SIZE_MATCH_OUTPUT,
    REFERENCE_SIZE_MAX_IDENTITY,
    combine_hybrid_visual_identity,
    prepare_reference_assets,
)
from ComfyUI_H3_Continuum_Join.reference_video import (
    REFERENCE_VIDEO_SIZE_BALANCED,
    REFERENCE_VIDEO_SIZE_EFFICIENT,
    REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
)
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumMemoryActionPolicyExperimental,
    H3ContinuumSamplerV38,
    H3ContinuumSamplerV38MemoryPolicyExperimental,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from ComfyUI_H3_Continuum_Join.v3.memory_action_policy import (
    ACTION_DISABLED,
    ACTION_REDUCE_REFERENCES_ONE_STEP,
    MEMORY_ACTION_POLICY_VERSION,
    format_memory_action_status,
    make_memory_action_policy,
    resolve_memory_action_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def _resolve(policy, *, image=True, video=True, image_mode=None, video_mode=None):
    return resolve_memory_action_policy(
        policy,
        image_mode=image_mode or REFERENCE_SIZE_MAX_IDENTITY,
        video_mode=video_mode or REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
        has_reference_images=image,
        has_video_guide=video,
    )


def test_policy_contract_is_versioned_deterministic_and_tamper_evident():
    first = make_memory_action_policy(ACTION_REDUCE_REFERENCES_ONE_STEP)
    second = make_memory_action_policy(ACTION_REDUCE_REFERENCES_ONE_STEP)
    assert first == second
    assert first["memory_action_policy_version"] == MEMORY_ACTION_POLICY_VERSION
    assert len(first["contract_hash"]) == 64
    assert first["auto_unload"] == "advisory_only"
    assert first["backend_switch"] == "advisory_only"
    assert first["same_run_actual_row_trigger"] is False

    modified = dict(first, action="unsupported")
    decision = _resolve(modified)
    assert decision.enabled is False
    assert decision.warning is not None
    assert decision.effective_image_mode == REFERENCE_SIZE_MAX_IDENTITY
    assert decision.effective_video_mode == REFERENCE_VIDEO_SIZE_MATCH_OUTPUT


def test_disabled_and_disconnected_are_complete_resolver_noops():
    disconnected = _resolve(None)
    disabled = _resolve(make_memory_action_policy(ACTION_DISABLED))
    for decision in (disconnected, disabled):
        assert decision.enabled is False
        assert decision.applied is False
        assert decision.effective_image_mode == REFERENCE_SIZE_MAX_IDENTITY
        assert decision.effective_video_mode == REFERENCE_VIDEO_SIZE_MATCH_OUTPUT
        assert format_memory_action_status(decision) is None


def test_one_step_reference_resolver_uses_only_existing_presets():
    policy = make_memory_action_policy(ACTION_REDUCE_REFERENCES_ONE_STEP)
    first = _resolve(policy)
    assert first.effective_image_mode == REFERENCE_SIZE_MATCH_OUTPUT
    assert first.effective_video_mode == REFERENCE_VIDEO_SIZE_BALANCED
    assert first.image_applied is True
    assert first.video_applied is True

    second = _resolve(
        policy,
        image_mode=REFERENCE_SIZE_MATCH_OUTPUT,
        video_mode=REFERENCE_VIDEO_SIZE_BALANCED,
    )
    assert second.effective_image_mode == REFERENCE_SIZE_MATCH_OUTPUT
    assert second.effective_video_mode == REFERENCE_VIDEO_SIZE_EFFICIENT
    assert second.image_applied is False
    assert second.video_applied is True

    minimum = _resolve(
        policy,
        image_mode=REFERENCE_SIZE_MATCH_OUTPUT,
        video_mode=REFERENCE_VIDEO_SIZE_EFFICIENT,
    )
    assert minimum.applied is False


def test_one_step_policy_changes_only_connected_reference_kinds():
    policy = make_memory_action_policy(ACTION_REDUCE_REFERENCES_ONE_STEP)
    image_only = _resolve(policy, video=False)
    assert image_only.image_applied is True
    assert image_only.video_applied is False
    assert image_only.effective_video_mode == REFERENCE_VIDEO_SIZE_MATCH_OUTPUT

    video_only = _resolve(policy, image=False)
    assert video_only.image_applied is False
    assert video_only.video_applied is True
    assert video_only.effective_image_mode == REFERENCE_SIZE_MAX_IDENTITY


def test_effective_reference_contract_changes_existing_run_storage_identity():
    image = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
    original = prepare_reference_assets(
        reference_image_1=image,
        reference_image_2=None,
        output_width=32,
        output_height=32,
        size_mode=REFERENCE_SIZE_MAX_IDENTITY,
    )
    downgraded = prepare_reference_assets(
        reference_image_1=image,
        reference_image_2=None,
        output_width=32,
        output_height=32,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )
    assert original is not None and downgraded is not None
    assert original.contract["size_mode"] == REFERENCE_SIZE_MAX_IDENTITY
    assert downgraded.contract["size_mode"] == REFERENCE_SIZE_MATCH_OUTPUT
    assert original.combined_hash != downgraded.combined_hash
    base = "a" * 64
    assert combine_hybrid_visual_identity(
        keyframe_identity_hash=base,
        reference_assets=original,
        has_first=False,
        has_last=False,
    ) != combine_hybrid_visual_identity(
        keyframe_identity_hash=base,
        reference_assets=downgraded,
        has_first=False,
        has_last=False,
    )


def test_existing_v38_schema_is_unchanged_and_experimental_input_is_additive():
    production = H3ContinuumSamplerV38.INPUT_TYPES()
    experimental = H3ContinuumSamplerV38MemoryPolicyExperimental.INPUT_TYPES()
    assert "memory_action_policy" not in production.get("required", {})
    assert "memory_action_policy" not in production.get("optional", {})
    assert production["required"] == experimental["required"]
    expected_optional = dict(production.get("optional", {}))
    actual_optional = dict(experimental.get("optional", {}))
    policy_input = actual_optional.pop("memory_action_policy")
    assert actual_optional == expected_optional
    assert policy_input[0] == "H3_CONTINUUM_MEMORY_ACTION_POLICY"


def test_nodes_are_additively_registered_under_advanced_category():
    policy_id = "H3ContinuumMemoryActionPolicyExperimental"
    sampler_id = "H3ContinuumSamplerV38MemoryPolicyExperimental"
    assert NODE_CLASS_MAPPINGS[policy_id] is H3ContinuumMemoryActionPolicyExperimental
    assert NODE_CLASS_MAPPINGS[sampler_id] is (
        H3ContinuumSamplerV38MemoryPolicyExperimental
    )
    assert policy_id not in root_nodes.NODE_CLASS_MAPPINGS
    assert sampler_id not in root_nodes.NODE_CLASS_MAPPINGS
    assert NODE_DISPLAY_NAME_MAPPINGS[policy_id].endswith("(Experimental)")
    assert NODE_DISPLAY_NAME_MAPPINGS[sampler_id].endswith("(Experimental)")
    assert H3ContinuumMemoryActionPolicyExperimental.CATEGORY.endswith("/Advanced")
    assert H3ContinuumSamplerV38MemoryPolicyExperimental.CATEGORY.endswith(
        "/Advanced"
    )


def test_experimental_disabled_delegates_exact_kwargs_and_output(monkeypatch):
    captured = []
    sentinel = object()

    def fake_run(self, **kwargs):
        captured.append(kwargs)
        return sentinel

    monkeypatch.setattr(H3ContinuumSamplerV38, "run", fake_run)
    node = H3ContinuumSamplerV38MemoryPolicyExperimental()
    kwargs = {
        "reference_size": REFERENCE_SIZE_MAX_IDENTITY,
        "video_reference_size": REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
        "reference_image_1": object(),
        "reference_video_1": object(),
        "base_seed": 123,
        "sigmas": object(),
    }
    assert node.run(**kwargs) is sentinel
    assert captured[-1] == kwargs
    assert node.run(
        memory_action_policy=make_memory_action_policy(ACTION_DISABLED),
        **kwargs,
    ) is sentinel
    assert captured[-1] == kwargs


def test_experimental_active_changes_only_effective_reference_modes(monkeypatch):
    captured = {}
    video = object()
    audio = object()
    plan = {"groups": [[1]]}

    def fake_run(self, **kwargs):
        captured.update(kwargs)
        return (video, audio, plan, "Production status", "driving")

    monkeypatch.setattr(H3ContinuumSamplerV38, "run", fake_run)
    seed = 42
    sigmas = object()
    result = H3ContinuumSamplerV38MemoryPolicyExperimental().run(
        memory_action_policy=make_memory_action_policy(
            ACTION_REDUCE_REFERENCES_ONE_STEP
        ),
        reference_size=REFERENCE_SIZE_MAX_IDENTITY,
        video_reference_size=REFERENCE_VIDEO_SIZE_BALANCED,
        reference_image_1=object(),
        reference_video_1=object(),
        base_seed=seed,
        sigmas=sigmas,
    )
    assert captured["reference_size"] == REFERENCE_SIZE_MATCH_OUTPUT
    assert captured["video_reference_size"] == REFERENCE_VIDEO_SIZE_EFFICIENT
    assert captured["base_seed"] == seed
    assert captured["sigmas"] is sigmas
    assert result[:3] == (video, audio, plan)
    assert result[4] == "driving"
    assert "automatic MODEL unload was not executed" in result[3]
    assert "external MODEL wrapper was preserved" in result[3]


def test_policy_code_has_no_runtime_unload_or_backend_switch_calls():
    called_names = set()
    for path in (ROOT / "v3" / "memory_action_policy.py", ROOT / "v3" / "driving_nodes.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)
    assert called_names.isdisjoint(
        {
            "unload_all_models",
            "unload_model_and_clones",
            "soft_empty_cache",
            "optimized_attention_override",
        }
    )
