from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import torch

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumAssembleSeamV35,
)
from ComfyUI_H3_Continuum_Join.v3.plan import _attach_second_pass_contract
from ComfyUI_H3_Continuum_Join.v3.second_pass import (
    SecondPassContractError,
    passthrough_audio_latents,
    update_second_pass_geometry,
    validate_second_pass_inputs,
)
from ComfyUI_H3_Continuum_Join.v3.second_pass_nodes import (
    H3ContinuumSecondPassV35,
)


def _entry(t: int, audio_t: int, *, prompt: str) -> dict:
    return {
        "video": torch.zeros((1, 24, t, 40, 40)),
        "audio": torch.zeros((1, 8, audio_t)),
        "prompt": prompt,
    }


def _terminal_fixture():
    logical = [
        _entry(31, 100, prompt="first"),
        _entry(35, 110, prompt="middle"),
        _entry(35, 110, prompt="last"),
    ]
    physical = [logical[0], _entry(65, 220, prompt="terminal")]
    plan = {
        "width": 640,
        "height": 640,
        "chunks": [
            {"total_frames": 124, "trim_frames": 0},
            {"total_frames": 141, "trim_frames": 22},
            {"total_frames": 141, "trim_frames": 22},
        ],
        "decode_groups": [
            {
                "logical_chunk_indices": [1],
                "terminal_merged": False,
                "total_frames": 124,
                "trim_frames": 0,
            },
            {
                "logical_chunk_indices": [2, 3],
                "terminal_merged": True,
                "total_frames": 260,
                "trim_frames": 22,
            },
        ],
    }
    _attach_second_pass_contract(
        plan,
        logical_entries=logical,
        physical_entries=physical,
        chunk_seconds=5.0,
    )
    videos = [{"samples": entry["video"]} for entry in physical]
    audios = [{"samples": entry["audio"]} for entry in physical]
    return plan, videos, audios


def _external_spatial_processor(video_latents, *, target_h: int, target_w: int):
    return [
        {
            **latent,
            "samples": latent["samples"].new_zeros(
                (*latent["samples"].shape[:3], target_h, target_w)
            ),
        }
        for latent in video_latents
    ]


def _without_target_geometry(plan: dict) -> dict:
    normalized = copy.deepcopy(plan)
    normalized.pop("width", None)
    normalized.pop("height", None)
    contract = normalized["second_pass_contract"]
    contract.pop("target_width", None)
    contract.pop("target_height", None)
    for group in contract["physical_groups"]:
        for key in (
            "target_latent_h",
            "target_latent_w",
            "target_width",
            "target_height",
        ):
            group.pop(key, None)
    return normalized


def test_external_processor_may_change_only_common_spatial_video_geometry():
    plan, videos, audios = _terminal_fixture()
    processed = _external_spatial_processor(videos, target_h=60, target_w=64)

    result = validate_second_pass_inputs(processed, audios, plan)

    assert result == {
        "physical_group_count": 2,
        "target_latent_h": 60,
        "target_latent_w": 64,
        "audio_passthrough": True,
    }
    terminal = plan["second_pass_contract"]["physical_groups"][1]
    assert terminal["logical_chunks"] == [2, 3]
    assert terminal["terminal_merged"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda values: values[:1], "group count"),
        (
            lambda values: [
                {"samples": torch.zeros((24, 31, 40, 40))},
                values[1],
            ],
            r"shape \[B,C,T,H,W\]",
        ),
        (
            lambda values: [
                values[0],
                {"samples": torch.zeros((1, 24, 64, 40, 40))},
            ],
            "temporal length",
        ),
        (
            lambda values: [
                values[0],
                {"samples": torch.zeros((1, 24, 65, 39, 40))},
            ],
            "preserved or enlarged",
        ),
    ),
)
def test_external_processor_rejects_group_temporal_or_downscale_changes(
    mutation,
    message,
):
    plan, videos, audios = _terminal_fixture()
    with pytest.raises(SecondPassContractError, match=message):
        validate_second_pass_inputs(mutation(videos), audios, plan)


@pytest.mark.parametrize(
    ("stream", "bad_value"),
    (("video", float("nan")), ("audio", float("inf"))),
)
def test_external_processor_rejects_nonfinite_latents_before_sampling(
    stream,
    bad_value,
):
    plan, videos, audios = _terminal_fixture()
    target = videos if stream == "video" else audios
    target[0]["samples"].view(-1)[0] = bad_value

    with pytest.raises(SecondPassContractError, match="contains NaN or Inf"):
        validate_second_pass_inputs(videos, audios, plan)


def test_external_processor_rejects_changed_audio_shape():
    plan, videos, audios = _terminal_fixture()
    audios[1] = {"samples": torch.zeros((1, 8, 219))}

    with pytest.raises(SecondPassContractError, match="audio latent shape"):
        validate_second_pass_inputs(videos, audios, plan)


def test_first_pass_audio_objects_and_input_plan_are_not_mutated():
    plan, videos, audios = _terminal_fixture()
    before = copy.deepcopy(plan)
    processed = _external_spatial_processor(videos, target_h=60, target_w=60)

    output_audio = passthrough_audio_latents(audios)
    updated = update_second_pass_geometry(plan, processed)

    assert plan == before
    assert updated is not plan
    assert all(left is right for left, right in zip(audios, output_audio, strict=True))
    assert _without_target_geometry(updated) == _without_target_geometry(plan)


def test_public_second_pass_and_finalize_keep_existing_comfy_type_boundary():
    second_pass = H3ContinuumSecondPassV35.INPUT_TYPES()
    finalize = H3ContinuumAssembleSeamV35.INPUT_TYPES()

    assert tuple(second_pass["required"]) == (
        "model",
        "clip",
        "sampler",
        "sigmas",
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "refine_seed",
    )
    assert second_pass["required"]["video_latents"] == ("LATENT",)
    assert second_pass["required"]["audio_latents"] == ("LATENT",)
    assert tuple(second_pass["optional"]) == ("refine_context", "video_vae")
    assert finalize["required"]["images"] == ("IMAGE",)
    assert finalize["required"]["audio"] == ("AUDIO",)
    assert "video_vae" not in finalize["required"]
    assert "audio_vae" not in finalize["required"]


def test_integration_modules_do_not_import_optional_accelerators_or_run_storage():
    root = Path(__file__).resolve().parents[1]
    forbidden_accelerators = {"sageattention", "sol_attn", "spectrum", "tensorrt"}
    modules = (
        root / "compatibility.py",
        root / "model_patch.py",
        root / "v3" / "second_pass.py",
    )

    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(
                    alias.name.split(".", 1)[0].lower() for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0].lower())
        assert imports.isdisjoint(forbidden_accelerators)

    second_pass_tree = ast.parse(
        (root / "v3" / "second_pass.py").read_text(encoding="utf-8")
    )
    imported_names = set()
    for node in ast.walk(second_pass_tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.lower())
    assert not any("run_storage" in name for name in imported_names)
