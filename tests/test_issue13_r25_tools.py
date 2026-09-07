from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch


TOOLS = Path(__file__).parents[1] / "tools"


def _load(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("h3_issue13_r25_gate_runner", "issue13_r25_gate_runner.py")
analyzer = _load("h3_issue13_r25_analyze", "issue13_r2_analyze.py")
soft_reanchor = _load(
    "h3_issue13_soft_reanchor",
    "gpu_diagnostic_node/soft_reanchor.py",
)


def _source_prompt() -> dict:
    return {
        "114": {"inputs": {"image": "old.png"}},
        "119": {"inputs": {}},
        "139": {"inputs": {"sampler_name": "res_multistep"}},
        "150": {"inputs": {"enabled": True}},
        "153": {"inputs": {"scheduler": "old", "steps": 20, "denoise": 0.5}},
        "188": {"inputs": {"value": "old"}},
        "191": {"inputs": {}},
        "249": {"inputs": {}},
        "286": {
            "inputs": {
                "lora_1": {"on": True, "lora": "old.safetensors", "strength": 1.0}
            }
        },
        "297": {"inputs": {"value": 0.3}},
        "305": {
            "class_type": "H3ContinuumSamplerV37",
            "inputs": {
                "continuation_backend": "Standard",
                "generation_mode": "Full Run",
                "review_action": "Continue / Next",
                "guide": ["900", 0],
                "last_frame": ["901", 0],
            },
        },
        "306": {"inputs": {}},
    }


def test_screening_order_and_contract_are_fixed():
    assert tuple(runner.CASE_SPECS)[:3] == (
        "screen_3x5_turbo",
        "screen_3x10_turbo",
        "screen_6x5_turbo",
    )
    prompt, loras = runner.configure_prompt(
        _source_prompt(),
        case="screen_3x10_turbo",
        output_prefix="video/test",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=7,
    )
    inputs = prompt["305"]["inputs"]
    assert inputs["chunks"] == 3
    assert inputs["chunk_seconds"] == 10.0
    assert inputs["width"] == 576
    assert inputs["height"] == 864
    assert prompt["139"]["inputs"]["sampler_name"] == "euler"
    assert prompt["153"]["inputs"]["steps"] == 8
    assert prompt["153"]["inputs"]["scheduler"] == "simple"
    assert prompt["114"]["inputs"]["image"] == "closeup.png"
    assert loras == [{"name": runner.TURBO_LORA, "strength": 1.0}]
    assert prompt["188"]["inputs"]["value"].count("\n---\n") == 2
    assert "dynv2" not in prompt["188"]["inputs"]["value"]


def test_six_by_five_uses_six_motion_steps():
    prompt, _ = runner.configure_prompt(
        _source_prompt(),
        case="screen_6x5_turbo",
        output_prefix="video/test",
        image_name="closeup.png",
        seed=1,
        width=576,
        height=864,
        diagnostic_nonce=8,
    )
    assert prompt["305"]["inputs"]["chunks"] == 6
    assert prompt["188"]["inputs"]["value"].count("\n---\n") == 5


def test_six_by_five_fixed_prompt_repeats_one_prompt():
    prompt, loras = runner.configure_prompt(
        _source_prompt(),
        case="screen_6x5_turbo_fixed_prompt",
        output_prefix="video/issue13_r26/fixed",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132600,
    )
    prompts = prompt["188"]["inputs"]["value"].split("\n---\n")
    assert len(prompts) == 6
    assert len(set(prompts)) == 1
    assert prompt["305"]["inputs"]["chunks"] == 6
    assert prompt["305"]["inputs"]["chunk_seconds"] == 5.0
    assert prompt["153"]["inputs"]["steps"] == 8
    assert loras == [{"name": runner.TURBO_LORA, "strength": 1.0}]


def test_six_by_five_fixed_prefix_changes_only_diagnostic_source_mode():
    source = _source_prompt()
    recursive, _ = runner.configure_prompt(
        source,
        case="screen_6x5_turbo_fixed_prompt",
        output_prefix="video/issue13_r26/recursive",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132600,
    )
    fixed, _ = runner.configure_prompt(
        source,
        case="screen_6x5_turbo_fixed_prefix",
        output_prefix="video/issue13_r26/fixed-prefix",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132601,
    )
    recursive_inputs = recursive["305"]["inputs"]
    fixed_inputs = fixed["305"]["inputs"]
    assert recursive_inputs["continuation_source_mode"] == "Recursive"
    assert fixed_inputs["continuation_source_mode"] == "Fixed Group 1 Tail"
    ignored = {"continuation_source_mode", "diagnostic_nonce"}
    assert {
        key: value for key, value in recursive_inputs.items() if key not in ignored
    } == {key: value for key, value in fixed_inputs.items() if key not in ignored}
    assert recursive["188"]["inputs"]["value"] == fixed["188"]["inputs"]["value"]


def test_six_by_five_soft_reanchor_changes_only_diagnostic_source_mode():
    source = _source_prompt()
    recursive, _ = runner.configure_prompt(
        source,
        case="screen_6x5_turbo_fixed_prompt",
        output_prefix="video/issue13_r27/recursive",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132700,
    )
    reanchored, _ = runner.configure_prompt(
        source,
        case="screen_6x5_turbo_soft_reanchor",
        output_prefix="video/issue13_r27/soft-reanchor",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132701,
    )
    recursive_inputs = recursive["305"]["inputs"]
    reanchored_inputs = reanchored["305"]["inputs"]
    assert recursive_inputs["continuation_source_mode"] == "Recursive"
    assert reanchored_inputs["continuation_source_mode"] == "Soft Re-anchor Group 4"
    ignored = {"continuation_source_mode", "diagnostic_nonce"}
    assert {
        key: value for key, value in recursive_inputs.items() if key not in ignored
    } == {key: value for key, value in reanchored_inputs.items() if key not in ignored}
    assert recursive["188"]["inputs"]["value"] == reanchored["188"]["inputs"]["value"]


def test_soft_reanchor_moves_global_appearance_halfway_without_geometry_change():
    current = torch.zeros((1, 4, 8, 8, 3), dtype=torch.float32)
    current[..., 0] = 0.8
    current[..., 1] = 0.3
    current[..., 2] = 0.2
    reference = torch.zeros_like(current)
    reference[..., 0] = 0.4
    reference[..., 1] = 0.5
    reference[..., 2] = 0.6
    adjusted, report = soft_reanchor.soft_match_rgb_appearance(
        current,
        reference,
        strength=0.5,
    )
    assert adjusted.shape == current.shape
    assert torch.equal(current[..., 0], torch.full_like(current[..., 0], 0.8))
    before_gap = abs(
        report["before"]["luma_mean"] - report["reference"]["luma_mean"]
    )
    after_gap = abs(
        report["after"]["luma_mean"] - report["reference"]["luma_mean"]
    )
    assert after_gap < before_gap
    assert report["strength"] == 0.5
    assert report["geometry_unchanged"] is True


def test_soft_reanchor_vae_roundtrip_preserves_shape_and_inputs():
    class FakeVideoVAE:
        def decode(self, latent):
            value = float(latent.mean().item())
            pixels = torch.full((1, 22, 32, 32, 3), value, dtype=torch.float32)
            pixels[..., 0] = value + 0.1
            return pixels.clamp(0.0, 1.0)

        def encode(self, pixels):
            value = float(pixels.mean().item())
            return torch.full((1, 24, 7, 2, 2), value, dtype=torch.float32)

    current = torch.full((1, 24, 7, 2, 2), 0.7, dtype=torch.float32)
    reference = torch.full((1, 24, 7, 2, 2), 0.3, dtype=torch.float32)
    current_before = current.clone()
    reference_before = reference.clone()

    encoded, report = soft_reanchor.soft_reanchor_context(
        current,
        reference,
        video_vae=FakeVideoVAE(),
        strength=0.5,
    )

    assert encoded.shape == current.shape
    assert encoded.dtype == current.dtype
    assert torch.isfinite(encoded).all()
    assert torch.equal(current, current_before)
    assert torch.equal(reference, reference_before)
    assert report["current_context_unchanged"] is True
    assert report["reference_context_unchanged"] is True
    assert report["latent_shape_unchanged"] is True


def test_boundary_metrics_report_flow_and_frame_jump():
    frames = []
    for index in range(12):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[:, max(0, min(20, index)) : max(1, min(24, index + 4))] = 80
        if index >= 6:
            frame = np.clip(frame + 80, 0, 255).astype(np.uint8)
        frames.append(frame)
    rows = analyzer._boundary_metrics(frames, chunks=2)
    assert len(rows) == 1
    assert rows[0]["target_group"] == 2
    assert rows[0]["frame_mae"] > 0.0
    assert rows[0]["trajectory_flow_discontinuity"] >= 0.0


def test_motion_pair_ab_uses_normal_twenty_step_and_no_turbo():
    off, off_loras = runner.configure_prompt(
        _source_prompt(),
        case="motion_off_3x5_normal",
        output_prefix="video/test",
        image_name="closeup.png",
        seed=1,
        width=576,
        height=864,
        diagnostic_nonce=9,
    )
    both, both_loras = runner.configure_prompt(
        _source_prompt(),
        case="motion_both_3x5_normal",
        output_prefix="video/test",
        image_name="closeup.png",
        seed=1,
        width=576,
        height=864,
        diagnostic_nonce=10,
    )
    assert off_loras == []
    assert off["153"]["inputs"]["steps"] == 20
    assert "dynv2" in off["188"]["inputs"]["value"]
    assert [item["name"] for item in both_loras] == [
        runner.MOTION_BOOSTER_LORA,
        runner.MOTION_REPAIR_LORA,
    ]
    assert both["153"]["inputs"]["steps"] == 20
    assert all(runner.TURBO_LORA != item["name"] for item in both_loras)


def test_continuation_rgb_proxy_uses_pre_boundary_context_window():
    frames = [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(30)]
    result = analyzer._continuation_input_rgb_proxy(
        frames, chunks=3, context_frames=4
    )
    assert result["context_frames"] == 4
    assert len(result["boundaries"]) == 2
    first = result["boundaries"][0]
    assert first["boundary_frame"] == 10
    assert first["window_start"] == 6
    assert first["window_end_exclusive"] == 10
    assert first["window_frames"] == 4
    assert abs(sum(first["luma_histogram"]) - 1.0) < 1e-9
