from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


TOOLS = Path(__file__).parents[1] / "tools"


def _load(name: str, filename: str):
    path = TOOLS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load("h3_issue13_r2_gate_runner", "issue13_r2_gate_runner.py")
analyzer = _load("h3_issue13_r2_analyze", "issue13_r2_analyze.py")


def _source_prompt() -> dict:
    return {
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
        "188": {"inputs": {"value": "old"}},
        "153": {"inputs": {"scheduler": "old", "steps": 20, "denoise": 0.5}},
        "297": {"inputs": {"value": 0.3}},
        "306": {"inputs": {}},
        "191": {"inputs": {}},
        "249": {"inputs": {}},
        "150": {"inputs": {"enabled": True}},
        "286": {
            "inputs": {
                "lora_1": {"on": True, "lora": "turbo.safetensors", "strength": 1.0}
            }
        },
    }


def test_g1_configures_reported_six_by_ten_masked_av_baseline():
    prompt = runner.configure_prompt(
        _source_prompt(),
        case="g1_standard_6x10",
        output_prefix="video/test",
        seed=130013,
        steps=8,
        size=384,
        diagnostic_nonce=7,
    )
    sampler = prompt["305"]
    inputs = sampler["inputs"]
    assert sampler["class_type"] == "H3ContinuumMaskedPrefixR1Diagnostic"
    assert inputs["chunks"] == 6
    assert inputs["chunk_seconds"] == 10.0
    assert inputs["continuation_transport"] == "masked_av_prefix_22_v1"
    assert inputs["continuity"] == "Balanced — 22 frames"
    assert inputs["audio_continuity"] is True
    assert inputs["run_storage"] == "Off"
    assert "continuation_backend" not in inputs
    assert "guide" not in inputs
    assert "last_frame" not in inputs
    assert prompt["188"]["inputs"]["value"].count("\n---\n") == 5
    assert prompt["150"]["inputs"]["enabled"] is False
    assert prompt["286"]["inputs"]["lora_1"]["on"] is False
    assert prompt["306"]["inputs"]["video_seam"] == "Off"
    assert prompt["306"]["inputs"]["audio_seam"] == "Off"


def test_g2_keeps_old_five_by_five_comparison_shape():
    prompt = runner.configure_prompt(
        _source_prompt(),
        case="g2_standard_5x5",
        output_prefix="video/test",
        seed=130013,
        steps=8,
        size=384,
        diagnostic_nonce=8,
    )
    inputs = prompt["305"]["inputs"]
    assert inputs["chunks"] == 5
    assert inputs["chunk_seconds"] == 5.0
    assert prompt["188"]["inputs"]["value"].count("\n---\n") == 4


def test_diagnostic_payload_is_extracted_from_status_suffix():
    assert runner._diagnostic_from_status(
        'Production status\nV3.6-R1 diagnostic: {"transport":"masked_av_prefix_22_v1"}'
    ) == {"transport": "masked_av_prefix_22_v1"}


def test_decoded_metrics_include_local_sharpness():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, 16:] = 255
    metrics = analyzer._frame_metrics(frame)
    assert metrics["contrast_p95_p05"] == 1.0
    assert metrics["sharpness_laplacian_variance"] > 0.0
    assert metrics["gradient_mean"] > 0.0


def test_latent_trend_requires_bit_exact_continuation_prefixes():
    diagnostic = {
        "sample_calls": [
            {"sample_number": 1, "output_tail": {"rms": 1.0, "std": 0.5}},
            {
                "sample_number": 2,
                "source_tail": {"rms": 1.0, "std": 0.5},
                "output_tail": {"rms": 1.2, "std": 0.7},
            },
        ],
        "final_prefix_pairs": [
            {
                "bit_exact": True,
                "audio_bit_exact": True,
                "max_abs_diff": 0.0,
                "audio_max_abs_diff": 0.0,
            }
        ],
    }
    result = analyzer._latent_trend(diagnostic)
    assert result["all_video_prefix_pairs_bit_exact"] is True
    assert result["all_audio_prefix_pairs_bit_exact"] is True
    trend = result["streams"]["output_tail"]["trend"]["rms"]
    assert trend["last_minus_first"] == pytest.approx(0.2)
    assert trend["increase_steps"] == 1
