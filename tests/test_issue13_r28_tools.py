from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


policy_module = _load(
    "h3_issue13_r28_continuation_policy",
    "gpu_diagnostic_node/continuation_policy.py",
)
runner = _load("h3_issue13_r28_gate_runner", "issue13_r28_gate_runner.py")
analyzer = _load("h3_issue13_r28_analyze", "issue13_r28_analyze.py")


class FakeNested:
    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    def unbind(self):
        return self.tensors


def _summary(tensor: torch.Tensor) -> dict:
    return {
        "shape": [int(value) for value in tensor.shape],
        "mean": float(tensor.float().mean().item()),
    }


def _latent():
    video = torch.randn((1, 24, 42, 4, 4), generator=torch.Generator().manual_seed(1))
    audio = torch.randn((1, 32, 2, 235), generator=torch.Generator().manual_seed(2))
    video_mask = torch.ones((1, 1, 42, 4, 4), dtype=torch.float32)
    video_mask[:, :, :7] = 0.0
    audio_mask = torch.ones((1, 1, 2, 235), dtype=torch.float32)
    audio_mask[..., :37] = 0.0
    return {
        "samples": FakeNested((video, audio)),
        "noise_mask": FakeNested((video_mask, audio_mask)),
    }


def _apply(mode: str):
    latent = _latent()
    original_samples = latent["samples"]
    original_video_mask, original_audio_mask = [
        value.clone() for value in latent["noise_mask"].unbind()
    ]
    policy = policy_module.R28ContinuationPolicy(mode, tensor_summary=_summary)
    result = policy.prepare_masked_latent(
        latent=latent,
        physical_group=2,
        logical_chunks=(2,),
        context_frames=22,
        video_context=torch.zeros((1, 24, 7, 4, 4)),
        audio_context=torch.zeros((1, 32, 2, 37)),
    )
    assert result["samples"] is original_samples
    assert torch.equal(latent["noise_mask"].unbind()[0], original_video_mask)
    assert torch.equal(latent["noise_mask"].unbind()[1], original_audio_mask)
    return policy, result


def test_depth_13_uses_last_four_slots_and_preserves_audio_contract():
    policy, result = _apply(policy_module.MODE_DEPTH_13)
    video_mask, audio_mask = result["noise_mask"].unbind()
    assert policy.active_video_slots == 4
    assert policy.active_video_frames == 13
    assert torch.all(video_mask[:, :, :3] == 1)
    assert torch.all(video_mask[:, :, 3:7] == 0)
    assert torch.all(video_mask[:, :, 7:] == 1)
    assert torch.all(audio_mask[..., :37] == 0)
    assert torch.all(audio_mask[..., 37:] == 1)


def test_depth_9_uses_last_three_slots_only():
    policy, result = _apply(policy_module.MODE_DEPTH_9)
    video_mask, _ = result["noise_mask"].unbind()
    assert policy.active_video_slots == 3
    assert policy.active_video_frames == 9
    assert torch.all(video_mask[:, :, :4] == 1)
    assert torch.all(video_mask[:, :, 4:7] == 0)


def test_strength_75_uses_core_fractional_mask_without_scaling_latent():
    latent = _latent()
    source_video, source_audio = latent["samples"].unbind()
    source_video_before = source_video.clone()
    source_audio_before = source_audio.clone()
    policy = policy_module.R28ContinuationPolicy(
        policy_module.MODE_STRENGTH_75, tensor_summary=_summary
    )
    result = policy.prepare_masked_latent(
        latent=latent,
        physical_group=2,
        logical_chunks=(2,),
        context_frames=22,
        video_context=torch.zeros((1, 24, 7, 4, 4)),
        audio_context=torch.zeros((1, 32, 2, 37)),
    )
    video_mask, audio_mask = result["noise_mask"].unbind()
    assert torch.all(video_mask[:, :, :7] == 0.25)
    assert torch.all(audio_mask[..., :37] == 0)
    assert torch.equal(source_video, source_video_before)
    assert torch.equal(source_audio, source_audio_before)


def test_baseline_returns_original_latent_and_mask_objects():
    latent = _latent()
    policy = policy_module.R28ContinuationPolicy(
        policy_module.MODE_BASELINE_22, tensor_summary=_summary
    )
    result = policy.prepare_masked_latent(
        latent=latent,
        physical_group=2,
        logical_chunks=(2,),
        context_frames=22,
        video_context=torch.zeros((1, 24, 7, 4, 4)),
        audio_context=torch.zeros((1, 32, 2, 37)),
    )
    assert result is latent
    assert policy.active_video_slots == 7
    assert policy.active_video_frames == 22


def _source_prompt() -> dict:
    return {
        "114": {"inputs": {"image": "old.png"}},
        "119": {"inputs": {}},
        "139": {"inputs": {"sampler_name": "old"}},
        "150": {"inputs": {"enabled": True}},
        "153": {"inputs": {"scheduler": "old", "steps": 20}},
        "188": {"inputs": {"value": "old"}},
        "191": {"inputs": {}},
        "249": {"inputs": {}},
        "286": {"inputs": {"lora_1": {"on": False}}},
        "297": {"inputs": {"value": 0.3}},
        "305": {
            "class_type": "H3ContinuumMaskedPrefixR1Diagnostic",
            "inputs": {
                "continuation_source_mode": "Recursive",
                "synchronize_sampling": True,
                "last_frame": ["900", 0],
            },
        },
        "306": {"inputs": {}},
    }


def test_runner_changes_only_the_approved_r28_experiment_contract():
    prompt, loras = runner.configure_prompt(
        _source_prompt(),
        case="depth_13f",
        output_prefix="video/issue13_r28/test",
        image_name="closeup.png",
        seed=131325,
        width=576,
        height=864,
        diagnostic_nonce=132800,
    )
    inputs = prompt["305"]["inputs"]
    assert prompt["305"]["class_type"] == "H3ContinuumContinuationR28Diagnostic"
    assert inputs["continuation_experiment"] == "Video Depth 13f"
    assert inputs["continuation_transport"] == "masked_av_prefix_22_v1"
    assert inputs["continuity"] == "Balanced — 22 frames"
    assert inputs["audio_continuity"] is True
    assert inputs["chunks"] == 6
    assert inputs["chunk_seconds"] == 5.0
    assert "continuation_source_mode" not in inputs
    assert "synchronize_sampling" not in inputs
    assert "last_frame" not in inputs
    assert prompt["188"]["inputs"]["value"].count("\n---\n") == 5
    assert len(set(prompt["188"]["inputs"]["value"].split("\n---\n"))) == 1
    assert loras == [{"name": runner.r25.TURBO_LORA, "strength": 1.0}]


def _analysis_case(scale: float) -> dict:
    rows = []
    for group in range(1, 7):
        row = {key: 0.0 for key in analyzer.base.METRIC_KEYS}
        for key in analyzer.DRIFT_KEYS:
            row[key] = float(group - 1) * scale
        rows.append(row)
    return {
        "end_trend": analyzer.base._trend(rows, analyzer.base.METRIC_KEYS),
        "boundary_metrics": [
            {"trajectory_flow_discontinuity": 0.1} for _ in range(5)
        ],
    }


def test_comparison_marks_a_clean_half_reduction_promising():
    result = analyzer._metric_comparison(_analysis_case(1.0), _analysis_case(0.5))
    assert result["screening_mean_reduction_percent"] == 50.0
    assert result["sharpness_not_worse"] is True
    assert result["gradient_not_worse"] is True
    assert result["numeric_promising"] is True
