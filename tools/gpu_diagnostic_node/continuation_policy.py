"""Diagnostic-only continuation depth and mask-strength policies for Issue #13."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch


MODE_BASELINE_22 = "Baseline 22f"
MODE_DEPTH_13 = "Video Depth 13f"
MODE_DEPTH_9 = "Video Depth 9f"
MODE_STRENGTH_75 = "Video Strength 75%"
MODE_DEPTH_13_STRENGTH_75 = "Video Depth 13f + Strength 75%"
MODE_OPTIONS = (
    MODE_BASELINE_22,
    MODE_DEPTH_13,
    MODE_DEPTH_9,
    MODE_STRENGTH_75,
    MODE_DEPTH_13_STRENGTH_75,
)

_PREFIX_SLOTS = 7
_AUDIO_PREFIX_STEPS = 37
_ACTIVE_SLOT_CONTRACT = {
    MODE_BASELINE_22: (_PREFIX_SLOTS, 22),
    MODE_DEPTH_13: (4, 13),
    MODE_DEPTH_9: (3, 9),
    MODE_STRENGTH_75: (_PREFIX_SLOTS, 22),
    MODE_DEPTH_13_STRENGTH_75: (4, 13),
}


def _mask_values(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in torch.unique(tensor.detach().float()).tolist()]


class R28ContinuationPolicy:
    """Scoped callback used only by the external R2.8 diagnostic node."""

    def __init__(
        self,
        mode: str,
        *,
        tensor_summary: Callable[[torch.Tensor], dict[str, Any]],
    ) -> None:
        if mode not in MODE_OPTIONS:
            raise ValueError(f"unknown R2.8 continuation policy: {mode!r}")
        self.mode = str(mode)
        self.tensor_summary = tensor_summary
        self._prepared: dict[int, dict[str, Any]] = {}
        self.sample_calls: list[dict[str, Any]] = []

    @property
    def active_video_slots(self) -> int:
        return int(_ACTIVE_SLOT_CONTRACT[self.mode][0])

    @property
    def active_video_frames(self) -> int:
        return int(_ACTIVE_SLOT_CONTRACT[self.mode][1])

    @property
    def prefix_denoise_mask_value(self) -> float:
        if self.mode in (MODE_STRENGTH_75, MODE_DEPTH_13_STRENGTH_75):
            return 0.25
        return 0.0

    def prepare_masked_latent(
        self,
        *,
        latent: dict[str, Any],
        physical_group: int,
        logical_chunks: tuple[int, ...],
        context_frames: int,
        video_context: torch.Tensor,
        audio_context: torch.Tensor | None,
    ) -> dict[str, Any]:
        if int(context_frames) != 22:
            raise ValueError("R2.8 policy requires the unchanged Balanced 22-frame transport")
        mask = latent.get("noise_mask")
        if mask is None or not hasattr(mask, "unbind"):
            raise ValueError("R2.8 policy requires the native nested AV noise mask")
        parts = list(mask.unbind())
        if len(parts) < 2:
            raise ValueError("R2.8 policy requires Video and Audio masks")
        video_mask, audio_mask = parts[0], parts[1]
        if int(video_mask.shape[2]) <= _PREFIX_SLOTS:
            raise ValueError("R2.8 Video target has no generated region after the prefix")
        if int(audio_mask.shape[-1]) <= _AUDIO_PREFIX_STEPS:
            raise ValueError("R2.8 Audio target has no generated region after the prefix")

        original_video_mask = video_mask.detach().clone()
        original_audio_mask = audio_mask.detach().clone()
        transformed = latent
        if self.mode != MODE_BASELINE_22:
            video_mask = video_mask.clone()
            oldest_inactive = _PREFIX_SLOTS - self.active_video_slots
            if oldest_inactive:
                video_mask[:, :, :oldest_inactive] = 1.0
            video_mask[
                :, :, oldest_inactive:_PREFIX_SLOTS
            ] = self.prefix_denoise_mask_value
            transformed = dict(latent)
            transformed["noise_mask"] = mask.__class__((video_mask, audio_mask.clone()))

        effective_mask = list(transformed["noise_mask"].unbind())
        effective_video_mask, effective_audio_mask = effective_mask[0], effective_mask[1]
        self._prepared[int(physical_group)] = {
            "physical_group": int(physical_group),
            "logical_chunks": [int(value) for value in logical_chunks],
            "transport_context_frames": int(context_frames),
            "transport_video_prefix_slots": _PREFIX_SLOTS,
            "transport_audio_prefix_steps": _AUDIO_PREFIX_STEPS,
            "active_video_slots": self.active_video_slots,
            "active_video_frames": self.active_video_frames,
            "video_prefix_denoise_mask_value": self.prefix_denoise_mask_value,
            "video_context": self.tensor_summary(video_context),
            "audio_context": self.tensor_summary(audio_context)
            if audio_context is not None
            else None,
            "source_samples_object_passthrough": transformed.get("samples")
            is latent.get("samples"),
            "input_video_mask_unchanged": bool(
                torch.equal(original_video_mask, list(mask.unbind())[0])
            ),
            "input_audio_mask_unchanged": bool(
                torch.equal(original_audio_mask, list(mask.unbind())[1])
            ),
            "effective_video_prefix_mask_values": _mask_values(
                effective_video_mask[:, :, :_PREFIX_SLOTS]
            ),
            "effective_video_generated_mask_values": _mask_values(
                effective_video_mask[:, :, _PREFIX_SLOTS:]
            ),
            "effective_audio_prefix_mask_values": _mask_values(
                effective_audio_mask[..., :_AUDIO_PREFIX_STEPS]
            ),
            "effective_audio_generated_mask_values": _mask_values(
                effective_audio_mask[..., _AUDIO_PREFIX_STEPS:]
            ),
        }
        return transformed

    def before_sampling(
        self,
        *,
        physical_group: int,
        logical_chunks: tuple[int, ...],
        context_frames: int,
        latent: dict[str, Any],
        conditioning: Any,
        seed: int,
        sigmas: torch.Tensor,
    ) -> dict[str, Any]:
        samples = list(latent["samples"].unbind())
        metadata = conditioning[0][1] if conditioning else {}
        record = {
            "physical_group": int(physical_group),
            "logical_chunks": [int(value) for value in logical_chunks],
            "context_frames": int(context_frames),
            "seed": int(seed),
            "sigmas": self.tensor_summary(sigmas),
            "input_video": self.tensor_summary(samples[0]),
            "input_audio": self.tensor_summary(samples[1]),
            "prompt_embedding": self.tensor_summary(conditioning[0][0])
            if conditioning and torch.is_tensor(conditioning[0][0])
            else None,
            "minimax_frame_count": metadata.get("minimax_frame_count"),
            "policy": self._prepared.get(int(physical_group)),
        }
        return {"started": time.perf_counter(), "record": record}

    def after_sampling(
        self,
        *,
        token: dict[str, Any],
        physical_group: int,
        logical_chunks: tuple[int, ...],
        context_frames: int,
        latent: dict[str, Any],
        sampled: dict[str, Any],
        seed: int,
        sigmas: torch.Tensor,
    ) -> None:
        record = dict(token["record"])
        record["sample_host_elapsed_seconds"] = float(
            time.perf_counter() - float(token["started"])
        )
        source_video, source_audio = list(latent["samples"].unbind())[:2]
        output_video, output_audio = list(sampled["samples"].unbind())[:2]
        record["output_video"] = self.tensor_summary(output_video)
        record["output_audio"] = self.tensor_summary(output_audio)
        if int(context_frames) == 22:
            source_video_prefix = source_video[:, :, :_PREFIX_SLOTS]
            output_video_prefix = output_video[:, :, :_PREFIX_SLOTS]
            source_audio_prefix = source_audio[..., :_AUDIO_PREFIX_STEPS]
            output_audio_prefix = output_audio[..., :_AUDIO_PREFIX_STEPS]
            record["restored_video_prefix_bit_exact"] = bool(
                torch.equal(source_video_prefix, output_video_prefix)
            )
            record["restored_audio_prefix_bit_exact"] = bool(
                torch.equal(source_audio_prefix, output_audio_prefix)
            )
            record["restored_video_prefix"] = self.tensor_summary(output_video_prefix)
            record["restored_audio_prefix"] = self.tensor_summary(output_audio_prefix)
        self.sample_calls.append(record)

    def finalize(self) -> dict[str, Any]:
        return {
            "format": "h3-continuum-issue13-r28-policy-v1",
            "mode": self.mode,
            "transport_context_frames": 22,
            "transport_video_prefix_slots": _PREFIX_SLOTS,
            "transport_audio_prefix_steps": _AUDIO_PREFIX_STEPS,
            "active_video_slots": self.active_video_slots,
            "active_video_frames": self.active_video_frames,
            "video_prefix_denoise_mask_value": self.prefix_denoise_mask_value,
            "audio_contract": "unchanged 37T protected prefix",
            "sample_calls": list(self.sample_calls),
            "sampling_count": len(self.sample_calls),
        }
