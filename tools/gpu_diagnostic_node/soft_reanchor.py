"""Diagnostic-only one-shot appearance re-anchor for Issue #13 R2.7."""

from __future__ import annotations

from typing import Any

import torch


SOFT_REANCHOR_STRENGTH = 0.5
_EPSILON = 1.0e-6


def _validate_pixels(value: torch.Tensor, *, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 5 or int(value.shape[-1]) != 3:
        raise ValueError(f"{label} must be [B,F,H,W,3] RGB pixels")
    measured = value.detach().float()
    if not bool(torch.isfinite(measured).all().item()):
        raise ValueError(f"{label} contains NaN or Inf")
    return measured


def rgb_appearance_summary(value: torch.Tensor) -> dict[str, float]:
    """Return global low-frequency RGB appearance statistics."""

    rgb = _validate_pixels(value, label="pixels")
    red, green, blue = rgb.unbind(dim=-1)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    maximum = rgb.amax(dim=-1)
    minimum = rgb.amin(dim=-1)
    saturation = (maximum - minimum) / maximum.clamp_min(_EPSILON)
    return {
        "luma_mean": float(luma.mean().item()),
        "luma_std": float(luma.std(unbiased=False).item()),
        "saturation_mean": float(saturation.mean().item()),
        "chroma_u_mean": float((blue - luma).mean().item()),
        "chroma_u_std": float((blue - luma).std(unbiased=False).item()),
        "chroma_v_mean": float((red - luma).mean().item()),
        "chroma_v_std": float((red - luma).std(unbiased=False).item()),
        "highlight_fraction": float((luma >= (250.0 / 255.0)).float().mean().item()),
    }


def _matched_channel(
    current: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    current_mean = current.mean()
    current_std = current.std(unbiased=False).clamp_min(_EPSILON)
    reference_mean = reference.mean()
    reference_std = reference.std(unbiased=False).clamp_min(_EPSILON)
    scale = (reference_std / current_std).clamp(0.5, 2.0)
    return (current - current_mean) * scale + reference_mean, float(scale.item())


def soft_match_rgb_appearance(
    current_pixels: torch.Tensor,
    reference_pixels: torch.Tensor,
    *,
    strength: float = SOFT_REANCHOR_STRENGTH,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Match only global luma/chroma statistics; preserve spatial coordinates."""

    amount = float(strength)
    if not 0.0 <= amount <= 1.0:
        raise ValueError("soft re-anchor strength must be in [0,1]")
    current = _validate_pixels(current_pixels, label="current_pixels")
    reference = _validate_pixels(reference_pixels, label="reference_pixels")
    if tuple(current.shape[-3:]) != tuple(reference.shape[-3:]):
        raise ValueError("current/reference pixel geometry must match")

    current_r, current_g, current_b = current.unbind(dim=-1)
    reference_r, reference_g, reference_b = reference.unbind(dim=-1)
    current_y = 0.2126 * current_r + 0.7152 * current_g + 0.0722 * current_b
    reference_y = (
        0.2126 * reference_r + 0.7152 * reference_g + 0.0722 * reference_b
    )
    current_u = current_b - current_y
    current_v = current_r - current_y
    reference_u = reference_b - reference_y
    reference_v = reference_r - reference_y

    matched_y, luma_scale = _matched_channel(current_y, reference_y)
    matched_u, chroma_u_scale = _matched_channel(current_u, reference_u)
    matched_v, chroma_v_scale = _matched_channel(current_v, reference_v)
    matched_r = matched_y + matched_v
    matched_b = matched_y + matched_u
    matched_g = (matched_y - 0.2126 * matched_r - 0.0722 * matched_b) / 0.7152
    fully_matched = torch.stack((matched_r, matched_g, matched_b), dim=-1).clamp(0.0, 1.0)
    adjusted = (current + amount * (fully_matched - current)).clamp(0.0, 1.0)
    delta = adjusted - current
    report = {
        "strength": amount,
        "method": "global Y/U/V mean+std match; 50% blend; no blur/warp/resize",
        "scales": {
            "luma": luma_scale,
            "chroma_u": chroma_u_scale,
            "chroma_v": chroma_v_scale,
        },
        "reference": rgb_appearance_summary(reference),
        "before": rgb_appearance_summary(current),
        "fully_matched": rgb_appearance_summary(fully_matched),
        "after": rgb_appearance_summary(adjusted),
        "pixel_delta_rms": float(delta.square().mean().sqrt().item()),
        "pixel_delta_max_abs": float(delta.abs().max().item()),
        "geometry_unchanged": tuple(adjusted.shape) == tuple(current.shape),
    }
    return adjusted.to(dtype=current_pixels.dtype), report


def soft_reanchor_context(
    current_context: torch.Tensor,
    reference_context: torch.Tensor,
    *,
    video_vae: Any,
    strength: float = SOFT_REANCHOR_STRENGTH,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode/adjust/encode one temporary context copy without mutating inputs."""

    if not torch.is_tensor(current_context) or current_context.ndim != 5:
        raise ValueError("current_context must be [B,C,T,H,W]")
    if not torch.is_tensor(reference_context) or reference_context.ndim != 5:
        raise ValueError("reference_context must be [B,C,T,H,W]")
    if tuple(current_context.shape) != tuple(reference_context.shape):
        raise ValueError("current/reference latent context shapes must match")
    if int(current_context.shape[0]) != 1:
        raise ValueError("R2.7 diagnostic expects a single context batch")
    if not callable(getattr(video_vae, "decode", None)) or not callable(
        getattr(video_vae, "encode", None)
    ):
        raise ValueError("R2.7 Soft Re-anchor requires the connected Video VAE")

    current_guard = current_context.detach().clone()
    reference_guard = reference_context.detach().clone()
    reference_pixels = video_vae.decode(reference_context)
    current_pixels = video_vae.decode(current_context)
    adjusted_pixels, appearance = soft_match_rgb_appearance(
        current_pixels,
        reference_pixels,
        strength=strength,
    )
    encoded = video_vae.encode(adjusted_pixels[0])
    if not torch.is_tensor(encoded) or tuple(encoded.shape) != tuple(current_context.shape):
        raise ValueError(
            "Soft Re-anchor VAE roundtrip changed the continuation latent shape: "
            f"{getattr(encoded, 'shape', None)} vs {tuple(current_context.shape)}"
        )
    encoded = encoded.to(
        device=current_context.device,
        dtype=current_context.dtype,
    ).contiguous()
    if not bool(torch.isfinite(encoded.float()).all().item()):
        raise ValueError("Soft Re-anchor encoded context contains NaN or Inf")
    appearance.update(
        {
            "current_context_unchanged": bool(torch.equal(current_context, current_guard)),
            "reference_context_unchanged": bool(
                torch.equal(reference_context, reference_guard)
            ),
            "latent_shape_unchanged": tuple(encoded.shape)
            == tuple(current_context.shape),
        }
    )
    return encoded, appearance
