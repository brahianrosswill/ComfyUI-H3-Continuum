"""Target-aware refinement sampling for frozen Video and Audio policies."""

from __future__ import annotations

from typing import Any, Callable

import torch

from ..state import extract_av_streams
from .refine_sampling import (
    RefineSamplingRuntimeError,
    _load_runtime_modules,
    _make_basic_guider,
    sample_refine_chunk,
)
from .refine_target import (
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
    RefineTarget,
    RefineTargetError,
    resolve_refine_target,
)
from .refine_window import GroupTemporalWindow


class TargetedRefineExecutionUnavailable(RefineTargetError):
    """Raised before Sampling when a future target mode is not implemented."""


def _validate_ranges(
    ranges: tuple[tuple[int, int], ...],
    length: int,
    *,
    name: str,
) -> tuple[tuple[int, int], ...]:
    if not ranges:
        raise RefineSamplingRuntimeError(f"{name} temporal window is empty")
    previous_stop = 0
    for start, stop in ranges:
        if start < previous_stop or stop <= start or stop > int(length):
            raise RefineSamplingRuntimeError(
                f"{name} temporal window is outside its latent length"
            )
        previous_stop = stop
    return ranges


def _window_masks(
    video: torch.Tensor,
    audio: torch.Tensor,
    temporal_window: GroupTemporalWindow,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(temporal_window, GroupTemporalWindow):
        raise RefineSamplingRuntimeError(
            "temporal window must be a resolved GroupTemporalWindow"
        )
    video_ranges = _validate_ranges(
        temporal_window.video_slot_ranges,
        int(video.shape[2]),
        name="Video",
    )
    audio_ranges = _validate_ranges(
        temporal_window.audio_tick_ranges,
        int(audio.shape[-1]),
        name="Audio",
    )
    video_mask = torch.zeros(
        (
            int(video.shape[0]),
            1,
            int(video.shape[2]),
            int(video.shape[3]),
            int(video.shape[4]),
        ),
        dtype=torch.float32,
        device="cpu",
    )
    audio_mask = torch.zeros(
        (
            int(audio.shape[0]),
            1,
            int(audio.shape[2]),
            int(audio.shape[3]),
        ),
        dtype=torch.float32,
        device="cpu",
    )
    for start, stop in video_ranges:
        video_mask[:, :, start:stop] = 1.0
    for start, stop in audio_ranges:
        audio_mask[..., start:stop] = 1.0
    return video_mask, audio_mask


def _prepare_video_only_noise_and_mask(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    seed: int,
    batch_inds: Any = None,
    prepare_noise_fn: Callable[[torch.Tensor, int, Any], torch.Tensor],
    nested_builder: Callable[[tuple[torch.Tensor, torch.Tensor]], Any],
    temporal_window: GroupTemporalWindow,
) -> tuple[Any, Any]:
    """Build windowed Video noise/mask while locking Audio completely."""

    video_noise = prepare_noise_fn(video, int(seed), batch_inds)
    if not torch.is_tensor(video_noise) or tuple(video_noise.shape) != tuple(video.shape):
        raise RefineSamplingRuntimeError(
            "ComfyUI returned invalid Video Only noise geometry"
        )
    video_mask, _audio_window_mask = _window_masks(video, audio, temporal_window)
    audio_noise = torch.zeros_like(audio, device="cpu")
    audio_mask = torch.zeros(
        (int(audio.shape[0]), 1, 1, 1),
        dtype=torch.float32,
        device="cpu",
    )
    return (
        nested_builder((video_noise, audio_noise)),
        nested_builder((video_mask, audio_mask)),
    )


def require_video_only_execution(
    refine_target: RefineTarget | str | None,
) -> RefineTarget:
    target = resolve_refine_target(refine_target)
    if target.mode != MODE_VIDEO_ONLY:
        raise TargetedRefineExecutionUnavailable(
            f"refine target {target.mode!r} is contract-only in Phase A1"
        )
    return target


def _prepare_audio_only_noise_and_mask(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    seed: int,
    batch_inds: Any = None,
    prepare_noise_fn: Callable[[torch.Tensor, int, Any], torch.Tensor],
    nested_builder: Callable[[tuple[torch.Tensor, torch.Tensor]], Any],
    temporal_window: GroupTemporalWindow | None = None,
) -> tuple[Any, Any]:
    """Build zero/locked Video and seeded-random/sampled Audio streams."""

    if video.ndim != 5 or audio.ndim != 4:
        raise RefineSamplingRuntimeError(
            "targeted latent must contain video [B,C,T,H,W] and audio [B,C,2,T]"
        )
    if int(video.shape[0]) != int(audio.shape[0]):
        raise RefineSamplingRuntimeError("targeted video/audio batch sizes differ")

    audio_noise = prepare_noise_fn(audio, int(seed), batch_inds)
    if not torch.is_tensor(audio_noise) or tuple(audio_noise.shape) != tuple(audio.shape):
        raise RefineSamplingRuntimeError(
            "ComfyUI returned invalid Audio Only noise geometry"
        )
    video_noise = torch.zeros_like(video, device="cpu")
    if temporal_window is None:
        video_mask = torch.zeros(
            (int(video.shape[0]), 1, 1, 1, 1),
            dtype=torch.float32,
            device="cpu",
        )
        audio_mask = torch.ones(
            (int(audio.shape[0]), 1, 1, 1),
            dtype=torch.float32,
            device="cpu",
        )
    else:
        _video_window_mask, audio_mask = _window_masks(
            video,
            audio,
            temporal_window,
        )
        video_mask = torch.zeros(
            (int(video.shape[0]), 1, 1, 1, 1),
            dtype=torch.float32,
            device="cpu",
        )
    noise = nested_builder((video_noise, audio_noise))
    noise_mask = nested_builder((video_mask, audio_mask))
    return noise, noise_mask


def _prepare_video_audio_noise_and_mask(
    video: torch.Tensor,
    audio: torch.Tensor,
    *,
    seed: int,
    batch_inds: Any = None,
    prepare_noise_fn: Callable[[torch.Tensor, int, Any], torch.Tensor],
    nested_builder: Callable[[tuple[torch.Tensor, torch.Tensor]], Any],
    temporal_window: GroupTemporalWindow | None = None,
) -> tuple[Any, Any]:
    """Build seeded-random noise and sampled masks for both AV streams."""

    if video.ndim != 5 or audio.ndim != 4:
        raise RefineSamplingRuntimeError(
            "targeted latent must contain video [B,C,T,H,W] and audio [B,C,2,T]"
        )
    if int(video.shape[0]) != int(audio.shape[0]):
        raise RefineSamplingRuntimeError("targeted video/audio batch sizes differ")

    video_noise = prepare_noise_fn(video, int(seed), batch_inds)
    audio_noise = prepare_noise_fn(audio, int(seed), batch_inds)
    if not torch.is_tensor(video_noise) or tuple(video_noise.shape) != tuple(video.shape):
        raise RefineSamplingRuntimeError(
            "ComfyUI returned invalid Video + Audio video noise geometry"
        )
    if not torch.is_tensor(audio_noise) or tuple(audio_noise.shape) != tuple(audio.shape):
        raise RefineSamplingRuntimeError(
            "ComfyUI returned invalid Video + Audio audio noise geometry"
        )
    if temporal_window is None:
        video_mask = torch.ones(
            (int(video.shape[0]), 1, 1, 1, 1),
            dtype=torch.float32,
            device="cpu",
        )
        audio_mask = torch.ones(
            (int(audio.shape[0]), 1, 1, 1),
            dtype=torch.float32,
            device="cpu",
        )
    else:
        video_mask, audio_mask = _window_masks(video, audio, temporal_window)
    noise = nested_builder((video_noise, audio_noise))
    noise_mask = nested_builder((video_mask, audio_mask))
    return noise, noise_mask


def _sample_with_target_noise_and_mask(
    *,
    model: Any,
    conditioning: list,
    latent: dict[str, Any],
    sampler: Any,
    sigmas: torch.Tensor,
    seed: int,
    enable_preview: bool,
    noise_mask_builder: Callable[..., tuple[Any, Any]],
    temporal_window: GroupTemporalWindow | None = None,
    restore_video_outside_window: bool = False,
    restore_audio_outside_window: bool = False,
) -> dict[str, Any]:
    """Run Core once after constructing one explicit AV target policy."""

    if not isinstance(latent, dict) or "samples" not in latent:
        raise RefineSamplingRuntimeError("latent must be a ComfyUI LATENT dictionary")
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1 or sigmas.numel() < 2:
        raise RefineSamplingRuntimeError("sigmas must contain at least two values")

    model_management, nested_tensor, comfy_sample, comfy_utils, preview = (
        _load_runtime_modules()
    )
    working = latent.copy()
    latent_image = comfy_sample.fix_empty_latent_channels(
        model,
        working["samples"],
        working.get("downscale_ratio_spacial"),
        working.get("downscale_ratio_temporal"),
    )
    working["samples"] = latent_image
    video, audio = extract_av_streams(working)
    noise, noise_mask = noise_mask_builder(
        video,
        audio,
        seed=int(seed),
        batch_inds=working.get("batch_index"),
        prepare_noise_fn=comfy_sample.prepare_noise,
        nested_builder=nested_tensor.NestedTensor,
        temporal_window=temporal_window,
    )
    guider = _make_basic_guider(model, conditioning)

    x0_output: dict[str, Any] = {}
    callback = None
    if enable_preview:
        callback = preview.prepare_callback(
            model,
            int(sigmas.shape[-1]) - 1,
            x0_output,
        )
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=callback,
        disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED,
        seed=int(seed),
    )
    samples = samples.to(model_management.intermediate_device())
    if temporal_window is not None:
        sampled_output = {"samples": samples}
        sampled_video, sampled_audio = extract_av_streams(sampled_output)
        if restore_video_outside_window:
            sampled_video = _restore_outside_temporal_ranges(
                sampled_video,
                video,
                temporal_window.video_slot_ranges,
                time_dim=2,
                name="Video",
            )
        if restore_audio_outside_window:
            sampled_audio = _restore_outside_temporal_ranges(
                sampled_audio,
                audio,
                temporal_window.audio_tick_ranges,
                time_dim=-1,
                name="Audio",
            )
        samples = nested_tensor.NestedTensor((sampled_video, sampled_audio))

    output = working.copy()
    output.pop("downscale_ratio_spacial", None)
    output.pop("downscale_ratio_temporal", None)
    output["samples"] = samples
    extract_av_streams(output)
    return output


def _restore_outside_temporal_ranges(
    sampled: torch.Tensor,
    source: torch.Tensor,
    ranges: tuple[tuple[int, int], ...],
    *,
    time_dim: int,
    name: str,
) -> torch.Tensor:
    """Restore every inactive temporal position from the input bit-exactly."""

    if not torch.is_tensor(sampled) or not torch.is_tensor(source):
        raise RefineSamplingRuntimeError(f"{name} temporal restore requires Tensors")
    if tuple(sampled.shape) != tuple(source.shape):
        raise RefineSamplingRuntimeError(
            f"{name} temporal restore input/output shapes differ"
        )
    if sampled.dtype != source.dtype:
        raise RefineSamplingRuntimeError(
            f"{name} temporal restore input/output dtypes differ"
        )
    dim = int(time_dim)
    if dim < 0:
        dim += sampled.ndim
    if dim < 0 or dim >= sampled.ndim:
        raise RefineSamplingRuntimeError(f"{name} temporal restore dimension is invalid")
    validated = _validate_ranges(ranges, int(sampled.shape[dim]), name=name)
    restored = sampled.clone()
    original = source.to(device=restored.device)
    cursor = 0
    for start, stop in validated:
        if start > cursor:
            restored.narrow(dim, cursor, start - cursor).copy_(
                original.narrow(dim, cursor, start - cursor)
            )
        cursor = stop
    if cursor < int(restored.shape[dim]):
        restored.narrow(dim, cursor, int(restored.shape[dim]) - cursor).copy_(
            original.narrow(dim, cursor, int(restored.shape[dim]) - cursor)
        )
    return restored


def sample_video_only_window_refine_chunk(
    *,
    model: Any,
    conditioning: list,
    latent: dict[str, Any],
    sampler: Any,
    sigmas: torch.Tensor,
    seed: int,
    temporal_window: GroupTemporalWindow,
    enable_preview: bool = True,
) -> dict[str, Any]:
    """Refine only selected Video slots and restore all other AV positions."""

    return _sample_with_target_noise_and_mask(
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=int(seed),
        enable_preview=bool(enable_preview),
        noise_mask_builder=_prepare_video_only_noise_and_mask,
        temporal_window=temporal_window,
        restore_video_outside_window=True,
        restore_audio_outside_window=True,
    )


def sample_audio_only_refine_chunk(
    *,
    model: Any,
    conditioning: list,
    latent: dict[str, Any],
    sampler: Any,
    sigmas: torch.Tensor,
    seed: int,
    temporal_window: GroupTemporalWindow | None = None,
    enable_preview: bool = True,
) -> dict[str, Any]:
    """Sample Audio while Core carries Video with zero noise and mask zero."""

    return _sample_with_target_noise_and_mask(
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=int(seed),
        enable_preview=bool(enable_preview),
        noise_mask_builder=_prepare_audio_only_noise_and_mask,
        temporal_window=temporal_window,
        restore_video_outside_window=temporal_window is not None,
        restore_audio_outside_window=temporal_window is not None,
    )


def sample_video_audio_refine_chunk(
    *,
    model: Any,
    conditioning: list,
    latent: dict[str, Any],
    sampler: Any,
    sigmas: torch.Tensor,
    seed: int,
    temporal_window: GroupTemporalWindow | None = None,
    enable_preview: bool = True,
) -> dict[str, Any]:
    """Sample and adopt both Video and Audio using mask one for each stream."""

    return _sample_with_target_noise_and_mask(
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=int(seed),
        enable_preview=bool(enable_preview),
        noise_mask_builder=_prepare_video_audio_noise_and_mask,
        temporal_window=temporal_window,
        restore_video_outside_window=temporal_window is not None,
        restore_audio_outside_window=temporal_window is not None,
    )


def sample_targeted_refine_chunk(
    *,
    model: Any,
    conditioning: list,
    latent: dict[str, Any],
    sampler: Any,
    sigmas: torch.Tensor,
    seed: int,
    refine_target: RefineTarget | str | None = MODE_VIDEO_ONLY,
    temporal_window: GroupTemporalWindow | None = None,
    enable_preview: bool = True,
    legacy_sample_fn: Callable[..., dict[str, Any]] | None = None,
    video_only_window_sample_fn: Callable[..., dict[str, Any]] | None = None,
    audio_only_sample_fn: Callable[..., dict[str, Any]] | None = None,
    video_audio_sample_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch an implemented target without changing its stream policy."""

    target = resolve_refine_target(refine_target)
    if target.mode == MODE_VIDEO_ONLY and temporal_window is not None:
        sample_fn = video_only_window_sample_fn or sample_video_only_window_refine_chunk
    elif target.mode == MODE_VIDEO_ONLY:
        sample_fn = legacy_sample_fn or sample_refine_chunk
    elif target.mode == MODE_AUDIO_ONLY:
        sample_fn = audio_only_sample_fn or sample_audio_only_refine_chunk
    elif target.mode == MODE_VIDEO_AUDIO:
        sample_fn = video_audio_sample_fn or sample_video_audio_refine_chunk
    else:  # pragma: no cover - resolve_refine_target validates the mode.
        raise TargetedRefineExecutionUnavailable(
            f"refine target {target.mode!r} is unavailable"
        )
    arguments = dict(
        model=model,
        conditioning=conditioning,
        latent=latent,
        sampler=sampler,
        sigmas=sigmas,
        seed=int(seed),
        enable_preview=bool(enable_preview),
    )
    if temporal_window is not None:
        arguments["temporal_window"] = temporal_window
    return sample_fn(**arguments)
