"""Target-aware Second Pass orchestration over complete physical groups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..state import extract_av_streams
from ..v2.sampling import latent_from_cpu
from .refine_schedule import (
    RefineScheduleError,
    resolve_refine_schedule,
    serializable_schedule_contract,
)
from .refine_scope import (
    MODE_ALL as SCOPE_ALL,
    resolve_refine_scope,
)
from .refine_target import (
    MODE_AUDIO_ONLY,
    MODE_VIDEO_AUDIO,
    MODE_VIDEO_ONLY,
    RefineTarget,
    build_target_execution_contract,
    derive_target_refine_seed,
    resolve_refine_target,
    serializable_target_contract,
)
from .refine_window import (
    MODE_TIME_WINDOW,
    GroupTemporalWindow,
    RefineWindow,
    resolve_refine_window,
)
from .second_pass import (
    SecondPassContractError,
    _normalize_selected_group_indices,
    _latent_samples,
    prepare_physical_refine_groups,
    run_second_pass_groups,
    validate_second_pass_inputs,
)
from .targeted_refine_sampling import (
    TargetedRefineExecutionUnavailable,
    sample_targeted_refine_chunk,
)


def _attach_target_contract(
    updated_plan: dict[str, Any],
    target: RefineTarget,
) -> None:
    contract = updated_plan.get("second_pass_contract")
    if not isinstance(contract, dict):
        raise SecondPassContractError(
            "targeted Second Pass output has no mutable contract"
        )
    if contract.get("version") != 1:
        raise SecondPassContractError(
            "targeted Second Pass must preserve contract version 1"
        )
    schedule_contract = contract.get("refine_schedule")
    contract["refine_target_contract"] = serializable_target_contract(target)
    contract["refine_execution_contract"] = build_target_execution_contract(
        target,
        schedule_contract,
    )


def _run_video_only_groups(
    *,
    target: RefineTarget,
    sample_fn,
    arguments: dict[str, Any],
    selected_group_indices: Sequence[int] | None = None,
    temporal_windows_by_group: Mapping[int, GroupTemporalWindow] | None = None,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any], str]:
    def targeted_sample(**kwargs):
        return sample_targeted_refine_chunk(
            refine_target=target,
            legacy_sample_fn=sample_fn,
            video_only_window_sample_fn=sample_fn,
            **kwargs,
        )

    videos, audios, updated_plan, status = run_second_pass_groups(
        **arguments,
        sample_fn=targeted_sample,
        selected_group_indices=selected_group_indices,
        temporal_windows_by_group=temporal_windows_by_group,
    )
    _attach_target_contract(updated_plan, target)
    return videos, audios, updated_plan, status


def _run_sampled_target_groups(
    *,
    target: RefineTarget,
    model: Any,
    clip: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    video_latents: Sequence[Any],
    audio_latents: Sequence[Any],
    assembly_plan: Mapping[str, Any],
    refine_seed: int,
    refine_context: Any,
    video_vae: Any,
    conditioning_upscale_method: str,
    enable_preview: bool,
    encode_prompt_fn,
    latent_builder,
    sample_fn,
    stream_extractor,
    clone_model_fn,
    validate_refine_context_fn,
    adapt_group_conditioning_fn,
    refine_schedule,
    selected_group_indices: Sequence[int] | None = None,
    temporal_windows_by_group: Mapping[int, GroupTemporalWindow] | None = None,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any], str]:
    """Sample one non-legacy target and adopt only its configured outputs."""

    if target.mode not in (MODE_AUDIO_ONLY, MODE_VIDEO_AUDIO):
        raise TargetedRefineExecutionUnavailable(
            f"refine target {target.mode!r} is not a sampled-stream target"
        )
    adopt_sampled_video = target.mode == MODE_VIDEO_AUDIO

    validation = validate_second_pass_inputs(
        video_latents,
        audio_latents,
        assembly_plan,
    )
    selected_indices, selection_active = _normalize_selected_group_indices(
        selected_group_indices,
        int(validation["physical_group_count"]),
    )
    try:
        resolved_schedule = resolve_refine_schedule(sigmas, refine_schedule)
    except RefineScheduleError as exc:
        raise SecondPassContractError(
            f"refine schedule is invalid: {exc}"
        ) from exc
    effective_sigmas = resolved_schedule.sigmas
    schedule_contract = serializable_schedule_contract(resolved_schedule)
    latent_builder = latent_builder or latent_from_cpu
    stream_extractor = stream_extractor or extract_av_streams
    refined_videos: list[dict[str, Any]] = (
        list(video_latents) if selection_active else []
    )
    refined_audios: list[dict[str, Any]] = (
        list(audio_latents) if selection_active else []
    )
    group_seeds: list[int] = []
    conditioning_sources: list[str] = []
    group_report_lines: list[str] = []

    def sample_prepared_group(
        group_index,
        video_latent,
        group,
        chunk_model,
        conditioning,
        detail,
    ):
        audio_latent = audio_latents[group_index]
        video_samples = _latent_samples(
            video_latent,
            name=f"video_latents[{group_index}]",
        )
        audio_samples = _latent_samples(
            audio_latent,
            name=f"audio_latents[{group_index}]",
        )
        physical_seed = derive_target_refine_seed(
            int(refine_seed),
            group_index,
            target,
        )
        group_seeds.append(physical_seed)
        conditioning_source = str(detail["conditioning_source"])
        conditioning_sources.append(conditioning_source)
        nested_latent = latent_builder(video_samples, audio_samples)
        sample_arguments = dict(
            model=chunk_model,
            conditioning=conditioning,
            latent=nested_latent,
            sampler=sampler,
            sigmas=effective_sigmas,
            seed=physical_seed,
            refine_target=target,
            enable_preview=bool(enable_preview),
            audio_only_sample_fn=sample_fn,
            video_audio_sample_fn=sample_fn,
        )
        if temporal_windows_by_group is not None:
            temporal_window = temporal_windows_by_group.get(group_index)
            if temporal_window is None:
                raise SecondPassContractError(
                    f"selected physical group {group_index + 1} has no temporal window"
                )
            sample_arguments["temporal_window"] = temporal_window
        sampled = sample_targeted_refine_chunk(**sample_arguments)
        sampled_video, sampled_audio = stream_extractor(sampled)
        if tuple(sampled_video.shape) != tuple(video_samples.shape):
            raise SecondPassContractError(
                f"sampled video group {group_index} changed B/C/T/H/W geometry"
            )
        if tuple(sampled_audio.shape) != tuple(audio_samples.shape):
            raise SecondPassContractError(
                f"refined audio group {group_index} changed B/C/stereo/T geometry"
            )
        if not bool(torch.isfinite(sampled_audio.float()).all().item()):
            raise SecondPassContractError(
                f"refined audio group {group_index} contains NaN or Inf"
            )
        if adopt_sampled_video:
            if not bool(torch.isfinite(sampled_video.float()).all().item()):
                raise SecondPassContractError(
                    f"refined video group {group_index} contains NaN or Inf"
                )
            output_video = dict(video_latent)
            output_video["samples"] = sampled_video
            if selection_active:
                refined_videos[group_index] = output_video
            else:
                refined_videos.append(output_video)
        output_audio = dict(audio_latent)
        output_audio["samples"] = sampled_audio
        if selection_active:
            refined_audios[group_index] = output_audio
        else:
            refined_audios.append(output_audio)
        adoption = (
            f"video_shape={tuple(sampled_video.shape)}, "
            "sampled_video_adopted=true, sampled_audio_adopted=true"
            if adopt_sampled_video
            else "temporary_video_discarded=true"
        )
        group_report_lines.append(
            "group "
            f"{group_index + 1}: logical_chunks={group.get('logical_chunks')}, "
            f"prompt_policy={group.get('prompt_policy', 'unknown')}, "
            f"conditioning_source={conditioning_source}, "
            f"refine_seed={physical_seed}, "
            f"audio_shape={tuple(sampled_audio.shape)}, sampling_passes=1, "
            f"{adoption}"
        )

    prepared = prepare_physical_refine_groups(
        model=model,
        clip=clip,
        video_latents=video_latents,
        assembly_plan=assembly_plan,
        refine_context=refine_context,
        video_vae=video_vae,
        conditioning_upscale_method=conditioning_upscale_method,
        encode_prompt_fn=encode_prompt_fn,
        clone_model_fn=clone_model_fn,
        validate_refine_context_fn=validate_refine_context_fn,
        adapt_group_conditioning_fn=adapt_group_conditioning_fn,
        group_consumer_fn=sample_prepared_group,
        retain_group_outputs=False,
        selected_group_indices=(selected_indices if selection_active else None),
    )
    expected_groups = int(validation["physical_group_count"])
    expected_sampled_groups = len(selected_indices)
    if len(group_seeds) != expected_sampled_groups:
        raise SecondPassContractError(
            "targeted Sampling count differs from selected physical group count"
        )
    if len(refined_audios) != expected_groups:
        raise SecondPassContractError(
            "targeted Audio output count differs from physical group count"
        )
    if adopt_sampled_video and len(refined_videos) != expected_groups:
        raise SecondPassContractError(
            "targeted Video output count differs from physical group count"
        )

    output_videos = refined_videos if adopt_sampled_video else list(video_latents)
    updated_plan = prepared["updated_assembly_plan"]
    contract = updated_plan["second_pass_contract"]
    execution_target = "both_sampled" if adopt_sampled_video else "video_locked_audio"
    if conditioning_sources and all(
        source == "refine_context" for source in conditioning_sources
    ):
        contract["execution"] = (
            f"context_aware_physical_groups_{execution_target}_v1"
        )
    elif any(source == "refine_context" for source in conditioning_sources):
        contract["execution"] = f"mixed_context_physical_groups_{execution_target}_v1"
    else:
        contract["execution"] = f"t2va_physical_groups_{execution_target}_v1"
    contract["conditioning_sources"] = conditioning_sources
    contract["refine_seed_base"] = int(refine_seed)
    contract["refine_group_seeds"] = group_seeds
    contract["video_output"] = (
        "sampled" if adopt_sampled_video else "bit_exact_first_pass_passthrough"
    )
    contract["video_sampling"] = (
        "seeded_random_mask_1" if adopt_sampled_video else "zero_noise_mask_locked"
    )
    contract["audio_output"] = "sampled"
    contract["audio_sampling"] = "seeded_random_mask_1"
    contract["refine_schedule"] = schedule_contract
    contract["refine_schedule_identity"] = schedule_contract["schedule_hash"]
    _attach_target_contract(updated_plan, target)

    if adopt_sampled_video:
        target_line = (
            "Target: Video + Audio "
            "(GPU Experimental PASS / Production HOLD)."
        )
        sampling_line = (
            "Sampling contract: video seeded random noise/mask=1 and adopted "
            "sampled output; audio seeded random noise/mask=1 and adopted "
            "sampled output."
        )
        output_line = (
            "Outputs: sampled Video and sampled Audio adopted for every complete "
            "physical group."
        )
    else:
        target_line = "Target: Audio Only (Experimental / Production HOLD)."
        sampling_line = (
            "Sampling contract: video zero noise/mask=0 and discarded sampled "
            "output; audio seeded random noise/mask=1 and adopted sampled output."
        )
        output_line = (
            "Video output: original first-pass physical Video LATENT objects "
            "returned bit-exact; temporary sampled Video discarded."
        )
    report_lines = [
        "H3 Continuum Targeted Second Pass",
        target_line,
        f"Physical groups: {validation['physical_group_count']}.",
        sampling_line,
        "RefineSchedule: "
        f"schema={schedule_contract['schema_version']}, "
        f"mode={schedule_contract['mode']}, "
        f"evaluations={schedule_contract['evaluation_count']}, "
        f"sigma_hash={schedule_contract['sigma_hash']}.",
        *prepared["warnings"],
        *group_report_lines,
        output_line,
    ]
    if selection_active:
        report_lines.insert(
            3,
            "Selected physical groups: "
            + ", ".join(str(index + 1) for index in selected_indices)
            + "; non-selected groups were exact object passthrough with no "
            "conditioning, MODEL clone, Sampling, or seed derivation.",
        )
    return output_videos, refined_audios, updated_plan, "\n".join(report_lines)


def run_targeted_second_pass_groups(
    *,
    model: Any,
    clip: Any,
    sampler: Any,
    sigmas: torch.Tensor,
    video_latents: Sequence[Any],
    audio_latents: Sequence[Any],
    assembly_plan: Mapping[str, Any],
    refine_seed: int,
    refine_target: RefineTarget | str | None = MODE_VIDEO_ONLY,
    refine_context: Any = None,
    video_vae: Any = None,
    conditioning_upscale_method: str = "bilinear",
    enable_preview: bool = True,
    encode_prompt_fn=None,
    latent_builder=None,
    sample_fn=None,
    stream_extractor=None,
    clone_model_fn=None,
    validate_refine_context_fn=None,
    adapt_group_conditioning_fn=None,
    refine_schedule=None,
    refine_scope: str | None = SCOPE_ALL,
    refine_scope_index: int = 1,
    window_start_sec: float = 0.0,
    window_end_sec: float = 5.0,
) -> tuple[list[dict[str, Any]], list[Any], dict[str, Any], str]:
    """Run an implemented target without changing the V3.5 public node."""

    target = resolve_refine_target(refine_target)
    window: RefineWindow | None = None
    if isinstance(refine_scope, str) and refine_scope.strip().lower() == MODE_TIME_WINDOW:
        scope = None
        window = resolve_refine_window(
            window_start_sec,
            window_end_sec,
            assembly_plan,
        )
        selected_group_indices = window.selected_group_indices
        temporal_windows_by_group = {
            group.group_index: group for group in window.groups
        }
    else:
        scope = resolve_refine_scope(
            refine_scope,
            refine_scope_index,
            assembly_plan,
        )
        selected_group_indices = (
            None if scope.is_all else scope.selected_group_indices
        )
        temporal_windows_by_group = None
    arguments = dict(
        model=model,
        clip=clip,
        sampler=sampler,
        sigmas=sigmas,
        video_latents=video_latents,
        audio_latents=audio_latents,
        assembly_plan=assembly_plan,
        refine_seed=int(refine_seed),
        refine_context=refine_context,
        video_vae=video_vae,
        conditioning_upscale_method=str(conditioning_upscale_method),
        enable_preview=bool(enable_preview),
        encode_prompt_fn=encode_prompt_fn,
        latent_builder=latent_builder,
        stream_extractor=stream_extractor,
        clone_model_fn=clone_model_fn,
        validate_refine_context_fn=validate_refine_context_fn,
        adapt_group_conditioning_fn=adapt_group_conditioning_fn,
        refine_schedule=refine_schedule,
    )
    if target.mode == MODE_VIDEO_ONLY:
        result = _run_video_only_groups(
            target=target,
            sample_fn=sample_fn,
            arguments=arguments,
            selected_group_indices=selected_group_indices,
            temporal_windows_by_group=temporal_windows_by_group,
        )
    elif target.mode == MODE_AUDIO_ONLY:
        result = _run_sampled_target_groups(
            target=target,
            sample_fn=sample_fn,
            selected_group_indices=selected_group_indices,
            temporal_windows_by_group=temporal_windows_by_group,
            **arguments,
        )
    elif target.mode == MODE_VIDEO_AUDIO:
        result = _run_sampled_target_groups(
            target=target,
            sample_fn=sample_fn,
            selected_group_indices=selected_group_indices,
            temporal_windows_by_group=temporal_windows_by_group,
            **arguments,
        )
    else:
        raise TargetedRefineExecutionUnavailable(
            f"refine target {target.mode!r} is unavailable"
        )

    videos, audios, updated_plan, status = result
    if window is not None:
        contract = window.contract
        window_line = (
            "Refine Window: timeline=final_visible_output, interval=["
            f"{contract['effective_start_sec']:.6f}, "
            f"{contract['effective_end_sec']:.6f}) sec, output_frames="
            f"{contract['output_frame_range']}, selected_physical_groups="
            f"{contract['selected_physical_groups']}, continuation_prefix=protected, "
            "outside_window=mask_zero_then_bit_exact_restore, "
            "physical_group_atomicity=preserved."
        )
        return videos, audios, updated_plan, f"{status}\n{window_line}"
    assert scope is not None
    if scope.is_all:
        return videos, audios, updated_plan, status
    scope_contract = scope.contract
    scope_line = (
        "Refine Scope: "
        f"mode={scope.mode}, requested_index={scope_contract['requested_index']}, "
        "selected_physical_groups="
        f"{list(scope_contract['selected_physical_groups'])}, "
        "selected_logical_chunks="
        f"{list(scope_contract['selected_logical_chunks'])}, "
        f"terminal_expanded={str(scope_contract['terminal_expanded']).lower()}."
    )
    return videos, audios, updated_plan, f"{status}\n{scope_line}"
