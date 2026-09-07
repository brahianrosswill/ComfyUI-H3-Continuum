"""Read-only V3.8 reliability diagnostics.

The helpers in this module inspect shapes and existing metadata only.  They do
not resize, encode, hash, copy, or otherwise modify user tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..compatibility import check_comfy_h3_runtime
from ..constants import (
    DIAGNOSTICS_BASIC,
    DIAGNOSTICS_FULL,
    DIAGNOSTICS_OFF,
    FPS,
    normalize_diagnostics_mode,
)
from ..reference import resolve_reference_image_size
from ..reference_video import (
    REFERENCE_VIDEO_SIZE_EFFICIENT,
    resolve_reference_video_frame_count,
    resolve_reference_video_size,
)
from ..temporal import (
    align_frame_count_up,
    audio_grid_offset,
    audio_latent_t,
    context_slots,
    is_valid_frame_count,
    video_latent_t,
)


VISUAL_LOAD_LOW_MAX = 0.75
VISUAL_LOAD_MEDIUM_MAX = 1.75


@dataclass(frozen=True, slots=True)
class DiagnosticSection:
    basic: tuple[str, ...] = ()
    detailed: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class V38Diagnostics:
    basic: tuple[str, ...]
    detailed: tuple[str, ...]


def _combine_sections(*sections: DiagnosticSection) -> V38Diagnostics:
    return V38Diagnostics(
        basic=tuple(line for section in sections for line in section.basic),
        detailed=tuple(line for section in sections for line in section.detailed),
    )


def _physical_groups(assembly_plan: Any) -> list[Any]:
    if not isinstance(assembly_plan, Mapping):
        return []
    decode_groups = assembly_plan.get("decode_groups")
    if isinstance(decode_groups, list) and decode_groups:
        return list(decode_groups)
    chunks = assembly_plan.get("chunks")
    return list(chunks) if isinstance(chunks, list) else []


def _latent_outputs(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _latent_samples(value: Any) -> Any:
    return value.get("samples") if isinstance(value, Mapping) else None


def _image_geometry(value: Any) -> tuple[int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError("IMAGE must have shape [B,H,W,C]")
    height = int(shape[1])
    width = int(shape[2])
    if height < 1 or width < 1:
        raise ValueError("IMAGE geometry must be positive")
    return width, height


def _video_guide_geometry(value: Any) -> tuple[int, int, int]:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError("Video Guide must have shape [T,H,W,C]")
    frames = int(shape[0])
    height = int(shape[1])
    width = int(shape[2])
    if frames < 1 or height < 1 or width < 1:
        raise ValueError("Video Guide geometry must be positive")
    return frames, width, height


def _native_chunk_frames(chunk_seconds: float) -> int:
    return align_frame_count_up(int(round(float(chunk_seconds) * FPS)))


def build_frame_alignment_diagnostics(
    *,
    video_latents: Any,
    assembly_plan: Any,
    video_guide: Any = None,
    chunk_seconds: float,
) -> DiagnosticSection:
    """Inspect Video Guide alignment and physical video latent lengths."""

    basic: list[str] = []
    detailed: list[str] = []
    if video_guide is not None:
        try:
            source_frames, _source_width, _source_height = _video_guide_geometry(
                video_guide
            )
            aligned_frames = resolve_reference_video_frame_count(
                source_frames,
                _native_chunk_frames(chunk_seconds),
            )
            if source_frames < aligned_frames:
                action = "final-frame pad"
            elif source_frames > aligned_frames:
                action = "trimmed to one-chunk H3 limit"
            else:
                action = "unchanged"
            line = (
                f"Frame alignment: Video Guide {source_frames} -> {aligned_frames} "
                f"({action}, H3 17k+5)."
            )
            if action != "unchanged":
                basic.append(line)
            detailed.append(line)
        except Exception as exc:
            line = (
                "Frame alignment advisory: Video Guide alignment could not be "
                f"inspected ({type(exc).__name__}: {exc})."
            )
            basic.append(line)
            detailed.append(line)

    groups = _physical_groups(assembly_plan)
    outputs = _latent_outputs(video_latents)
    if not groups:
        line = "Frame alignment advisory: physical group metadata is unavailable."
        basic.append(line)
        detailed.append(line)
        return DiagnosticSection(tuple(basic), tuple(detailed))
    if len(outputs) != len(groups):
        line = (
            "Frame alignment advisory: physical video output count "
            f"{len(outputs)} does not match group count {len(groups)}."
        )
        basic.append(line)
        detailed.append(line)

    for index, group in enumerate(groups, start=1):
        problems: list[str] = []
        try:
            if not isinstance(group, Mapping):
                raise ValueError("group metadata is not a mapping")
            frames = int(group["total_frames"])
            trim = int(group["trim_frames"])
            net = int(group["net_frames"])
            metadata_t = int(group["expected_video_latent_t"])
            if not is_valid_frame_count(frames):
                problems.append("frames are not on H3 17k+5")
                expected_t = None
            else:
                expected_t = video_latent_t(frames)
                if metadata_t != expected_t:
                    problems.append("metadata T differs from native grid")
            if frames - trim != net:
                problems.append("total-trim differs from net")
            samples = (
                _latent_samples(outputs[index - 1])
                if index <= len(outputs)
                else None
            )
            if not torch.is_tensor(samples) or samples.ndim < 3:
                actual_t = None
                problems.append("video LATENT samples are missing or invalid")
            else:
                actual_t = int(samples.shape[2])
                if actual_t != metadata_t:
                    problems.append("actual T differs from metadata")
                if expected_t is not None and actual_t != expected_t:
                    problems.append("actual T differs from native grid")
            result = "PASS" if not problems else "ADVISORY"
            expected_text = "n/a" if expected_t is None else str(expected_t)
            actual_text = "n/a" if actual_t is None else str(actual_t)
            detailed.append(
                f"Frame grid: group={index} frames={frames} trim={trim} net={net} "
                f"video_t={actual_text} metadata={metadata_t} "
                f"expected={expected_text} {result}."
            )
        except Exception as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
            detailed.append(
                f"Frame grid: group={index} unavailable ADVISORY "
                f"({problems[-1]})."
            )
        if problems:
            basic.append(
                f"Frame alignment advisory: group {index}: "
                + "; ".join(problems)
                + "."
            )
    return DiagnosticSection(tuple(basic), tuple(detailed))


def validate_audio_latent_lengths(
    *,
    audio_latents: Any,
    assembly_plan: Any,
) -> DiagnosticSection:
    """Validate physical audio latent shapes without reading tensor values."""

    basic: list[str] = []
    detailed: list[str] = []
    groups = _physical_groups(assembly_plan)
    outputs = _latent_outputs(audio_latents)
    if not groups:
        line = "Audio latent advisory: physical group metadata is unavailable."
        return DiagnosticSection((line,), (line,))
    if len(outputs) != len(groups):
        line = (
            f"Audio latent advisory: physical output count {len(outputs)} "
            f"does not match group count {len(groups)}."
        )
        basic.append(line)
        detailed.append(line)

    for index, group in enumerate(groups, start=1):
        problems: list[str] = []
        frames = metadata_t = nominal_t = actual_t = None
        phase = None
        try:
            if not isinstance(group, Mapping):
                raise ValueError("group metadata is not a mapping")
            frames = int(group["total_frames"])
            metadata_t = int(group["expected_audio_latent_t"])
            nominal_t = audio_latent_t(frames)
            samples = (
                _latent_samples(outputs[index - 1])
                if index <= len(outputs)
                else None
            )
            if not torch.is_tensor(samples):
                problems.append("LATENT samples are missing or not a Tensor")
            elif samples.ndim != 4:
                problems.append(f"rank is {samples.ndim}, expected 4")
            else:
                if int(samples.shape[0]) < 1:
                    problems.append("batch dimension is empty")
                if int(samples.shape[1]) != 32:
                    problems.append(f"channels={int(samples.shape[1])}, expected 32")
                if int(samples.shape[2]) != 2:
                    problems.append(f"stereo={int(samples.shape[2])}, expected 2")
                actual_t = int(samples.shape[-1])
                if actual_t <= 0:
                    problems.append("T must be positive")
                else:
                    phase = audio_grid_offset(frames, actual_t)
                    if actual_t != metadata_t:
                        problems.append(
                            f"metadata={metadata_t} actual={actual_t} "
                            f"delta={actual_t - metadata_t}"
                        )
                    if not -0.500001 <= phase <= 0.500001:
                        problems.append(f"phase={phase:.6f} is outside [-0.5,0.5]")
            result = "PASS" if not problems else "ADVISORY"
            detailed.append(
                f"Audio latent check: group={index} frames={frames} "
                f"metadata={metadata_t} actual="
                f"{'n/a' if actual_t is None else actual_t} nominal={nominal_t} "
                f"phase={'n/a' if phase is None else f'{phase:.6f}'} {result}."
            )
        except Exception as exc:
            problems.append(f"{type(exc).__name__}: {exc}")
            detailed.append(
                f"Audio latent check: group={index} unavailable ADVISORY "
                f"({problems[-1]})."
            )
        if problems:
            basic.append(
                f"Audio latent advisory: group {index}: "
                + "; ".join(problems)
                + "."
            )
    return DiagnosticSection(tuple(basic), tuple(detailed))


def collect_core_compatibility_advisory(
    checker: Callable[[], list[str]] | None = None,
) -> DiagnosticSection:
    """Collect the existing feature-based Core advisory after successful generation."""

    active_checker = checker or check_comfy_h3_runtime
    try:
        issues = tuple(dict.fromkeys(str(issue) for issue in active_checker() if issue))
    except Exception as exc:
        basic = (
            "Core compatibility advisory unavailable; generation result was preserved.",
        )
        detailed = (
            basic[0][:-1] + f" ({type(exc).__name__}: {exc}).",
        )
        return DiagnosticSection(basic, detailed)
    if not issues:
        return DiagnosticSection(
            (),
            ("Core compatibility: native MiniMax H3 contracts detected.",),
        )
    return DiagnosticSection(
        (
            "Core compatibility advisory: "
            f"{len(issues)} native H3 contract difference(s) detected; "
            "generation completed; enable Detailed Report for details.",
        ),
        tuple(f"Core compatibility advisory: {issue}." for issue in issues),
    )


def _visual_load_class(total: float) -> str:
    if total < VISUAL_LOAD_LOW_MAX:
        return "Low"
    if total < VISUAL_LOAD_MEDIUM_MAX:
        return "Medium"
    return "High"


def estimate_visual_conditioning_load(
    *,
    output_width: int,
    output_height: int,
    reference_images: Sequence[Any],
    reference_size: str,
    video_guide: Any,
    video_guide_size: str = REFERENCE_VIDEO_SIZE_EFFICIENT,
    chunk_seconds: float,
    still_guide_active: bool,
) -> DiagnosticSection:
    """Estimate peak per-group relative visual conditioning from geometry only."""

    output_area = int(output_width) * int(output_height)
    if output_area <= 0:
        line = "Visual conditioning load advisory: output geometry is invalid."
        return DiagnosticSection((line,), (line,))

    components: list[tuple[str, float]] = []
    problems: list[str] = []
    for socket_index, image in enumerate(reference_images, start=1):
        if image is None:
            continue
        try:
            source_width, source_height = _image_geometry(image)
            target_width, target_height = resolve_reference_image_size(
                source_width,
                source_height,
                output_width=int(output_width),
                output_height=int(output_height),
                size_mode=str(reference_size),
            )
            load = (target_width * target_height) / float(output_area)
            components.append(
                (f"Ref{socket_index}={target_width}x{target_height}", load)
            )
        except Exception as exc:
            problems.append(
                f"Reference Image {socket_index} unavailable "
                f"({type(exc).__name__}: {exc})"
            )

    if video_guide is not None:
        try:
            source_frames, source_width, source_height = _video_guide_geometry(
                video_guide
            )
            native_frames = _native_chunk_frames(chunk_seconds)
            aligned_frames = resolve_reference_video_frame_count(
                source_frames,
                native_frames,
            )
            target_width, target_height = resolve_reference_video_size(
                source_width,
                source_height,
                output_width=int(output_width),
                output_height=int(output_height),
                size_mode=str(video_guide_size),
            )
            spatial_ratio = (target_width * target_height) / float(output_area)
            temporal_ratio = video_latent_t(aligned_frames) / float(
                video_latent_t(native_frames)
            )
            components.append(
                (
                    f"Video Guide={target_width}x{target_height}/{aligned_frames}f",
                    spatial_ratio * temporal_ratio,
                )
            )
        except Exception as exc:
            problems.append(
                f"Video Guide unavailable ({type(exc).__name__}: {exc})"
            )

    basic: list[str] = []
    detailed: list[str] = []
    if components:
        total = sum(load for _name, load in components)
        classification = _visual_load_class(total)
        basic_line = (
            f"Visual conditioning load: {classification} "
            "(peak per physical group)."
        )
        if still_guide_active:
            basic_line = basic_line[:-1] + "; Still Guide active."
        basic.append(basic_line)
        details = "; ".join(
            f"{name}={load:.3f}" for name, load in components
        )
        detailed_line = (
            f"Visual conditioning load: total={total:.3f} "
            f"class={classification}; {details}"
        )
        if still_guide_active:
            detailed_line += "; Still Guide=active"
        detailed.append(detailed_line + ".")
    elif still_guide_active:
        basic.append("Still Guide: active.")
        detailed.append("Still Guide: active (excluded from numeric visual load).")

    for problem in problems:
        line = f"Visual conditioning load advisory: {problem}."
        basic.append(line)
        detailed.append(line)
    return DiagnosticSection(tuple(basic), tuple(detailed))


@dataclass(frozen=True, slots=True)
class _LogicalChunkGeometry:
    index: int
    total_frames: int | None
    trim_frames: int | None
    net_frames: int | None
    derived_start: int
    derived_stop: int
    visible_start: int | None
    visible_stop: int | None
    problems: tuple[str, ...]


def _chunk_boundary_advisory(message: str) -> str:
    return f"Chunk boundary advisory: {str(message).rstrip('.')}."


def _logical_chunk_geometries(
    assembly_plan: Any,
) -> tuple[list[_LogicalChunkGeometry], list[str]]:
    problems: list[str] = []
    if not isinstance(assembly_plan, Mapping):
        return [], ["Assembly Plan metadata is unavailable"]
    raw_chunks = assembly_plan.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        return [], ["logical chunk metadata is unavailable"]

    geometries: list[_LogicalChunkGeometry] = []
    cursor = 0
    for position, chunk in enumerate(raw_chunks, start=1):
        local: list[str] = []
        total = trim = net = None
        visible_start = visible_stop = None
        index = position
        if not isinstance(chunk, Mapping):
            local.append("metadata is not a mapping")
        else:
            try:
                index = int(chunk.get("chunk_index", chunk.get("sequence_index", position)))
                if index != position:
                    local.append(
                        f"logical order has Chunk {index} at position {position}"
                    )
            except Exception:
                index = position
                local.append("logical chunk index is invalid")
            try:
                total = int(chunk["total_frames"])
                trim = int(chunk["trim_frames"])
                net = int(chunk["net_frames"])
                if total < 0:
                    local.append("total_frames is negative")
                if trim < 0:
                    local.append("trim_frames is negative")
                if trim > total:
                    local.append("trim_frames exceeds total_frames")
                if net < 0:
                    local.append("net_frames is negative")
                if total - trim != net:
                    local.append("total_frames - trim_frames differs from net_frames")
            except Exception as exc:
                local.append(
                    "required geometry is unavailable "
                    f"({type(exc).__name__}: {exc})"
                )

        derived_start = cursor
        derived_stop = cursor + (net if net is not None and net >= 0 else 0)
        if isinstance(chunk, Mapping) and "frame_start" in chunk:
            try:
                visible_start = int(chunk["frame_start"])
                if visible_start != derived_start:
                    local.append(
                        f"frame_start={visible_start} differs from derived {derived_start}"
                    )
            except Exception:
                local.append("frame_start is invalid")
        else:
            visible_start = derived_start if net is not None else None
        if isinstance(chunk, Mapping) and "frame_stop" in chunk:
            try:
                visible_stop = int(chunk["frame_stop"])
                if visible_stop != derived_stop:
                    local.append(
                        f"frame_stop={visible_stop} differs from derived {derived_stop}"
                    )
            except Exception:
                local.append("frame_stop is invalid")
        else:
            visible_stop = derived_stop if net is not None else None

        geometries.append(
            _LogicalChunkGeometry(
                index=index,
                total_frames=total,
                trim_frames=trim,
                net_frames=net,
                derived_start=derived_start,
                derived_stop=derived_stop,
                visible_start=visible_start,
                visible_stop=visible_stop,
                problems=tuple(local),
            )
        )
        problems.extend(f"Chunk {position}: {problem}" for problem in local)
        cursor = derived_stop
    return geometries, problems


def _physical_boundary_mapping(
    *,
    assembly_plan: Any,
    logical_count: int,
    video_latents: Any,
    audio_latents: Any,
) -> tuple[list[Mapping[str, Any]], list[tuple[int, ...]], dict[int, int], list[str]]:
    problems: list[str] = []
    if not isinstance(assembly_plan, Mapping):
        return [], [], {}, ["physical group metadata is unavailable"]

    if "decode_groups" in assembly_plan:
        raw_groups = assembly_plan.get("decode_groups")
        if not isinstance(raw_groups, list) or not raw_groups:
            return [], [], {}, ["physical groups are not a non-empty list"]
        groups = list(raw_groups)
        fallback = False
    else:
        raw_chunks = assembly_plan.get("chunks")
        if not isinstance(raw_chunks, list) or not raw_chunks:
            return [], [], {}, ["physical group metadata is unavailable"]
        groups = list(raw_chunks)
        fallback = True

    normalized_groups: list[Mapping[str, Any]] = []
    logical_groups: list[tuple[int, ...]] = []
    logical_to_group: dict[int, int] = {}
    flattened: list[int] = []
    for group_index, group in enumerate(groups, start=1):
        if not isinstance(group, Mapping):
            problems.append(f"physical group {group_index} metadata is not a mapping")
            normalized_groups.append({})
            logical_groups.append(())
            continue
        normalized_groups.append(group)
        raw_logical = (
            [group.get("chunk_index", group.get("sequence_index", group_index))]
            if fallback
            else group.get("logical_chunk_indices")
        )
        if not isinstance(raw_logical, (list, tuple)) or not raw_logical:
            problems.append(f"physical group {group_index} has no logical chunks")
            logical_groups.append(())
            continue
        try:
            logical = tuple(int(value) for value in raw_logical)
        except Exception:
            problems.append(f"physical group {group_index} logical chunks are invalid")
            logical_groups.append(())
            continue
        logical_groups.append(logical)
        flattened.extend(logical)
        if tuple(sorted(logical)) != logical:
            problems.append(f"physical group {group_index} logical order is reversed")
        terminal = bool(group.get("terminal_merged", False))
        if terminal:
            expected_terminal = (logical_count - 1, logical_count)
            if len(logical) != 2 or logical != expected_terminal:
                problems.append(
                    f"physical group {group_index} Terminal Merge mapping is invalid"
                )
        elif len(logical) != 1:
            problems.append(
                f"physical group {group_index} has multiple logical chunks without Terminal Merge"
            )
        for chunk_index in logical:
            if chunk_index < 1 or chunk_index > logical_count:
                problems.append(
                    f"physical group {group_index} references invalid Chunk {chunk_index}"
                )
            elif chunk_index in logical_to_group:
                problems.append(f"logical Chunk {chunk_index} is mapped more than once")
            else:
                logical_to_group[chunk_index] = group_index

    expected = list(range(1, logical_count + 1))
    if flattened != expected:
        missing = [value for value in expected if value not in flattened]
        if missing:
            problems.append(
                "physical group mapping does not cover all logical chunks "
                f"(missing={missing})"
            )
        if not missing and flattened != sorted(flattened):
            problems.append("physical group logical chunk order is not contiguous")

    video_count = len(_latent_outputs(video_latents))
    audio_count = len(_latent_outputs(audio_latents))
    if video_count != len(groups) or audio_count != len(groups):
        problems.append(
            "physical output/group counts differ "
            f"(groups={len(groups)}, video={video_count}, audio={audio_count})"
        )
    return normalized_groups, logical_groups, logical_to_group, problems


def _second_pass_boundary_contract_problems(
    *,
    assembly_plan: Any,
    groups: Sequence[Mapping[str, Any]],
    logical_groups: Sequence[tuple[int, ...]],
) -> tuple[list[str], str | None]:
    if not isinstance(assembly_plan, Mapping) or "second_pass_contract" not in assembly_plan:
        return [], None
    contract = assembly_plan.get("second_pass_contract")
    if not isinstance(contract, Mapping):
        return ["Second Pass contract is not a mapping"], "Second Pass contract: UNAVAILABLE."
    contract_groups = contract.get("physical_groups")
    if not isinstance(contract_groups, list):
        return ["Second Pass physical_groups is not a list"], "Second Pass contract: UNAVAILABLE."

    problems: list[str] = []
    if len(contract_groups) != len(groups):
        problems.append(
            "Second Pass physical group count differs from decode groups "
            f"({len(contract_groups)} != {len(groups)})"
        )
    for index, (decode_group, logical) in enumerate(
        zip(groups, logical_groups, strict=False)
    ):
        if index >= len(contract_groups):
            break
        contract_group = contract_groups[index]
        if not isinstance(contract_group, Mapping):
            problems.append(f"Second Pass group {index} is not a mapping")
            continue
        raw_contract_logical = contract_group.get("logical_chunks")
        contract_logical = (
            tuple(raw_contract_logical)
            if isinstance(raw_contract_logical, (list, tuple))
            else ()
        )
        checks = (
            ("group_id", contract_group.get("group_id"), index),
            ("logical_chunks", contract_logical, logical),
            (
                "physical_frames",
                contract_group.get("physical_frames"),
                decode_group.get("total_frames"),
            ),
            (
                "trim_prefix_frames",
                contract_group.get("trim_prefix_frames"),
                decode_group.get("trim_frames"),
            ),
            (
                "terminal_merged",
                bool(contract_group.get("terminal_merged", False)),
                bool(decode_group.get("terminal_merged", False)),
            ),
        )
        for name, actual, expected in checks:
            if actual != expected:
                problems.append(
                    f"Second Pass group {index} {name} differs "
                    f"({actual!r} != {expected!r})"
                )
    result = "PASS" if not problems else "ADVISORY"
    return problems, f"Second Pass contract: groups={len(contract_groups)} {result}."


def build_chunk_boundary_diagnostics(
    *,
    video_latents: Any,
    audio_latents: Any,
    assembly_plan: Any,
) -> DiagnosticSection:
    """Inspect logical/physical chunk boundaries using metadata and shapes only."""

    detailed = [
        "Chunk boundary contract: structural diagnostic only; "
        "no image/audio similarity measurement."
    ]
    basic: list[str] = []
    geometries, logical_problems = _logical_chunk_geometries(assembly_plan)
    groups, logical_groups, logical_to_group, mapping_problems = (
        _physical_boundary_mapping(
            assembly_plan=assembly_plan,
            logical_count=len(geometries),
            video_latents=video_latents,
            audio_latents=audio_latents,
        )
    )
    second_pass_problems, second_pass_line = _second_pass_boundary_contract_problems(
        assembly_plan=assembly_plan,
        groups=groups,
        logical_groups=logical_groups,
    )
    global_problems = logical_problems + mapping_problems + second_pass_problems
    for problem in dict.fromkeys(global_problems):
        basic.append(_chunk_boundary_advisory(problem))
    if second_pass_line is not None:
        detailed.append(second_pass_line)

    for boundary_index in range(1, len(geometries)):
        left = geometries[boundary_index - 1]
        right = geometries[boundary_index]
        local = list(left.problems) + list(right.problems)
        if left.visible_stop is None or right.visible_start is None:
            gap = None
            local.append("visible range is unavailable")
        else:
            gap = int(right.visible_start) - int(left.visible_stop)
            if gap != 0:
                local.append("non-contiguous visible range")

        context_frames = right.trim_frames
        context_video_t = context_audio_t = None
        if context_frames is None:
            local.append("incoming context is unavailable")
        else:
            try:
                context_video_t = context_slots(context_frames)
                context_audio_t = audio_latent_t(context_frames)
            except Exception as exc:
                local.append(
                    f"incoming context {context_frames}f is invalid "
                    f"({type(exc).__name__}: {exc})"
                )

        left_group = logical_to_group.get(left.index)
        right_group = logical_to_group.get(right.index)
        if left_group is None or right_group is None:
            local.append("physical group mapping is unavailable")
            classification = "UNAVAILABLE"
            detailed.append(
                f"Boundary {left.index}→{right.index}: physical mapping unavailable; "
                f"UNAVAILABLE ({'; '.join(dict.fromkeys(local))})."
            )
        elif left_group == right_group:
            group = groups[right_group - 1] if right_group <= len(groups) else {}
            terminal = bool(group.get("terminal_merged", False))
            expected_pair = (left.index, right.index)
            actual_pair = (
                logical_groups[right_group - 1]
                if right_group <= len(logical_groups)
                else ()
            )
            if not terminal:
                local.append("shared physical group lacks terminal_merged=yes")
            if actual_pair != expected_pair:
                local.append("shared physical group does not match this boundary")
            classification = "PASS" if not local else "ADVISORY"
            detailed.append(
                f"Boundary {left.index}→{right.index}: shared physical group={right_group}; "
                f"terminal_merged={'yes' if terminal else 'no'}; "
                "no external decode boundary; "
                f"{classification}"
                + (f" ({'; '.join(dict.fromkeys(local))})" if local else "")
                + "."
            )
        else:
            classification = "PASS" if not local else "ADVISORY"
            visible_left = "n/a" if left.visible_stop is None else str(left.visible_stop)
            visible_right = "n/a" if right.visible_start is None else str(right.visible_start)
            gap_text = "n/a" if gap is None else str(gap)
            context_text = "n/a" if context_frames is None else f"{context_frames}f"
            video_text = "n/a" if context_video_t is None else f"{context_video_t}T"
            audio_text = "n/a" if context_audio_t is None else f"{context_audio_t}T"
            right_text = (
                "right=unavailable"
                if right.total_frames is None
                or right.trim_frames is None
                or right.net_frames is None
                else (
                    f"right={right.total_frames}f trim={right.trim_frames} "
                    f"net={right.net_frames}"
                )
            )
            detailed.append(
                f"Boundary {left.index}→{right.index}: physical; "
                f"visible={visible_left}→{visible_right} gap={gap_text}; "
                f"context={context_text} video={video_text} audio={audio_text}; "
                f"{right_text}; {classification}"
                + (f" ({'; '.join(dict.fromkeys(local))})" if local else "")
                + "."
            )

        if local:
            if gap not in (None, 0):
                message = (
                    f"boundary {left.index}→{right.index} has a non-contiguous "
                    "visible range"
                )
            elif classification == "UNAVAILABLE":
                message = f"boundary {left.index}→{right.index} is unavailable"
            else:
                message = (
                    f"boundary {left.index}→{right.index}: "
                    + "; ".join(dict.fromkeys(local))
                )
            advisory = _chunk_boundary_advisory(message)
            if advisory not in basic:
                basic.append(advisory)

    return DiagnosticSection(tuple(basic), tuple(detailed))


def build_v38_diagnostics(
    *,
    video_latents: Any,
    audio_latents: Any,
    assembly_plan: Any,
    output_width: int,
    output_height: int,
    chunk_seconds: float,
    reference_images: Sequence[Any] = (),
    reference_size: str = "Match Output",
    video_guide: Any = None,
    video_guide_size: str = REFERENCE_VIDEO_SIZE_EFFICIENT,
    still_guide_active: bool = False,
    core_checker: Callable[[], list[str]] | None = None,
) -> V38Diagnostics:
    return _combine_sections(
        build_frame_alignment_diagnostics(
            video_latents=video_latents,
            assembly_plan=assembly_plan,
            video_guide=video_guide,
            chunk_seconds=float(chunk_seconds),
        ),
        validate_audio_latent_lengths(
            audio_latents=audio_latents,
            assembly_plan=assembly_plan,
        ),
        build_chunk_boundary_diagnostics(
            video_latents=video_latents,
            audio_latents=audio_latents,
            assembly_plan=assembly_plan,
        ),
        collect_core_compatibility_advisory(core_checker),
        estimate_visual_conditioning_load(
            output_width=int(output_width),
            output_height=int(output_height),
            reference_images=tuple(reference_images),
            reference_size=str(reference_size),
            video_guide=video_guide,
            video_guide_size=str(video_guide_size),
            chunk_seconds=float(chunk_seconds),
            still_guide_active=bool(still_guide_active),
        ),
    )


def format_v38_diagnostics(
    diagnostics: V38Diagnostics,
    *,
    mode: str = DIAGNOSTICS_BASIC,
) -> str:
    normalized = normalize_diagnostics_mode(str(mode))
    if normalized == DIAGNOSTICS_OFF:
        return ""
    lines = diagnostics.detailed if normalized == DIAGNOSTICS_FULL else diagnostics.basic
    if not lines:
        return ""
    return "V3.8 Reliability\n" + "\n".join(lines)


def append_v38_status(
    status: Any,
    diagnostics: V38Diagnostics,
    *,
    mode: str = DIAGNOSTICS_BASIC,
) -> str:
    suffix = format_v38_diagnostics(diagnostics, mode=mode)
    original = str(status)
    return original if not suffix else original.rstrip() + "\n" + suffix
