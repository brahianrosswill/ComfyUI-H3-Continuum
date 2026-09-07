"""Pure visible-timeline resolution for Experimental temporal refinement."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..temporal import audio_latent_t, latent_slot_offsets, pixel_frames_for_latent_t


MAGIC = "H3_CONTINUUM_REFINE_WINDOW"
SCHEMA_VERSION = 1
MODE_TIME_WINDOW = "time_window"


class RefineWindowError(ValueError):
    """Raised before preparation when a visible time window is invalid."""


@dataclass(frozen=True)
class GroupTemporalWindow:
    """One physical group's immutable temporal mask ranges."""

    group_index: int
    logical_chunks: tuple[int, ...]
    terminal_merged: bool
    visible_frame_ranges: tuple[tuple[int, int], ...]
    physical_frame_ranges: tuple[tuple[int, int], ...]
    video_slot_ranges: tuple[tuple[int, int], ...]
    audio_tick_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RefineWindow:
    """One versioned visible-output window with no Tensor or runtime object."""

    selected_group_indices: tuple[int, ...]
    groups: tuple[GroupTemporalWindow, ...]
    contract: Mapping[str, Any]

    def group(self, group_index: int) -> GroupTemporalWindow | None:
        for group in self.groups:
            if group.group_index == int(group_index):
                return group
        return None


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _merge_ranges(values: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(
        (int(start), int(stop))
        for start, stop in values
        if int(stop) > int(start)
    )
    merged: list[list[int]] = []
    for start, stop in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((start, stop) for start, stop in merged)


def _indices_to_ranges(indices: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(set(int(value) for value in indices))
    if not ordered:
        return ()
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append((start, previous + 1))
        start = previous = value
    ranges.append((start, previous + 1))
    return tuple(ranges)


def _overlaps(
    start: int,
    stop: int,
    ranges: Sequence[tuple[int, int]],
) -> bool:
    return any(start < range_stop and stop > range_start for range_start, range_stop in ranges)


def _contract_groups(assembly_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(assembly_plan, Mapping):
        raise RefineWindowError("assembly_plan must be a mapping")
    contract = assembly_plan.get("second_pass_contract")
    if not isinstance(contract, Mapping) or contract.get("version") != 1:
        raise RefineWindowError("assembly_plan has no Second Pass contract version 1")
    groups = contract.get("physical_groups")
    if (
        not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes))
        or not groups
        or not all(isinstance(group, Mapping) for group in groups)
    ):
        raise RefineWindowError("assembly_plan has no physical groups")
    return list(groups)


def _logical_chunks(group: Mapping[str, Any], group_index: int) -> tuple[int, ...]:
    values = group.get("logical_chunks")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(type(value) is not int or value <= 0 for value in values)
    ):
        raise RefineWindowError(
            f"physical group {group_index + 1} has invalid logical chunk identity"
        )
    chunks = tuple(int(value) for value in values)
    if tuple(sorted(chunks)) != chunks or len(set(chunks)) != len(chunks):
        raise RefineWindowError(
            f"physical group {group_index + 1} logical chunks are not ordered and unique"
        )
    return chunks


def _visible_units(
    assembly_plan: Mapping[str, Any],
    groups: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, int]], int]:
    decode_groups = assembly_plan.get("decode_groups")
    if isinstance(decode_groups, Sequence) and not isinstance(
        decode_groups, (str, bytes)
    ):
        if len(decode_groups) != len(groups) or not all(
            isinstance(group, Mapping) for group in decode_groups
        ):
            raise RefineWindowError(
                "assembly_plan decode groups differ from Second Pass physical groups"
            )
        source_units: Sequence[Mapping[str, Any]] = decode_groups
    else:
        source_units = groups

    cursor = 0
    units: list[dict[str, int]] = []
    for group_index, (group, source) in enumerate(zip(groups, source_units, strict=True)):
        physical_frames = int(group.get("physical_frames", 0))
        trim_frames = int(group.get("trim_prefix_frames", -1))
        source_latent_t = int(group.get("source_latent_t", 0))
        audio_shape = group.get("source_audio_shape")
        if physical_frames <= 0 or trim_frames < 0 or trim_frames >= physical_frames:
            raise RefineWindowError(
                f"physical group {group_index + 1} frame geometry is invalid"
            )
        if source_latent_t <= 0 or pixel_frames_for_latent_t(source_latent_t) != physical_frames:
            raise RefineWindowError(
                f"physical group {group_index + 1} Video latent T differs from its frame grid"
            )
        if (
            not isinstance(audio_shape, Sequence)
            or isinstance(audio_shape, (str, bytes))
            or len(audio_shape) != 4
            or int(audio_shape[-1]) != audio_latent_t(physical_frames)
        ):
            raise RefineWindowError(
                f"physical group {group_index + 1} Audio latent T differs from its frame grid"
            )
        net_frames = physical_frames - trim_frames
        source_total = int(source.get("total_frames", physical_frames))
        source_trim = int(source.get("trim_frames", trim_frames))
        source_net = int(source.get("net_frames", source_total - source_trim))
        source_start = int(source.get("frame_start", cursor))
        source_stop = int(source.get("frame_stop", source_start + source_net))
        if (
            source_total != physical_frames
            or source_trim != trim_frames
            or source_net != net_frames
            or source_start != cursor
            or source_stop != cursor + net_frames
        ):
            raise RefineWindowError(
                f"physical group {group_index + 1} visible range is inconsistent"
            )
        units.append(
            {
                "group_index": group_index,
                "visible_start": source_start,
                "visible_stop": source_stop,
                "physical_frames": physical_frames,
                "trim_frames": trim_frames,
                "video_t": source_latent_t,
                "audio_t": int(audio_shape[-1]),
            }
        )
        cursor = source_stop
    return units, cursor


def _output_to_natural_ranges(
    output_start: int,
    output_stop: int,
    *,
    natural_frames: int,
    target_frames: int,
    preserve_final_frame: bool,
) -> tuple[tuple[int, int], ...]:
    mappings: list[tuple[int, int, int, bool]] = []
    if target_frames >= natural_frames:
        mappings.append((0, natural_frames, 0, False))
        if target_frames > natural_frames:
            mappings.append((natural_frames, target_frames, natural_frames - 1, True))
    elif preserve_final_frame and target_frames >= 2:
        mappings.extend(
            (
                (0, target_frames - 1, 0, False),
                (target_frames - 1, target_frames, natural_frames - 1, True),
            )
        )
    else:
        mappings.append((0, target_frames, 0, False))

    natural_ranges: list[tuple[int, int]] = []
    for destination_start, destination_stop, source_start, repeated in mappings:
        overlap_start = max(output_start, destination_start)
        overlap_stop = min(output_stop, destination_stop)
        if overlap_stop <= overlap_start:
            continue
        if repeated:
            natural_ranges.append((source_start, source_start + 1))
        else:
            mapped_start = source_start + overlap_start - destination_start
            mapped_stop = source_start + overlap_stop - destination_start
            natural_ranges.append((mapped_start, mapped_stop))
    return _merge_ranges(natural_ranges)


def _group_window(
    group: Mapping[str, Any],
    unit: Mapping[str, int],
    natural_ranges: Sequence[tuple[int, int]],
) -> GroupTemporalWindow | None:
    visible_start = int(unit["visible_start"])
    visible_stop = int(unit["visible_stop"])
    visible_ranges = _merge_ranges(
        (
            max(start, visible_start),
            min(stop, visible_stop),
        )
        for start, stop in natural_ranges
        if start < visible_stop and stop > visible_start
    )
    if not visible_ranges:
        return None

    trim_frames = int(unit["trim_frames"])
    physical_ranges = _merge_ranges(
        (
            trim_frames + start - visible_start,
            trim_frames + stop - visible_start,
        )
        for start, stop in visible_ranges
    )
    video_t = int(unit["video_t"])
    offsets = latent_slot_offsets(video_t)
    selected_slots = []
    for slot, slot_start in enumerate(offsets):
        slot_stop = offsets[slot + 1] if slot + 1 < video_t else int(unit["physical_frames"])
        if _overlaps(slot_start, slot_stop, physical_ranges):
            selected_slots.append(slot)
    video_ranges = _indices_to_ranges(selected_slots)
    if video_ranges and offsets[video_ranges[0][0]] < trim_frames:
        raise RefineWindowError(
            f"physical group {int(unit['group_index']) + 1} Video grid would expose "
            "the protected continuation prefix"
        )

    audio_ranges = _merge_ranges(
        (
            max(0, min(int(unit["audio_t"]), audio_latent_t(start))),
            max(0, min(int(unit["audio_t"]), audio_latent_t(stop))),
        )
        for start, stop in physical_ranges
    )
    protected_audio_stop = audio_latent_t(trim_frames)
    if audio_ranges and audio_ranges[0][0] < protected_audio_stop:
        raise RefineWindowError(
            f"physical group {int(unit['group_index']) + 1} Audio grid would expose "
            "the protected continuation prefix"
        )
    if not video_ranges or not audio_ranges:
        raise RefineWindowError(
            f"physical group {int(unit['group_index']) + 1} window resolves to an empty AV grid"
        )
    return GroupTemporalWindow(
        group_index=int(unit["group_index"]),
        logical_chunks=_logical_chunks(group, int(unit["group_index"])),
        terminal_merged=bool(group.get("terminal_merged", False)),
        visible_frame_ranges=visible_ranges,
        physical_frame_ranges=physical_ranges,
        video_slot_ranges=video_ranges,
        audio_tick_ranges=audio_ranges,
    )


def resolve_refine_window(
    start_sec: float,
    end_sec: float,
    assembly_plan: Mapping[str, Any],
) -> RefineWindow:
    """Resolve final visible output time to protected physical AV ranges.

    The requested interval is half-open.  Fractions select every output frame
    interval they overlap.  Continuation trim prefixes are never selected.
    """

    try:
        requested_start = float(start_sec)
        requested_end = float(end_sec)
    except (TypeError, ValueError) as exc:
        raise RefineWindowError("refine window seconds must be numeric") from exc
    if not math.isfinite(requested_start) or not math.isfinite(requested_end):
        raise RefineWindowError("refine window seconds must be finite")
    if requested_start >= requested_end:
        raise RefineWindowError("refine window start must be less than end")

    groups = _contract_groups(assembly_plan)
    units, natural_frames = _visible_units(assembly_plan, groups)
    fps = int(assembly_plan.get("fps", 0))
    target_frames = int(assembly_plan.get("target_frames", 0))
    if fps <= 0 or target_frames <= 0 or natural_frames <= 0:
        raise RefineWindowError("assembly_plan visible timeline is invalid")

    clamped_start = min(max(requested_start, 0.0), target_frames / fps)
    clamped_end = min(max(requested_end, 0.0), target_frames / fps)
    output_start = max(0, min(target_frames, math.floor(clamped_start * fps + 1e-9)))
    output_stop = max(0, min(target_frames, math.ceil(clamped_end * fps - 1e-9)))
    if output_stop <= output_start:
        raise RefineWindowError("refine window does not overlap the visible output")

    natural_ranges = _output_to_natural_ranges(
        output_start,
        output_stop,
        natural_frames=natural_frames,
        target_frames=target_frames,
        preserve_final_frame=bool(assembly_plan.get("preserve_final_frame", False)),
    )
    if not natural_ranges:
        raise RefineWindowError("refine window has no natural timeline source")

    resolved_groups = tuple(
        group_window
        for group, unit in zip(groups, units, strict=True)
        if (
            group_window := _group_window(
                group,
                unit,
                natural_ranges,
            )
        )
        is not None
    )
    if not resolved_groups:
        raise RefineWindowError("refine window selects no physical group")
    selected = tuple(group.group_index for group in resolved_groups)
    group_contracts = [
        {
            "group_index": group.group_index,
            "physical_group": group.group_index + 1,
            "logical_chunks": list(group.logical_chunks),
            "terminal_merged": group.terminal_merged,
            "visible_frame_ranges": [list(value) for value in group.visible_frame_ranges],
            "physical_frame_ranges": [list(value) for value in group.physical_frame_ranges],
            "video_slot_ranges": [list(value) for value in group.video_slot_ranges],
            "audio_tick_ranges": [list(value) for value in group.audio_tick_ranges],
        }
        for group in resolved_groups
    ]
    contract: dict[str, Any] = {
        "magic": MAGIC,
        "schema_version": SCHEMA_VERSION,
        "mode": MODE_TIME_WINDOW,
        "timeline": "final_visible_output",
        "interval": "half_open",
        "requested_start_sec": requested_start,
        "requested_end_sec": requested_end,
        "effective_start_sec": output_start / fps,
        "effective_end_sec": output_stop / fps,
        "output_frame_range": [output_start, output_stop],
        "target_frames": target_frames,
        "natural_frames": natural_frames,
        "natural_frame_ranges": [list(value) for value in natural_ranges],
        "selected_physical_groups": [index + 1 for index in selected],
        "groups": group_contracts,
        "continuation_prefix_policy": "always_protected",
        "outside_window_policy": "mask_zero_then_bit_exact_restore",
        "physical_group_atomicity": "preserved",
        "temporal_quantization": "intersecting_video_slots_and_audio_ticks",
    }
    contract["window_hash"] = _identity(contract)
    return RefineWindow(
        selected_group_indices=selected,
        groups=resolved_groups,
        contract=MappingProxyType(contract),
    )


def serializable_window_contract(window: RefineWindow) -> dict[str, Any]:
    if not isinstance(window, RefineWindow):
        raise RefineWindowError("refine_window must be a RefineWindow")
    return json.loads(json.dumps(dict(window.contract)))
