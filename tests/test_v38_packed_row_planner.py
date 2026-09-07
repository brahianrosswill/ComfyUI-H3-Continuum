from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join.model_patch import _wrapper_factory
from ComfyUI_H3_Continuum_Join.v3.packed_row_planner import (
    MAX_CONCURRENT_PHYSICAL_GROUPS,
    MemoryObservation,
    PackedRowPlanner,
    PackedRowPlanningError,
    analyze_packed_layout,
    planner_event_fail_soft,
)


ROOT = Path(__file__).resolve().parents[1]


def _layout(*segments):
    return SimpleNamespace(seq_len=segments[-1][1], segments=list(segments))


def _baseline_layout(*, reference_rows=0, target_video_rows=20):
    cursor = 0
    segments = [(cursor, cursor + 4, "text")]
    cursor += 4
    if reference_rows:
        segments.append((cursor, cursor + reference_rows, "ref_img"))
        cursor += reference_rows
    segments.append((cursor, cursor + 6, "audio"))
    cursor += 6
    segments.append((cursor, cursor + target_video_rows, "video"))
    return _layout(*segments)


def _memory(value):
    return MemoryObservation(
        system_available_bytes=value,
        cuda_allocated_bytes=value + 1,
        cuda_reserved_bytes=value + 2,
        core_visible_free_bytes=value + 3,
    )


def test_actual_segment_accounting_uses_core_layout_partition():
    layout = _layout(
        (0, 5, "text"),
        (5, 9, "cond"),
        (9, 12, "ref_img"),
        (12, 14, "cond_audio"),
        (14, 18, "ref_audio"),
        (18, 26, "audio"),
        (26, 56, "video"),
    )
    result = analyze_packed_layout(layout)
    assert result.seq_len == 56
    assert result.text_rows == 5
    assert result.condition_video_rows == 7
    assert result.condition_audio_rows == 6
    assert result.target_audio_rows == 8
    assert result.target_video_rows == 30
    assert result.condition_rows == 13
    assert result.target_rows == 38
    assert result.condition_to_target_ratio == pytest.approx(13 / 38)
    assert sum(stop - start for start, stop, _ in result.segment_signature) == 56


@pytest.mark.parametrize(
    "layout",
    [
        SimpleNamespace(seq_len=10, segments=[]),
        _layout((0, 3, "text"), (4, 6, "audio"), (6, 10, "video")),
        SimpleNamespace(
            seq_len=11,
            segments=[(0, 3, "text"), (3, 6, "audio"), (6, 10, "video")],
        ),
        _layout((0, 3, "text"), (3, 6, "audio"), (6, 8, "video"), (8, 10, "video")),
    ],
)
def test_invalid_layout_accounting_is_diagnostic_error(layout):
    with pytest.raises(PackedRowPlanningError):
        analyze_packed_layout(layout)


def test_unknown_segment_is_counted_without_changing_core_layout():
    layout = _layout(
        (0, 2, "text"),
        (2, 5, "future_core_segment"),
        (5, 9, "audio"),
        (9, 15, "video"),
    )
    before = tuple(layout.segments)
    result = analyze_packed_layout(layout)
    assert result.unknown_rows == 3
    assert tuple(layout.segments) == before


def test_planner_keeps_fixed_order_terminal_atomicity_and_earliest_peak_tie():
    planner = PackedRowPlanner(memory_reader=lambda: _memory(1024))
    for group, logical, terminal, layout in (
        (1, (1,), False, _baseline_layout(target_video_rows=20)),
        (2, (2,), False, _baseline_layout(reference_rows=4, target_video_rows=24)),
        (3, (3, 4), True, _baseline_layout(reference_rows=4, target_video_rows=24)),
    ):
        planner.begin_group(
            physical_group=group,
            logical_chunks=logical,
            terminal_atomic=terminal,
        )
        planner.observe_layout(layout)
        planner.observe_layout(layout)
        planner.finish_group()

    lines = planner.report_lines()
    assert MAX_CONCURRENT_PHYSICAL_GROUPS == 1
    assert "execution_order=[1, 2, 3]" in lines[0]
    assert "peak_candidate=2" in lines[0]
    assert "VRAM/time prediction=not provided" in lines[0]
    assert "physical_group=3" in lines[3]
    assert "logical_chunks=[3, 4]" in lines[3]
    assert "terminal_atomic=true" in lines[3]


def test_topology_change_within_group_becomes_unavailable_advisory():
    planner = PackedRowPlanner(memory_reader=lambda: MemoryObservation())
    planner.begin_group(physical_group=1, logical_chunks=(1,), terminal_atomic=False)
    planner.observe_layout(_baseline_layout())
    planner.observe_layout(_baseline_layout(reference_rows=2))
    planner.finish_group()
    lines = planner.report_lines()
    assert "observed_groups=0/1" in lines[0]
    assert "unavailable/advisory=PackedLayout topology changed" in lines[1]


def test_missing_layout_and_memory_counter_failure_are_fail_soft():
    def unavailable_memory():
        raise RuntimeError("counter unavailable")

    planner = PackedRowPlanner(memory_reader=unavailable_memory)
    planner.begin_group(physical_group=1, logical_chunks=(1,), terminal_atomic=False)
    planner.observe_layout(None)
    planner.finish_group()
    lines = planner.report_lines()
    assert "peak_candidate=unavailable" in lines[0]
    assert "Core PackedLayout is unavailable" in lines[1]


def test_model_wrapper_observes_target_only_layout_before_normal_passthrough():
    class Executor:
        def __call__(self, *args, **kwargs):
            return kwargs["minimax_payload"]

    planner = PackedRowPlanner(memory_reader=lambda: MemoryObservation())
    planner.begin_group(physical_group=1, logical_chunks=(1,), terminal_atomic=False)
    payload = {"layout": _baseline_layout(), "seed": 7}
    wrapper = _wrapper_factory(
        strict=True,
        debug=False,
        packed_row_planner=planner,
    )
    result = wrapper(Executor(), object(), minimax_payload=payload)
    planner.finish_group()
    assert result is payload
    assert result == {"layout": payload["layout"], "seed": 7}
    assert planner.records[0]["breakdown"].seq_len == payload["layout"].seq_len


def test_lifecycle_helper_never_propagates_diagnostic_failure():
    class BrokenPlanner:
        def explode(self):
            raise RuntimeError("diagnostic failure")

        def note_observation_failure(self, exc):
            raise RuntimeError("secondary diagnostic failure")

    assert planner_event_fail_soft(BrokenPlanner(), "explode") is None


def test_a7a_source_has_no_execution_or_allocator_mutation_calls():
    source = (ROOT / "v3" / "packed_row_planner.py").read_text(encoding="utf-8")
    for forbidden in (
        ".synchronize(",
        ".empty_cache(",
        ".reset_peak_memory_stats(",
        "unload_all_models(",
        "pynvml",
    ):
        assert forbidden not in source


def test_a7a_is_detailed_only_and_has_no_public_widget():
    sequence = (ROOT / "v2" / "sequence.py").read_text(encoding="utf-8")
    planner_creation = sequence.index("packed_row_planner=PackedRowPlanner()")
    detailed_guard = sequence.rindex(
        "if diagnostics_mode==DIAGNOSTICS_FULL:", 0, planner_creation
    )
    assert detailed_guard < planner_creation
    public_nodes = (ROOT / "v3" / "nodes.py").read_text(encoding="utf-8")
    selective_node = (ROOT / "v3" / "second_pass_nodes.py").read_text(
        encoding="utf-8"
    )
    assert "Packed-row" not in public_nodes
    assert "Packed-row" not in selective_node
