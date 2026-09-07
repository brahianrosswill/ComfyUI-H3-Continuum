from __future__ import annotations

from pathlib import Path

import pytest

from ComfyUI_H3_Continuum_Join.v3.adaptive_continuity import (
    ADAPTIVE_CONTINUITY_OBSERVER_VERSION,
    ADAPTIVE_CONTINUITY_PLANNER_HASH,
    AdaptiveContinuityObservationError,
    AdaptiveContinuityObserver,
    EXECUTION_APPLIED,
    INITIAL_TRANSPORT,
    adaptive_observer_event_fail_soft,
    make_adaptive_continuity_decision,
)


ROOT = Path(__file__).resolve().parents[1]


def _entry(context_frames, motion_score):
    return {
        "context_frames": context_frames,
        "motion_score": motion_score,
        "video": object(),
        "audio": object(),
    }


def test_planner_contract_and_decision_hash_are_deterministic():
    kwargs = {
        "physical_group": 2,
        "logical_chunks": (2,),
        "observed_motion_score": 0.075,
        "current_context_frames": 22,
        "resolved_transport": "masked_av_prefix_22_v1",
        "terminal_atomic": False,
        "boundary_index": 1,
        "boundary_count": 2,
        "reason": "fixed profile",
    }
    first = make_adaptive_continuity_decision(**kwargs)
    second = make_adaptive_continuity_decision(**kwargs)
    assert ADAPTIVE_CONTINUITY_OBSERVER_VERSION == 1
    assert len(ADAPTIVE_CONTINUITY_PLANNER_HASH) == 64
    assert first == second
    assert len(first["decision_hash"]) == 64
    assert first["execution_applied"] is EXECUTION_APPLIED is False
    assert first["issue_13_mitigation_claim"] is False
    assert first["confidence_scope"].endswith("not mitigation efficacy")


def test_initial_group_is_trace_only_and_has_no_continuation_action():
    decision = make_adaptive_continuity_decision(
        physical_group=1,
        logical_chunks=(1,),
        observed_motion_score=0.0,
        current_context_frames=0,
        resolved_transport=INITIAL_TRANSPORT,
        terminal_atomic=False,
        boundary_index=0,
        boundary_count=0,
        reason="initial clip",
    )
    assert decision["recommendation"] == "no_continuation_action_initial_group"
    assert decision["execution_applied"] is False


def test_reference_context_and_masked_contracts_are_only_recommended_not_applied():
    common = {
        "physical_group": 2,
        "logical_chunks": (2,),
        "observed_motion_score": 0.1,
        "current_context_frames": 22,
        "terminal_atomic": False,
        "boundary_index": 1,
        "boundary_count": 2,
        "reason": "conservative balanced fallback",
    }
    reference = make_adaptive_continuity_decision(
        **common, resolved_transport="reference_context_v1"
    )
    masked = make_adaptive_continuity_decision(
        **common, resolved_transport="masked_av_prefix_22_v1"
    )
    assert reference["recommendation"] == "keep_current_reference_context_contract"
    assert masked["recommendation"] == "keep_current_masked_prefix_contract"
    assert reference["execution_applied"] is masked["execution_applied"] is False


def test_terminal_recommendation_keeps_atomic_fixed_contract():
    decision = make_adaptive_continuity_decision(
        physical_group=2,
        logical_chunks=(2, 3),
        observed_motion_score=0.05,
        current_context_frames=22,
        resolved_transport="masked_av_prefix_22_v1",
        terminal_atomic=True,
        boundary_index=1,
        boundary_count=2,
        reason="terminal merged 10-second sample",
    )
    assert decision["terminal_atomic"] is True
    assert decision["recommendation"] == "keep_fixed_terminal_merge_contract"
    assert decision["execution_applied"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"current_context_frames": 13},
        {"resolved_transport": "dynamic_transport"},
        {"observed_motion_score": float("nan")},
        {"boundary_index": 3},
    ],
)
def test_malformed_observation_is_rejected_by_pure_builder(override):
    kwargs = {
        "physical_group": 2,
        "logical_chunks": (2,),
        "observed_motion_score": 0.1,
        "current_context_frames": 22,
        "resolved_transport": "masked_av_prefix_22_v1",
        "terminal_atomic": False,
        "boundary_index": 1,
        "boundary_count": 2,
        "reason": "fixed profile",
    }
    kwargs.update(override)
    with pytest.raises(AdaptiveContinuityObservationError):
        make_adaptive_continuity_decision(**kwargs)


def test_fail_soft_event_records_advisory_without_raising():
    observer = AdaptiveContinuityObserver()
    result = adaptive_observer_event_fail_soft(
        observer,
        "observe",
        physical_group=2,
        logical_chunks=(2,),
        observed_motion_score=0.1,
        current_context_frames=13,
        resolved_transport="masked_av_prefix_22_v1",
        terminal_atomic=False,
        boundary_index=1,
        boundary_count=2,
        reason="invalid diagnostic input",
    )
    assert result is None
    lines = observer.report_lines()
    assert "observed_groups=0" in lines[0]
    assert "unavailable/advisory" not in lines[0]
    assert "Adaptive Continuity advisory [A8a v1]" in lines[1]
    assert "execution_applied=false" in lines[1]


def test_replayed_normal_prefix_uses_metadata_without_retaining_payloads():
    entries = [_entry(0, 0.0), _entry(22, 0.07), _entry(22, 0.08)]
    observer = AdaptiveContinuityObserver()
    observer.replay_prefix(
        entries=entries,
        total_chunks=3,
        terminal_merge_enabled=False,
        resolved_transport="masked_av_prefix_22_v1",
    )
    assert [record["physical_group"] for record in observer.records] == [1, 2, 3]
    assert all(record["reused"] for record in observer.records)
    assert all("video" not in record and "audio" not in record for record in observer.records)


def test_replayed_terminal_pair_is_one_atomic_physical_record():
    entries = [_entry(0, 0.0), _entry(22, 0.09), _entry(22, 0.09)]
    observer = AdaptiveContinuityObserver()
    observer.replay_prefix(
        entries=entries,
        total_chunks=3,
        terminal_merge_enabled=True,
        resolved_transport="masked_av_prefix_22_v1",
    )
    records = observer.records
    assert len(records) == 2
    assert records[1]["logical_chunks"] == [2, 3]
    assert records[1]["terminal_atomic"] is True
    assert records[1]["boundary_index"] == 1
    assert records[1]["boundary_count"] == 2


def test_report_contains_required_contract_fields_and_no_execution_claim():
    observer = AdaptiveContinuityObserver()
    observer.observe(
        physical_group=1,
        logical_chunks=(1,),
        observed_motion_score=0.0,
        current_context_frames=0,
        resolved_transport=INITIAL_TRANSPORT,
        terminal_atomic=False,
        boundary_index=0,
        boundary_count=2,
        reason="initial clip",
    )
    text = "\n".join(observer.report_lines())
    for token in (
        "planner_hash=",
        "motion=",
        "context=",
        "transport=",
        "terminal_atomic=",
        "boundary=",
        "recommendation=",
        "confidence=",
        "reason=",
        "fallback_reason=",
        "execution_applied=false",
    ):
        assert token in text
    assert "Issue #13 mitigation claim=none" in text


def test_a8a_is_detailed_only_and_has_no_public_schema_or_identity_field():
    sequence = (ROOT / "v2" / "sequence.py").read_text(encoding="utf-8")
    observer_creation = sequence.index(
        "adaptive_continuity_observer=AdaptiveContinuityObserver()"
    )
    detailed_guard = sequence.rindex(
        "if diagnostics_mode==DIAGNOSTICS_FULL:", 0, observer_creation
    )
    assert detailed_guard < observer_creation
    for path in (
        ROOT / "v3" / "nodes.py",
        ROOT / "v3" / "driving_nodes.py",
        ROOT / "v3" / "second_pass_nodes.py",
    ):
        assert "adaptive_continuity" not in path.read_text(encoding="utf-8")
    storage = (ROOT / "run_storage.py").read_text(encoding="utf-8")
    assert "adaptive_continuity" not in storage


def test_a8a_source_has_no_tensor_or_execution_mutation_api():
    source = (ROOT / "v3" / "adaptive_continuity.py").read_text(encoding="utf-8")
    for forbidden in (
        "import torch",
        ".clone(",
        ".copy_(",
        "noise_mask",
        "prepare_conditioning",
        "sample_chunk",
        "unload_all_models",
        "optimized_attention_override",
        "PackedLayout",
    ):
        assert forbidden not in source

