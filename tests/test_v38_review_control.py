from __future__ import annotations

import pytest

from ComfyUI_H3_Continuum_Join.v3.review_control import (
    EXECUTION_MODE_FULL_RUN,
    EXECUTION_MODE_REVIEW_CONTINUE,
    EXECUTION_MODE_REVIEW_FINISH,
    EXECUTION_MODE_REVIEW_REGENERATE,
    EXECUTION_MODE_REVIEW_CONTINUE_FROM_TAKE,
    EXECUTION_MODE_REVIEW_USE_TAKE,
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_REVIEW,
    NONCE_POLICY_ADVANCE,
    NONCE_POLICY_INHERIT,
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_FINISH_REMAINING,
    REVIEW_ACTION_REGENERATE_CURRENT,
    REVIEW_PAUSE_REASON,
    REVISION_STATUS_COMPLETE,
    REVISION_STATUS_IN_PROGRESS,
    REVISION_STATUS_INTERRUPTED,
    REVISION_STATUS_REVIEW_READY,
    RUN_STORAGE_OFF,
    RUN_STORAGE_SAVE_AUTO_RESUME,
    ReviewControlError,
    ReviewUnit,
    make_review_pause_metadata,
    resolve_next_review_unit,
    resolve_review_execution,
    resolve_take_execution,
)
from ComfyUI_H3_Continuum_Join.branch_provenance import (
    TAKE_ACTION_CONTINUE,
    TAKE_ACTION_USE,
)


def _resolve(**overrides):
    values = {
        "generation_mode": GENERATION_MODE_REVIEW,
        "review_action": REVIEW_ACTION_CONTINUE,
        "configured_chunks": 6,
        "validated_prefix_count": 0,
        "terminal_merge_enabled": False,
        "terminal_pair_start": None,
        "manual_regenerate_from": "Auto",
        "run_storage_mode": RUN_STORAGE_SAVE_AUTO_RESUME,
        "latest_review_unit": None,
        "latest_revision_status": None,
        "latest_effective_nonce": 0,
        "latest_branch_regenerate_from": 0,
    }
    values.update(overrides)
    return resolve_review_execution(**values)


def _review_unit(start, end=None):
    end = start if end is None else end
    return {"start": start, "end": end, "physical_group": start}


def test_full_run_has_no_review_execution_limit_or_storage_requirement():
    result = _resolve(
        generation_mode=GENERATION_MODE_FULL_RUN,
        run_storage_mode=RUN_STORAGE_OFF,
    )
    assert result.execution_mode == EXECUTION_MODE_FULL_RUN
    assert result.max_new_physical_groups is None
    assert result.requires_run_storage is False
    assert result.partial_review is False


@pytest.mark.parametrize(
    ("prefix", "expected"),
    ((0, 1), (1, 2), (2, 3), (5, 6)),
)
def test_normal_review_resolves_one_next_chunk(prefix, expected):
    result = _resolve(
        validated_prefix_count=prefix,
        latest_revision_status=(REVISION_STATUS_REVIEW_READY if prefix else None),
        latest_review_unit=(_review_unit(prefix) if prefix else None),
    )
    assert result.execution_mode == EXECUTION_MODE_REVIEW_CONTINUE
    assert result.max_new_physical_groups == 1
    assert (result.next_review_unit_start, result.next_review_unit_end) == (
        expected,
        expected,
    )


def test_review_complete_has_no_next_unit():
    result = _resolve(
        validated_prefix_count=6,
        latest_revision_status=REVISION_STATUS_COMPLETE,
        latest_review_unit=_review_unit(6),
    )
    assert result.next_review_unit_start is None
    assert result.next_review_unit_end is None
    assert result.projected_prefix_count == 6
    assert result.partial_review is False
    assert "complete" in result.status_hint.lower()


def test_finish_remaining_removes_physical_group_limit_and_is_one_shot():
    result = _resolve(
        review_action=REVIEW_ACTION_FINISH_REMAINING,
        validated_prefix_count=3,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(3),
    )
    assert result.execution_mode == EXECUTION_MODE_REVIEW_FINISH
    assert result.max_new_physical_groups is None
    assert result.finish_remaining is True
    assert result.one_shot_reset_required is True
    assert result.projected_prefix_count == 6
    assert result.partial_review is False


@pytest.mark.parametrize(
    ("prefix", "expected"),
    (
        (0, (1, 1)),
        (1, (2, 3)),
        (3, None),
    ),
)
def test_terminal_merge_next_unit_is_atomic(prefix, expected):
    unit = resolve_next_review_unit(
        configured_chunks=3,
        validated_prefix_count=prefix,
        terminal_merge_enabled=True,
        terminal_pair_start=2,
    )
    if expected is None:
        assert unit is None
    else:
        assert unit is not None
        assert (unit.start, unit.end) == expected


def test_terminal_pair_partial_prefix_is_an_invariant_error():
    with pytest.raises(ReviewControlError, match="inside the atomic Terminal Merge pair"):
        resolve_next_review_unit(
            configured_chunks=3,
            validated_prefix_count=2,
            terminal_merge_enabled=True,
            terminal_pair_start=2,
        )


def test_terminal_merge_must_be_the_final_two_chunks():
    with pytest.raises(ReviewControlError, match="final two"):
        resolve_next_review_unit(
            configured_chunks=5,
            validated_prefix_count=0,
            terminal_merge_enabled=True,
            terminal_pair_start=3,
        )


def test_smart_regenerate_resolves_current_normal_unit():
    result = _resolve(
        review_action=REVIEW_ACTION_REGENERATE_CURRENT,
        validated_prefix_count=3,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(3),
    )
    assert result.execution_mode == EXECUTION_MODE_REVIEW_REGENERATE
    assert result.effective_regenerate_from == 3
    assert result.max_new_physical_groups == 1
    assert result.smart_regenerate is True
    assert result.one_shot_reset_required is True
    assert result.requested_effective_nonce == 1


def test_smart_regenerate_resolves_terminal_pair_start():
    result = _resolve(
        review_action=REVIEW_ACTION_REGENERATE_CURRENT,
        configured_chunks=3,
        validated_prefix_count=3,
        terminal_merge_enabled=True,
        terminal_pair_start=2,
        latest_revision_status=REVISION_STATUS_COMPLETE,
        latest_review_unit=_review_unit(2, 3),
    )
    assert result.effective_regenerate_from == 2
    assert (result.next_review_unit_start, result.next_review_unit_end) == (2, 3)
    assert result.next_review_physical_group == 2
    assert result.partial_review is False


def test_regenerate_current_rejects_empty_prefix():
    with pytest.raises(ReviewControlError, match="existing reviewed unit"):
        _resolve(review_action=REVIEW_ACTION_REGENERATE_CURRENT)


def test_regenerate_current_rejects_missing_review_unit():
    with pytest.raises(ReviewControlError, match="review_unit metadata"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=2,
            latest_revision_status=REVISION_STATUS_REVIEW_READY,
        )


def test_regenerate_current_rejects_manual_regenerate_from():
    with pytest.raises(ReviewControlError, match="cannot be combined"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=2,
            manual_regenerate_from=1,
            latest_revision_status=REVISION_STATUS_REVIEW_READY,
            latest_review_unit=_review_unit(2),
        )


def test_review_requires_run_storage():
    with pytest.raises(ReviewControlError, match="requires Run Storage"):
        _resolve(run_storage_mode=RUN_STORAGE_OFF)


@pytest.mark.parametrize(
    "status",
    (REVISION_STATUS_IN_PROGRESS, REVISION_STATUS_INTERRUPTED),
)
def test_regenerate_current_rejects_unready_revision(status):
    with pytest.raises(ReviewControlError, match="review_ready revision"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=2,
            latest_revision_status=status,
            latest_review_unit=_review_unit(2),
        )


def test_review_complete_with_review_unit_can_regenerate():
    result = _resolve(
        review_action=REVIEW_ACTION_REGENERATE_CURRENT,
        validated_prefix_count=6,
        latest_revision_status=REVISION_STATUS_COMPLETE,
        latest_review_unit=_review_unit(6),
    )
    assert result.effective_regenerate_from == 6


def test_full_complete_without_review_unit_cannot_smart_regenerate():
    with pytest.raises(ReviewControlError, match="review_unit metadata"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=6,
            latest_revision_status=REVISION_STATUS_COMPLETE,
            latest_review_unit=None,
        )


def test_review_unit_must_be_the_validated_prefix_head():
    with pytest.raises(ReviewControlError, match="head of the validated prefix"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=4,
            latest_revision_status=REVISION_STATUS_REVIEW_READY,
            latest_review_unit=_review_unit(3),
        )


def test_review_pause_metadata_is_non_identity_shape():
    metadata = make_review_pause_metadata(ReviewUnit(4, 5, 4))
    assert metadata == {
        "review_unit": {"start": 4, "end": 5, "physical_group": 4},
        "review_pause_reason": REVIEW_PAUSE_REASON,
    }
    assert "sampling" not in metadata


def test_review_pause_metadata_rejects_mismatched_physical_group():
    with pytest.raises(ReviewControlError, match="physical group"):
        make_review_pause_metadata(ReviewUnit(4, 5, 5))


@pytest.mark.parametrize(
    "status",
    (
        REVISION_STATUS_IN_PROGRESS,
        REVISION_STATUS_INTERRUPTED,
        REVISION_STATUS_REVIEW_READY,
    ),
)
def test_continue_inherits_existing_regenerate_branch_without_new_nonce(status):
    result = _resolve(
        validated_prefix_count=2,
        latest_revision_status=status,
        latest_review_unit=(
            _review_unit(2) if status == REVISION_STATUS_REVIEW_READY else None
        ),
        latest_effective_nonce=2,
        latest_branch_regenerate_from=2,
    )
    assert result.nonce_policy == NONCE_POLICY_INHERIT
    assert result.effective_regenerate_from == 2
    assert result.branch_regenerate_from == 2
    assert result.requested_effective_nonce == 2


@pytest.mark.parametrize(
    ("latest_nonce", "next_nonce"),
    ((0, 1), (1, 2), (2, 3)),
)
def test_consecutive_smart_regenerate_advances_current_branch_nonce(
    latest_nonce,
    next_nonce,
):
    result = _resolve(
        review_action=REVIEW_ACTION_REGENERATE_CURRENT,
        validated_prefix_count=2,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(2),
        latest_effective_nonce=latest_nonce,
        latest_branch_regenerate_from=(2 if latest_nonce else 0),
    )
    assert result.nonce_policy == NONCE_POLICY_ADVANCE
    assert result.branch_regenerate_from == 2
    assert result.requested_effective_nonce == next_nonce


def test_smart_regenerate_rejects_effective_nonce_overflow():
    with pytest.raises(ReviewControlError, match="variation space is exhausted"):
        _resolve(
            review_action=REVIEW_ACTION_REGENERATE_CURRENT,
            validated_prefix_count=2,
            latest_revision_status=REVISION_STATUS_REVIEW_READY,
            latest_review_unit=_review_unit(2),
            latest_effective_nonce=0xFFFFFFFF,
            latest_branch_regenerate_from=2,
        )


def test_regenerate_then_continue_keeps_the_regenerated_branch():
    regenerate = _resolve(
        review_action=REVIEW_ACTION_REGENERATE_CURRENT,
        validated_prefix_count=2,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(2),
        latest_effective_nonce=1,
        latest_branch_regenerate_from=2,
    )
    assert regenerate.requested_effective_nonce == 2

    continued = _resolve(
        validated_prefix_count=2,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(2),
        latest_effective_nonce=regenerate.requested_effective_nonce,
        latest_branch_regenerate_from=regenerate.branch_regenerate_from,
    )
    assert continued.effective_regenerate_from == 2
    assert continued.requested_effective_nonce == 2
    assert (continued.next_review_unit_start, continued.next_review_unit_end) == (3, 3)


def test_positive_nonce_without_branch_boundary_is_rejected():
    with pytest.raises(ReviewControlError, match="requires its regenerate branch"):
        _resolve(
            validated_prefix_count=2,
            latest_revision_status=REVISION_STATUS_REVIEW_READY,
            latest_review_unit=_review_unit(2),
            latest_effective_nonce=2,
            latest_branch_regenerate_from=0,
        )


def test_manual_review_regeneration_maps_terminal_second_chunk_to_atomic_pair():
    result = _resolve(
        configured_chunks=3,
        validated_prefix_count=3,
        terminal_merge_enabled=True,
        terminal_pair_start=2,
        manual_regenerate_from=3,
        latest_revision_status=REVISION_STATUS_COMPLETE,
        latest_review_unit=_review_unit(2, 3),
    )
    assert result.effective_regenerate_from == 3
    assert (result.next_review_unit_start, result.next_review_unit_end) == (2, 3)


def test_partial_review_describes_the_projected_output_prefix():
    partial = _resolve()
    terminal_completion = _resolve(
        configured_chunks=3,
        validated_prefix_count=1,
        terminal_merge_enabled=True,
        terminal_pair_start=2,
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_review_unit=_review_unit(1),
    )
    assert partial.partial_review is True
    assert partial.projected_prefix_count == 1
    assert terminal_completion.partial_review is False
    assert terminal_completion.projected_prefix_count == 3


def test_use_take_selects_without_generating_and_continue_starts_new_boundary():
    selected = resolve_take_execution(
        take_action=TAKE_ACTION_USE,
        configured_chunks=5,
        selected_prefix_count=3,
        terminal_merge_enabled=False,
        terminal_pair_start=None,
        selected_review_unit=_review_unit(3),
        selected_effective_nonce=1,
        selected_branch_regenerate_from=3,
        next_effective_nonce=2,
    )
    assert selected.execution_mode == EXECUTION_MODE_REVIEW_USE_TAKE
    assert selected.max_new_physical_groups == 0
    assert selected.projected_prefix_count == 3

    continued = resolve_take_execution(
        take_action=TAKE_ACTION_CONTINUE,
        configured_chunks=5,
        selected_prefix_count=3,
        terminal_merge_enabled=False,
        terminal_pair_start=None,
        selected_review_unit=_review_unit(3),
        selected_effective_nonce=1,
        selected_branch_regenerate_from=3,
        next_effective_nonce=2,
    )
    assert continued.execution_mode == EXECUTION_MODE_REVIEW_CONTINUE_FROM_TAKE
    assert continued.effective_regenerate_from == 4
    assert continued.requested_effective_nonce == 2
    assert continued.max_new_physical_groups == 1


def test_continue_from_take_keeps_terminal_merge_atomic():
    continued = resolve_take_execution(
        take_action=TAKE_ACTION_CONTINUE,
        configured_chunks=3,
        selected_prefix_count=1,
        terminal_merge_enabled=True,
        terminal_pair_start=2,
        selected_review_unit=_review_unit(1),
        selected_effective_nonce=0,
        selected_branch_regenerate_from=0,
        next_effective_nonce=1,
    )
    assert (
        continued.next_review_unit_start,
        continued.next_review_unit_end,
        continued.next_review_physical_group,
    ) == (2, 3, 2)
