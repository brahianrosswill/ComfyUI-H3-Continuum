from __future__ import annotations

import inspect

import pytest

from ComfyUI_H3_Continuum_Join.temporal import audio_latent_t, context_slots
from ComfyUI_H3_Continuum_Join.v3.reliability_v38 import (
    build_chunk_boundary_diagnostics,
)


def _chunk(index: int, *, context: int = 22, start: int | None = None):
    if index == 1:
        total = net = 124
        trim = 0
    else:
        trim = context
        total = context + 119
        net = 119
    if start is None:
        start = 0 if index == 1 else 124 + (index - 2) * 119
    return {
        "sequence_index": index,
        "chunk_index": index,
        "total_frames": total,
        "trim_frames": trim,
        "net_frames": net,
        "context_frames": trim,
        "frame_start": start,
        "frame_stop": start + net,
    }


def _decode_group(chunk, *, logical=None, terminal=False):
    group = dict(chunk)
    group["logical_chunk_indices"] = list(logical or [chunk["chunk_index"]])
    group["terminal_merged"] = bool(terminal)
    return group


def _second_pass_group(group_id: int, group):
    return {
        "group_id": group_id,
        "logical_chunks": list(group["logical_chunk_indices"]),
        "physical_frames": group["total_frames"],
        "trim_prefix_frames": group["trim_frames"],
        "terminal_merged": bool(group.get("terminal_merged", False)),
    }


def _plan(*chunks, groups=None, second_pass=True):
    active_groups = (
        list(groups)
        if groups is not None
        else [_decode_group(chunk) for chunk in chunks]
    )
    plan = {"chunks": list(chunks), "decode_groups": active_groups}
    if second_pass:
        plan["second_pass_contract"] = {
            "version": 1,
            "physical_groups": [
                _second_pass_group(index, group)
                for index, group in enumerate(active_groups)
            ],
        }
    return plan


def _diagnose(plan, *, video_count=None, audio_count=None):
    groups = plan.get("decode_groups", plan.get("chunks", []))
    return build_chunk_boundary_diagnostics(
        video_latents=[object()] * (len(groups) if video_count is None else video_count),
        audio_latents=[object()] * (len(groups) if audio_count is None else audio_count),
        assembly_plan=plan,
    )


def test_single_chunk_has_zero_boundaries_and_silent_basic():
    result = _diagnose(_plan(_chunk(1)))

    assert result.basic == ()
    assert result.detailed == (
        "Chunk boundary contract: structural diagnostic only; "
        "no image/audio similarity measurement.",
        "Second Pass contract: groups=1 PASS.",
    )


@pytest.mark.parametrize(
    ("context", "expected_video", "expected_audio"),
    (
        (5, 2, 8),
        (22, 7, 37),
        (39, 12, 65),
    ),
)
def test_fast_balanced_strong_context_uses_shared_temporal_helpers(
    context, expected_video, expected_audio
):
    first = _chunk(1)
    second = _chunk(2, context=context)
    result = _diagnose(_plan(first, second))

    assert context_slots(context) == expected_video
    assert audio_latent_t(context) == expected_audio
    assert result.basic == ()
    assert len([line for line in result.detailed if line.startswith("Boundary ")]) == 1
    assert (
        f"context={context}f video={expected_video}T audio={expected_audio}T"
        in result.detailed[-1]
    )
    assert result.detailed[-1].endswith("PASS.")


def test_three_chunks_have_two_boundaries_and_silent_basic():
    result = _diagnose(_plan(_chunk(1), _chunk(2), _chunk(3)))

    assert result.basic == ()
    boundaries = [line for line in result.detailed if line.startswith("Boundary ")]
    assert len(boundaries) == 2
    assert boundaries[0].startswith("Boundary 1→2: physical;")
    assert boundaries[1].startswith("Boundary 2→3: physical;")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("trim_frames", 200, "trim_frames exceeds total_frames"),
        ("net_frames", 118, "total_frames - trim_frames differs from net_frames"),
    ),
)
def test_invalid_right_geometry_is_advisory(field, value, message):
    right = _chunk(2)
    right[field] = value
    result = _diagnose(_plan(_chunk(1), right))

    assert any(message in line for line in result.basic)
    assert result.detailed[-1].endswith(").")
    assert "ADVISORY" in result.detailed[-1]


@pytest.mark.parametrize(("right_start", "gap"), ((125, 1), (123, -1)))
def test_artificial_gap_or_overlap_is_advisory(right_start, gap):
    result = _diagnose(_plan(_chunk(1), _chunk(2, start=right_start)))

    assert any("non-contiguous visible range" in line for line in result.basic)
    boundary = result.detailed[-1]
    assert f"gap={gap}" in boundary
    assert "ADVISORY" in boundary


def test_missing_geometry_is_unavailable_without_hard_failure():
    right = _chunk(2)
    del right["total_frames"]
    result = _diagnose(_plan(_chunk(1), right, second_pass=False))

    assert result.basic
    assert "required geometry is unavailable" in "\n".join(result.basic)
    assert "ADVISORY" in result.detailed[-1]


def test_non_contiguous_logical_order_is_advisory():
    right = _chunk(2)
    right["chunk_index"] = 3
    groups = [_decode_group(_chunk(1)), _decode_group(right, logical=[2])]
    result = _diagnose(_plan(_chunk(1), right, groups=groups, second_pass=False))

    assert any("logical order" in line for line in result.basic)


def test_physical_mapping_missing_logical_chunk():
    chunks = (_chunk(1), _chunk(2), _chunk(3))
    groups = [_decode_group(chunks[0]), _decode_group(chunks[2])]
    result = _diagnose(_plan(*chunks, groups=groups, second_pass=False))

    assert any("does not cover all logical chunks" in line for line in result.basic)
    assert any("UNAVAILABLE" in line for line in result.detailed)


def test_physical_mapping_duplicate_logical_chunk():
    chunks = (_chunk(1), _chunk(2))
    groups = [_decode_group(chunks[0]), _decode_group(chunks[1], logical=[1])]
    result = _diagnose(_plan(*chunks, groups=groups, second_pass=False))

    assert any("mapped more than once" in line for line in result.basic)


def test_physical_mapping_reversed_order():
    chunks = (_chunk(1), _chunk(2))
    group = _decode_group(chunks[0], logical=[2, 1], terminal=True)
    result = _diagnose(_plan(*chunks, groups=[group], second_pass=False))

    joined = "\n".join(result.basic)
    assert "logical order is reversed" in joined
    assert "Terminal Merge mapping is invalid" in joined


def test_physical_output_group_count_mismatch_is_summarized():
    plan = _plan(_chunk(1), _chunk(2), second_pass=False)
    result = _diagnose(plan, video_count=1, audio_count=2)

    assert any("physical output/group counts differ" in line for line in result.basic)


def _terminal_plan(*, terminal_flag=True, second_pass=True):
    chunks = [_chunk(1), _chunk(2), _chunk(3)]
    terminal = dict(chunks[1])
    terminal.update(
        {
            "total_frames": 260,
            "trim_frames": 22,
            "net_frames": 238,
            "frame_start": 124,
            "frame_stop": 362,
            "logical_chunk_indices": [2, 3],
            "terminal_merged": terminal_flag,
        }
    )
    groups = [_decode_group(chunks[0]), terminal]
    return _plan(*chunks, groups=groups, second_pass=second_pass)


def test_terminal_merge_is_shared_physical_boundary_not_external_seam():
    result = _diagnose(_terminal_plan())

    assert result.basic == ()
    terminal = [line for line in result.detailed if line.startswith("Boundary 2→3")][0]
    assert "shared physical group=2" in terminal
    assert "terminal_merged=yes" in terminal
    assert "no external decode boundary" in terminal
    assert terminal.endswith("PASS.")


def test_terminal_flag_mismatch_is_advisory():
    result = _diagnose(_terminal_plan(terminal_flag=False, second_pass=False))

    assert any("without Terminal Merge" in line for line in result.basic)
    terminal = [line for line in result.detailed if line.startswith("Boundary 2→3")][0]
    assert "terminal_merged=no" in terminal
    assert "ADVISORY" in terminal


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("logical_chunks", [1, 2], "logical_chunks differs"),
        ("physical_frames", 999, "physical_frames differs"),
        ("trim_prefix_frames", 5, "trim_prefix_frames differs"),
    ),
)
def test_second_pass_contract_mismatch_is_advisory(field, value, message):
    plan = _terminal_plan()
    plan["second_pass_contract"]["physical_groups"][1][field] = value
    result = _diagnose(plan)

    assert any(message in line for line in result.basic)
    assert "Second Pass contract: groups=2 ADVISORY." in result.detailed


class _ReadOnlyTensorProbe:
    shape = (1, 24, 37, 4, 4)
    dtype = "probe-dtype"
    device = "probe-device"

    def data_ptr(self):
        return 12345

    def clone(self):
        raise AssertionError("clone must not be called")

    def cpu(self):
        raise AssertionError("cpu must not be called")

    def float(self):
        raise AssertionError("float must not be called")

    def numpy(self):
        raise AssertionError("numpy must not be called")

    def __hash__(self):
        raise AssertionError("hash must not be called")


def test_diagnostic_is_read_only_and_preserves_all_objects():
    video_tensor = _ReadOnlyTensorProbe()
    audio_tensor = _ReadOnlyTensorProbe()
    video_latents = [{"samples": video_tensor}]
    audio_latents = [{"samples": audio_tensor}]
    plan = _plan(_chunk(1))
    before = (
        id(video_latents),
        id(audio_latents),
        id(plan),
        id(video_tensor),
        id(audio_tensor),
        video_tensor.data_ptr(),
        video_tensor.dtype,
        video_tensor.device,
        video_tensor.shape,
    )

    build_chunk_boundary_diagnostics(
        video_latents=video_latents,
        audio_latents=audio_latents,
        assembly_plan=plan,
    )

    after = (
        id(video_latents),
        id(audio_latents),
        id(plan),
        id(video_tensor),
        id(audio_tensor),
        video_tensor.data_ptr(),
        video_tensor.dtype,
        video_tensor.device,
        video_tensor.shape,
    )
    assert after == before


def test_boundary_implementation_has_no_tensor_content_operations():
    source = inspect.getsource(build_chunk_boundary_diagnostics)
    for forbidden in (
        ".clone(",
        ".cpu(",
        ".float(",
        ".numpy(",
        "ContextDiagnosticsTracker",
        "torch.equal",
    ):
        assert forbidden not in source

