from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join.constants import (
    DIAGNOSTICS_BASIC,
    PROMPT_MODE_FIXED,
    V2_CONTINUITY_OPTIONS,
)
from ComfyUI_H3_Continuum_Join.temporal import audio_latent_t, video_latent_t
from ComfyUI_H3_Continuum_Join.v2 import sequence
from ComfyUI_H3_Continuum_Join.v2.h3_builder import IdentityAssets
from ComfyUI_H3_Continuum_Join.v2.nodes import H3ContinuumSamplerV2
from ComfyUI_H3_Continuum_Join.v2.prompts import make_prompt_plan
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)
from ComfyUI_H3_Continuum_Join.v3.nodes import (
    H3ContinuumSamplerProduction,
    H3ContinuumSamplerV3,
)
from ComfyUI_H3_Continuum_Join.v3.plan import prepare_physical_decode_entries


class _Nested:
    def __init__(self, parts):
        self.parts = list(parts)

    def unbind(self):
        return self.parts


class _FakeClip:
    def tokenize(self, prompt, images=None, **_kwargs):
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 1, 2), {"prompt": tokens}]]


def _latent(width, height, frames):
    return {
        "samples": _Nested(
            (
                torch.zeros(
                    1,
                    24,
                    video_latent_t(frames),
                    height // 16,
                    width // 16,
                ),
                torch.zeros(1, 32, 2, audio_latent_t(frames)),
            )
        )
    }


def _install_fake_runtime(monkeypatch, *, first_frame=None, last_frame=None, timeline=None):
    width, height = 96, 64
    timeline = [] if timeline is None else timeline
    first_latent = (
        torch.full((1, 24, 1, height // 16, width // 16), 0.25)
        if first_frame is not None
        else None
    )
    last_latent = (
        torch.full((1, 24, 1, height // 16, width // 16), 0.75)
        if last_frame is not None
        else None
    )
    assets = IdentityAssets(
        first_frame,
        first_latent,
        last_frame,
        last_latent,
        "review-cap-identity" if first_frame is not None else "none",
    )
    model = SimpleNamespace(
        model=SimpleNamespace(diffusion_model=SimpleNamespace()),
        model_options={},
        wrappers={},
        model_dtype=lambda: torch.bfloat16,
        model_size=lambda: 0,
    )
    samples = []

    monkeypatch.setattr(sequence, "check_comfy_h3_runtime", lambda: [])
    monkeypatch.setattr(sequence, "prepare_identity_assets", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "encode_identity_latents", lambda *a, **k: assets)
    monkeypatch.setattr(sequence, "empty_h3_latent", _latent)
    monkeypatch.setattr(sequence, "accelerator_summary", lambda _model: "accelerators")
    monkeypatch.setattr(sequence, "clone_model_for_chunk", lambda model, **_kwargs: model)
    monkeypatch.setattr(
        sequence,
        "latent_from_cpu",
        lambda video, audio: {"samples": _Nested((video, audio))},
    )

    def sample_chunk(**kwargs):
        video, audio = kwargs["latent"]["samples"].unbind()
        video = video.clone()
        audio = audio.clone()
        value = float(int(kwargs["seed"]) % 251 + 1)
        video.fill_(value)
        audio.fill_(value + 0.5)
        metadata = kwargs["conditioning"][0][1]
        keyframes = list(metadata.get("minimax_keyframes") or [])
        event = {
            "seed": int(kwargs["seed"]),
            "keyframes": keyframes,
            "video_shape": tuple(video.shape),
            "audio_shape": tuple(audio.shape),
        }
        samples.append(event)
        timeline.append(("sample", int(kwargs["seed"])))
        return {"samples": _Nested((video, audio))}

    monkeypatch.setattr(sequence, "sample_chunk", sample_chunk)
    return SimpleNamespace(
        width=width,
        height=height,
        model=model,
        clip=_FakeClip(),
        assets=assets,
        samples=samples,
        timeline=timeline,
    )


def _run_sequence(
    runtime,
    *,
    chunks,
    chunk_seconds=5.0,
    session=None,
    limit=None,
    first_frame=None,
    last_frame=None,
    capture_refine_context=False,
):
    plan = make_prompt_plan(
        mode=PROMPT_MODE_FIXED,
        script="continuous shot",
        chunks=chunks,
        chunk_seconds=chunk_seconds,
    )
    return sequence.run_sequence(
        model=runtime.model,
        clip=runtime.clip,
        video_vae=object(),
        audio_vae=None,
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        first_frame=first_frame,
        last_frame=last_frame,
        prompt_plan=plan,
        width=runtime.width,
        height=runtime.height,
        continuity=V2_CONTINUITY_OPTIONS[1],
        base_seed=42,
        audio_continuity=True,
        exact_total_duration=False,
        diagnostics_mode=DIAGNOSTICS_BASIC,
        reroll_from_chunk=0,
        reroll_nonce=0,
        strict_compatibility=False,
        debug=False,
        session=session,
        latent_only=True,
        capture_refine_context=capture_refine_context,
        max_new_physical_groups=limit,
    )


def _run_v3_node(
    runtime,
    *,
    chunks,
    session=None,
    limit=None,
    first_frame=None,
    last_frame=None,
):
    advanced = {"last_frame": last_frame}
    if session is not None:
        advanced["session"] = session
    return H3ContinuumSamplerV3().run(
        model=runtime.model,
        clip=runtime.clip,
        video_vae=object(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        sequence_prompt="continuous shot",
        prompt_mode=PROMPT_MODE_FIXED,
        chunks=chunks,
        chunk_seconds=5.0,
        width=runtime.width,
        height=runtime.height,
        continuity=V2_CONTINUITY_OPTIONS[1],
        base_seed=42,
        first_frame=first_frame,
        advanced=advanced,
        max_new_physical_groups=limit,
    )


def _entry_contract(entries):
    return [
        {
            "seed": entry["seed"],
            "prompt_hash": entry["prompt_hash"],
            "plan": entry["plan"],
            "video": entry["video"],
            "audio": entry["audio"],
        }
        for entry in entries
    ]


def _assert_entry_parity(left, right):
    left_contract = _entry_contract(left)
    right_contract = _entry_contract(right)
    assert len(left_contract) == len(right_contract)
    for left_entry, right_entry in zip(left_contract, right_contract, strict=True):
        assert left_entry["seed"] == right_entry["seed"]
        assert left_entry["prompt_hash"] == right_entry["prompt_hash"]
        assert left_entry["plan"] == right_entry["plan"]
        assert torch.equal(left_entry["video"], right_entry["video"])
        assert torch.equal(left_entry["audio"], right_entry["audio"])


def test_full_run_none_keeps_all_six_chunks(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    entries, last_state, session, report = _run_sequence(
        runtime,
        chunks=6,
        limit=None,
    )
    assert len(entries) == len(session["chunks"]) == 6
    assert last_state["clip_index"] == 6
    assert len(runtime.samples) == 6
    assert "Physical review execution" not in report


def test_limit_one_returns_one_new_chunk_and_prefix_assembly(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    entries, last_state, session, report = _run_sequence(
        runtime,
        chunks=6,
        limit=1,
    )
    decode_entries, assembly_plan = prepare_physical_decode_entries(
        entries,
        chunk_seconds=5.0,
        preserve_final_frame=False,
        terminal_merged=False,
    )
    assert len(entries) == len(session["chunks"]) == len(decode_entries) == 1
    assert last_state["clip_index"] == 1
    assert len(runtime.samples) == 1
    assert len(assembly_plan["chunks"]) == 1
    assert assembly_plan["target_frames"] == 120
    assert assembly_plan["preserve_final_frame"] is False
    assert "max_new_physical_groups" not in session["settings"]
    assert "1 new physical group(s) completed; 1/6 logical chunks" in report


def test_reused_prefix_does_not_consume_physical_group_budget(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    first_entries, _, first_session, _ = _run_sequence(
        runtime,
        chunks=6,
        limit=1,
    )
    first_video = first_entries[0]["video"].clone()
    before = len(runtime.samples)
    entries, last_state, session, report = _run_sequence(
        runtime,
        chunks=6,
        session=first_session,
        limit=1,
    )
    assert len(entries) == len(session["chunks"]) == 2
    assert len(runtime.samples) - before == 1
    assert entries[0]["reused"] is True
    assert torch.equal(entries[0]["video"], first_video)
    assert last_state["clip_index"] == 2
    assert "1 new physical group(s) completed; 2/6 logical chunks" in report


def test_prefix_five_generates_only_chunk_six_and_completes(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    session = None
    entries = None
    for expected in range(1, 6):
        entries, _, session, _ = _run_sequence(
            runtime,
            chunks=6,
            session=session,
            limit=1,
        )
        assert len(entries) == expected
    before = len(runtime.samples)
    entries, last_state, session, _ = _run_sequence(
        runtime,
        chunks=6,
        session=session,
        limit=1,
    )
    assert len(entries) == len(session["chunks"]) == 6
    assert len(runtime.samples) - before == 1
    assert last_state["clip_index"] == 6


def test_incremental_normal_run_matches_full_run_latents_and_assembly(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    full_entries, _, _, _ = _run_sequence(runtime, chunks=6, limit=None)
    session = None
    incremental_entries = None
    for _ in range(6):
        incremental_entries, _, session, _ = _run_sequence(
            runtime,
            chunks=6,
            session=session,
            limit=1,
        )
    _assert_entry_parity(full_entries, incremental_entries)
    _, full_plan = prepare_physical_decode_entries(
        full_entries,
        chunk_seconds=5.0,
        preserve_final_frame=False,
        terminal_merged=False,
    )
    _, incremental_plan = prepare_physical_decode_entries(
        incremental_entries,
        chunk_seconds=5.0,
        preserve_final_frame=False,
        terminal_merged=False,
    )
    assert incremental_plan == full_plan


def test_v3_node_returns_partial_latents_and_assembly_plan(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    video, audio, plan, result = _run_v3_node(runtime, chunks=6, limit=1)
    assert len(video) == len(audio) == len(plan["chunks"]) == 1
    assert plan["target_frames"] == 120
    assert plan["preserve_final_frame"] is False
    assert len(result["session"]["chunks"]) == 1
    assert "1/6 logical chunks" in result["report"]


def test_fl2va_review_stops_before_terminal_then_completes_atomic_pair(monkeypatch):
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones(1, 64, 96, 3)
    runtime = _install_fake_runtime(
        monkeypatch,
        first_frame=first,
        last_frame=last,
    )
    video1, audio1, plan1, result1 = _run_v3_node(
        runtime,
        chunks=3,
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    assert len(video1) == len(audio1) == len(plan1["chunks"]) == 1
    assert plan1["preserve_final_frame"] is False
    assert plan1["target_frames"] == 120
    assert len(runtime.samples) == 1
    assert all(
        keyframe["latent"] is not runtime.assets.last_latent
        for keyframe in runtime.samples[0]["keyframes"]
    )

    before = len(runtime.samples)
    video2, audio2, plan2, result2 = _run_v3_node(
        runtime,
        chunks=3,
        session=result1["session"],
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    assert len(runtime.samples) - before == 1
    assert len(result2["session"]["chunks"]) == 3
    assert len(video2) == len(audio2) == 2
    assert len(plan2["chunks"]) == 3
    assert plan2["target_frames"] == 360
    assert plan2["preserve_final_frame"] is True
    assert plan2["physical_decode_group_count"] == 2
    terminal_group = plan2["decode_groups"][-1]
    assert terminal_group["logical_chunk_indices"] == [2, 3]
    assert terminal_group["terminal_merged"] is True
    assert any(
        keyframe["latent"] is runtime.assets.last_latent
        for keyframe in runtime.samples[-1]["keyframes"]
    )


def test_partial_terminal_session_is_atomically_reset_not_returned(monkeypatch):
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones(1, 64, 96, 3)
    runtime = _install_fake_runtime(
        monkeypatch,
        first_frame=first,
        last_frame=last,
    )
    _, _, complete_session, _ = _run_sequence(
        runtime,
        chunks=3,
        limit=None,
        first_frame=first,
        last_frame=last,
    )
    partial_pair_session = copy.deepcopy(complete_session)
    partial_pair_session["chunks"] = partial_pair_session["chunks"][:2]
    before = len(runtime.samples)
    entries, _, session, report = _run_sequence(
        runtime,
        chunks=3,
        session=partial_pair_session,
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    assert len(runtime.samples) - before == 1
    assert len(entries) == len(session["chunks"]) == 3
    assert "terminal merged pair is atomic" in report
    assert entries[1]["reused"] is False
    assert entries[2]["reused"] is False


def test_incremental_terminal_run_matches_full_run(monkeypatch):
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones(1, 64, 96, 3)
    runtime = _install_fake_runtime(
        monkeypatch,
        first_frame=first,
        last_frame=last,
    )
    full_entries, _, _, _ = _run_sequence(
        runtime,
        chunks=3,
        limit=None,
        first_frame=first,
        last_frame=last,
    )
    first_entries, _, first_session, _ = _run_sequence(
        runtime,
        chunks=3,
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    assert len(first_entries) == 1
    incremental_entries, _, _, _ = _run_sequence(
        runtime,
        chunks=3,
        session=first_session,
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    _assert_entry_parity(full_entries, incremental_entries)
    full_decode, full_plan = prepare_physical_decode_entries(
        full_entries,
        chunk_seconds=5.0,
        preserve_final_frame=True,
        terminal_merged=True,
    )
    incremental_decode, incremental_plan = prepare_physical_decode_entries(
        incremental_entries,
        chunk_seconds=5.0,
        preserve_final_frame=True,
        terminal_merged=True,
    )
    assert incremental_plan == full_plan
    assert len(incremental_decode) == len(full_decode) == 2
    for left, right in zip(full_decode, incremental_decode, strict=True):
        assert torch.equal(left["video"], right["video"])
        assert torch.equal(left["audio"], right["audio"])


class _CommitRecorder:
    effective_reroll_nonce = 0
    reused_count = 0
    review_execution = None

    def __init__(self, timeline):
        self.timeline = timeline
        self.prepare_kwargs = None
        self.review_groups = []

    def prepare(self, **kwargs):
        self.prepare_kwargs = kwargs
        return None

    def commit_chunk(self, entry, *, position):
        self.timeline.append(("commit", position))

    def mark_review_group(self, *, start, end, physical_group):
        self.review_groups.append((start, end, physical_group))


def test_storage_review_execution_drives_internal_cap_and_group_marker(monkeypatch):
    timeline = []
    runtime = _install_fake_runtime(monkeypatch, timeline=timeline)
    storage = _CommitRecorder(timeline)
    storage.review_execution = SimpleNamespace(
        effective_regenerate_from=0,
        max_new_physical_groups=1,
    )
    import ComfyUI_H3_Continuum_Join.run_storage as run_storage

    monkeypatch.setattr(run_storage, "get_active_run_storage", lambda: storage)
    entries, _, _, _ = _run_sequence(runtime, chunks=3, limit=None)
    assert len(entries) == 1
    assert storage.review_groups == [(1, 1, 1)]


def test_storage_use_take_returns_validated_prefix_without_sampling(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    _, _, selected_session, _ = _run_sequence(runtime, chunks=3, limit=2)
    selected_session["settings"]["run_storage_validated_prefix"] = True
    runtime.samples.clear()
    runtime.timeline.clear()
    storage = _CommitRecorder(runtime.timeline)
    storage.review_execution = SimpleNamespace(
        effective_regenerate_from=2,
        max_new_physical_groups=0,
    )

    def prepare(**kwargs):
        storage.prepare_kwargs = kwargs
        return selected_session

    storage.prepare = prepare
    import ComfyUI_H3_Continuum_Join.run_storage as run_storage

    monkeypatch.setattr(run_storage, "get_active_run_storage", lambda: storage)
    entries, _, session, report = _run_sequence(runtime, chunks=3, limit=None)
    assert len(entries) == len(session["chunks"]) == 2
    assert runtime.samples == []
    assert runtime.timeline == []
    assert storage.review_groups == []
    assert "0 new physical group(s) completed; 2/3 logical chunks" in report


def test_normal_group_stops_only_after_commit(monkeypatch):
    timeline = []
    runtime = _install_fake_runtime(monkeypatch, timeline=timeline)
    storage = _CommitRecorder(timeline)
    import ComfyUI_H3_Continuum_Join.run_storage as run_storage

    monkeypatch.setattr(run_storage, "get_active_run_storage", lambda: storage)
    entries, _, _, _ = _run_sequence(runtime, chunks=3, limit=1)
    assert len(entries) == 1
    assert [event[0] for event in timeline] == ["sample", "commit"]
    assert timeline[-1] == ("commit", 0)
    assert "max_new_physical_groups" not in storage.prepare_kwargs


def test_terminal_group_counts_once_after_both_logical_commits(monkeypatch):
    first = torch.zeros(1, 64, 96, 3)
    last = torch.ones(1, 64, 96, 3)
    timeline = []
    runtime = _install_fake_runtime(
        monkeypatch,
        first_frame=first,
        last_frame=last,
        timeline=timeline,
    )
    storage = _CommitRecorder(timeline)
    import ComfyUI_H3_Continuum_Join.run_storage as run_storage

    monkeypatch.setattr(run_storage, "get_active_run_storage", lambda: storage)
    entries, _, _, report = _run_sequence(
        runtime,
        chunks=2,
        limit=1,
        first_frame=first,
        last_frame=last,
    )
    assert len(entries) == 2
    assert [event[0] for event in timeline] == ["sample", "commit", "commit"]
    assert timeline[-2:] == [("commit", 0), ("commit", 1)]
    assert "1 new physical group(s) completed; 2/2 logical chunks" in report
    assert "max_new_physical_groups" not in storage.prepare_kwargs


def test_partial_refine_context_is_explicitly_incomplete(monkeypatch):
    runtime = _install_fake_runtime(monkeypatch)
    entries, _, session, report, refine_context = _run_sequence(
        runtime,
        chunks=3,
        limit=1,
        capture_refine_context=True,
    )
    assert len(entries) == len(session["chunks"]) == 1
    assert refine_context["complete"] is False
    assert len(refine_context["groups"]) == 1
    assert any("physical review execution" in note for note in refine_context["notes"])
    assert "1/3 logical chunks" in report


@pytest.mark.parametrize("invalid", (0, -1, True, 1.5))
def test_invalid_physical_group_limit_is_rejected(monkeypatch, invalid):
    runtime = _install_fake_runtime(monkeypatch)
    with pytest.raises(sequence.SequenceRuntimeError, match="positive integer"):
        _run_sequence(runtime, chunks=3, limit=invalid)


def test_physical_group_limit_is_internal_and_not_a_public_widget():
    for node_class in (
        H3ContinuumSamplerV2,
        H3ContinuumSamplerV3,
        H3ContinuumSamplerProduction,
        H3ContinuumSamplerV37,
        H3ContinuumSamplerV38,
    ):
        schema = node_class.INPUT_TYPES()
        assert "max_new_physical_groups" not in schema.get("required", {})
        assert "max_new_physical_groups" not in schema.get("optional", {})
