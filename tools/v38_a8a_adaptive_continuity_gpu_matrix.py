"""Run the A8a read-only Adaptive Continuity GPU Matrix.

The runner is evidence-only.  It compares the public V3.8 Production path with
accepted pre-A8 evidence, exercises the accepted Issue #13 same-prompt stress
path, and splits the Review/Run Storage gate across a real backend restart.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping
import uuid


TOOLS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from v38_a2_audio_only_gpu_canary import _history_entry, _output_latents, _output_text
from v38_a2_audio_only_gpu_canary import _audio_health
from v38_a3c_selective_refine_gpu_matrix import (
    GateFailure,
    _decoded_stream_hash,
    _execute,
    _marker_json,
)
from v38_a7a_packed_row_gpu_matrix import (
    FirstPassCase,
    _configure_first_pass,
    _cross_run_plan_identity,
)
from v38_review_phase_f_gpu_gate_runner import _manifest_snapshot
from v38_review_phase_f_recovery_runner import _cached_nodes, _preserved_prefix
from v38_easy_gpu_gate_runner import _output_media, _probe_media

_OBSERVER_SPEC = importlib.util.spec_from_file_location(
    "h3_a8a_observer_contract",
    REPO_ROOT / "v3" / "adaptive_continuity.py",
)
if _OBSERVER_SPEC is None or _OBSERVER_SPEC.loader is None:
    raise RuntimeError("could not load the A8a observer contract")
_OBSERVER_MODULE = importlib.util.module_from_spec(_OBSERVER_SPEC)
_OBSERVER_SPEC.loader.exec_module(_OBSERVER_MODULE)
ADAPTIVE_CONTINUITY_PLANNER_HASH = _OBSERVER_MODULE.ADAPTIVE_CONTINUITY_PLANNER_HASH


SUMMARY_PATTERN = re.compile(
    r"planner_hash=([0-9a-f]{64}),\s*observed_groups=(\d+),\s*"
    r"execution_applied=false"
)
DECISION_PATTERN = re.compile(
    r"physical_group=(\d+),\s*logical_chunks=(\[[^\]]+\]),\s*"
    r"boundary=(\d+)/(\d+),\s*motion=([0-9.]+),\s*context=(\d+),\s*"
    r"transport=([^,]+),\s*terminal_atomic=(true|false),\s*"
    r"reused=(true|false),\s*recommendation=([^,]+),\s*"
    r"confidence=([0-9.]+),\s*reason=(.*?),\s*"
    r"fallback_reason=(.*?),\s*decision_hash=([0-9a-f]{64}),\s*"
    r"execution_applied=false\."
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = _load_json(path)
    checks = value.get("checks") or {}
    if "observer_not_persisted" in checks and value.get("run_storage"):
        # Early runner revisions searched values too, which incorrectly treated
        # the existing report_summary status field as a schema mutation.
        manifest = value["run_storage"]["manifest"]
        keys: list[str] = []

        def collect(current: Any) -> None:
            if isinstance(current, Mapping):
                for key, item in current.items():
                    keys.append(str(key).lower())
                    collect(item)
            elif isinstance(current, list):
                for item in current:
                    collect(item)

        collect(manifest)
        checks.pop("observer_not_persisted", None)
        checks["observer_not_persisted_as_schema"] = not any(
            key.startswith("adaptive_continuity") or key == "decision_hash"
            for key in keys
        )
        value["checks"] = checks
        value["pass"] = all(checks.values())
        _write_json(path, value)
    if bool(value.get("pass")) and all((value.get("checks") or {}).values()):
        value["resumed_without_resampling"] = True
        return value
    return None


def _parse_observer(status: str) -> dict[str, Any]:
    summary_lines = [
        line
        for line in status.splitlines()
        if line.startswith("Adaptive Continuity observer [A8a v1]:")
    ]
    advisory_lines = [
        line
        for line in status.splitlines()
        if line.startswith("Adaptive Continuity advisory [A8a v1]:")
        or "Adaptive Continuity observer [A8a v1]: unavailable" in line
    ]
    if len(summary_lines) != 1:
        raise GateFailure(f"expected one A8a observer summary, found {len(summary_lines)}")
    match = SUMMARY_PATTERN.search(summary_lines[0])
    if match is None:
        raise GateFailure("A8a observer summary could not be parsed")
    records: list[dict[str, Any]] = []
    for line in status.splitlines():
        if not line.startswith("Adaptive Continuity decision [A8a v1]:"):
            continue
        current = DECISION_PATTERN.search(line)
        if current is None:
            raise GateFailure(f"A8a decision could not be parsed: {line}")
        fallback = current.group(13)
        record = {
            "physical_group": int(current.group(1)),
            "logical_chunks": json.loads(current.group(2)),
            "boundary_index": int(current.group(3)),
            "boundary_count": int(current.group(4)),
            "observed_motion_score": float(current.group(5)),
            "current_context_frames": int(current.group(6)),
            "resolved_transport": current.group(7),
            "terminal_atomic": current.group(8) == "true",
            "reused": current.group(9) == "true",
            "recommendation": current.group(10),
            "confidence": float(current.group(11)),
            "reason": current.group(12),
            "fallback_reason": None if fallback == "none" else fallback,
            "decision_hash": current.group(14),
            "execution_applied": False,
        }
        # Motion is intentionally formatted to six decimal places in the human
        # report while the decision hash retains the full observed float.  The
        # report can therefore validate hash presence/shape, not recompute it.
        record["hash_well_formed"] = bool(re.fullmatch(r"[0-9a-f]{64}", record["decision_hash"]))
        records.append(record)
    return {
        "planner_hash": match.group(1),
        "observed_groups": int(match.group(2)),
        "records": records,
        "advisories": advisory_lines,
        "summary": summary_lines[0],
    }


def _observer_checks(
    observer: Mapping[str, Any],
    expected: list[tuple[list[int], int, str, bool, bool]],
    *,
    total_chunks: int | None = None,
) -> dict[str, bool]:
    records = list(observer["records"])
    return {
        "planner_hash": observer["planner_hash"] == ADAPTIVE_CONTINUITY_PLANNER_HASH,
        "one_record_per_physical_group": len(records) == len(expected),
        "summary_group_count": int(observer["observed_groups"]) == len(expected),
        "physical_order": [item["physical_group"] for item in records]
        == list(range(1, len(expected) + 1)),
        "logical_groups": [item["logical_chunks"] for item in records]
        == [item[0] for item in expected],
        "context_frames": [item["current_context_frames"] for item in records]
        == [item[1] for item in expected],
        "transport": [item["resolved_transport"] for item in records]
        == [item[2] for item in expected],
        "terminal_atomic": [item["terminal_atomic"] for item in records]
        == [item[3] for item in expected],
        "reused": [item["reused"] for item in records]
        == [item[4] for item in expected],
        "boundary_count": all(
            int(record["boundary_count"])
            == max(
                0,
                int(total_chunks if total_chunks is not None else sum(len(item[0]) for item in expected)) - 1,
            )
            for record in records
        ),
        "decision_hashes": all(item["hash_well_formed"] for item in records),
        "execution_applied_false": all(not item["execution_applied"] for item in records),
        "no_observer_advisory": not observer["advisories"],
    }


def _baseline_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    probe = result["probe"]
    return {
        "video": list(probe.get("input_video_sha256") or []),
        "audio": list(probe.get("input_audio_sha256") or []),
        "assembly": copy.deepcopy((probe.get("input_plan") or {}).get("normalized")),
        "decoded": copy.deepcopy(result.get("decoded") or {}),
    }


def _run_public_case(
    args: argparse.Namespace,
    *,
    name: str,
    case: FirstPassCase,
    template_path: Path,
    baseline_path: Path,
    expected: list[tuple[list[int], int, str, bool, bool]],
) -> dict[str, Any]:
    template = _load_json(template_path)
    output_prefix = f"v38_a8a/{args.run_tag}/{name}"
    prompt = _configure_first_pass(
        template,
        case=case,
        diagnostics="Detailed Report",
        output_prefix=output_prefix,
        reference_image="unused.png",
        reference_video="unused.mp4",
    )
    result, entry = _execute(
        args,
        name=name,
        prompt=prompt,
        expected_seconds=case.expected_seconds,
        boundaries=tuple(float(index * 5) for index in range(1, case.chunks)),
    )
    status = _output_text(entry, "900")
    probe = _marker_json(_output_text(entry, "908"), "A3C_PROBE_JSON=")
    observer = _parse_observer(status)
    baseline = _baseline_signature(_load_json(baseline_path))
    current = {
        "video": list(probe.get("input_video_sha256") or []),
        "audio": list(probe.get("input_audio_sha256") or []),
        "assembly": copy.deepcopy((probe.get("input_plan") or {}).get("normalized")),
        "decoded": copy.deepcopy(result.get("decoded") or {}),
    }
    checks = {
        **_observer_checks(observer, expected, total_chunks=case.chunks),
        "video_latent_sha_parity": current["video"] == baseline["video"],
        "audio_latent_sha_parity": current["audio"] == baseline["audio"],
        "assembly_plan_parity": _cross_run_plan_identity(current["assembly"])
        == _cross_run_plan_identity(baseline["assembly"]),
        "decoded_video_parity": current["decoded"].get("video_sha256")
        == baseline["decoded"].get("video_sha256"),
        "decoded_audio_parity": current["decoded"].get("audio_pcm_sha256")
        == baseline["decoded"].get("audio_pcm_sha256"),
        "audio_health": bool(result["audio_health"]["pass"]),
        "no_reference_video": not any(
            key.startswith("reference_video_") for key in prompt["305"]["inputs"]
        ),
    }
    payload = {
        **result,
        "status": status,
        "probe": probe,
        "observer": observer,
        "baseline_path": str(baseline_path),
        "baseline": baseline,
        "current": current,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(args.evidence_root / name / "result.json", payload)
    if not payload["pass"]:
        raise GateFailure(f"{name} failed: {[key for key, value in checks.items() if not value]}")
    print(f"A8a {name}: PASS ({result['api_elapsed_seconds']:.3f}s)", flush=True)
    return payload


def _reachable_prompt(prompt: Mapping[str, Any], roots: tuple[str, ...]) -> dict[str, Any]:
    keep: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in keep:
            return
        node = prompt.get(node_id)
        if not isinstance(node, Mapping):
            raise GateFailure(f"prompt dependency is missing: {node_id}")
        keep.add(node_id)
        for value in (node.get("inputs") or {}).values():
            if isinstance(value, list) and len(value) == 2 and str(value[0]) in prompt:
                visit(str(value[0]))

    for root in roots:
        visit(root)
    return {key: copy.deepcopy(value) for key, value in prompt.items() if key in keep}


def _diagnostic_signature(value: Mapping[str, Any]) -> dict[str, Any]:
    def tensor_shas(key: str) -> list[str]:
        return [str(item.get("sha256") or "") for item in (value.get(key) or [])]

    calls = []
    for call in value.get("sample_calls") or []:
        calls.append(
            {
                "sample_number": call.get("sample_number"),
                "seed": call.get("seed"),
                "sigmas": (call.get("sigmas") or {}).get("sha256"),
                "prompt_embedding": (call.get("prompt_embedding") or {}).get("sha256"),
                "output_video": (call.get("output_video") or {}).get("sha256"),
                "output_audio": (call.get("output_audio") or {}).get("sha256"),
            }
        )
    return {
        "physical_group_count": value.get("physical_group_count"),
        "logical_chunk_count": value.get("logical_chunk_count"),
        "physical_decode_group_count": value.get("physical_decode_group_count"),
        "production_transport": value.get("production_transport"),
        "transport": value.get("transport"),
        "context_frames": value.get("context_frames"),
        "audio_prefix_steps": value.get("audio_prefix_steps"),
        "video_prefix_slots": value.get("video_prefix_slots"),
        "synchronize_sampling": value.get("synchronize_sampling"),
        "audio_continuity_forced": value.get("audio_continuity_forced"),
        "requested_audio_continuity": value.get("requested_audio_continuity"),
        "physical_video_sha256": tensor_shas("physical_video_groups"),
        "physical_audio_sha256": tensor_shas("physical_audio_groups"),
        "sample_calls": calls,
        "final_prefix_bit_exact": [
            [item.get("bit_exact"), item.get("audio_bit_exact")]
            for item in (value.get("final_prefix_pairs") or [])
        ],
    }


def _completed_history_result(
    args: argparse.Namespace,
    *,
    name: str,
    expected_seconds: float,
    boundaries: tuple[float, ...],
) -> tuple[dict[str, Any], Mapping[str, Any]] | None:
    """Recover evidence after Sampling when only runner post-processing failed."""

    history_path = args.evidence_root / name / "history.json"
    if not history_path.exists():
        return None
    history = _load_json(history_path)
    if not isinstance(history, Mapping) or len(history) != 1:
        return None
    entry = _history_entry(history, str(next(iter(history))))
    media_path = _output_media(history, args.output_root)
    media = _probe_media(media_path, args.ffprobe)
    result = {
        "name": name,
        "prompt_id": str(next(iter(history))),
        "api_elapsed_seconds": None,
        "prompt": str(args.evidence_root / name / "prompt.json"),
        "history": str(history_path),
        "media_path": str(media_path),
        "media": media,
        "decoded": {
            "video_sha256": _decoded_stream_hash(
                media_path, ffmpeg=args.ffmpeg, stream="video"
            ),
            "audio_pcm_sha256": _decoded_stream_hash(
                media_path, ffmpeg=args.ffmpeg, stream="audio"
            ),
        },
        "audio_health": _audio_health(
            media_path,
            ffmpeg=args.ffmpeg,
            expected_seconds=expected_seconds,
            boundaries=boundaries,
        ),
        "resources": {"salvaged_from_completed_history": True},
        "completed_history_reused_without_resampling": True,
    }
    return result, entry


def _run_issue13(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _reachable_prompt(_load_json(args.issue_template), ("191", "249"))
    # The accepted R2.6 wrapper later gained this required diagnostic-only
    # selector.  "Recursive" exactly preserves the accepted R2.6 source path.
    prompt["305"]["inputs"]["continuation_source_mode"] = "Recursive"
    prompt["191"]["inputs"]["filename_prefix"] = (
        f"v38_a8a/{args.run_tag}/g3_issue13_6x5_same_prompt"
    )
    recovered = _completed_history_result(
        args,
        name="g3_issue13_6x5_same_prompt",
        expected_seconds=30.0,
        boundaries=(5.0, 10.0, 15.0, 20.0, 25.0),
    )
    if recovered is None:
        result, entry = _execute(
            args,
            name="g3_issue13_6x5_same_prompt",
            prompt=prompt,
            expected_seconds=30.0,
            boundaries=(5.0, 10.0, 15.0, 20.0, 25.0),
        )
    else:
        result, entry = recovered
    status = _output_text(entry, "249")
    diagnostic_line = next(
        (
            line
            for line in status.splitlines()
            if line.startswith("V3.6-R1 diagnostic: ")
        ),
        "",
    )
    if not diagnostic_line:
        raise GateFailure("V3.6-R1 diagnostic JSON is missing")
    diagnostic = json.loads(diagnostic_line[len("V3.6-R1 diagnostic: ") :])
    observer = _parse_observer(status)
    baseline_diagnostic = _load_json(args.issue_baseline)["diagnostic"]
    baseline = _diagnostic_signature(baseline_diagnostic)
    current = _diagnostic_signature(diagnostic)
    expected = [
        ([index], 0 if index == 1 else 22, "initial" if index == 1 else "masked_av_prefix_22_v1", False, False)
        for index in range(1, 7)
    ]
    checks = {
        **_observer_checks(observer, expected, total_chunks=6),
        "accepted_r26_diagnostic_parity": current == baseline,
        "six_physical_groups": diagnostic.get("physical_group_count") == 6,
        "five_exact_prefix_pairs": len(diagnostic.get("final_prefix_pairs") or []) == 5
        and all(
            bool(item.get("bit_exact")) and bool(item.get("audio_bit_exact"))
            for item in (diagnostic.get("final_prefix_pairs") or [])
        ),
        "same_prompt": len(set(diagnostic.get("logical_prompt_hashes") or [])) <= 1,
        "audio_health": bool(result["audio_health"]["pass"]),
        "no_reference_video": "reference_video_1" not in prompt["305"]["inputs"],
    }
    payload = {
        **result,
        "status": status,
        "diagnostic": diagnostic,
        "observer": observer,
        "baseline_path": str(args.issue_baseline),
        "baseline_signature": baseline,
        "current_signature": current,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(args.evidence_root / "g3_issue13_6x5_same_prompt" / "result.json", payload)
    if not payload["pass"]:
        raise GateFailure(
            "g3_issue13_6x5_same_prompt failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    elapsed = result.get("api_elapsed_seconds")
    elapsed_text = "completed-history" if elapsed is None else f"{float(elapsed):.3f}s"
    print(f"A8a g3_issue13_6x5_same_prompt: PASS ({elapsed_text})", flush=True)
    return payload


def _review_prompt(
    args: argparse.Namespace,
    *,
    output_prefix: str,
    run_name: str,
) -> dict[str, Any]:
    template = _load_json(args.t2_template)
    case = FirstPassCase("g5_review", 3, False, "none", "accepted")
    prompt = _configure_first_pass(
        template,
        case=case,
        diagnostics="Detailed Report",
        output_prefix=output_prefix,
        reference_image="unused.png",
        reference_video="unused.mp4",
    )
    inputs = prompt["305"]["inputs"]
    inputs.update(
        {
            "run_storage": "Save + Auto Resume",
            "run_name": run_name,
            "project_id": str(uuid.uuid5(uuid.NAMESPACE_URL, run_name)),
            "generation_mode": "Review Each Chunk",
            "review_action": "Continue / Next",
        }
    )
    return prompt


def _manifest_entry_metadata(storage: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for stored in (storage["manifest"].get("chunks") or []):
        entry = stored.get("entry") or {}
        result.append(
            {
                "context_frames": int(entry.get("context_frames", 0)),
                "motion_score": float(entry.get("motion_score", 0.0)),
            }
        )
    return result


def _review_metadata_checks(
    observer: Mapping[str, Any], storage: Mapping[str, Any]
) -> dict[str, bool]:
    records = list(observer["records"])
    metadata = _manifest_entry_metadata(storage)
    count = min(len(records), len(metadata))
    schema_keys: list[str] = []

    def collect_keys(current: Any) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                schema_keys.append(str(key).lower())
                collect_keys(item)
        elif isinstance(current, list):
            for item in current:
                collect_keys(item)

    collect_keys(storage["manifest"])
    return {
        "trace_matches_saved_context": count == len(records)
        and all(records[index]["current_context_frames"] == metadata[index]["context_frames"] for index in range(count)),
        "trace_matches_saved_motion": count == len(records)
        and all(abs(records[index]["observed_motion_score"] - metadata[index]["motion_score"]) <= 0.000001 for index in range(count)),
        "observer_not_persisted_as_schema": not any(
            key.startswith("adaptive_continuity") or key == "decision_hash"
            for key in schema_keys
        ),
    }


def _run_review_queue(
    args: argparse.Namespace,
    *,
    queue_index: int,
    run_name: str,
    before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prompt = _review_prompt(
        args,
        output_prefix=f"v38_a8a/{args.run_tag}/g5_review_q{queue_index}",
        run_name=run_name,
    )
    result, entry = _execute(
        args,
        name=f"g5_review_q{queue_index}",
        prompt=prompt,
        expected_seconds=float(queue_index * 5),
        boundaries=tuple(float(index * 5) for index in range(1, queue_index)),
    )
    status = _output_text(entry, "900")
    observer = _parse_observer(status)
    storage = _manifest_snapshot(args.run_storage_root / run_name)
    expected = [
        ([index], 0 if index == 1 else 22, "initial" if index == 1 else "masked_av_prefix_22_v1", False, index < queue_index)
        for index in range(1, queue_index + 1)
    ]
    cached = sorted(_cached_nodes(_load_json(Path(result["history"]))))
    checks = {
        **_observer_checks(observer, expected, total_chunks=3),
        **_review_metadata_checks(observer, storage),
        "sampler_executed": "305" not in cached,
        "chunk_count": len(storage["chunks"]) == queue_index,
        "status": storage["status"] == ("complete" if queue_index == 3 else "review_ready"),
        "one_new_group": before is None or len(storage["chunks"]) == len(before["chunks"]) + 1,
        "prefix_exact_reuse": before is None or _preserved_prefix(before, storage),
        "audio_health": bool(result["audio_health"]["pass"]),
        "no_reference_video": "reference_video_1" not in prompt["305"]["inputs"],
    }
    payload = {
        **result,
        "status_text": status,
        "observer": observer,
        "run_storage": storage,
        "cached_nodes": cached,
        "checks": checks,
        "pass": all(checks.values()),
    }
    _write_json(args.evidence_root / f"g5_review_q{queue_index}" / "result.json", payload)
    if not payload["pass"]:
        raise GateFailure(
            f"g5_review_q{queue_index} failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    print(f"A8a g5_review_q{queue_index}: PASS ({result['api_elapsed_seconds']:.3f}s)", flush=True)
    return payload


def _compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pass": result.get("pass"),
        "api_elapsed_seconds": result.get("api_elapsed_seconds"),
        "resources": result.get("resources"),
        "checks": result.get("checks"),
        "observer": result.get("observer"),
        "result_path": str(Path(result["history"]).parent / "result.json"),
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.evidence_root / "gate_summary.json"
    summary = _load_json(summary_path) if summary_path.exists() else {
        "format": "h3-continuum-v38-a8a-adaptive-continuity-gpu-matrix-v1",
        "run_tag": args.run_tag,
        "server": args.server,
        "results": {},
        "pass": False,
        "production_status": "read-only observer; execution_applied=false; A8b HOLD; A5 HOLD",
    }
    summary["backend_pids"] = list(dict.fromkeys([*(summary.get("backend_pids") or []), args.backend_pid]))
    summary.pop("failure_type", None)
    summary.pop("failure", None)
    try:
        if args.stage == "pre-restart":
            public_root = args.public_baseline_root
            cases = (
                (
                    "g1_1x5",
                    FirstPassCase("g1", 1, False, "none", "accepted"),
                    args.t2_template,
                    public_root / "g1_1x5_t2va_detailed" / "result.json",
                    [([1], 0, "initial", False, False)],
                ),
                (
                    "g2_3x5",
                    FirstPassCase("g2", 3, False, "none", "accepted"),
                    args.t2_template,
                    public_root / "g2_3x5_continuation_detailed" / "result.json",
                    [
                        ([1], 0, "initial", False, False),
                        ([2], 22, "masked_av_prefix_22_v1", False, False),
                        ([3], 22, "masked_av_prefix_22_v1", False, False),
                    ],
                ),
            )
            for name, case, template, baseline, expected in cases:
                result_path = args.evidence_root / name / "result.json"
                current = _load_completed(result_path) or _run_public_case(
                    args,
                    name=name,
                    case=case,
                    template_path=template,
                    baseline_path=baseline,
                    expected=expected,
                )
                summary["results"][name] = _compact_result(current)
                _write_json(summary_path, summary)

            issue_path = args.evidence_root / "g3_issue13_6x5_same_prompt" / "result.json"
            issue = _load_completed(issue_path) or _run_issue13(args)
            summary["results"]["g3_issue13_6x5_same_prompt"] = _compact_result(issue)
            _write_json(summary_path, summary)

            terminal_path = args.evidence_root / "g4_fl2va_terminal" / "result.json"
            terminal = _load_completed(terminal_path) or _run_public_case(
                args,
                name="g4_fl2va_terminal",
                case=FirstPassCase("g4", 3, True, "none", "accepted"),
                template_path=args.terminal_template,
                baseline_path=public_root / "g3_fl2va_terminal_detailed" / "result.json",
                expected=[
                    ([1], 0, "initial", False, False),
                    ([2, 3], 22, "masked_av_prefix_22_v1", True, False),
                ],
            )
            summary["results"]["g4_fl2va_terminal"] = _compact_result(terminal)
            _write_json(summary_path, summary)

            run_name = f"v38_a8a_review_restart_{args.run_tag}"
            q1_path = args.evidence_root / "g5_review_q1" / "result.json"
            q1 = _load_completed(q1_path) or _run_review_queue(
                args,
                queue_index=1,
                run_name=run_name,
                before=None,
            )
            summary["results"]["g5_review_q1"] = _compact_result(q1)
            summary["restart_state"] = {
                "run_name": run_name,
                "q1_manifest": q1["run_storage"],
                "pre_restart_backend_pid": args.backend_pid,
            }
            summary["status"] = "A8a pre-restart gates PASS; backend restart required"
            _write_json(summary_path, summary)
            return summary, 0

        state = summary.get("restart_state") or {}
        if not state.get("run_name") or not state.get("q1_manifest"):
            raise GateFailure("post-restart stage requires completed pre-restart state")
        if int(state.get("pre_restart_backend_pid")) == int(args.backend_pid):
            raise GateFailure("post-restart stage must use a fresh backend PID")
        run_name = str(state["run_name"])
        before = state["q1_manifest"]
        disk_before = _manifest_snapshot(args.run_storage_root / run_name)
        if not _preserved_prefix(before, disk_before) or len(disk_before["chunks"]) != 1:
            raise GateFailure("Run Storage prefix was not restored exactly after restart")

        q2 = _run_review_queue(
            args,
            queue_index=2,
            run_name=run_name,
            before=disk_before,
        )
        summary["results"]["g5_review_q2_after_restart"] = _compact_result(q2)
        _write_json(summary_path, summary)
        q3 = _run_review_queue(
            args,
            queue_index=3,
            run_name=run_name,
            before=q2["run_storage"],
        )
        summary["results"]["g5_review_q3_complete"] = _compact_result(q3)
        summary["restart_state"]["post_restart_backend_pid"] = args.backend_pid
        summary["pass"] = all(bool(item.get("pass")) for item in summary["results"].values())
        summary["status"] = (
            "A8a Adaptive Continuity Observer GPU Experimental PASS"
            if summary["pass"]
            else "A8a STOP / Production unchanged"
        )
        _write_json(summary_path, summary)
        return summary, 0 if summary["pass"] else 2
    except Exception as exc:
        summary["pass"] = False
        summary["status"] = "A8a STOP / Production unchanged"
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        _write_json(summary_path, summary)
        return summary, 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pre-restart", "post-restart"), required=True)
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--t2-template", type=Path, required=True)
    parser.add_argument("--terminal-template", type=Path, required=True)
    parser.add_argument("--public-baseline-root", type=Path, required=True)
    parser.add_argument("--issue-template", type=Path, required=True)
    parser.add_argument("--issue-baseline", type=Path, required=True)
    parser.add_argument(
        "--run-storage-root",
        type=Path,
        default=Path(r"D:\output\video\comfy_video\h3_continuum\runs"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\output\video\comfy_video"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    summary, code = run(args)
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "pass": summary.get("pass"),
                "failure": summary.get("failure"),
                "summary": str(args.evidence_root / "gate_summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
