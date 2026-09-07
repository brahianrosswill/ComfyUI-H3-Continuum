"""Resume the stopped Phase F G1 run after the Review cache fix.

The failed Queue 2 prompt is submitted unchanged for both recovery queues.  This
keeps the sampler inputs (and the complete API graph) identical while proving
that Review mode no longer reuses the sampler's ComfyUI output cache.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from v38_review_phase_f_gpu_gate_runner import (
    GateFailure,
    _find_one,
    _manifest_snapshot,
    _parity,
    _run_queue,
    _sha256_file,
)


def _cached_nodes(history: Mapping[str, Any]) -> set[str]:
    nodes: set[str] = set()
    for prompt_history in history.values():
        for message in (prompt_history.get("status") or {}).get("messages") or []:
            if len(message) != 2 or message[0] != "execution_cached":
                continue
            nodes.update(str(node_id) for node_id in (message[1].get("nodes") or []))
    return nodes


def _chunk_identity(chunk: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "sequence_index",
        "file_sha256",
        "video_latent_sha256",
        "audio_latent_sha256",
        "seed",
        "prompt_hash",
        "plan",
    )
    return {field: copy.deepcopy(chunk[field]) for field in fields}


def _preserved_prefix(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    before_chunks = before["chunks"]
    after_chunks = after["chunks"]
    return len(after_chunks) >= len(before_chunks) and all(
        _chunk_identity(old) == _chunk_identity(after_chunks[index])
        for index, old in enumerate(before_chunks)
    )


def _sampler_inputs(prompt: Mapping[str, Any]) -> tuple[str, Mapping[str, Any]]:
    node_id, sampler = _find_one(prompt, "H3ContinuumSamplerV38")
    return node_id, sampler["inputs"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.prior_summary.read_text(encoding="utf-8"))
    g0 = prior["results"]["g0"]
    failed_q2 = prior["results"]["g1"][-1]
    prompt_path = Path(failed_q2["prompt"])
    prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
    sampler_id, sampler_inputs = _sampler_inputs(prompt)
    review_run_name = str(prior["run_names"]["g1"])
    run_root = args.run_storage_root / review_run_name
    before = _manifest_snapshot(run_root)
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-review-phase-f-r1-recovery-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "prior_summary": str(args.prior_summary),
        "failed_queue_prompt": str(prompt_path),
        "failed_queue_prompt_sha256": _sha256_file(prompt_path),
        "sampler_node_id": sampler_id,
        "sampler_inputs": sampler_inputs,
        "run_name": review_run_name,
        "initial_run_storage": before,
        "pass": False,
        "stopped": False,
        "results": {},
    }
    summary_path = args.output_root / "recovery_summary.json"
    try:
        if len(before["chunks"]) != 1 or before["status"] != "review_ready":
            raise GateFailure(
                "Recovery must start at Chunk 1 review_ready: "
                f"chunks={len(before['chunks'])}, status={before['status']}"
            )

        q2 = _run_queue(
            args,
            name="g1_recovery_q2",
            prompt=prompt,
            run_root=run_root,
        )
        q2_cached = sorted(_cached_nodes(json.loads(Path(q2["history"]).read_text(encoding="utf-8"))))
        q2_storage = q2["run_storage"]
        q2_checks = {
            "exact_failed_prompt_reused": _sha256_file(Path(q2["prompt"]))
            == summary["failed_queue_prompt_sha256"],
            "sampler_not_cached": sampler_id not in q2_cached,
            "status_review_ready": q2_storage["status"] == "review_ready",
            "exactly_two_chunks": len(q2_storage["chunks"]) == 2,
            "exactly_one_new_physical_chunk": len(q2_storage["chunks"])
            == len(before["chunks"]) + 1,
            "chunk_1_exactly_reused": _preserved_prefix(before, q2_storage),
            "manifest_updated": q2_storage["mtime_ns"] != before["mtime_ns"],
        }
        summary["results"]["q2"] = {
            **q2,
            "cached_nodes": q2_cached,
            "checks": q2_checks,
        }
        if not all(q2_checks.values()):
            raise GateFailure(f"Recovery Queue 2 failed: {q2_checks}")

        q3 = _run_queue(
            args,
            name="g1_recovery_q3",
            prompt=prompt,
            run_root=run_root,
        )
        q3_cached = sorted(_cached_nodes(json.loads(Path(q3["history"]).read_text(encoding="utf-8"))))
        q3_storage = q3["run_storage"]
        q3_checks = {
            "same_prompt_as_queue_2": _sha256_file(Path(q3["prompt"]))
            == _sha256_file(Path(q2["prompt"])),
            "sampler_not_cached": sampler_id not in q3_cached,
            "status_complete": q3_storage["status"] == "complete",
            "exactly_three_chunks": len(q3_storage["chunks"]) == 3,
            "exactly_one_new_physical_chunk": len(q3_storage["chunks"])
            == len(q2_storage["chunks"]) + 1,
            "chunks_1_2_exactly_reused": _preserved_prefix(q2_storage, q3_storage),
            "manifest_updated": q3_storage["mtime_ns"] != q2_storage["mtime_ns"],
        }
        parity = _parity(g0, q3)
        summary["results"]["q3"] = {
            **q3,
            "cached_nodes": q3_cached,
            "checks": q3_checks,
        }
        summary["parity"] = parity
        if not all(q3_checks.values()):
            raise GateFailure(f"Recovery Queue 3 failed: {q3_checks}")
        if not parity["pass"]:
            raise GateFailure("Recovered G1 does not exactly match the Phase F G0 baseline")
        summary["pass"] = True
        return summary
    except Exception as exc:
        summary["stopped"] = True
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        return summary
    finally:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--run-storage-root", type=Path, required=True)
    parser.add_argument("--backend-pid", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "pass": summary["pass"],
                "stopped": summary["stopped"],
                "failure": summary.get("failure"),
                "summary": str(args.output_root / "recovery_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
