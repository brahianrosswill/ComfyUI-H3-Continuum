"""G4 FL2VA conditioning determinism Gate (evidence-only, no Sampling)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import submit_and_wait


CHECKPOINTS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
FIRST_IMAGE = "20260830030614-Anima-Luc_v2_00008_-1635935150-ER SDE.webp"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
PROMPT = (
    "A continuous cinematic shot begins from the supplied First Image. "
    "The subject moves naturally while identity, clothing, lighting, camera "
    "direction, footsteps, and ambience remain coherent."
)


class GateFailure(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(*, context: str, focus_mode: str, nonce: int) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "LoadImage",
            "inputs": {"image": FIRST_IMAGE},
            "_meta": {"title": "LoadImage"},
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VIDEO_VAE},
            "_meta": {"title": "Video VAE"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"},
            "_meta": {"title": "MiniMax H3 Text Encoder"},
        },
        "4": {
            "class_type": "H3G4ConditioningDeterminismDiagnostic",
            "inputs": {
                "video_vae": ["2", 0],
                "clip": ["3", 0],
                "first_image": ["1", 0],
                "prompt": PROMPT,
                "width": 544,
                "height": 544,
                "frame_count": 124,
                "execution_context": context,
                "focus_mode": focus_mode,
                "diagnostic_nonce": int(nonce),
            },
            "_meta": {"title": "G4 Conditioning Determinism Collector"},
        },
    }


def _extract(history: Mapping[str, Any], prompt_id: str) -> dict[str, Any]:
    record = history.get(prompt_id)
    if not isinstance(record, Mapping):
        raise GateFailure(f"history record missing for {prompt_id}")
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GateFailure("history outputs are missing")
    node = outputs.get("4")
    if not isinstance(node, Mapping):
        raise GateFailure("diagnostic node output is missing")
    text = node.get("text")
    if not isinstance(text, list) or len(text) != 1:
        raise GateFailure(f"unexpected diagnostic text output: {text!r}")
    result = json.loads(str(text[0]))
    if result.get("format") != "h3-v38-g4-conditioning-determinism-v1":
        raise GateFailure("unexpected diagnostic output format")
    missing = [key for key in CHECKPOINTS if key not in result.get("checkpoints", {})]
    if missing:
        raise GateFailure(f"missing checkpoints: {missing}")
    return result


def _run_case(args, *, name: str, context: str, focus_mode: str, nonce: int) -> dict[str, Any]:
    case_root = args.evidence_root / args.stage / name
    prompt = _prompt(context=context, focus_mode=focus_mode, nonce=nonce)
    _write(case_root / "prompt.json", prompt)
    print(f"G4 diagnostic start: {name}", flush=True)
    prompt_id, history, elapsed = submit_and_wait(
        server=args.server,
        prompt=prompt,
        client_id=f"g4diag-{uuid.uuid4().hex}",
        timeout_seconds=float(args.timeout_seconds),
        poll_seconds=float(args.poll_seconds),
    )
    _write(case_root / "history.json", history)
    result = _extract(history, prompt_id)
    _write(case_root / "result.json", result)
    output = {
        "name": name,
        "prompt_id": prompt_id,
        "api_elapsed_seconds": elapsed,
        "prompt": str(case_root / "prompt.json"),
        "history": str(case_root / "history.json"),
        "result_path": str(case_root / "result.json"),
        "result": result,
    }
    print(f"G4 diagnostic end: {name}; {elapsed:.3f}s", flush=True)
    return output


def _checkpoint_sha(run: Mapping[str, Any], checkpoint: str) -> str:
    return str(run["result"]["checkpoints"][checkpoint]["sha256"])


def _compare(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    first = None
    for checkpoint in CHECKPOINTS:
        left_sha = _checkpoint_sha(left, checkpoint)
        right_sha = _checkpoint_sha(right, checkpoint)
        match = left_sha == right_sha
        rows[checkpoint] = {
            "left_sha256": left_sha,
            "right_sha256": right_sha,
            "match": match,
        }
        if first is None and not match:
            first = checkpoint
    return {"first_divergence": first, "checkpoints": rows, "all_match": first is None}


def _focus_mode(*firsts: str | None) -> str:
    if "P2" in firsts:
        return "vae_repeat"
    if "P5" in firsts:
        return "qwen_repeat"
    return "standard"


def _focus_checks(run: Mapping[str, Any]) -> dict[str, Any]:
    result = run["result"]
    mode = str(result["focus_mode"])
    if mode == "vae_repeat":
        rows = result["focus"]["vae_repeat"]
        hashes = [str(row["sha256"]) for row in rows]
        return {"mode": mode, "hashes": hashes, "all_match": len(set(hashes)) == 1}
    if mode == "qwen_repeat":
        rows = result["focus"]["qwen_repeat"]
        token_hashes = [str(row["tokens"]["sha256"]) for row in rows]
        cond_hashes = [str(row["conditioning"]["sha256"]) for row in rows]
        return {
            "mode": mode,
            "token_hashes": token_hashes,
            "conditioning_hashes": cond_hashes,
            "tokens_all_match": len(set(token_hashes)) == 1,
            "conditioning_all_match": len(set(cond_hashes)) == 1,
        }
    return {"mode": "standard"}


def _prepare(args) -> dict[str, Any]:
    cold = _run_case(args, name="cold_a", context="Cold A", focus_mode="standard", nonce=1)
    warm = _run_case(args, name="warm_b", context="Warm B", focus_mode="standard", nonce=2)
    full = _run_case(args, name="full_context", context="Full", focus_mode="standard", nonce=3)
    review = _run_case(args, name="review_context", context="Review", focus_mode="standard", nonce=4)
    cold_warm = _compare(cold, warm)
    full_review = _compare(full, review)
    mode = _focus_mode(cold_warm["first_divergence"], full_review["first_divergence"])
    focus = None
    focus_checks = None
    if mode != "standard":
        focus = _run_case(args, name=mode, context="Warm B", focus_mode=mode, nonce=5)
        focus_checks = _focus_checks(focus)
    summary = {
        "format": "h3-v38-g4-conditioning-determinism-prepare-v1",
        "stage": "prepare",
        "server": args.server,
        "runner_sha256": _sha256_file(Path(__file__)),
        "runs": {"cold_a": cold, "warm_b": warm, "full": full, "review": review, "focus": focus},
        "comparisons": {"cold_vs_warm": cold_warm, "full_vs_review": full_review},
        "focus_mode": mode,
        "focus_checks": focus_checks,
    }
    _write(args.evidence_root / "prepare_summary.json", summary)
    return summary


def _classify(prepare: Mapping[str, Any], restart_compare: Mapping[str, Any]) -> dict[str, Any]:
    cold_first = prepare["comparisons"]["cold_vs_warm"]["first_divergence"]
    context_first = prepare["comparisons"]["full_vs_review"]["first_divergence"]
    restart_first = restart_compare["first_divergence"]
    first = context_first or cold_first or restart_first
    standalone_first = cold_first or restart_first
    if standalone_first is None and context_first is not None:
        case = "CASE F"
        reason = "standalone repeats match; Full/Review context is the only divergence"
    elif first in ("P0", "P1"):
        case = "CASE A"
        reason = "resize/input path is the first divergence"
    elif first == "P2":
        case = "CASE B"
        reason = "Video VAE identity latent is the first divergence"
    elif first in ("P3", "P4", "P5", "P6"):
        case = "CASE C"
        reason = "Qwen/CLIP image-conditioning path is the first divergence"
    elif first == "P7":
        case = "CASE D"
        reason = "conditioning assembly/keyframe attachment is the first divergence"
    else:
        case = "CASE E"
        reason = "P0-P7 are bit-exact; the G4 hypothesis must move to Sampling or later runtime ordering"
    return {"case": case, "first_divergence": first, "reason": reason}


def _short(value: str) -> str:
    return value[:12]


def _report_markdown(summary: Mapping[str, Any]) -> str:
    prepare = summary["prepare"]
    restart = summary["restart_c"]
    full = prepare["runs"]["full"]
    review = prepare["runs"]["review"]
    repeat_cmp = prepare["comparisons"]["cold_vs_warm"]
    context_cmp = prepare["comparisons"]["full_vs_review"]
    restart_cmp = summary["comparisons"]["cold_vs_restart"]
    lines = [
        "# G4-Diagnostic FL2VA Conditioning Determinism — STOP Report",
        "",
        f"Classification: **{summary['classification']['case']}**  ",
        f"First divergence: **{summary['classification']['first_divergence'] or 'none through P7'}**  ",
        f"Result: {summary['classification']['reason']}",
        "",
        "Production code was not changed. GPU Sampling was not executed.",
        "",
        "| Checkpoint | Full | Review | Full/Review | Repeat | Restart |",
        "|---|---:|---:|---|---|---|",
    ]
    for checkpoint in CHECKPOINTS:
        full_sha = _checkpoint_sha(full, checkpoint)
        review_sha = _checkpoint_sha(review, checkpoint)
        lines.append(
            f"| {checkpoint} | `{_short(full_sha)}` | `{_short(review_sha)}` | "
            f"{'PASS' if context_cmp['checkpoints'][checkpoint]['match'] else 'FAIL'} | "
            f"{'PASS' if repeat_cmp['checkpoints'][checkpoint]['match'] else 'FAIL'} | "
            f"{'PASS' if restart_cmp['checkpoints'][checkpoint]['match'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Focused repeat",
            "",
            "```json",
            json.dumps(prepare.get("focus_checks"), indent=2, ensure_ascii=False),
            "```",
            "",
            "## Runtime state",
            "",
            "Cold, warm, Full-context, Review-context, and restart state snapshots are retained in the JSON evidence. Patch UUID creation differences across restart are recorded but are not treated as tensor divergence by themselves.",
            "",
            "## Next action",
            "",
            "STOP. Present the first divergence, impact, and a minimal branch-specific fix proposal for approval. Do not add caches, deterministic-mode forcing, model reloads, dtype/seed changes, or Run Storage contract changes in this Gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _restart(args) -> dict[str, Any]:
    prepare_path = args.evidence_root / "prepare_summary.json"
    if not prepare_path.is_file():
        raise GateFailure(f"prepare summary missing: {prepare_path}")
    prepare = json.loads(prepare_path.read_text(encoding="utf-8"))
    restart = _run_case(args, name="restart_c", context="Restart C", focus_mode="standard", nonce=101)
    cold_restart = _compare(prepare["runs"]["cold_a"], restart)
    classification = _classify(prepare, cold_restart)
    summary = {
        "format": "h3-v38-g4-conditioning-determinism-final-v1",
        "stage": "stopped",
        "prepare": prepare,
        "restart_c": restart,
        "comparisons": {"cold_vs_restart": cold_restart},
        "classification": classification,
    }
    _write(args.evidence_root / "g4_conditioning_determinism_summary.json", summary)
    report = _report_markdown(summary)
    (args.evidence_root / "G4_CONDITIONING_DETERMINISM_STOP_REPORT.md").write_text(
        report,
        encoding="utf-8",
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "restart"), required=True)
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path(
            r"D:\Codex\_test_results\ComfyUI-H3-Continuum"
            r"\v38-g4-conditioning-determinism-20260831_150837"
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    summary = _prepare(args) if args.stage == "prepare" else _restart(args)
    print(json.dumps(summary.get("classification", {"stage": args.stage}), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
