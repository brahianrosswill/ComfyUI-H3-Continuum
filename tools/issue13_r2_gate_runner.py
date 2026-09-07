"""Run the Issue #13 R2 long-duration continuation diagnostic.

This is evidence tooling only. It selects the already accepted Masked AV
transport explicitly through the external GPU diagnostic node and does not
modify the Production sampler, conditioning, or assembly path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import submit_and_wait


BASE_PROMPT = (
    "A continuous cinematic live-action tracking shot of one woman in an "
    "orange jacket riding a skateboard steadily forward along a waterfront "
    "promenade. The camera follows her in one smooth direction. Preserve her "
    "face, clothing, skateboard, warm sunlight, detailed texture, rich color, "
    "and continuous forward motion. Natural skateboard wheels, wind, and "
    "distant city ambience are audible."
)

CASE_SPECS = {
    "g1_standard_6x10": {"chunks": 6, "chunk_seconds": 10.0},
    "g2_standard_5x5": {"chunks": 5, "chunk_seconds": 5.0},
}

DIAGNOSTIC_MARKER = "V3.6-R1 diagnostic: "


def _same_prompt_list(chunks: int) -> str:
    return "\n---\n".join([BASE_PROMPT] * int(chunks))


def configure_prompt(
    source: dict,
    *,
    case: str,
    output_prefix: str,
    seed: int,
    steps: int,
    size: int,
    diagnostic_nonce: int,
) -> dict:
    if case not in CASE_SPECS:
        raise ValueError(f"unknown Issue #13 R2 case: {case}")
    spec = CASE_SPECS[case]
    chunks = int(spec["chunks"])
    chunk_seconds = float(spec["chunk_seconds"])
    prompt = copy.deepcopy(source)

    sampler = prompt["305"]
    sampler["class_type"] = "H3ContinuumMaskedPrefixR1Diagnostic"
    sampler["_meta"] = {"title": f"Issue #13 R2 Diagnostic — {case}"}
    inputs = sampler["inputs"]
    inputs.update(
        {
            "prompt_mode": "List",
            "chunks": chunks,
            "chunk_seconds": chunk_seconds,
            "width": int(size),
            "height": int(size),
            "continuity": "Balanced — 22 frames",
            "base_seed": int(seed),
            "audio_continuity": True,
            "diagnostics": "Detailed Report",
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "strict_compatibility": False,
            "debug": False,
            "show_preview": False,
            "run_storage": "Off",
            "run_name": "",
            "project_id": "",
            "continuation_transport": "masked_av_prefix_22_v1",
            "synchronize_sampling": True,
            "diagnostic_nonce": int(diagnostic_nonce),
            "first_frame": ["119", 0],
        }
    )
    for key in (
        "continuation_backend",
        "generation_mode",
        "review_action",
        "max_new_physical_groups",
        "aspect",
        "preset",
        "custom_mp",
        "guide",
        "last_frame",
    ):
        inputs.pop(key, None)

    prompt["188"]["inputs"]["value"] = _same_prompt_list(chunks)
    prompt["153"]["inputs"].update(
        {"scheduler": "simple", "steps": int(steps), "denoise": 1.0}
    )
    prompt["297"]["inputs"]["value"] = max(
        0.05, (float(size) * float(size)) / 1_000_000.0
    )
    prompt["306"]["inputs"].update(
        {
            "exact_total_duration": True,
            "audio_seam": "Off",
            "video_seam": "Off",
            "buffer_backend": "Auto",
            "diagnostics": "Detailed Report",
        }
    )
    prompt["191"]["inputs"]["filename_prefix"] = str(output_prefix)
    prompt["249"]["inputs"]["source"] = ["305", 3]

    spectrum = prompt.get("150", {}).get("inputs", {})
    if spectrum:
        spectrum["enabled"] = False
    lora_loader = prompt.get("286", {}).get("inputs", {})
    for name, value in lora_loader.items():
        if name.startswith("lora_") and isinstance(value, dict):
            value["on"] = False
    return prompt


def _output_video(history: dict, prompt_id: str) -> dict | None:
    entry = history.get(str(prompt_id), {})
    outputs = entry.get("outputs", {}) if isinstance(entry, dict) else {}
    save = outputs.get("191", {}) if isinstance(outputs, dict) else {}
    images = save.get("images", []) if isinstance(save, dict) else []
    return images[0] if images else None


def _status_text(history: dict, prompt_id: str) -> str:
    entry = history.get(str(prompt_id), {})
    outputs = entry.get("outputs", {}) if isinstance(entry, dict) else {}
    preview = outputs.get("249", {}) if isinstance(outputs, dict) else {}
    values = preview.get("text", []) if isinstance(preview, dict) else []
    return str(values[0]) if values else ""


def _diagnostic_from_status(status: str) -> dict:
    if DIAGNOSTIC_MARKER not in status:
        raise RuntimeError("Issue #13 R2 diagnostic marker is missing from status")
    payload = status.rsplit(DIAGNOSTIC_MARKER, 1)[1].strip()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise RuntimeError("Issue #13 R2 diagnostic payload is not an object")
    return value


def run(args: argparse.Namespace) -> dict:
    source = json.loads(args.prompt.read_text(encoding="utf-8"))
    cases = list(args.cases or CASE_SPECS)
    unknown = [case for case in cases if case not in CASE_SPECS]
    if unknown:
        raise SystemExit(f"unknown --cases: {', '.join(unknown)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    results = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        results = list(existing.get("cases") or [])
    completed_cases = {str(item.get("case")) for item in results}

    for offset, case in enumerate(cases):
        if case in completed_cases:
            print(f"ISSUE13-R2 SKIP completed case={case}", flush=True)
            continue
        spec = CASE_SPECS[case]
        configured = configure_prompt(
            source,
            case=case,
            output_prefix=f"video/issue13_r2/Issue13_R2_{case}_seed{args.seed}",
            seed=args.seed,
            steps=args.steps,
            size=args.size,
            diagnostic_nonce=args.diagnostic_nonce + offset,
        )
        case_root = args.output_root / case
        case_root.mkdir(parents=True, exist_ok=True)
        prompt_path = case_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(configured, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"ISSUE13-R2 START case={case}", flush=True)
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=configured,
            client_id=f"issue13-r2-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=3.0,
        )
        history_path = case_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status = _status_text(history, prompt_id)
        diagnostic = _diagnostic_from_status(status)
        result = {
            "case": case,
            "chunks": int(spec["chunks"]),
            "chunk_seconds": float(spec["chunk_seconds"]),
            "prompt_id": prompt_id,
            "api_elapsed_seconds": elapsed,
            "prompt": str(prompt_path),
            "history": str(history_path),
            "output_video": _output_video(history, prompt_id),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "diagnostic": diagnostic,
        }
        (case_root / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        results.append(result)
        print(
            f"ISSUE13-R2 COMPLETE case={case} prompt_id={prompt_id} "
            f"elapsed={elapsed:.3f}s",
            flush=True,
        )

        summary = {
            "format": "h3-continuum-issue13-r2-gate-v1",
            "server": args.server,
            "source_prompt": str(args.prompt),
            "seed": int(args.seed),
            "steps": int(args.steps),
            "size": int(args.size),
            "sampler": "res_multistep",
            "scheduler": "simple",
            "continuity": "Balanced — 22 frames",
            "audio_continuity": True,
            "transport": "masked_av_prefix_22_v1",
            "spectrum": False,
            "lora": False,
            "video_seam": "Off",
            "audio_seam": "Off",
            "cases": results,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=130013)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--diagnostic-nonce", type=int, default=131300)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--cases", nargs="*", choices=tuple(CASE_SPECS))
    return parser


def main() -> int:
    summary = run(_parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
