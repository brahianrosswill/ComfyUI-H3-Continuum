"""Run the staged Issue #13 R2.8 continuation depth/strength GPU gate."""

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

import issue13_r25_gate_runner as r25
from p32_api_runner import submit_and_wait


CASE_SPECS = {
    "baseline_22f": {"experiment": "Baseline 22f"},
    "depth_13f": {"experiment": "Video Depth 13f"},
    "depth_9f": {"experiment": "Video Depth 9f"},
    "strength_75": {"experiment": "Video Strength 75%"},
    "depth_13f_strength_75": {
        "experiment": "Video Depth 13f + Strength 75%"
    },
}

DIAGNOSTIC_MARKER = "R2.8 diagnostic: "


def configure_prompt(
    source: dict,
    *,
    case: str,
    output_prefix: str,
    image_name: str,
    seed: int,
    width: int,
    height: int,
    diagnostic_nonce: int,
) -> tuple[dict, list[dict]]:
    if case not in CASE_SPECS:
        raise ValueError(f"unknown Issue #13 R2.8 case: {case}")
    prompt = copy.deepcopy(source)
    sampler = prompt["305"]
    sampler["class_type"] = "H3ContinuumContinuationR28Diagnostic"
    sampler["_meta"] = {"title": f"Issue #13 R2.8 Diagnostic — {case}"}
    inputs = sampler["inputs"]
    inputs.update(
        {
            "prompt_mode": "List",
            "chunks": 6,
            "chunk_seconds": 5.0,
            "width": int(width),
            "height": int(height),
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
            "continuation_experiment": CASE_SPECS[case]["experiment"],
            "diagnostic_nonce": int(diagnostic_nonce),
            "first_frame": ["119", 0],
        }
    )
    for key in (
        "continuation_backend",
        "continuation_source_mode",
        "synchronize_sampling",
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

    prompt["188"]["inputs"]["value"] = r25._prompt_list(
        6, motion_trigger=False, fixed_prompt=True
    )
    prompt["114"]["inputs"]["image"] = str(image_name)
    prompt["139"]["inputs"]["sampler_name"] = "euler"
    prompt["153"]["inputs"].update(
        {"scheduler": "simple", "steps": 8, "denoise": 1.0}
    )
    prompt["297"]["inputs"]["value"] = max(
        0.05, (float(width) * float(height)) / 1_000_000.0
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
    selected = r25._configure_loras(prompt["286"]["inputs"], "turbo")
    return prompt, selected


def _diagnostic_from_status(status: str) -> dict:
    if DIAGNOSTIC_MARKER not in status:
        raise RuntimeError("Issue #13 R2.8 diagnostic marker is missing")
    value = json.loads(status.rsplit(DIAGNOSTIC_MARKER, 1)[1].strip())
    if not isinstance(value, dict):
        raise RuntimeError("Issue #13 R2.8 diagnostic payload is not an object")
    return value


def run(args: argparse.Namespace) -> dict:
    source = json.loads(args.prompt.read_text(encoding="utf-8"))
    cases = list(args.cases or ("depth_13f",))
    image_sha256 = hashlib.sha256(args.image_source.read_bytes()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    results = []
    if summary_path.exists():
        results = list(
            json.loads(summary_path.read_text(encoding="utf-8")).get("cases") or []
        )
    completed = {str(item.get("case")) for item in results}

    for offset, case in enumerate(cases):
        if case in completed:
            print(f"ISSUE13-R28 SKIP completed case={case}", flush=True)
            continue
        configured, selected = configure_prompt(
            source,
            case=case,
            output_prefix=f"video/issue13_r28/Issue13_R28_{case}_seed{args.seed}",
            image_name=args.image_name,
            seed=args.seed,
            width=args.width,
            height=args.height,
            diagnostic_nonce=args.diagnostic_nonce + offset,
        )
        inventory = r25._inventory_loras(args.lora_root, selected)
        case_root = args.output_root / case
        case_root.mkdir(parents=True, exist_ok=True)
        prompt_path = case_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(configured, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"ISSUE13-R28 START case={case}", flush=True)
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=configured,
            client_id=f"issue13-r28-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=3.0,
        )
        history_path = case_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        status = r25._status_text(history, prompt_id)
        result = {
            "case": case,
            "chunks": 6,
            "chunk_seconds": 5.0,
            "profile": "turbo",
            "steps": 8,
            "sampler": "euler",
            "scheduler": "simple",
            "prompt_id": prompt_id,
            "api_elapsed_seconds": float(elapsed),
            "prompt": str(prompt_path),
            "history": str(history_path),
            "output_video": r25._output_video(history, prompt_id),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "lora_inventory": inventory,
            "diagnostic": _diagnostic_from_status(status),
        }
        (case_root / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(result)
        summary = {
            "format": "h3-continuum-issue13-r28-gate-v1",
            "server": args.server,
            "source_prompt": str(args.prompt),
            "baseline_summary": str(args.baseline_summary)
            if args.baseline_summary
            else None,
            "input_image": {
                "source": str(args.image_source),
                "runtime_name": args.image_name,
                "sha256": image_sha256,
            },
            "seed": int(args.seed),
            "width": int(args.width),
            "height": int(args.height),
            "sampler": "euler",
            "scheduler": "simple",
            "continuity_transport": "Balanced 22f / masked_av_prefix_22_v1",
            "audio_continuity": True,
            "cases": results,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"ISSUE13-R28 COMPLETE case={case} prompt_id={prompt_id} elapsed={elapsed:.3f}s",
            flush=True,
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-source", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--baseline-summary", type=Path)
    parser.add_argument(
        "--lora-root",
        type=Path,
        default=Path(r"D:\StabilityMatrix\Data\Packages\ComfyUI_W\models\loras"),
    )
    parser.add_argument("--seed", type=int, default=131325)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--diagnostic-nonce", type=int, default=132800)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--cases", nargs="*", choices=tuple(CASE_SPECS))
    return parser


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
