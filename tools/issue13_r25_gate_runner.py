"""Run the staged Issue #13 R2.5 cause-isolation GPU gate.

This is evidence tooling only. It uses the external diagnostic node and keeps
the Production sampler, conditioning, Run Storage, and assembly code intact.
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


TURBO_LORA = "minimax\\minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
MOTION_BOOSTER_LORA = "minimax\\H3_Motion_BoosterV2.safetensors"
MOTION_REPAIR_LORA = "minimax\\Motion_Repair.safetensors"

CASE_SPECS = {
    "screen_3x5_turbo": {"chunks": 3, "chunk_seconds": 5.0, "steps": 8, "profile": "turbo"},
    "screen_3x10_turbo": {"chunks": 3, "chunk_seconds": 10.0, "steps": 8, "profile": "turbo"},
    "screen_6x5_turbo": {"chunks": 6, "chunk_seconds": 5.0, "steps": 8, "profile": "turbo"},
    "screen_6x5_turbo_fixed_prompt": {
        "chunks": 6,
        "chunk_seconds": 5.0,
        "steps": 8,
        "profile": "turbo",
        "fixed_prompt": True,
    },
    "screen_6x5_turbo_fixed_prefix": {
        "chunks": 6,
        "chunk_seconds": 5.0,
        "steps": 8,
        "profile": "turbo",
        "fixed_prompt": True,
        "continuation_source_mode": "Fixed Group 1 Tail",
    },
    "screen_6x5_turbo_soft_reanchor": {
        "chunks": 6,
        "chunk_seconds": 5.0,
        "steps": 8,
        "profile": "turbo",
        "fixed_prompt": True,
        "continuation_source_mode": "Soft Re-anchor Group 4",
    },
    "extend_4x10_turbo": {"chunks": 4, "chunk_seconds": 10.0, "steps": 8, "profile": "turbo"},
    "extend_5x10_turbo": {"chunks": 5, "chunk_seconds": 10.0, "steps": 8, "profile": "turbo"},
    "extend_6x10_turbo": {"chunks": 6, "chunk_seconds": 10.0, "steps": 8, "profile": "turbo"},
    "motion_off_3x5_normal": {"chunks": 3, "chunk_seconds": 5.0, "steps": 20, "profile": "motion_off"},
    "motion_both_3x5_normal": {"chunks": 3, "chunk_seconds": 5.0, "steps": 20, "profile": "motion_both"},
    "motion_booster_3x5_normal": {"chunks": 3, "chunk_seconds": 5.0, "steps": 20, "profile": "motion_booster"},
    "motion_repair_3x5_normal": {"chunks": 3, "chunk_seconds": 5.0, "steps": 20, "profile": "motion_repair"},
}

PROMPT_ACTIONS = (
    "She holds a calm gaze while a light breeze moves a few strands of hair and a thin curl of smoke rises.",
    "Her eyes shift gently toward the camera as her cigarette hand lifts slightly and her hair moves in the breeze.",
    "She blinks and turns her head a little while a strand of hair brushes her cheek and the smoke curls upward.",
    "Her shoulders settle into a subtly different pose as the cigarette lowers slightly and her bangs move softly.",
    "She looks just past the camera and then back while the smoke disperses and the movement of her hair continues.",
    "She returns to a steady gaze with her cigarette hand near her shoulder and her hair settling naturally.",
)

DIAGNOSTIC_MARKER = "V3.6-R1 diagnostic: "


def _prompt(action: str, *, motion_trigger: bool) -> str:
    trigger = "dynv2, " if motion_trigger else ""
    return (
        "integrated_multimodal_description: [Shot 1] "
        f"{trigger}Live-action close-up of the same blonde woman at a Route 66 gas station, "
        "wearing the same white lace camisole and holding the same cigarette. "
        "The camera remains static, the daylight exposure stays stable, and her natural skin texture, facial features, clothing, and background remain consistent. "
        f"{action}\n\n"
        "overall_soundscape: A soft outdoor breeze, distant road ambience, and faint fabric and hair movement remain audible.\n\n"
        "non_diegetic_music: N/A"
    )


def _prompt_list(
    chunks: int, *, motion_trigger: bool, fixed_prompt: bool = False
) -> str:
    if fixed_prompt:
        return "\n---\n".join(
            _prompt(PROMPT_ACTIONS[0], motion_trigger=motion_trigger)
            for _ in range(int(chunks))
        )
    indices = (0, 2, 4) if int(chunks) == 3 else tuple(range(int(chunks)))
    return "\n---\n".join(
        _prompt(PROMPT_ACTIONS[index], motion_trigger=motion_trigger)
        for index in indices
    )


def _configure_loras(loader_inputs: dict, profile: str) -> list[dict]:
    for name, value in list(loader_inputs.items()):
        if name.startswith("lora_") and isinstance(value, dict):
            value["on"] = False

    selected: list[dict] = []

    def set_slot(slot: int, name: str, strength: float = 1.0) -> None:
        loader_inputs[f"lora_{slot}"] = {
            "on": True,
            "lora": name,
            "strength": float(strength),
        }
        selected.append({"name": name, "strength": float(strength)})

    if profile == "turbo":
        set_slot(1, TURBO_LORA)
    elif profile == "motion_both":
        set_slot(1, MOTION_BOOSTER_LORA)
        set_slot(2, MOTION_REPAIR_LORA)
    elif profile == "motion_booster":
        set_slot(1, MOTION_BOOSTER_LORA)
    elif profile == "motion_repair":
        set_slot(1, MOTION_REPAIR_LORA)
    elif profile != "motion_off":
        raise ValueError(f"unknown LoRA profile: {profile}")
    return selected


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
        raise ValueError(f"unknown Issue #13 R2.5 case: {case}")
    spec = CASE_SPECS[case]
    chunks = int(spec["chunks"])
    chunk_seconds = float(spec["chunk_seconds"])
    profile = str(spec["profile"])
    prompt = copy.deepcopy(source)

    sampler = prompt["305"]
    sampler["class_type"] = "H3ContinuumMaskedPrefixR1Diagnostic"
    sampler["_meta"] = {"title": f"Issue #13 R2.5 Diagnostic — {case}"}
    inputs = sampler["inputs"]
    inputs.update(
        {
            "prompt_mode": "List",
            "chunks": chunks,
            "chunk_seconds": chunk_seconds,
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
            "synchronize_sampling": True,
            "diagnostic_nonce": int(diagnostic_nonce),
            "continuation_source_mode": str(
                spec.get("continuation_source_mode", "Recursive")
            ),
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

    motion_profile = profile.startswith("motion_")
    prompt["188"]["inputs"]["value"] = _prompt_list(
        chunks,
        motion_trigger=motion_profile,
        fixed_prompt=bool(spec.get("fixed_prompt", False)),
    )
    prompt["114"]["inputs"]["image"] = str(image_name)
    prompt["139"]["inputs"]["sampler_name"] = "euler"
    prompt["153"]["inputs"].update(
        {"scheduler": "simple", "steps": int(spec["steps"]), "denoise": 1.0}
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
    selected_loras = _configure_loras(prompt["286"]["inputs"], profile)
    return prompt, selected_loras


def _relative_lora_path(name: str) -> Path:
    return Path(*str(name).replace("\\", "/").split("/"))


def _inventory_loras(lora_root: Path, selected: list[dict]) -> list[dict]:
    inventory = []
    for item in selected:
        path = lora_root / _relative_lora_path(str(item["name"]))
        if not path.is_file():
            raise FileNotFoundError(f"required LoRA is missing: {path}")
        inventory.append(
            {
                **item,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return inventory


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
        raise RuntimeError("Issue #13 R2.5 diagnostic marker is missing")
    value = json.loads(status.rsplit(DIAGNOSTIC_MARKER, 1)[1].strip())
    if not isinstance(value, dict):
        raise RuntimeError("Issue #13 R2.5 diagnostic payload is not an object")
    return value


def run(args: argparse.Namespace) -> dict:
    source = json.loads(args.prompt.read_text(encoding="utf-8"))
    cases = list(
        args.cases
        or ("screen_3x5_turbo", "screen_3x10_turbo", "screen_6x5_turbo")
    )
    image_sha256 = hashlib.sha256(args.image_source.read_bytes()).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "summary.json"
    results = []
    if summary_path.exists():
        results = list(
            json.loads(summary_path.read_text(encoding="utf-8")).get("cases") or []
        )
    completed_cases = {str(item.get("case")) for item in results}

    for offset, case in enumerate(cases):
        if case in completed_cases:
            print(f"ISSUE13-R25 SKIP completed case={case}", flush=True)
            continue
        configured, selected = configure_prompt(
            source,
            case=case,
            output_prefix=f"video/issue13_r25/Issue13_R25_{case}_seed{args.seed}",
            image_name=args.image_name,
            seed=args.seed,
            width=args.width,
            height=args.height,
            diagnostic_nonce=args.diagnostic_nonce + offset,
        )
        lora_inventory = _inventory_loras(args.lora_root, selected)
        case_root = args.output_root / case
        case_root.mkdir(parents=True, exist_ok=True)
        prompt_path = case_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(configured, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"ISSUE13-R25 START case={case}", flush=True)
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=configured,
            client_id=f"issue13-r25-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=3.0,
        )
        history_path = case_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        status = _status_text(history, prompt_id)
        spec = CASE_SPECS[case]
        result = {
            "case": case,
            "chunks": int(spec["chunks"]),
            "chunk_seconds": float(spec["chunk_seconds"]),
            "profile": str(spec["profile"]),
            "steps": int(spec["steps"]),
            "sampler": "euler",
            "scheduler": "simple",
            "prompt_id": prompt_id,
            "api_elapsed_seconds": elapsed,
            "prompt": str(prompt_path),
            "history": str(history_path),
            "output_video": _output_video(history, prompt_id),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "lora_inventory": lora_inventory,
            "diagnostic": _diagnostic_from_status(status),
        }
        (case_root / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(result)
        summary = {
            "format": "h3-continuum-issue13-r25-gate-v1",
            "server": args.server,
            "source_prompt": str(args.prompt),
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
            "continuity": "Balanced — 22 frames",
            "audio_continuity": True,
            "transport": "masked_av_prefix_22_v1",
            "spectrum": False,
            "video_seam": "Off",
            "audio_seam": "Off",
            "cases": results,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"ISSUE13-R25 COMPLETE case={case} prompt_id={prompt_id} elapsed={elapsed:.3f}s",
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
    parser.add_argument(
        "--lora-root",
        type=Path,
        default=Path(r"D:\StabilityMatrix\Data\Packages\ComfyUI_W\models\loras"),
    )
    parser.add_argument("--seed", type=int, default=131325)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--height", type=int, default=864)
    parser.add_argument("--diagnostic-nonce", type=int, default=132500)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--cases", nargs="*", choices=tuple(CASE_SPECS))
    return parser


def main() -> int:
    print(json.dumps(run(_parser().parse_args()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
