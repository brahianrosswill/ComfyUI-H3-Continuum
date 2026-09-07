"""Run the V3.8 Easy Mode Phase 3 GPU acceptance matrix.

The runner starts from one successful ComfyUI history prompt, preserves its
installed model/LoRA/VAE stack, and changes only the Easy inputs, image
presence, attention route, output prefix, and fixed seed needed by the gate.
It writes every submitted API prompt, returned history, resource sample, and
media probe under one evidence directory.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping
import uuid


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import _json_request, submit_and_wait
from p32_gpu_profile_probe import configure_profile
from v36_r1_stress_gate_runner import ResourceMonitor


PROFILES = ("sage", "sage_sol", "sage_spectrum", "sage_sol_spectrum")
REFERENCE_PROFILE_CASE = "i2va_reference1_10s"


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GateCase:
    name: str
    duration: int
    chunks: int
    chunk_seconds: float
    first: bool
    last: bool
    reference_slot: int | None
    prompt: str
    expected_mode: str
    expected_width: int
    expected_height: int


CASES = {
    REFERENCE_PROFILE_CASE: GateCase(
        name=REFERENCE_PROFILE_CASE,
        duration=10,
        chunks=1,
        chunk_seconds=10.0,
        first=True,
        last=False,
        reference_slot=1,
        prompt=(
            "A continuous cinematic tracking shot begins from the supplied First "
            "Image. Use <Picture 2> as the identity, face, clothing, and color "
            "reference. The woman walks naturally toward camera through the city, "
            "with coherent body motion, stable facial identity, realistic wind, "
            "footsteps, and urban ambience."
        ),
        expected_mode="Hybrid I2VA + Reference + Continuation",
        expected_width=480,
        expected_height=640,
    ),
    "fl2va_reference1_15s": GateCase(
        name="fl2va_reference1_15s",
        duration=15,
        chunks=2,
        chunk_seconds=7.5,
        first=True,
        last=True,
        reference_slot=1,
        prompt=(
            "A continuous cinematic shot starts from the supplied First Image and "
            "converges naturally to the supplied Last Image. Use <Picture 3> as the "
            "identity, face, clothing, and color reference throughout. The woman "
            "walks forward, turns smoothly, and reaches the final pose without a "
            "cut. Preserve stable anatomy, natural motion, footsteps, and ambience."
        ),
        expected_mode="Hybrid FL2VA + Reference + Continuation",
        expected_width=480,
        expected_height=640,
    ),
    "i2va_reference2_only_20s": GateCase(
        name="i2va_reference2_only_20s",
        duration=20,
        chunks=2,
        chunk_seconds=10.0,
        first=True,
        last=False,
        reference_slot=2,
        prompt=(
            "A single continuous cinematic shot begins from the supplied First "
            "Image. Use <Picture 2> as the identity, face, clothing, and color "
            "reference; the second helper socket is the only connected reference "
            "and must compact to the first active reference. The woman walks, looks "
            "toward camera, and continues forward with stable identity and ambience."
        ),
        expected_mode="Hybrid I2VA + Reference + Continuation",
        expected_width=480,
        expected_height=640,
    ),
    "t2va_no_reference_30s": GateCase(
        name="t2va_no_reference_30s",
        duration=30,
        chunks=3,
        chunk_seconds=10.0,
        first=False,
        last=False,
        reference_slot=None,
        prompt=(
            "A single continuous cinematic wide shot follows a woman walking through "
            "a modern city plaza. She changes direction smoothly while the camera "
            "tracks beside her, preserving natural anatomy, coherent motion, stable "
            "lighting, footsteps, wind, and distant urban ambience."
        ),
        expected_mode="T2VA + Continuation",
        expected_width=736,
        expected_height=416,
    ),
}


SPECTRUM_DEFAULTS = {
    "enabled": False,
    "blend_weight": 0.5,
    "degree": 1,
    "ridge_lambda": 0.1,
    "window_size": 2.0,
    "flex_window": 0.75,
    "warmup_steps": 1,
    "tail_actual_steps": 1,
    "max_history": 8,
    "debug": False,
    "history_storage": "system_ram",
    "bootstrap_first_forecast": True,
    "anchor_residual_feedback": False,
    "selective_rollback_correction": False,
    "offline_smoothing_replay": True,
    "audio_blend_weight": 0.0,
    "offline_archive_storage": "system_ram",
    "model_aware_mode": "off",
    "model_aware_risk_threshold": 0.65,
    "model_aware_trust_shrinkage": False,
    "model_aware_replay_generic_correction": False,
    "generic_correction_mode": "coordinate_rls",
    "generic_correction_limiter": "hard_clip",
    "generic_correction_limit": 0.4,
    "generic_correction_attenuation": "no_attenuation",
}


def _find_nodes(prompt: Mapping[str, Any], class_type: str) -> list[str]:
    return [
        str(node_id)
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == class_type
    ]


def _one_node(prompt: Mapping[str, Any], class_type: str) -> str:
    values = _find_nodes(prompt, class_type)
    if len(values) != 1:
        raise GateFailure(f"expected one {class_type}, found {len(values)}")
    return values[0]


def _next_node_id(prompt: Mapping[str, Any]) -> str:
    numeric = [int(value) for value in prompt if str(value).isdigit()]
    return str((max(numeric) if numeric else 0) + 1)


def _history_prompt(server: str, prompt_id: str) -> tuple[dict[str, Any], Any]:
    history = _json_request(f"{server.rstrip('/')}/history/{prompt_id}")
    entry = history.get(str(prompt_id))
    if not isinstance(entry, Mapping):
        raise GateFailure(f"history does not contain prompt {prompt_id}")
    raw = entry.get("prompt")
    if not isinstance(raw, list) or len(raw) < 3 or not isinstance(raw[2], Mapping):
        raise GateFailure("history entry has no API prompt")
    return copy.deepcopy(dict(raw[2])), history


def _ensure_load_image(
    prompt: dict[str, Any], *, image: str, title: str
) -> list[Any]:
    for node_id in _find_nodes(prompt, "LoadImage"):
        if prompt[node_id].get("inputs", {}).get("image") == image:
            return [node_id, 0]
    node_id = _next_node_id(prompt)
    prompt[node_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": str(image)},
        "_meta": {"title": str(title)},
    }
    return [node_id, 0]


def _ensure_profile_route(prompt: dict[str, Any]) -> None:
    easy_id = _one_node(prompt, "H3ContinuumEasyV38")
    easy_inputs = prompt[easy_id].setdefault("inputs", {})
    upstream = easy_inputs.get("model")
    if not isinstance(upstream, list) or len(upstream) != 2:
        raise GateFailure("Easy node has no MODEL link")
    spectrum_nodes = _find_nodes(prompt, "SpectrumApplyMiniMaxH3")
    if len(spectrum_nodes) > 1:
        raise GateFailure("multiple Spectrum nodes are not supported by this gate")
    if spectrum_nodes:
        spectrum_id = spectrum_nodes[0]
        spectrum_inputs = prompt[spectrum_id].setdefault("inputs", {})
        if easy_inputs.get("model") != [spectrum_id, 0]:
            spectrum_inputs["model"] = list(upstream)
            easy_inputs["model"] = [spectrum_id, 0]
        return
    spectrum_id = _next_node_id(prompt)
    prompt[spectrum_id] = {
        "class_type": "SpectrumApplyMiniMaxH3",
        "inputs": {**SPECTRUM_DEFAULTS, "model": list(upstream)},
        "_meta": {"title": "Spectrum"},
    }
    easy_inputs["model"] = [spectrum_id, 0]


def configure_case(
    source: Mapping[str, Any],
    *,
    case: GateCase,
    profile: str,
    seed: int,
    output_prefix: str,
    reference_image_1: str,
    reference_image_2: str,
    last_image: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(source))
    easy_id = _one_node(prompt, "H3ContinuumEasyV38")
    helper_id = _one_node(prompt, "H3ContinuumEasyReferences")
    easy_inputs = prompt[easy_id].setdefault("inputs", {})
    first_link = easy_inputs.get("first_frame")
    if case.first and not isinstance(first_link, list):
        raise GateFailure("source prompt has no First Image link")

    if case.first:
        easy_inputs["first_frame"] = list(first_link)
        easy_inputs["aspect"] = "Auto from First Image"
    else:
        easy_inputs.pop("first_frame", None)
        easy_inputs["aspect"] = "Landscape 16:9"
    if case.last:
        easy_inputs["last_frame"] = _ensure_load_image(
            prompt, image=last_image, title="Last Image"
        )
    else:
        easy_inputs.pop("last_frame", None)

    helper_inputs = prompt[helper_id].setdefault("inputs", {})
    helper_inputs.clear()
    if case.reference_slot is not None:
        reference_file = (
            reference_image_1 if case.reference_slot == 1 else reference_image_2
        )
        helper_inputs[f"reference_image_{case.reference_slot}"] = _ensure_load_image(
            prompt,
            image=reference_file,
            title=f"Reference Image {case.reference_slot}",
        )

    easy_inputs.update(
        {
            "duration": int(case.duration),
            "seed": int(seed),
            "seed_mode": "Fixed",
            "preset": "Draft — 0.30 MP",
            "custom_mp": 0.30,
            "run_storage": "Off",
        }
    )
    prompt_link = easy_inputs.get("prompt_text")
    if not isinstance(prompt_link, list) or len(prompt_link) != 2:
        raise GateFailure("Easy node has no Prompt link")
    prompt_node = str(prompt_link[0])
    prompt[prompt_node].setdefault("inputs", {})["value"] = case.prompt

    scheduler_id = _one_node(prompt, "BasicScheduler")
    prompt[scheduler_id].setdefault("inputs", {}).update(
        {"scheduler": "simple", "steps": 8, "denoise": 1.0}
    )
    assembly_id = _one_node(prompt, "H3ContinuumAssembleSeamV35")
    prompt[assembly_id].setdefault("inputs", {}).update(
        {
            "exact_total_duration": True,
            "audio_seam": "Auto",
            "video_seam": "Auto",
            "buffer_backend": "Auto",
            "diagnostics": "Detailed Report",
        }
    )
    for save_id in _find_nodes(prompt, "SaveVideo"):
        prompt[save_id].setdefault("inputs", {})["filename_prefix"] = str(
            output_prefix
        )

    _ensure_profile_route(prompt)
    return configure_profile(prompt, profile)


def _walk_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_strings(item)


def _output_media(history: Mapping[str, Any], output_root: Path) -> Path:
    for value in history.values():
        if not isinstance(value, Mapping):
            continue
        outputs = value.get("outputs")
        if not isinstance(outputs, Mapping):
            continue
        for output in outputs.values():
            if not isinstance(output, Mapping):
                continue
            for field in ("images", "videos", "gifs"):
                for media in output.get(field, []) or []:
                    if not isinstance(media, Mapping):
                        continue
                    filename = media.get("filename")
                    if not filename or Path(str(filename)).suffix.lower() != ".mp4":
                        continue
                    path = output_root / str(media.get("subfolder") or "") / str(
                        filename
                    )
                    if path.is_file():
                        return path
    raise GateFailure("history returned no existing MP4 output")


def _probe_media(path: Path, ffprobe: str) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-show_entries",
            "stream=codec_type,width,height,r_frame_rate,nb_read_frames,sample_rate,channels,duration:format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    streams = list(payload.get("streams") or [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    if not isinstance(video, Mapping) or not isinstance(audio, Mapping):
        raise GateFailure("output lacks video or audio stream")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(path.stat().st_size),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "frames": int(video.get("nb_read_frames", 0)),
        "fps": str(video.get("r_frame_rate", "")),
        "video_duration": float(video.get("duration", 0.0)),
        "audio_duration": float(audio.get("duration", 0.0)),
        "format_duration": float(payload.get("format", {}).get("duration", 0.0)),
        "audio_sample_rate": int(audio.get("sample_rate", 0)),
        "audio_channels": int(audio.get("channels", 0)),
    }


def _validate_result(
    history: Mapping[str, Any], *, case: GateCase, media: Mapping[str, Any]
) -> dict[str, Any]:
    text = "\n".join(_walk_strings(history))
    required = [
        case.expected_mode,
        f"/ {case.duration} sec",
        f"{case.chunks} chunks",
        f"{case.chunk_seconds:.3f}s per chunk",
        f"{case.expected_width} x {case.expected_height}",
    ]
    if case.reference_slot is not None:
        required.append("Reference conditioning: 1 image(s), size=Match Output")
    missing = [value for value in required if value not in text]
    if missing:
        raise GateFailure(f"status evidence is missing: {missing}")
    if "H3C-P103 Warning" in text or "H3C-P102 Warning" in text:
        raise GateFailure("Picture numbering produced a reference prompt warning")
    expected_frames = int(case.duration * 24)
    media_checks = {
        "geometry": (media["width"], media["height"])
        == (case.expected_width, case.expected_height),
        "frames": int(media["frames"]) == expected_frames,
        "fps": str(media["fps"]) == "24/1",
        "duration": abs(float(media["format_duration"]) - case.duration) <= 0.02,
        "audio_rate": int(media["audio_sample_rate"]) == 32000,
        "audio_channels": int(media["audio_channels"]) == 2,
    }
    if not all(media_checks.values()):
        raise GateFailure(f"media contract failed: {media_checks}")
    return {"required_status": required, "media_checks": media_checks}


def _find_backend_pid() -> int | None:
    try:
        import psutil
    except Exception:
        return None
    candidates = []
    for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "ComfyUI_W" in command and "main.py" in command:
                rss = int(process.info["memory_info"].rss)
                candidates.append((rss, int(process.info["pid"])))
        except Exception:
            continue
    return max(candidates)[1] if candidates else None


def run(args) -> dict[str, Any]:
    source, source_history = _history_prompt(args.server, args.history_prompt_id)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "source_history.json").write_text(
        json.dumps(source_history, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_root / "source_prompt.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ffprobe = shutil.which("ffprobe")
    if not args.dry_run and ffprobe is None:
        raise GateFailure("ffprobe is required for GPU acceptance")
    backend_pid = _find_backend_pid()
    matrix = [
        (CASES[REFERENCE_PROFILE_CASE], profile) for profile in PROFILES
    ] + [
        (CASES["fl2va_reference1_15s"], "sage"),
        (CASES["i2va_reference2_only_20s"], "sage"),
        (CASES["t2va_no_reference_30s"], "sage"),
    ]
    results = []
    for case, profile in matrix:
        label = f"{case.name}__{profile}"
        run_root = args.output_root / label
        run_root.mkdir(parents=True, exist_ok=True)
        prompt = configure_case(
            source,
            case=case,
            profile=profile,
            seed=args.seed,
            output_prefix=f"video/V38_Easy_Phase3/{label}",
            reference_image_1=args.reference_image_1,
            reference_image_2=args.reference_image_2,
            last_image=args.last_image,
        )
        prompt_path = run_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record: dict[str, Any] = {
            "label": label,
            "case": asdict(case),
            "profile": profile,
            "seed": int(args.seed),
            "prompt": str(prompt_path),
        }
        if args.dry_run:
            record["dry_run"] = True
            results.append(record)
            continue
        queue = _json_request(f"{args.server.rstrip('/')}/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise GateFailure("ComfyUI queue is not empty")
        print(f"START {label}", flush=True)
        started = time.perf_counter()
        with ResourceMonitor(backend_pid, args.monitor_interval) as monitor:
            prompt_id, history, elapsed = submit_and_wait(
                server=args.server,
                prompt=prompt,
                client_id=f"v38-easy-phase3-{uuid.uuid4().hex}",
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        history_path = run_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        media = _probe_media(_output_media(history, args.media_root), ffprobe)
        validation = _validate_result(history, case=case, media=media)
        record.update(
            {
                "prompt_id": prompt_id,
                "api_elapsed_seconds": float(elapsed),
                "wall_elapsed_seconds": float(time.perf_counter() - started),
                "history": str(history_path),
                "resources": monitor.as_dict(),
                "media": media,
                "validation": validation,
                "status": "PASS",
            }
        )
        (run_root / "summary.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(record)
        print(f"PASS {label} elapsed={elapsed:.3f}s", flush=True)
    summary = {
        "format": "h3-continuum-v38-easy-phase3-gpu-gate-v1",
        "source_history_prompt_id": args.history_prompt_id,
        "server": args.server,
        "media_root": str(args.media_root),
        "seed": int(args.seed),
        "backend_pid": backend_pid,
        "dry_run": bool(args.dry_run),
        "results": results,
        "status": "DRY_RUN" if args.dry_run else "PASS",
    }
    (args.output_root / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.6:8188")
    parser.add_argument("--history-prompt-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--media-root",
        type=Path,
        default=Path(r"D:\output\video\comfy_video"),
    )
    parser.add_argument("--seed", type=int, default=927966463230733)
    parser.add_argument("--reference-image-1", required=True)
    parser.add_argument("--reference-image-2", required=True)
    parser.add_argument("--last-image", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run(args)
    print(json.dumps({"status": summary["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
