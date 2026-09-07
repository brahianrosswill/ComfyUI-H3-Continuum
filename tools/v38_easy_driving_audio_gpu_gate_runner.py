"""Run the Easy Load Audio + Driving Audio GPU acceptance gate."""

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

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import _json_request, submit_and_wait
from p32_gpu_profile_probe import configure_profile
from v36_r1_stress_gate_runner import ResourceMonitor
from v38_easy_gpu_gate_runner import _ensure_profile_route


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GateCase:
    name: str
    duration: int
    enabled: bool
    sequence: str


CASES = (
    GateCase("g0_10s_off", 10, False, "G0"),
    GateCase("g1_10s_on", 10, True, "G1"),
    GateCase("g2_20s_on", 20, True, "G2"),
    GateCase("g3_10s_on_1", 10, True, "G3 ON-1"),
    GateCase("g3_10s_off", 10, False, "G3 OFF"),
    GateCase("g3_10s_on_2", 10, True, "G3 ON-2"),
)


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
    return str(max(int(value) for value in prompt if str(value).isdigit()) + 1)


def configure_case(
    source: Mapping[str, Any],
    *,
    case: GateCase,
    audio_file: str,
    seed: int,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(source))
    easy_id = _one_node(prompt, "H3ContinuumEasyV38")
    easy_inputs = prompt[easy_id].setdefault("inputs", {})
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
    helper_id = _one_node(prompt, "H3ContinuumEasyReferences")
    prompt[helper_id].setdefault("inputs", {}).clear()
    audio_vae_id = None
    for candidate in _find_nodes(prompt, "VAELoader"):
        vae_name = str(prompt[candidate].get("inputs", {}).get("vae_name", ""))
        if "audio" in vae_name.lower():
            audio_vae_id = candidate
            break
    if audio_vae_id is None:
        raise GateFailure("source prompt has no Audio VAE loader")
    easy_inputs["audio_vae"] = [audio_vae_id, 0]
    if case.enabled:
        loader_id = _next_node_id(prompt)
        prompt[loader_id] = {
            "inputs": {"audio": audio_file},
            "class_type": "H3EasyLoadAudio",
            "_meta": {"title": "H3 Easy Load Audio"},
        }
        easy_inputs["driving_audio"] = [loader_id, 0]
        save_audio_id = str(int(loader_id) + 1)
        prompt[save_audio_id] = {
            "inputs": {
                "audio": [easy_id, 4],
                "filename_prefix": f"audio/V38_Easy_Driving/{case.name}",
            },
            "class_type": "SaveAudio",
            "_meta": {"title": "Save Audio (FLAC) (DEPRECATED)"},
        }
    else:
        easy_inputs.pop("driving_audio", None)

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
        prompt[save_id].setdefault("inputs", {})["filename_prefix"] = (
            f"video/V38_Easy_Driving/{case.name}"
        )
    _ensure_profile_route(prompt)
    return configure_profile(prompt, "sage")


def _history_media(
    history: Mapping[str, Any], output_root: Path, suffix: str
) -> Path:
    for entry in history.values():
        outputs = entry.get("outputs", {}) if isinstance(entry, Mapping) else {}
        for output in outputs.values():
            if not isinstance(output, Mapping):
                continue
            for field in ("audio", "videos", "gifs", "images"):
                for media in output.get(field, []) or []:
                    filename = str(media.get("filename", ""))
                    if Path(filename).suffix.lower() != suffix:
                        continue
                    path = output_root / str(media.get("subfolder") or "") / filename
                    if path.is_file():
                        return path
    raise GateFailure(f"history returned no existing {suffix} output")


def _ffprobe(path: Path, ffprobe: str) -> dict[str, Any]:
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
    return json.loads(result.stdout)


def _media_contract(path: Path, ffprobe: str) -> dict[str, Any]:
    payload = _ffprobe(path, ffprobe)
    streams = list(payload.get("streams") or [])
    video = next((x for x in streams if x.get("codec_type") == "video"), {})
    audio = next((x for x in streams if x.get("codec_type") == "audio"), {})
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
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


def _decode_pcm(path: Path, ffmpeg: str, duration: int | None = None) -> np.ndarray:
    command = [ffmpeg, "-v", "error", "-i", str(path)]
    if duration is not None:
        command += ["-t", str(int(duration))]
    command += ["-ar", "32000", "-ac", "2", "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).copy()


def _audio_evidence(
    *,
    input_path: Path,
    saved_path: Path,
    video_path: Path,
    duration: int,
    ffmpeg: str,
) -> dict[str, Any]:
    source = _decode_pcm(input_path, ffmpeg, duration)
    saved = _decode_pcm(saved_path, ffmpeg)
    final = _decode_pcm(video_path, ffmpeg)
    count = min(len(source), len(saved))
    source_bytes = source[:count].astype("<f4", copy=False).tobytes()
    saved_bytes = saved[:count].astype("<f4", copy=False).tobytes()
    boundary = int(10 * 32000)
    boundary_delta = None
    if duration > 10 and len(final) > boundary:
        boundary_delta = float(np.max(np.abs(final[boundary] - final[boundary - 1])))
    diffs = np.abs(np.diff(final, axis=0)) if len(final) > 1 else np.zeros((0, 2))
    return {
        "source_frames": int(len(source)),
        "saved_frames": int(len(saved)),
        "final_frames": int(len(final)),
        "source_prefix_pcm_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "saved_pcm_sha256": hashlib.sha256(saved_bytes).hexdigest(),
        "saved_pcm_bit_exact": source_bytes == saved_bytes,
        "finite": bool(np.isfinite(final).all()),
        "max_abs": float(np.max(np.abs(final))) if final.size else 0.0,
        "max_adjacent_delta": float(np.max(diffs)) if diffs.size else 0.0,
        "p999_adjacent_delta": float(np.quantile(diffs, 0.999)) if diffs.size else 0.0,
        "ten_second_boundary_delta": boundary_delta,
    }


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
                candidates.append((int(process.info["memory_info"].rss), process.pid))
        except Exception:
            continue
    return max(candidates)[1] if candidates else None


def run(args) -> dict[str, Any]:
    source = json.loads(args.source_prompt.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "source_prompt.json").write_text(
        json.dumps(source, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not args.dry_run and (ffmpeg is None or ffprobe is None):
        raise GateFailure("ffmpeg and ffprobe are required")
    backend_pid = _find_backend_pid()
    results = []
    for case in CASES:
        case_root = args.output_root / case.name
        case_root.mkdir(parents=True, exist_ok=True)
        prompt = configure_case(
            source,
            case=case,
            audio_file=args.audio_file,
            seed=args.seed,
        )
        prompt_path = case_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        record: dict[str, Any] = {
            "case": asdict(case),
            "prompt": str(prompt_path),
            "seed": int(args.seed),
        }
        if args.dry_run:
            record["status"] = "DRY_RUN"
            results.append(record)
            continue
        queue = _json_request(f"{args.server.rstrip('/')}/queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise GateFailure("ComfyUI queue is not empty")
        print(f"START {case.name}", flush=True)
        started = time.perf_counter()
        with ResourceMonitor(backend_pid, args.monitor_interval) as monitor:
            prompt_id, history, elapsed = submit_and_wait(
                server=args.server,
                prompt=prompt,
                client_id=f"v38-easy-driving-{uuid.uuid4().hex}",
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        history_path = case_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        video_path = _history_media(history, args.media_root, ".mp4")
        media = _media_contract(video_path, ffprobe)
        expected_frames = int(case.duration * 24)
        checks = {
            "frames": media["frames"] == expected_frames,
            "fps": media["fps"] == "24/1",
            "duration": abs(media["format_duration"] - case.duration) <= 0.02,
            "audio_rate": media["audio_sample_rate"] == 32000,
            "audio_channels": media["audio_channels"] == 2,
        }
        if not all(checks.values()):
            raise GateFailure(f"media contract failed for {case.name}: {checks}")
        audio = None
        if case.enabled:
            saved_path = _history_media(history, args.media_root, ".flac")
            audio = _audio_evidence(
                input_path=args.input_root / args.audio_file,
                saved_path=saved_path,
                video_path=video_path,
                duration=case.duration,
                ffmpeg=ffmpeg,
            )
            if not audio["finite"]:
                raise GateFailure(f"non-finite final audio in {case.name}")
        record.update(
            {
                "prompt_id": prompt_id,
                "api_elapsed_seconds": float(elapsed),
                "wall_elapsed_seconds": float(time.perf_counter() - started),
                "history": str(history_path),
                "resources": monitor.as_dict(),
                "media": media,
                "audio": audio,
                "checks": checks,
                "status": "PASS",
            }
        )
        (case_root / "summary.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        results.append(record)
        print(f"PASS {case.name} elapsed={elapsed:.3f}s", flush=True)
    summary = {
        "format": "h3-continuum-v38-easy-driving-audio-gpu-gate-v1",
        "server": args.server,
        "source_prompt": str(args.source_prompt),
        "audio_file": args.audio_file,
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
    parser.add_argument("--source-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, default=Path(r"D:\output\video\comfy_video"))
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\StabilityMatrix\Data\Packages\ComfyUI_W\input"))
    parser.add_argument("--audio-file", required=True)
    parser.add_argument("--seed", type=int, default=927966463230733)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=0.25)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(json.dumps(run(args), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
