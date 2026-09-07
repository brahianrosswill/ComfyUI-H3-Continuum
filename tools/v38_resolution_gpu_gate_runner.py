"""Run the three-case V3.8 main-sampler resolution GPU mini Gate."""

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
from typing import Any, Mapping
import uuid

import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import _json_request, submit_and_wait
from v36_r1_stress_gate_runner import ResourceMonitor
from v38_easy_gpu_gate_runner import _output_media, _probe_media


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GateCase:
    name: str
    aspect: str
    preset: str
    custom_mp: float
    first_image: bool
    expected_width: int
    expected_height: int
    prompt: str


CASES = (
    GateCase(
        name="g1_draft_landscape",
        aspect="Landscape 16:9",
        preset="Draft — 0.30 MP",
        custom_mp=0.30,
        first_image=False,
        expected_width=736,
        expected_height=416,
        prompt=(
            "A continuous cinematic landscape shot tracks a woman walking across "
            "a modern plaza. Natural body motion, stable lighting, footsteps, wind, "
            "and distant city ambience continue without a cut."
        ),
    ),
    GateCase(
        name="g2_balanced_portrait",
        aspect="Portrait 9:16",
        preset="Balanced — 0.60 MP",
        custom_mp=0.30,
        first_image=False,
        expected_width=576,
        expected_height=1024,
        prompt=(
            "A continuous portrait-format cinematic shot follows a woman walking "
            "toward camera through a city street. Preserve coherent anatomy, natural "
            "motion, stable exposure, footsteps, and urban ambience."
        ),
    ),
    GateCase(
        name="g3_native_auto_first_image",
        aspect="Auto from First Image",
        preset="Native 768",
        custom_mp=0.30,
        first_image=True,
        expected_width=768,
        expected_height=1024,
        prompt=(
            "A continuous cinematic shot begins from the supplied First Image. The "
            "subject moves naturally while identity, clothing, lighting, camera "
            "direction, footsteps, and ambience remain coherent."
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_origin(
    node_id: str,
    output_slot: int,
    *,
    nodes: Mapping[str, Mapping[str, Any]],
    links: Mapping[str, list[Any]],
    seen: set[tuple[str, int]] | None = None,
) -> list[Any] | None:
    seen = set(seen or ())
    key = (str(node_id), int(output_slot))
    if key in seen:
        return None
    seen.add(key)
    node = nodes.get(str(node_id))
    if not isinstance(node, Mapping):
        return None
    mode = int(node.get("mode", 0) or 0)
    if mode == 0:
        return [str(node_id), int(output_slot)]
    if mode != 4:
        return None
    outputs = list(node.get("outputs") or [])
    output_type = outputs[output_slot].get("type") if output_slot < len(outputs) else None
    candidates = []
    for input_value in node.get("inputs") or []:
        link_id = input_value.get("link")
        input_type = input_value.get("type")
        if link_id is None:
            continue
        if output_type not in (None, "*") and input_type not in (output_type, "*"):
            continue
        candidates.append(str(link_id))
    if len(candidates) != 1:
        return None
    link = links.get(candidates[0])
    if not link:
        return None
    return _resolve_origin(
        str(link[1]),
        int(link[2]),
        nodes=nodes,
        links=links,
        seen=seen,
    )


def _build_prompt(
    workflow: Mapping[str, Any],
    object_info: Mapping[str, Any],
    *,
    case: GateCase,
    image_name: str,
    output_prefix: str,
    seed: int,
) -> dict[str, Any]:
    nodes = {str(node["id"]): copy.deepcopy(node) for node in workflow["nodes"]}
    links = {str(link[0]): list(link) for link in workflow["links"]}
    sampler_ids = [node_id for node_id, node in nodes.items() if node["type"] == "H3ContinuumSamplerV37"]
    if len(sampler_ids) != 1:
        raise GateFailure(f"expected one V3.7 source sampler, found {len(sampler_ids)}")
    sampler_id = sampler_ids[0]
    nodes[sampler_id]["type"] = "H3ContinuumSamplerV38"
    nodes[sampler_id].setdefault("properties", {})["Node name for S&R"] = (
        "H3ContinuumSamplerV38"
    )
    sampler_named = dict(nodes[sampler_id].get("widgets_values_named") or {})
    sampler_named.pop("width", None)
    sampler_named.pop("height", None)
    sampler_named.update(
        {
            "aspect": case.aspect,
            "preset": case.preset,
            "custom_mp": case.custom_mp,
            "chunks": 1,
            "chunk_seconds": 5.0,
            "base_seed": int(seed),
            "diagnostics": "Detailed Report",
            "show_preview": False,
            "run_storage": "Off",
            "run_name": "",
            "project_id": f"v38-resolution-{case.name}-{seed}",
        }
    )
    nodes[sampler_id]["widgets_values_named"] = sampler_named

    for node in nodes.values():
        if node["type"] == "PrimitiveStringMultiline":
            node.setdefault("widgets_values_named", {})["value"] = case.prompt
        elif node["type"] == "LoadImage" and int(node["id"]) == 114:
            node.setdefault("widgets_values_named", {})["image"] = image_name
        elif node["type"] == "SaveVideo":
            node.setdefault("widgets_values_named", {})["filename_prefix"] = output_prefix

    prompt: dict[str, Any] = {}

    def add_node(node_id: str) -> None:
        if node_id in prompt:
            return
        node = nodes[node_id]
        if int(node.get("mode", 0) or 0) != 0:
            return
        class_type = str(node["type"])
        info = object_info.get(class_type)
        if not isinstance(info, Mapping):
            raise GateFailure(f"object_info has no node class {class_type}")
        valid_inputs = set()
        input_schema = info.get("input") or {}
        valid_inputs.update((input_schema.get("required") or {}).keys())
        valid_inputs.update((input_schema.get("optional") or {}).keys())
        api_inputs: dict[str, Any] = {}
        for input_value in node.get("inputs") or []:
            name = str(input_value.get("name"))
            if name in {"width", "height"}:
                continue
            if node_id == sampler_id and name == "first_frame" and not case.first_image:
                continue
            link_id = input_value.get("link")
            if link_id is None or name not in valid_inputs:
                continue
            link = links.get(str(link_id))
            if not link:
                continue
            origin = _resolve_origin(
                str(link[1]),
                int(link[2]),
                nodes=nodes,
                links=links,
            )
            if origin is None:
                continue
            api_inputs[name] = origin
            add_node(str(origin[0]))
        for name, value in (node.get("widgets_values_named") or {}).items():
            if name in valid_inputs and name not in api_inputs:
                api_inputs[str(name)] = value
        prompt[node_id] = {
            "class_type": class_type,
            "inputs": api_inputs,
            "_meta": {"title": str(node.get("title") or class_type)},
        }

    output_ids = [
        node_id
        for node_id, node in nodes.items()
        if node["type"] == "SaveVideo" and int(node.get("mode", 0) or 0) == 0
    ]
    if len(output_ids) != 1:
        raise GateFailure(f"expected one active SaveVideo, found {len(output_ids)}")
    add_node(output_ids[0])
    if sampler_id not in prompt:
        raise GateFailure("V3.8 sampler is not connected to the SaveVideo output")
    return prompt


def _media_health(path: Path, ffmpeg: str) -> dict[str, Any]:
    video_result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "fps=1,scale=64:64",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame_size = 64 * 64 * 3
    frame_count = len(video_result.stdout) // frame_size
    frames = np.frombuffer(video_result.stdout[: frame_count * frame_size], dtype=np.uint8)
    frames = frames.reshape(frame_count, 64, 64, 3)
    frame_means = frames.mean(axis=(1, 2, 3))
    video_std = float(frames.astype(np.float32).std())

    audio_result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    audio = np.frombuffer(audio_result.stdout, dtype=np.float32)
    return {
        "sampled_video_frames": int(frame_count),
        "video_mean": float(frames.mean()),
        "video_std": video_std,
        "black_sample_fraction": float(np.mean(frame_means < 2.0)),
        "video_not_black_or_collapsed": bool(
            frame_count > 0 and video_std > 2.0 and np.any(frame_means >= 2.0)
        ),
        "audio_samples": int(audio.size),
        "audio_finite": bool(audio.size > 0 and np.isfinite(audio).all()),
        "audio_peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
    }


def run(args) -> dict[str, Any]:
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    object_info = _json_request(f"{args.server.rstrip('/')}/object_info")
    if "H3ContinuumSamplerV38" not in object_info:
        raise GateFailure("runtime has no H3ContinuumSamplerV38")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise GateFailure("ffmpeg and ffprobe are required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(CASES, start=1):
        case_root = args.output_root / case.name
        case_root.mkdir(parents=True, exist_ok=True)
        prompt = _build_prompt(
            workflow,
            object_info,
            case=case,
            image_name=args.image_name,
            output_prefix=f"v38_resolution_gate/{case.name}",
            seed=args.seed,
        )
        prompt_path = case_root / "prompt.json"
        prompt_path.write_text(
            json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"V3.8 Resolution Gate start {index}/{len(CASES)} {case.name}",
            flush=True,
        )
        if args.dry_run:
            results.append({"case": asdict(case), "prompt": str(prompt_path)})
            continue
        with ResourceMonitor(args.backend_pid, args.monitor_interval) as monitor:
            prompt_id, history, elapsed = submit_and_wait(
                server=args.server,
                prompt=prompt,
                client_id=f"v38-resolution-{uuid.uuid4().hex}",
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        history_path = case_root / "history.json"
        history_path.write_text(
            json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        media_path = _output_media(history, args.media_root)
        media = _probe_media(media_path, ffprobe)
        health = _media_health(media_path, ffmpeg)
        checks = {
            "geometry": (media["width"], media["height"])
            == (case.expected_width, case.expected_height),
            "frames": int(media["frames"]) == 120,
            "fps": str(media["fps"]) == "24/1",
            "duration": abs(float(media["format_duration"]) - 5.0) <= 0.02,
            "audio_rate": int(media["audio_sample_rate"]) == 32000,
            "audio_channels": int(media["audio_channels"]) == 2,
            "audio_finite": bool(health["audio_finite"]),
            "video_not_black_or_collapsed": bool(
                health["video_not_black_or_collapsed"]
            ),
        }
        if not all(checks.values()):
            raise GateFailure(f"{case.name} failed: {checks}")
        result = {
            "case": asdict(case),
            "prompt_id": prompt_id,
            "api_elapsed_seconds": float(elapsed),
            "prompt": str(prompt_path),
            "history": str(history_path),
            "media": media,
            "health": health,
            "resource": monitor.as_dict(),
            "checks": checks,
        }
        results.append(result)
        print(
            f"V3.8 Resolution Gate PASS {case.name} "
            f"{media['width']}x{media['height']} {elapsed:.3f}s",
            flush=True,
        )
    summary = {
        "format": "h3-continuum-v38-main-resolution-gpu-gate-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "workflow": str(args.workflow),
        "workflow_sha256": _sha256(args.workflow),
        "image_name": args.image_name,
        "seed": args.seed,
        "dry_run": bool(args.dry_run),
        "pass": bool(not args.dry_run and len(results) == len(CASES)),
        "results": results,
    }
    (args.output_root / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--backend-pid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=927966463230733)
    parser.add_argument("--timeout-seconds", type=float, default=1200.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
