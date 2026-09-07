"""Run the three-case V3.8 A2 Audio Only GPU Canary."""

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
import torch
from safetensors.torch import load_file


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import submit_and_wait
from v36_r1_stress_gate_runner import ResourceMonitor
from v38_easy_gpu_gate_runner import _output_media, _probe_media


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class GateCase:
    name: str
    chunks: int
    terminal_merge: bool
    expected_physical_groups: int
    boundary_seconds: tuple[float, ...]


CASES = (
    GateCase("g1_1x5_tail6", 1, False, 1, ()),
    GateCase("g2_3x5_balanced22", 3, False, 3, (5.0, 10.0)),
    GateCase("g3_terminal_merge", 3, True, 2, (5.0, 10.0)),
)


PROMPT = (
    "A continuous cinematic shot follows a woman walking through a modern plaza. "
    "Natural footsteps, clothing movement, wind, and steady city ambience remain "
    "audible throughout without cuts or silence."
)
FIRST_PASS_SEED = 3802701
REFINE_SEED = 3802702


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tensor(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _find_one(prompt: Mapping[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    matches = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == class_type
    ]
    if len(matches) != 1:
        raise GateFailure(f"expected one {class_type}, found {len(matches)}")
    return matches[0]


def _remove_class(prompt: dict[str, Any], class_type: str) -> None:
    for node_id in [
        str(key)
        for key, node in prompt.items()
        if str(node.get("class_type")) == class_type
    ]:
        prompt.pop(node_id, None)


def _configure_prompt(
    template: Mapping[str, Any],
    *,
    case: GateCase,
    output_prefix: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(template))
    sampler_id, continuum = _find_one(prompt, "H3ContinuumSamplerV38")
    inputs = continuum["inputs"]
    inputs.update(
        {
            "prompt_mode": "Fixed",
            "chunks": int(case.chunks),
            "chunk_seconds": 5.0,
            "continuity": "Balanced — 22 frames",
            "base_seed": FIRST_PASS_SEED,
            "audio_continuity": True,
            "diagnostics": "Detailed Report",
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "strict_compatibility": False,
            "debug": True,
            "show_preview": False,
            "run_storage": "Off",
            "run_name": "",
            "project_id": "",
            "generation_mode": "Full Run",
            "review_action": "Continue / Next",
            "aspect": "Square 1:1",
            "preset": "Custom",
            "custom_mp": 0.331776,
        }
    )
    for name in (
        "reference_video_1",
        "reference_video_2",
        "reference_video_3",
        "driving_audio",
    ):
        inputs.pop(name, None)
    if not case.terminal_merge:
        inputs.pop("first_frame", None)
        inputs.pop("last_frame", None)
        _remove_class(prompt, "LoadImage")
    _remove_class(prompt, "H3ContinuumLoadVideo")

    prompt_id = str(inputs["sequence_prompt"][0])
    prompt[prompt_id]["inputs"]["value"] = PROMPT
    first_scheduler_id = str(inputs["sigmas"][0])
    prompt[first_scheduler_id]["inputs"].update(
        {"scheduler": "simple", "steps": 8, "denoise": 1.0}
    )
    scheduler_matches = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == "BasicScheduler"
        and str(node_id) != first_scheduler_id
    ]
    if len(scheduler_matches) != 1:
        raise GateFailure(
            "expected one refine BasicScheduler, "
            f"found {len(scheduler_matches)}"
        )
    refine_scheduler_id, refine_scheduler = scheduler_matches[0]
    refine_scheduler["inputs"].update(
        {"scheduler": "simple", "steps": 10, "denoise": 0.35}
    )

    prompt["302"] = {
        "class_type": "H3A2AudioOnlyCanary",
        "inputs": {
            "model": inputs["model"],
            "clip": inputs["clip"],
            "sampler": inputs["sampler"],
            "sigmas": [refine_scheduler_id, 0],
            "video_latents": [sampler_id, 0],
            "audio_latents": [sampler_id, 1],
            "assembly_plan": [sampler_id, 2],
            "refine_context": [sampler_id, 5],
            "video_vae": inputs["video_vae"],
            "refine_seed": REFINE_SEED,
            "schedule_mode": "Tail",
            "tail_evaluations": 6,
        },
        "_meta": {"title": "A2 Audio Only Canary"},
    }
    for node_id in ("900", "901", "902", "903", "904", "905", "906"):
        prompt.pop(node_id, None)
    prompt["900"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": [sampler_id, 3]},
    }
    prompt["901"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["302", 3]},
    }
    prompt["902"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": [sampler_id, 0],
            "filename_prefix": f"{output_prefix}_first_video",
        },
    }
    prompt["903"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": [sampler_id, 1],
            "filename_prefix": f"{output_prefix}_first_audio",
        },
    }
    prompt["904"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": ["302", 0],
            "filename_prefix": f"{output_prefix}_refined_video",
        },
    }
    prompt["905"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": ["302", 1],
            "filename_prefix": f"{output_prefix}_refined_audio",
        },
    }
    prompt["906"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["302", 2]},
    }
    _save_id, save_video = _find_one(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = output_prefix
    return prompt


def _history_entry(history: Mapping[str, Any], prompt_id: str) -> Mapping[str, Any]:
    entry = history.get(prompt_id)
    if not isinstance(entry, Mapping):
        raise GateFailure(f"history is missing prompt {prompt_id}")
    return entry


def _output_text(entry: Mapping[str, Any], node_id: str) -> str:
    value = (entry.get("outputs") or {}).get(node_id) or {}
    items = value.get("text")
    if isinstance(items, list):
        return "\n".join(str(item) for item in items)
    if isinstance(items, str):
        return items
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _output_latents(
    entry: Mapping[str, Any],
    node_id: str,
    output_root: Path,
) -> list[dict[str, Any]]:
    output = (entry.get("outputs") or {}).get(node_id) or {}
    records = []
    for value in output.get("latents") or []:
        path = output_root / str(value.get("subfolder") or "") / str(
            value["filename"]
        )
        latent = load_file(str(path), device="cpu")["latent_tensor"]
        records.append(
            {
                "path": str(path),
                "shape": [int(item) for item in latent.shape],
                "dtype": str(latent.dtype),
                "tensor_sha256": _sha256_tensor(latent),
            }
        )
    return records


def _canary_json(status: str) -> dict[str, Any]:
    line = next(
        (item for item in status.splitlines() if item.startswith("A2_CANARY_JSON=")),
        "",
    )
    if not line:
        raise GateFailure("A2 canary status JSON is missing")
    return json.loads(line.split("=", 1)[1])


def _decode_audio(path: Path, ffmpeg: str) -> np.ndarray:
    process = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            "32000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode:
        raise GateFailure(
            "ffmpeg audio decode failed: "
            + process.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    audio = np.frombuffer(process.stdout, dtype="<f4")
    if audio.size == 0 or audio.size % 2:
        raise GateFailure("decoded stereo PCM is empty or malformed")
    return audio.reshape(-1, 2)


def _rms(value: np.ndarray) -> float:
    if value.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def _audio_health(
    media_path: Path,
    *,
    ffmpeg: str,
    expected_seconds: float,
    boundaries: tuple[float, ...],
) -> dict[str, Any]:
    audio = _decode_audio(media_path, ffmpeg)
    sample_rate = 32000
    global_rms = _rms(audio)
    channel_rms = [_rms(audio[:, channel]) for channel in range(2)]
    window = max(1, int(round(0.5 * sample_rate)))
    window_rms = [
        _rms(audio[start : start + window])
        for start in range(0, len(audio), window)
        if len(audio[start : start + window]) >= window // 2
    ]
    boundary_metrics = []
    for seconds in boundaries:
        index = int(round(seconds * sample_rate))
        if not 1 <= index < len(audio):
            raise GateFailure(f"audio boundary {seconds}s is outside decoded PCM")
        radius = int(round(0.1 * sample_rate))
        local = audio[max(0, index - radius) : min(len(audio), index + radius)]
        local_diff = np.abs(np.diff(local, axis=0))
        q999 = float(np.quantile(local_diff, 0.999)) if local_diff.size else 0.0
        sample_jump = float(np.max(np.abs(audio[index] - audio[index - 1])))
        threshold = max(0.25, q999 * 8.0)
        before_rms = _rms(audio[max(0, index - radius) : index])
        after_rms = _rms(audio[index : min(len(audio), index + radius)])
        boundary_metrics.append(
            {
                "seconds": seconds,
                "sample_index": index,
                "sample_jump": sample_jump,
                "local_diff_q999": q999,
                "click_threshold": threshold,
                "no_click_advisory": sample_jump <= threshold,
                "rms_before_100ms": before_rms,
                "rms_after_100ms": after_rms,
                "no_boundary_silence": before_rms > 1e-7 and after_rms > 1e-7,
            }
        )
    expected_samples = int(round(expected_seconds * sample_rate))
    checks = {
        "finite": bool(np.isfinite(audio).all()),
        "not_silent": global_rms > 1e-7,
        "channels_not_collapsed": all(value > 1e-7 for value in channel_rms),
        "windows_not_collapsed": bool(window_rms)
        and min(window_rms) > 1e-8,
        "duration_no_drift": abs(len(audio) - expected_samples) <= 1024,
        "boundaries_no_click": all(
            item["no_click_advisory"] for item in boundary_metrics
        ),
        "boundaries_not_silent": all(
            item["no_boundary_silence"] for item in boundary_metrics
        ),
    }
    return {
        "sample_rate": sample_rate,
        "channels": 2,
        "samples_per_channel": int(len(audio)),
        "duration_seconds": float(len(audio) / sample_rate),
        "expected_samples_per_channel": expected_samples,
        "global_rms": global_rms,
        "channel_rms": channel_rms,
        "peak": float(np.max(np.abs(audio))),
        "minimum_window_rms": min(window_rms) if window_rms else 0.0,
        "boundaries": boundary_metrics,
        "checks": checks,
        "pass": all(checks.values()),
    }


def _run_case(
    args: argparse.Namespace,
    *,
    case: GateCase,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    case_root = args.evidence_root / case.name
    case_root.mkdir(parents=True, exist_ok=True)
    output_prefix = f"v38_a2_audio_only/{args.run_tag}/{case.name}"
    prompt = _configure_prompt(template, case=case, output_prefix=output_prefix)
    prompt_path = case_root / "prompt.json"
    prompt_path.write_text(
        json.dumps(prompt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"A2 GPU Canary start: {case.name}", flush=True)
    with ResourceMonitor(args.backend_pid, interval_seconds=0.5) as monitor:
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=prompt,
            client_id=f"v38-a2-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=2.0,
        )
    history_path = case_root / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    entry = _history_entry(history, prompt_id)
    first_status = _output_text(entry, "900")
    canary_status = _output_text(entry, "901")
    canary = _canary_json(canary_status)
    first_videos = _output_latents(entry, "902", args.output_root)
    first_audios = _output_latents(entry, "903", args.output_root)
    refined_videos = _output_latents(entry, "904", args.output_root)
    refined_audios = _output_latents(entry, "905", args.output_root)
    media_path = _output_media(history, args.output_root)
    media = _probe_media(media_path, args.ffprobe)
    audio_health = _audio_health(
        media_path,
        ffmpeg=args.ffmpeg,
        expected_seconds=case.chunks * 5.0,
        boundaries=case.boundary_seconds,
    )
    first_video_hashes = [item["tensor_sha256"] for item in first_videos]
    refined_video_hashes = [item["tensor_sha256"] for item in refined_videos]
    first_audio_hashes = [item["tensor_sha256"] for item in first_audios]
    refined_audio_hashes = [item["tensor_sha256"] for item in refined_audios]
    groups = canary.get("physical_groups") or []
    terminal_check = True
    if case.terminal_merge:
        terminal_check = (
            len(groups) == 2
            and groups[-1].get("logical_chunks") == [2, 3]
            and groups[-1].get("terminal_merged") is True
        )
    checks = {
        "physical_group_count": len(groups) == case.expected_physical_groups,
        "first_video_count": len(first_videos) == case.expected_physical_groups,
        "refined_video_count": len(refined_videos) == case.expected_physical_groups,
        "first_audio_count": len(first_audios) == case.expected_physical_groups,
        "refined_audio_count": len(refined_audios) == case.expected_physical_groups,
        "video_sha_exact": first_video_hashes == refined_video_hashes,
        "video_object_identity": len(canary.get("video_object_identity") or [])
        == case.expected_physical_groups
        and all(canary.get("video_object_identity") or []),
        "video_tensor_identity": len(canary.get("video_tensor_identity") or [])
        == case.expected_physical_groups
        and all(canary.get("video_tensor_identity") or []),
        "audio_changed": len(first_audio_hashes) == len(refined_audio_hashes)
        and all(
            before != after
            for before, after in zip(
                first_audio_hashes,
                refined_audio_hashes,
                strict=True,
            )
        ),
        "audio_shape_preserved": canary.get("audio_input_shapes")
        == canary.get("audio_output_shapes"),
        "audio_seed_namespace": (
            (canary.get("target") or {}).get("seed_namespace")
            == "h3-continuum-refine-audio-v1"
        ),
        "tail_6": canary.get("schedule_mode") == "tail"
        and canary.get("schedule_evaluations") == 6,
        "second_pass_contract_v1": canary.get("second_pass_contract_version") == 1,
        "terminal_atomic_group": terminal_check,
        "media_duration": abs(float(media["format_duration"]) - case.chunks * 5.0)
        <= 0.05,
        "media_frames": int(media["frames"]) == case.chunks * 5 * 24,
        "media_audio_contract": int(media["audio_sample_rate"]) == 32000
        and int(media["audio_channels"]) == 2,
        "audio_health": bool(audio_health["pass"]),
        "temporary_video_discarded": (
            canary_status.count("temporary_video_discarded=true")
            == case.expected_physical_groups
        ),
    }
    result = {
        "case": asdict(case),
        "prompt_id": prompt_id,
        "api_elapsed_seconds": float(elapsed),
        "prompt": str(prompt_path),
        "history": str(history_path),
        "first_status": first_status,
        "canary_status": canary_status,
        "canary": canary,
        "first_video_latents": first_videos,
        "refined_video_latents": refined_videos,
        "first_audio_latents": first_audios,
        "refined_audio_latents": refined_audios,
        "media_path": str(media_path),
        "media_sha256": _sha256_file(media_path),
        "media": media,
        "audio_health": audio_health,
        "resources": monitor.as_dict(),
        "checks": checks,
        "pass": all(checks.values()),
    }
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (case_root / "canary_status.txt").write_text(
        canary_status,
        encoding="utf-8",
    )
    print(
        f"A2 GPU Canary end: {case.name}; pass={result['pass']}; "
        f"elapsed={elapsed:.3f}s; video_exact={checks['video_sha_exact']}; "
        f"audio_changed={checks['audio_changed']}",
        flush=True,
    )
    if not result["pass"]:
        failed = [name for name, value in checks.items() if not value]
        raise GateFailure(f"{case.name} failed checks: {failed}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    t2_template = json.loads(args.t2_template.read_text(encoding="utf-8"))
    terminal_template = json.loads(
        args.terminal_template.read_text(encoding="utf-8")
    )
    summary = {
        "format": "h3-continuum-v38-a2-audio-only-gpu-canary-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "first_pass_seed": FIRST_PASS_SEED,
        "refine_seed": REFINE_SEED,
        "settings": {
            "resolution": "576x576",
            "first_pass_steps": 8,
            "refine_source_steps": 10,
            "refine_denoise": 0.35,
            "refine_schedule": "Tail 6",
            "continuity": "Balanced — 22 frames",
            "audio_continuity": True,
            "run_storage": "Off",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "sage": True,
        },
        "templates": {
            "t2": str(args.t2_template),
            "terminal": str(args.terminal_template),
        },
        "results": {},
        "pass": False,
        "production_status": "HOLD",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        for case in CASES:
            template = terminal_template if case.terminal_merge else t2_template
            summary["results"][case.name] = _run_case(
                args,
                case=case,
                template=template,
            )
        summary["pass"] = True
        summary["status"] = "GPU Experimental PASS / Production HOLD"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "GPU Canary STOP / Production HOLD"
        return summary
    finally:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--t2-template", type=Path, required=True)
    parser.add_argument("--terminal-template", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(r"D:\output\video\comfy_video"),
    )
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe:
        raise SystemExit("ffmpeg and ffprobe are required")
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
