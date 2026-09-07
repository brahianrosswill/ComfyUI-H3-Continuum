"""Run the A3c Selective Refine public-node GPU Matrix."""

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


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import submit_and_wait
from v36_r1_stress_gate_runner import ResourceMonitor
from v38_a2_audio_only_gpu_canary import (
    _audio_health,
    _history_entry,
    _output_latents,
    _output_text,
)
from v38_easy_gpu_gate_runner import _output_media, _probe_media


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class MatrixCase:
    name: str
    target: str
    chunks: int
    terminal_merge: bool
    expected_physical_groups: int
    boundary_seconds: tuple[float, ...]


G2_TO_G5 = (
    MatrixCase("g2_audio_only_1x5", "Audio Only", 1, False, 1, ()),
    MatrixCase("g3_video_audio_1x5", "Video + Audio", 1, False, 1, ()),
    MatrixCase(
        "g4_video_audio_3x5",
        "Video + Audio",
        3,
        False,
        3,
        (5.0, 10.0),
    ),
    MatrixCase(
        "g5_video_audio_terminal",
        "Video + Audio",
        3,
        True,
        2,
        (5.0, 10.0),
    ),
)


PROMPT = (
    "A continuous cinematic shot follows a woman walking through a modern plaza. "
    "Natural footsteps, clothing movement, wind, and steady city ambience remain "
    "audible throughout without cuts or silence."
)
FIRST_PASS_SEED = 3802801
REFINE_SEED = 3802802
SELECTIVE_CLASS = "H3ContinuumSelectiveSecondPassExperimental"
LEGACY_CLASS = "H3ContinuumSecondPassV35"


def _find_one(prompt: Mapping[str, Any], class_type: str) -> tuple[str, dict[str, Any]]:
    values = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == class_type
    ]
    if len(values) != 1:
        raise GateFailure(f"expected one {class_type}, found {len(values)}")
    return values[0]


def _remove_class(prompt: dict[str, Any], class_type: str) -> None:
    for node_id in [
        str(value)
        for value, node in prompt.items()
        if str(node.get("class_type")) == class_type
    ]:
        prompt.pop(node_id, None)


def _second_pass_inputs(prompt: Mapping[str, Any], *, target: str | None) -> dict[str, Any]:
    sampler_id, continuum = _find_one(prompt, "H3ContinuumSamplerV38")
    inputs = continuum["inputs"]
    scheduler_matches = [
        (str(node_id), node)
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == "BasicScheduler"
        and str(node_id) != str(inputs["sigmas"][0])
    ]
    if len(scheduler_matches) != 1:
        raise GateFailure(
            f"expected one refine BasicScheduler, found {len(scheduler_matches)}"
        )
    refine_scheduler_id, _scheduler = scheduler_matches[0]
    result = {
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
    }
    if target is not None:
        result["refine_target"] = target
    return result


def _configure_base(
    template: Mapping[str, Any],
    *,
    chunks: int,
    terminal_merge: bool,
    output_prefix: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(template))
    sampler_id, continuum = _find_one(prompt, "H3ContinuumSamplerV38")
    inputs = continuum["inputs"]
    inputs.update(
        {
            "prompt_mode": "Fixed",
            "chunks": int(chunks),
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
    if not terminal_merge:
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
    refine_schedulers = [
        node
        for node_id, node in prompt.items()
        if str(node.get("class_type")) == "BasicScheduler"
        and str(node_id) != first_scheduler_id
    ]
    if len(refine_schedulers) != 1:
        raise GateFailure(
            f"expected one refine BasicScheduler, found {len(refine_schedulers)}"
        )
    refine_schedulers[0]["inputs"].update(
        {"scheduler": "simple", "steps": 10, "denoise": 0.35}
    )

    for node_id in ("907", "908", "909", "910", "911"):
        prompt.pop(node_id, None)
    _save_id, save_video = _find_one(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = output_prefix
    return prompt


def _redirect_refined_media(prompt: dict[str, Any], node_id: str) -> None:
    _video_id, video_decode = _find_one(prompt, "VAEDecode")
    _audio_id, audio_decode = _find_one(prompt, "VAEDecodeAudio")
    _assemble_id, assemble = _find_one(prompt, "H3ContinuumAssembleSeamV35")
    video_decode["inputs"]["samples"] = [node_id, 0]
    audio_decode["inputs"]["samples"] = [node_id, 1]
    assemble["inputs"]["assembly_plan"] = [node_id, 2]


def _configure_g1(template: Mapping[str, Any], *, output_prefix: str) -> dict[str, Any]:
    prompt = _configure_base(
        template,
        chunks=1,
        terminal_merge=False,
        output_prefix=output_prefix,
    )
    sampler_id, _continuum = _find_one(prompt, "H3ContinuumSamplerV38")
    prompt["302"] = {
        "class_type": LEGACY_CLASS,
        "inputs": _second_pass_inputs(prompt, target=None),
        "_meta": {"title": "Legacy V3.5 Video Only"},
    }
    prompt["303"] = {
        "class_type": SELECTIVE_CLASS,
        "inputs": _second_pass_inputs(prompt, target="Video Only"),
        "_meta": {"title": "Selective Video Only"},
    }
    _redirect_refined_media(prompt, "303")
    prompt["904"]["inputs"].update(
        {"samples": ["303", 0], "filename_prefix": f"{output_prefix}_selective_video"}
    )
    prompt["905"]["inputs"].update(
        {"samples": ["303", 1], "filename_prefix": f"{output_prefix}_selective_audio"}
    )
    prompt["909"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": ["302", 0],
            "filename_prefix": f"{output_prefix}_legacy_video",
        },
    }
    prompt["910"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": ["302", 1],
            "filename_prefix": f"{output_prefix}_legacy_audio",
        },
    }
    prompt["907"] = {
        "class_type": "H3A3cVideoOnlyParityProbe",
        "inputs": {
            "input_audio_latents": [sampler_id, 1],
            "legacy_video_latents": ["302", 0],
            "legacy_audio_latents": ["302", 1],
            "legacy_assembly_plan": ["302", 2],
            "legacy_status": ["302", 3],
            "selective_video_latents": ["303", 0],
            "selective_audio_latents": ["303", 1],
            "selective_assembly_plan": ["303", 2],
            "selective_status": ["303", 3],
        },
    }
    prompt["908"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["907", 0]},
    }
    prompt["911"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["303", 3]},
    }
    return prompt


def _configure_selective(
    template: Mapping[str, Any],
    *,
    case: MatrixCase,
    output_prefix: str,
) -> dict[str, Any]:
    prompt = _configure_base(
        template,
        chunks=case.chunks,
        terminal_merge=case.terminal_merge,
        output_prefix=output_prefix,
    )
    sampler_id, _continuum = _find_one(prompt, "H3ContinuumSamplerV38")
    prompt["302"] = {
        "class_type": SELECTIVE_CLASS,
        "inputs": _second_pass_inputs(prompt, target=case.target),
        "_meta": {"title": f"Selective {case.target}"},
    }
    _redirect_refined_media(prompt, "302")
    prompt["904"]["inputs"].update(
        {"samples": ["302", 0], "filename_prefix": f"{output_prefix}_refined_video"}
    )
    prompt["905"]["inputs"].update(
        {"samples": ["302", 1], "filename_prefix": f"{output_prefix}_refined_audio"}
    )
    prompt["907"] = {
        "class_type": "H3A3cSelectiveRefineProbe",
        "inputs": {
            "input_video_latents": [sampler_id, 0],
            "input_audio_latents": [sampler_id, 1],
            "output_video_latents": ["302", 0],
            "output_audio_latents": ["302", 1],
            "input_assembly_plan": [sampler_id, 2],
            "output_assembly_plan": ["302", 2],
            "refine_status": ["302", 3],
            "target_label": case.target,
        },
    }
    prompt["908"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["907", 0]},
    }
    return prompt


def _marker_json(text: str, marker: str) -> dict[str, Any]:
    line = next((value for value in text.splitlines() if value.startswith(marker)), "")
    if not line:
        raise GateFailure(f"status marker is missing: {marker}")
    return json.loads(line.split("=", 1)[1])


def _decoded_stream_hash(media_path: Path, *, ffmpeg: str, stream: str) -> str:
    if stream == "video":
        arguments = ["-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "rawvideo"]
    elif stream == "audio":
        arguments = [
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
        ]
    else:
        raise ValueError(f"unsupported stream {stream!r}")
    process = subprocess.Popen(
        [ffmpeg, "-v", "error", "-i", str(media_path), *arguments, "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise GateFailure("ffmpeg pipe creation failed")
    digest = hashlib.sha256()
    while True:
        block = process.stdout.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode:
        raise GateFailure(
            f"ffmpeg {stream} decode failed: "
            + stderr.decode("utf-8", errors="replace")[-2000:]
        )
    return digest.hexdigest()


def _execute(
    args: argparse.Namespace,
    *,
    name: str,
    prompt: dict[str, Any],
    expected_seconds: float,
    boundaries: tuple[float, ...],
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    case_root = args.evidence_root / name
    case_root.mkdir(parents=True, exist_ok=True)
    prompt_path = case_root / "prompt.json"
    prompt_path.write_text(
        json.dumps(prompt, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"A3c start: {name}", flush=True)
    with ResourceMonitor(args.backend_pid, interval_seconds=0.5) as monitor:
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=prompt,
            client_id=f"v38-a3c-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=2.0,
        )
    history_path = case_root / "history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    entry = _history_entry(history, prompt_id)
    media_path = _output_media(history, args.output_root)
    media = _probe_media(media_path, args.ffprobe)
    audio_health = _audio_health(
        media_path,
        ffmpeg=args.ffmpeg,
        expected_seconds=expected_seconds,
        boundaries=boundaries,
    )
    decoded = {
        "video_sha256": _decoded_stream_hash(
            media_path,
            ffmpeg=args.ffmpeg,
            stream="video",
        ),
        "audio_pcm_sha256": _decoded_stream_hash(
            media_path,
            ffmpeg=args.ffmpeg,
            stream="audio",
        ),
    }
    result = {
        "name": name,
        "prompt_id": prompt_id,
        "api_elapsed_seconds": float(elapsed),
        "prompt": str(prompt_path),
        "history": str(history_path),
        "media_path": str(media_path),
        "media": media,
        "decoded": decoded,
        "audio_health": audio_health,
        "resources": monitor.as_dict(),
    }
    return result, entry


def _run_g1(
    args: argparse.Namespace,
    *,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    output_prefix = f"v38_a3c/{args.run_tag}/g1_video_only_parity"
    prompt = _configure_g1(template, output_prefix=output_prefix)
    result, entry = _execute(
        args,
        name="g1_video_only_parity",
        prompt=prompt,
        expected_seconds=5.0,
        boundaries=(),
    )
    parity_status = _output_text(entry, "908")
    parity = _marker_json(parity_status, "A3C_PARITY_JSON=")
    first_videos = _output_latents(entry, "902", args.output_root)
    first_audios = _output_latents(entry, "903", args.output_root)
    selective_videos = _output_latents(entry, "904", args.output_root)
    selective_audios = _output_latents(entry, "905", args.output_root)
    legacy_videos = _output_latents(entry, "909", args.output_root)
    legacy_audios = _output_latents(entry, "910", args.output_root)
    group_count = int(parity.get("physical_group_count") or 0)
    checks = {
        "one_physical_group": group_count == 1,
        "first_output_counts": len(first_videos) == len(first_audios) == 1,
        "legacy_output_counts": len(legacy_videos) == len(legacy_audios) == 1,
        "selective_output_counts": len(selective_videos)
        == len(selective_audios)
        == 1,
        "video_latent_sha_parity": parity.get("legacy_video_sha256")
        == parity.get("selective_video_sha256"),
        "audio_latent_sha_parity": parity.get("legacy_audio_sha256")
        == parity.get("selective_audio_sha256"),
        "legacy_audio_passthrough": all(
            parity.get("legacy_audio_object_identity") or []
        )
        and all(parity.get("legacy_audio_tensor_identity") or []),
        "selective_audio_passthrough": all(
            parity.get("selective_audio_object_identity") or []
        )
        and all(parity.get("selective_audio_tensor_identity") or []),
        "audio_matches_first_pass": parity.get("legacy_audio_sha256")
        == [item["tensor_sha256"] for item in first_audios]
        == parity.get("selective_audio_sha256"),
        "refine_seed_parity": parity.get("legacy_refine_group_seeds")
        == parity.get("selective_refine_group_seeds"),
        "sigmas_parity": parity.get("legacy_schedule_sigma_hash")
        == parity.get("selective_schedule_sigma_hash"),
        "sampling_count_parity": parity.get("legacy_sampling_passes_reported")
        == parity.get("selective_sampling_passes_reported")
        == group_count,
        "assembly_plan_parity": parity.get("legacy_plan_without_target_identity")
        == parity.get("selective_plan_without_target_identity"),
        "target_contract": parity.get("selective_target_mode") == "video_only"
        and parity.get("selective_seed_namespace")
        == "h3-continuum-refine-v1",
        "shape_parity": parity.get("legacy_video_shapes")
        == parity.get("selective_video_shapes")
        and parity.get("legacy_audio_shapes")
        == parity.get("selective_audio_shapes"),
        "finite": all(parity.get("legacy_video_finite") or [])
        and all(parity.get("selective_video_finite") or [])
        and all(parity.get("legacy_audio_finite") or [])
        and all(parity.get("selective_audio_finite") or []),
        "media": int(result["media"]["frames"]) == 120
        and abs(float(result["media"]["format_duration"]) - 5.0) <= 0.05
        and int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
    }
    result.update(
        {
            "parity_status": parity_status,
            "parity": parity,
            "first_video_latents": first_videos,
            "first_audio_latents": first_audios,
            "legacy_video_latents": legacy_videos,
            "legacy_audio_latents": legacy_audios,
            "selective_video_latents": selective_videos,
            "selective_audio_latents": selective_audios,
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    case_root = args.evidence_root / "g1_video_only_parity"
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"A3c end: g1_video_only_parity; pass={result['pass']}", flush=True)
    if not result["pass"]:
        failed = [name for name, value in checks.items() if not value]
        raise GateFailure(f"G1 Video Only parity failed: {failed}")
    return result


def _run_selective_case(
    args: argparse.Namespace,
    *,
    case: MatrixCase,
    template: Mapping[str, Any],
) -> dict[str, Any]:
    output_prefix = f"v38_a3c/{args.run_tag}/{case.name}"
    prompt = _configure_selective(
        template,
        case=case,
        output_prefix=output_prefix,
    )
    result, entry = _execute(
        args,
        name=case.name,
        prompt=prompt,
        expected_seconds=case.chunks * 5.0,
        boundaries=case.boundary_seconds,
    )
    probe_status = _output_text(entry, "908")
    probe = _marker_json(probe_status, "A3C_PROBE_JSON=")
    first_videos = _output_latents(entry, "902", args.output_root)
    first_audios = _output_latents(entry, "903", args.output_root)
    refined_videos = _output_latents(entry, "904", args.output_root)
    refined_audios = _output_latents(entry, "905", args.output_root)
    groups = list(probe.get("physical_groups") or [])
    expected_mode = "audio_only" if case.target == "Audio Only" else "video_audio"
    expected_namespace = (
        "h3-continuum-refine-audio-v1"
        if case.target == "Audio Only"
        else "h3-continuum-refine-av-v1"
    )
    terminal_atomic = True
    if case.terminal_merge:
        terminal_atomic = (
            len(groups) == 2
            and groups[-1].get("logical_chunks") == [2, 3]
            and groups[-1].get("terminal_merged") is True
        )
    common = {
        "physical_group_count": len(groups) == case.expected_physical_groups,
        "latent_counts": len(first_videos)
        == len(first_audios)
        == len(refined_videos)
        == len(refined_audios)
        == case.expected_physical_groups,
        "target_contract": probe.get("target_mode") == expected_mode
        and probe.get("seed_namespace") == expected_namespace,
        "contract_version": probe.get("second_pass_contract_version") == 1,
        "seed_count": len(probe.get("refine_group_seeds") or [])
        == case.expected_physical_groups,
        "sigmas_recorded": bool(probe.get("schedule_sigma_hash")),
        "sampling_count": probe.get("sampling_passes_reported")
        == case.expected_physical_groups,
        "video_shape": probe.get("input_video_shapes")
        == probe.get("output_video_shapes"),
        "audio_shape": probe.get("input_audio_shapes")
        == probe.get("output_audio_shapes"),
        "finite": all(probe.get("output_video_finite") or [])
        and all(probe.get("output_audio_finite") or []),
        "terminal_atomic": terminal_atomic,
        "media_frames": int(result["media"]["frames"]) == case.chunks * 5 * 24,
        "media_duration": abs(
            float(result["media"]["format_duration"]) - case.chunks * 5.0
        )
        <= 0.05,
        "media_audio": int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
    }
    if case.target == "Audio Only":
        target_checks = {
            "video_object_passthrough": all(
                probe.get("input_video_object_identity") or []
            )
            and all(probe.get("input_video_tensor_identity") or []),
            "video_sha_passthrough": probe.get("input_video_sha256")
            == probe.get("output_video_sha256"),
            "audio_changed": all(
                before != after
                for before, after in zip(
                    probe.get("input_audio_sha256") or [],
                    probe.get("output_audio_sha256") or [],
                    strict=True,
                )
            ),
        }
    else:
        target_checks = {
            "video_changed": all(
                before != after
                for before, after in zip(
                    probe.get("input_video_sha256") or [],
                    probe.get("output_video_sha256") or [],
                    strict=True,
                )
            ),
            "audio_changed": all(
                before != after
                for before, after in zip(
                    probe.get("input_audio_sha256") or [],
                    probe.get("output_audio_sha256") or [],
                    strict=True,
                )
            ),
        }
    checks = {**common, **target_checks}
    result.update(
        {
            "case": asdict(case),
            "probe_status": probe_status,
            "probe": probe,
            "first_video_latents": first_videos,
            "first_audio_latents": first_audios,
            "refined_video_latents": refined_videos,
            "refined_audio_latents": refined_audios,
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    case_root = args.evidence_root / case.name
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"A3c end: {case.name}; pass={result['pass']}", flush=True)
    if not result["pass"]:
        failed = [name for name, value in checks.items() if not value]
        raise GateFailure(f"{case.name} failed: {failed}")
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    t2_template = json.loads(args.t2_template.read_text(encoding="utf-8"))
    terminal_template = json.loads(
        args.terminal_template.read_text(encoding="utf-8")
    )
    summary = {
        "format": "h3-continuum-v38-a3c-selective-refine-gpu-matrix-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "first_pass_seed": FIRST_PASS_SEED,
        "refine_seed": REFINE_SEED,
        "settings": {
            "resolution": "576x576",
            "first_pass_steps": 8,
            "refine_steps": 10,
            "refine_denoise": 0.35,
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
        "production_status": "HOLD for audio-targeted modes",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        summary["results"]["g1_video_only_parity"] = _run_g1(
            args,
            template=t2_template,
        )
        for case in G2_TO_G5:
            template = terminal_template if case.terminal_merge else t2_template
            summary["results"][case.name] = _run_selective_case(
                args,
                case=case,
                template=template,
            )
        summary["pass"] = True
        summary["status"] = "Selective Refine GPU Experimental PASS"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "A3c STOP / Production HOLD"
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
