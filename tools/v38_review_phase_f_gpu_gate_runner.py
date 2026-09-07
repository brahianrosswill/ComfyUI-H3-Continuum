"""Run the Phase F Full-vs-Review GPU parity checkpoint.

This first checkpoint intentionally submits identical V3.8 sampler inputs for
successive Review/Continue queues.  It therefore exercises the same ComfyUI
cache contract as a fixed-seed workflow instead of adding a test-only cache
nonce that could hide a Production integration defect.
"""

from __future__ import annotations

import argparse
import copy
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
from v38_easy_gpu_gate_runner import _output_media, _probe_media


class GateFailure(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def _build_prompt(
    template: Mapping[str, Any],
    *,
    run_name: str,
    generation_mode: str,
    review_action: str,
    output_prefix: str,
    seed: int,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(template))
    _, sampler = _find_one(prompt, "H3ContinuumSamplerV38")
    inputs = sampler["inputs"]
    inputs.update(
        {
            "aspect": "Landscape 16:9",
            "preset": "Draft — 0.30 MP",
            "custom_mp": 0.30,
            "chunks": 3,
            "chunk_seconds": 10.0,
            "base_seed": int(seed),
            "diagnostics": "Detailed Report",
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "strict_compatibility": False,
            "debug": True,
            "show_preview": False,
            "run_storage": "Save + Auto Resume",
            "run_name": run_name,
            "project_id": str(uuid.uuid5(uuid.NAMESPACE_URL, run_name)),
            "generation_mode": generation_mode,
            "review_action": review_action,
        }
    )
    _, save_video = _find_one(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = output_prefix
    return prompt


def _latest_manifest(run_root: Path) -> tuple[Path, dict[str, Any]]:
    candidates = list((run_root / "revisions").glob("*/manifest.json"))
    if not candidates:
        raise GateFailure(f"Run Storage manifest is missing: {run_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    path = candidates[0]
    return path, json.loads(path.read_text(encoding="utf-8"))


def _tensor_sha256(path: Path, key: str) -> str:
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key).contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _manifest_snapshot(run_root: Path) -> dict[str, Any]:
    manifest_path, manifest = _latest_manifest(run_root)
    revision_root = manifest_path.parent
    chunks = []
    for stored in manifest.get("chunks") or []:
        chunk_path = revision_root / "chunks" / str(stored["filename"])
        entry = dict(stored.get("entry") or {})
        chunks.append(
            {
                "sequence_index": int(stored["sequence_index"]),
                "filename": str(stored["filename"]),
                "file_sha256": _sha256_file(chunk_path),
                "manifest_file_sha256": str(stored["file_sha256"]),
                "video_latent_sha256": _tensor_sha256(chunk_path, "video"),
                "audio_latent_sha256": _tensor_sha256(chunk_path, "audio"),
                "seed": int(entry["seed"]),
                "prompt_hash": str(entry["prompt_hash"]),
                "plan": entry.get("plan"),
            }
        )
    return {
        "path": str(manifest_path),
        "mtime_ns": manifest_path.stat().st_mtime_ns,
        "status": str(manifest.get("status")),
        "review_control_version": manifest.get("review_control_version"),
        "review_pause_reason": manifest.get("review_pause_reason"),
        "review_unit": manifest.get("review_unit"),
        "branch_regenerate_from": manifest.get("branch_regenerate_from"),
        "effective_reroll_nonce": manifest.get("effective_reroll_nonce"),
        "chunks": chunks,
        "report_summary": str(manifest.get("report_summary") or ""),
        "manifest": manifest,
    }


def _stream_sha256(command: list[str]) -> tuple[str, int]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise GateFailure("failed to open ffmpeg stdout")
    digest = hashlib.sha256()
    size = 0
    while True:
        block = process.stdout.read(1024 * 1024)
        if not block:
            break
        digest.update(block)
        size += len(block)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    code = process.wait()
    if code:
        raise GateFailure(f"ffmpeg raw decode failed ({code}): {stderr[-2000:]}")
    return digest.hexdigest(), size


def _decoded_hashes(media_path: Path, ffmpeg: str, probe: Mapping[str, Any]) -> dict[str, Any]:
    video_sha, video_bytes = _stream_sha256(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(media_path),
            "-map",
            "0:v:0",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "-",
        ]
    )
    pcm_sha, pcm_bytes = _stream_sha256(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(media_path),
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
        ]
    )
    return {
        "raw_rgb24_sha256": video_sha,
        "raw_rgb24_bytes": video_bytes,
        "pcm_f32le_32k_stereo_sha256": pcm_sha,
        "pcm_f32le_32k_stereo_bytes": pcm_bytes,
        "decoded_frames": int(probe["frames"]),
        "width": int(probe["width"]),
        "height": int(probe["height"]),
        "fps": str(probe["fps"]),
        "audio_sample_rate": int(probe["audio_sample_rate"]),
        "audio_channels": int(probe["audio_channels"]),
        "format_duration": float(probe["format_duration"]),
    }


def _run_queue(
    args: argparse.Namespace,
    *,
    name: str,
    prompt: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    case_root = args.output_root / name
    case_root.mkdir(parents=True, exist_ok=True)
    prompt_path = case_root / "prompt.json"
    prompt_path.write_text(json.dumps(prompt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Phase F queue start: {name}", flush=True)
    with ResourceMonitor(args.backend_pid, args.monitor_interval) as monitor:
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=prompt,
            client_id=f"v38-phase-f-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    history_path = case_root / "history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    media_path = _output_media(history, args.media_root)
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        raise GateFailure("ffmpeg and ffprobe are required")
    media = _probe_media(media_path, ffprobe)
    decoded = _decoded_hashes(media_path, ffmpeg, media)
    storage = _manifest_snapshot(run_root)
    result = {
        "name": name,
        "prompt_id": prompt_id,
        "api_elapsed_seconds": float(elapsed),
        "prompt": str(prompt_path),
        "history": str(history_path),
        "media_path": str(media_path),
        "media_file_sha256": _sha256_file(media_path),
        "media": media,
        "decoded": decoded,
        "resource": monitor.as_dict(),
        "run_storage": storage,
    }
    (case_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Phase F queue end: {name}; chunks={len(storage['chunks'])}; "
        f"status={storage['status']}; {elapsed:.3f}s",
        flush=True,
    )
    return result


def _parity(full: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    full_chunks = full["run_storage"]["chunks"]
    review_chunks = review["run_storage"]["chunks"]
    fields = ("seed", "prompt_hash", "video_latent_sha256", "audio_latent_sha256", "plan")
    chunk_checks = []
    for index in range(max(len(full_chunks), len(review_chunks))):
        if index >= len(full_chunks) or index >= len(review_chunks):
            chunk_checks.append({"index": index + 1, "present": False})
            continue
        checks = {field: full_chunks[index][field] == review_chunks[index][field] for field in fields}
        chunk_checks.append({"index": index + 1, "present": True, "checks": checks})
    decoded_fields = (
        "raw_rgb24_sha256",
        "pcm_f32le_32k_stereo_sha256",
        "decoded_frames",
        "width",
        "height",
        "fps",
        "audio_sample_rate",
        "audio_channels",
        "format_duration",
    )
    decoded_checks = {
        field: full["decoded"][field] == review["decoded"][field]
        for field in decoded_fields
    }
    passed = (
        len(full_chunks) == len(review_chunks) == 3
        and all(item.get("present") and all(item["checks"].values()) for item in chunk_checks)
        and all(decoded_checks.values())
    )
    return {"pass": passed, "chunks": chunk_checks, "decoded": decoded_checks}


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    template = json.loads(args.template_prompt.read_text(encoding="utf-8"))
    stamp = args.run_tag
    full_run_name = f"v38_phase_f_g0_full_{stamp}"
    review_run_name = f"v38_phase_f_g1_review_{stamp}"
    runs_root = args.run_storage_root
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-review-phase-f-g0-g1-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "template_prompt": str(args.template_prompt),
        "template_prompt_sha256": _sha256_file(args.template_prompt),
        "seed": int(args.seed),
        "run_names": {"g0": full_run_name, "g1": review_run_name},
        "pass": False,
        "stopped": False,
        "results": {},
    }
    summary_path = args.output_root / "gate_summary.json"
    try:
        g0_prompt = _build_prompt(
            template,
            run_name=full_run_name,
            generation_mode="Full Run",
            review_action="Continue / Next",
            output_prefix=f"v38_phase_f/{stamp}/g0_full",
            seed=args.seed,
        )
        g0 = _run_queue(
            args,
            name="g0_full",
            prompt=g0_prompt,
            run_root=runs_root / full_run_name,
        )
        summary["results"]["g0"] = g0
        if len(g0["run_storage"]["chunks"]) != 3 or g0["run_storage"]["status"] != "complete":
            raise GateFailure("G0 Full baseline did not complete exactly three chunks")

        review_results = []
        previous_mtime = None
        for queue_index in range(1, 4):
            prompt = _build_prompt(
                template,
                run_name=review_run_name,
                generation_mode="Review Each Chunk",
                review_action="Continue / Next",
                output_prefix=f"v38_phase_f/{stamp}/g1_review_q{queue_index}",
                seed=args.seed,
            )
            result = _run_queue(
                args,
                name=f"g1_review_q{queue_index}",
                prompt=prompt,
                run_root=runs_root / review_run_name,
            )
            review_results.append(result)
            summary["results"]["g1"] = review_results
            actual_chunks = len(result["run_storage"]["chunks"])
            expected_status = "complete" if queue_index == 3 else "review_ready"
            if actual_chunks != queue_index or result["run_storage"]["status"] != expected_status:
                unchanged = previous_mtime == result["run_storage"]["mtime_ns"]
                raise GateFailure(
                    "G1 Review queue progression failed: "
                    f"queue={queue_index}, chunks={actual_chunks}, "
                    f"status={result['run_storage']['status']}, "
                    f"manifest_unchanged={unchanged}"
                )
            previous_mtime = result["run_storage"]["mtime_ns"]

        parity = _parity(g0, review_results[-1])
        summary["parity"] = parity
        if not parity["pass"]:
            raise GateFailure("G0 Full and G1 Review exact parity failed")
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
    parser.add_argument("--template-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--run-storage-root", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--backend-pid", type=int, default=None)
    parser.add_argument("--seed", type=int, default=927966463230733)
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
