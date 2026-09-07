"""Run the A7a read-only Packed-row Observability GPU Matrix."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from v38_a2_audio_only_gpu_canary import (
    _history_entry,
    _output_latents,
    _output_text,
)
from v38_a3c_selective_refine_gpu_matrix import (
    GateFailure,
    PROMPT,
    _configure_base,
    _execute,
    _find_one,
    _marker_json,
    _remove_class,
)


@dataclass(frozen=True)
class FirstPassCase:
    name: str
    chunks: int
    terminal_merge: bool
    conditioning: str
    baseline_kind: str
    baseline_history: Path | None = None

    @property
    def expected_groups(self) -> int:
        return 2 if self.terminal_merge else self.chunks

    @property
    def expected_seconds(self) -> float:
        return float(self.chunks * 5)


PLAN_PATTERN = re.compile(
    r"execution_order=(\[[^\]]*\]),\s*"
    r"max_concurrent_physical_groups=(\d+),\s*"
    r"observed_groups=(\d+)/(\d+),\s*"
    r"peak_candidate=(\d+|unavailable),\s*"
    r"peak_total_rows=(\d+|unavailable)"
)
GROUP_PATTERN = re.compile(
    r"physical_group=(\d+),\s*logical_chunks=(\[[^\]]*\]),\s*"
    r"terminal_atomic=(true|false),\s*total=(\d+),\s*text=(\d+),\s*"
    r"condition_video=(\d+),\s*condition_audio=(\d+),\s*"
    r"target_video=(\d+),\s*target_audio=(\d+),\s*unknown=(\d+)"
)


def _history_only_entry(path: Path) -> Mapping[str, Any]:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(history, Mapping) or len(history) != 1:
        raise GateFailure(f"baseline history must contain one prompt: {path}")
    return _history_entry(history, str(next(iter(history))))


def _baseline_from_a6(path: Path) -> dict[str, Any]:
    entry = _history_only_entry(path)
    probe_status = _output_text(entry, "908")
    probe = _marker_json(probe_status, "A3C_PROBE_JSON=")
    return {
        "origin": str(path),
        "video_sha256": list(probe.get("input_video_sha256") or []),
        "audio_sha256": list(probe.get("input_audio_sha256") or []),
        "assembly_sha256": str((probe.get("input_plan") or {}).get("sha256") or ""),
        "assembly_normalized": copy.deepcopy(
            (probe.get("input_plan") or {}).get("normalized")
        ),
        "decoded": None,
    }


def _cross_run_plan_identity(value: Any) -> Any:
    """Ignore only the pre-existing per-execution monotonic start marker."""

    normalized = copy.deepcopy(value)
    if isinstance(normalized, dict):
        normalized.pop("_runtime_started_at", None)
    return normalized


def _parse_packed_rows(status: str) -> dict[str, Any]:
    plan_line = next(
        (line for line in status.splitlines() if line.startswith("Packed-row plan [A7a v1]:")),
        "",
    )
    if not plan_line:
        raise GateFailure("A7a packed-row plan is missing from Detailed status")
    match = PLAN_PATTERN.search(plan_line)
    if match is None:
        raise GateFailure("A7a packed-row plan could not be parsed")
    groups = []
    for line in status.splitlines():
        if not line.startswith("Packed-row group [A7a v1]:"):
            continue
        current = GROUP_PATTERN.search(line)
        if current is None:
            raise GateFailure(f"A7a packed-row group could not be parsed: {line}")
        group = {
            "physical_group": int(current.group(1)),
            "logical_chunks": json.loads(current.group(2)),
            "terminal_atomic": current.group(3) == "true",
            "total": int(current.group(4)),
            "text": int(current.group(5)),
            "condition_video": int(current.group(6)),
            "condition_audio": int(current.group(7)),
            "target_video": int(current.group(8)),
            "target_audio": int(current.group(9)),
            "unknown": int(current.group(10)),
            "point_in_time_memory_observation": (
                "pre_system_available=" in line
                and "post_system_available=" in line
                and "pre_CUDA_allocated=" in line
                and "post_CUDA_allocated=" in line
                and "pre_Core_free=" in line
                and "post_Core_free=" in line
            ),
        }
        group["component_sum"] = sum(
            group[key]
            for key in (
                "text",
                "condition_video",
                "condition_audio",
                "target_video",
                "target_audio",
                "unknown",
            )
        )
        groups.append(group)
    peak_candidate = None if match.group(5) == "unavailable" else int(match.group(5))
    peak_rows = None if match.group(6) == "unavailable" else int(match.group(6))
    return {
        "execution_order": json.loads(match.group(1)),
        "max_concurrent_physical_groups": int(match.group(2)),
        "observed_groups": int(match.group(3)),
        "recorded_groups": int(match.group(4)),
        "peak_row_candidate": peak_candidate,
        "peak_total_rows": peak_rows,
        "groups": groups,
        "report_contract": {
            "actual_core_rows_only": "actual Core rows only" in plan_line,
            "no_vram_time_prediction": "VRAM/time prediction=not provided" in plan_line,
            "execution_policy_unchanged": "execution policy=unchanged" in plan_line,
        },
    }


def _configure_first_pass(
    template: Mapping[str, Any],
    *,
    case: FirstPassCase,
    diagnostics: str,
    output_prefix: str,
    reference_image: str,
    reference_video: str,
) -> dict[str, Any]:
    prompt = _configure_base(
        template,
        chunks=case.chunks,
        terminal_merge=case.terminal_merge,
        output_prefix=output_prefix,
    )
    _remove_class(prompt, "H3ContinuumSecondPassV35")
    _remove_class(prompt, "H3ContinuumSelectiveSecondPassExperimental")
    sampler_id, sampler_node = _find_one(prompt, "H3ContinuumSamplerV38")
    inputs = sampler_node["inputs"]
    inputs["diagnostics"] = diagnostics
    inputs["run_storage"] = "Off"
    prompt_id = str(inputs["sequence_prompt"][0])
    prompt[prompt_id]["inputs"]["value"] = PROMPT

    for node_id in ("900", "901", "902", "903", "904", "905", "906", "907", "908", "909", "910", "911"):
        prompt.pop(node_id, None)

    if case.conditioning == "reference_images_3":
        prompt[prompt_id]["inputs"]["value"] = (
            PROMPT
            + " Preserve the identities and natural appearance from <Picture 1>, "
            "<Picture 2>, and <Picture 3>."
        )
        for offset, input_name in enumerate(
            ("reference_image_1", "reference_image_2", "reference_image_3"),
            start=950,
        ):
            prompt[str(offset)] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image},
                "_meta": {"title": f"A7a Reference Image {offset - 949}"},
            }
            inputs[input_name] = [str(offset), 0]
    elif case.conditioning == "reference_video":
        prompt[prompt_id]["inputs"]["value"] = (
            PROMPT
            + " Follow the motion and appearance guidance from <Video 1> while preserving continuity."
        )
        prompt["950"] = {
            "class_type": "H3ContinuumLoadVideo",
            "inputs": {
                "enable_video": True,
                "file": reference_video,
                "force_rate": 24.0,
            },
            "_meta": {"title": "A7a High-row Video Guide"},
        }
        inputs["reference_video_1"] = ["950", 0]
        inputs["video_reference_size"] = "Efficient - 0.4 MP"

    _video_decode_id, video_decode = _find_one(prompt, "VAEDecode")
    _audio_decode_id, audio_decode = _find_one(prompt, "VAEDecodeAudio")
    _assemble_id, assemble = _find_one(prompt, "H3ContinuumAssembleSeamV35")
    video_decode["inputs"]["samples"] = [sampler_id, 0]
    audio_decode["inputs"]["samples"] = [sampler_id, 1]
    assemble["inputs"]["assembly_plan"] = [sampler_id, 2]
    prompt["900"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": [sampler_id, 3]},
    }
    prompt["902"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": [sampler_id, 0],
            "filename_prefix": f"{output_prefix}_video",
        },
    }
    prompt["903"] = {
        "class_type": "SaveLatent",
        "inputs": {
            "samples": [sampler_id, 1],
            "filename_prefix": f"{output_prefix}_audio",
        },
    }
    prompt["907"] = {
        "class_type": "H3A3cSelectiveRefineProbe",
        "inputs": {
            "input_video_latents": [sampler_id, 0],
            "input_audio_latents": [sampler_id, 1],
            "output_video_latents": [sampler_id, 0],
            "output_audio_latents": [sampler_id, 1],
            "input_assembly_plan": [sampler_id, 2],
            "output_assembly_plan": [sampler_id, 2],
            "refine_status": [sampler_id, 3],
            "target_label": "Video Only",
        },
    }
    prompt["908"] = {
        "class_type": "PreviewAny",
        "inputs": {"source": ["907", 0]},
    }
    return prompt


def _run_once(
    args: argparse.Namespace,
    *,
    case: FirstPassCase,
    template: Mapping[str, Any],
    diagnostics: str,
) -> dict[str, Any]:
    variant = "basic" if diagnostics == "Basic" else "detailed"
    name = f"{case.name}_{variant}"
    output_prefix = f"v38_a7a/{args.run_tag}/{name}"
    prompt = _configure_first_pass(
        template,
        case=case,
        diagnostics=diagnostics,
        output_prefix=output_prefix,
        reference_image=args.reference_image,
        reference_video=args.reference_video,
    )
    result, entry = _execute(
        args,
        name=name,
        prompt=prompt,
        expected_seconds=case.expected_seconds,
        boundaries=tuple(float(index * 5) for index in range(1, case.chunks)),
    )
    sampler_status = _output_text(entry, "900")
    probe_status = _output_text(entry, "908")
    probe = _marker_json(probe_status, "A3C_PROBE_JSON=")
    video_records = _output_latents(entry, "902", args.output_root)
    audio_records = _output_latents(entry, "903", args.output_root)
    result.update(
        {
            "diagnostics": diagnostics,
            "sampler_status": sampler_status,
            "probe": probe,
            "video_latents": video_records,
            "audio_latents": audio_records,
            "packed_rows": (
                _parse_packed_rows(sampler_status)
                if diagnostics == "Detailed Report"
                else None
            ),
        }
    )
    checks = {
        "latent_count": len(video_records)
        == len(audio_records)
        == case.expected_groups,
        "saved_video_sha_matches_probe": [item["tensor_sha256"] for item in video_records]
        == list(probe.get("input_video_sha256") or []),
        "saved_audio_sha_matches_probe": [item["tensor_sha256"] for item in audio_records]
        == list(probe.get("input_audio_sha256") or []),
        "finite": all(probe.get("output_video_finite") or [])
        and all(probe.get("output_audio_finite") or []),
        "media_frames": int(result["media"]["frames"])
        == int(case.expected_seconds * 24),
        "media_duration": abs(
            float(result["media"]["format_duration"]) - case.expected_seconds
        )
        <= 0.05,
        "media_audio": int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
        "run_storage_off": prompt[_find_one(prompt, "H3ContinuumSamplerV38")[0]][
            "inputs"
        ]["run_storage"]
        == "Off",
    }
    if diagnostics == "Basic":
        checks["planner_absent_in_basic"] = "Packed-row plan [A7a" not in sampler_status
    else:
        packed = result["packed_rows"]
        rows = list(packed["groups"])
        expected_order = list(range(1, case.expected_groups + 1))
        actual_peak = min(
            (group["physical_group"] for group in rows if group["total"] == max(item["total"] for item in rows)),
            default=None,
        )
        checks.update(
            {
                "execution_order": packed["execution_order"] == expected_order,
                "max_concurrent_one": packed["max_concurrent_physical_groups"] == 1,
                "all_groups_observed": packed["observed_groups"]
                == packed["recorded_groups"]
                == len(rows)
                == case.expected_groups,
                "component_sum_equals_seq_len": all(
                    group["component_sum"] == group["total"] for group in rows
                ),
                "one_valid_target_pair_per_observed_layout": all(
                    group["target_video"] > 0 and group["target_audio"] > 0
                    for group in rows
                ),
                "peak_row_candidate": packed["peak_row_candidate"] == actual_peak,
                "peak_row_count": packed["peak_total_rows"]
                == max((group["total"] for group in rows), default=None),
                "memory_observation_only": all(
                    group["point_in_time_memory_observation"] for group in rows
                ),
                "report_contract": all(packed["report_contract"].values()),
                "terminal_atomic": (
                    not case.terminal_merge
                    or (
                        rows[-1]["logical_chunks"] == [2, 3]
                        and rows[-1]["terminal_atomic"] is True
                    )
                ),
            }
        )
    result["checks"] = checks
    result["pass"] = all(checks.values())
    result_path = args.evidence_root / name / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not result["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise GateFailure(f"{name} failed: {failed}")
    return result


def _baseline_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    probe = result["probe"]
    return {
        "origin": str(result["history"]),
        "video_sha256": list(probe.get("input_video_sha256") or []),
        "audio_sha256": list(probe.get("input_audio_sha256") or []),
        "assembly_sha256": str((probe.get("input_plan") or {}).get("sha256") or ""),
        "assembly_normalized": copy.deepcopy(
            (probe.get("input_plan") or {}).get("normalized")
        ),
        "decoded": copy.deepcopy(result.get("decoded")),
    }


def _load_completed_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    checks = result.get("checks") or {}
    if not bool(result.get("pass")) or not checks or not all(checks.values()):
        raise GateFailure(f"existing result is not a completed run-level PASS: {path}")
    result["resumed_without_resampling"] = True
    return result


def _parity_checks(
    detailed: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, bool]:
    probe = detailed["probe"]
    checks = {
        "video_latent_sha_parity": list(probe.get("input_video_sha256") or [])
        == list(baseline["video_sha256"]),
        "audio_latent_sha_parity": list(probe.get("input_audio_sha256") or [])
        == list(baseline["audio_sha256"]),
        "assembly_plan_identity": _cross_run_plan_identity(
            (probe.get("input_plan") or {}).get("normalized")
        )
        == _cross_run_plan_identity(baseline.get("assembly_normalized")),
    }
    if baseline.get("decoded") is not None:
        checks["decoded_video_parity"] = (
            detailed.get("decoded") or {}
        ).get("video_sha256") == baseline["decoded"].get("video_sha256")
        checks["decoded_audio_parity"] = (
            detailed.get("decoded") or {}
        ).get("audio_pcm_sha256") == baseline["decoded"].get("audio_pcm_sha256")
    return checks


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    t2_template = json.loads(args.t2_template.read_text(encoding="utf-8"))
    terminal_template = json.loads(args.terminal_template.read_text(encoding="utf-8"))
    cases = (
        FirstPassCase("g1_1x5_t2va", 1, False, "none", "same_runtime_basic"),
        FirstPassCase(
            "g2_3x5_continuation",
            3,
            False,
            "none",
            "accepted_a6",
            args.a6_g2_history,
        ),
        FirstPassCase(
            "g3_fl2va_terminal",
            3,
            True,
            "none",
            "accepted_a6",
            args.a6_g3_history,
        ),
        FirstPassCase(
            "g4_reference_images_3",
            1,
            False,
            "reference_images_3",
            "same_runtime_basic",
        ),
        FirstPassCase(
            "g5_reference_video_high_row",
            1,
            False,
            "reference_video",
            "same_runtime_basic",
        ),
    )
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-a7a-packed-row-gpu-matrix-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "settings": {
            "resolution": "576x576",
            "steps": 8,
            "continuity": "Balanced - 22 frames",
            "audio_continuity": True,
            "run_storage": "Off",
            "sampler": "res_multistep",
            "scheduler": "simple",
            "sage": True,
        },
        "baseline_policy": {
            "g1_g4_g5": "same-runtime Basic vs Detailed",
            "g2_g3": "accepted pre-A7 A6 First Pass latent/Assembly evidence",
            "decoded_parity": "required where same-runtime Basic baseline exists",
        },
        "results": {},
        "pass": False,
        "production_status": "read-only Experimental; A7b execution policy HOLD",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        g1_rows = None
        for case in cases:
            template = terminal_template if case.terminal_merge else t2_template
            basic = None
            if case.baseline_kind == "same_runtime_basic":
                basic_path = args.evidence_root / f"{case.name}_basic" / "result.json"
                basic = (
                    _load_completed_result(basic_path)
                    if args.resume_existing and basic_path.exists()
                    else _run_once(
                        args,
                        case=case,
                        template=template,
                        diagnostics="Basic",
                    )
                )
                baseline = _baseline_from_result(basic)
            else:
                if case.baseline_history is None:
                    raise GateFailure(f"accepted baseline path missing for {case.name}")
                baseline = _baseline_from_a6(case.baseline_history)
            detailed_path = (
                args.evidence_root / f"{case.name}_detailed" / "result.json"
            )
            detailed = (
                _load_completed_result(detailed_path)
                if args.resume_existing and detailed_path.exists()
                else _run_once(
                    args,
                    case=case,
                    template=template,
                    diagnostics="Detailed Report",
                )
            )
            parity = _parity_checks(detailed, baseline)
            extra: dict[str, bool] = {}
            current_rows = detailed["packed_rows"]["groups"]
            if case.name == "g1_1x5_t2va":
                g1_rows = current_rows[0]
            elif case.conditioning in {"reference_images_3", "reference_video"}:
                if g1_rows is None:
                    raise GateFailure("G1 row baseline is unavailable")
                extra["condition_video_rows_increased_vs_g1"] = (
                    current_rows[0]["condition_video"] > g1_rows["condition_video"]
                )
            all_checks = {**parity, **extra}
            case_result = {
                "case": {
                    "name": case.name,
                    "chunks": case.chunks,
                    "terminal_merge": case.terminal_merge,
                    "conditioning": case.conditioning,
                    "baseline_kind": case.baseline_kind,
                },
                "baseline": baseline,
                "basic": basic,
                "detailed": detailed,
                "parity_checks": parity,
                "conditioning_checks": extra,
                "pass": all(all_checks.values()),
            }
            summary["results"][case.name] = case_result
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"A7a {case.name}: pass={case_result['pass']} "
                f"peak-row-candidate={detailed['packed_rows']['peak_row_candidate']} "
                f"rows={detailed['packed_rows']['peak_total_rows']}",
                flush=True,
            )
            if not case_result["pass"]:
                failed = [key for key, value in all_checks.items() if not value]
                raise GateFailure(f"{case.name} parity failed: {failed}")
        summary["pass"] = True
        summary["status"] = "A7a Packed-row Observability GPU Experimental PASS"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "A7a STOP / Production unchanged"
        return summary
    finally:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--backend-pid", type=int, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--t2-template", type=Path, required=True)
    parser.add_argument("--terminal-template", type=Path, required=True)
    parser.add_argument("--a6-g2-history", type=Path, required=True)
    parser.add_argument("--a6-g3-history", type=Path, required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--reference-video", required=True)
    parser.add_argument("--resume-existing", action="store_true")
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
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "pass": summary.get("pass"),
                "failure": summary.get("failure"),
                "summary": str(args.evidence_root / "summary.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
