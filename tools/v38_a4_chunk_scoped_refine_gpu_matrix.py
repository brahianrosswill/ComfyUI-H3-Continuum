"""Run the A4 Chunk-scoped Refine GPU Matrix against the public node."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from v38_a2_audio_only_gpu_canary import _output_latents, _output_text
from v38_a3c_selective_refine_gpu_matrix import (
    GateFailure,
    MatrixCase,
    _configure_selective,
    _execute,
    _marker_json,
)


@dataclass(frozen=True)
class ScopeCase:
    name: str
    scope: str
    scope_index: int
    terminal_merge: bool
    selected_group_index: int | None
    selected_logical_chunks: tuple[int, ...]


CASES = (
    ScopeCase("g1_all_video_audio_3x5", "All", 1, False, None, (1, 2, 3)),
    ScopeCase("g2_physical_group_2", "Physical Group", 2, False, 1, (2,)),
    ScopeCase("g3_logical_chunk_2", "Logical Chunk", 2, False, 1, (2,)),
    ScopeCase("g4_terminal_logical_chunk_2", "Logical Chunk", 2, True, 1, (2, 3)),
    ScopeCase("g5_terminal_logical_chunk_3", "Logical Chunk", 3, True, 1, (2, 3)),
)


def _key_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            path = f"{prefix}.{key}"
            paths.append(path)
            paths.extend(_key_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_key_paths(item, f"{prefix}[]"))
    return paths


def _cross_run_plan_identity(value: Any) -> Any:
    """Ignore only the pre-existing per-execution monotonic start marker."""

    normalized = copy.deepcopy(value)
    if isinstance(normalized, dict):
        normalized.pop("_runtime_started_at", None)
    return normalized


def _configure(
    template: Mapping[str, Any],
    *,
    case: ScopeCase,
    output_prefix: str,
) -> dict[str, Any]:
    base_case = MatrixCase(
        name=case.name,
        target="Video + Audio",
        chunks=3,
        terminal_merge=case.terminal_merge,
        expected_physical_groups=2 if case.terminal_merge else 3,
        boundary_seconds=(5.0, 10.0),
    )
    prompt = _configure_selective(
        template,
        case=base_case,
        output_prefix=output_prefix,
    )
    selective = prompt["302"]["inputs"]
    selective["refine_scope"] = case.scope
    selective["refine_scope_index"] = int(case.scope_index)
    prompt["302"]["_meta"]["title"] = (
        f"Selective Video + Audio / {case.scope} {case.scope_index}"
    )
    return prompt


def _load_a3c_baseline(path: Path, result_name: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not bool(summary.get("pass")):
        raise GateFailure("A3c baseline summary is not PASS")
    result = (summary.get("results") or {}).get(result_name)
    if not isinstance(result, dict) or not bool(result.get("pass")):
        raise GateFailure(f"A3c baseline {result_name!r} is missing or not PASS")
    return result


def _identity_checks(
    probe: Mapping[str, Any],
    *,
    selected_group_index: int,
) -> dict[str, bool]:
    count = len(probe.get("physical_groups") or [])
    expected_selected = [index == selected_group_index for index in range(count)]
    video_object = list(probe.get("input_video_object_identity") or [])
    video_tensor = list(probe.get("input_video_tensor_identity") or [])
    audio_object = list(probe.get("input_audio_object_identity") or [])
    audio_tensor = list(probe.get("input_audio_tensor_identity") or [])
    input_video = list(probe.get("input_video_sha256") or [])
    output_video = list(probe.get("output_video_sha256") or [])
    input_audio = list(probe.get("input_audio_sha256") or [])
    output_audio = list(probe.get("output_audio_sha256") or [])
    return {
        "identity_vector_lengths": all(
            len(values) == count
            for values in (
                video_object,
                video_tensor,
                audio_object,
                audio_tensor,
                input_video,
                output_video,
                input_audio,
                output_audio,
            )
        ),
        "video_object_passthrough": all(
            identity == (not selected)
            for identity, selected in zip(
                video_object,
                expected_selected,
                strict=True,
            )
        ),
        "video_tensor_passthrough": all(
            identity == (not selected)
            for identity, selected in zip(
                video_tensor,
                expected_selected,
                strict=True,
            )
        ),
        "audio_object_passthrough": all(
            identity == (not selected)
            for identity, selected in zip(
                audio_object,
                expected_selected,
                strict=True,
            )
        ),
        "audio_tensor_passthrough": all(
            identity == (not selected)
            for identity, selected in zip(
                audio_tensor,
                expected_selected,
                strict=True,
            )
        ),
        "video_sha_selection": all(
            (before != after) is selected
            for before, after, selected in zip(
                input_video,
                output_video,
                expected_selected,
                strict=True,
            )
        ),
        "audio_sha_selection": all(
            (before != after) is selected
            for before, after, selected in zip(
                input_audio,
                output_audio,
                expected_selected,
                strict=True,
            )
        ),
    }


def _run_case(
    args: argparse.Namespace,
    *,
    case: ScopeCase,
    template: Mapping[str, Any],
    baseline_probe: Mapping[str, Any],
    all_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output_prefix = f"v38_a4/{args.run_tag}/{case.name}"
    prompt = _configure(template, case=case, output_prefix=output_prefix)
    result, entry = _execute(
        args,
        name=case.name,
        prompt=prompt,
        expected_seconds=15.0,
        boundaries=(5.0, 10.0),
    )
    probe_status = _output_text(entry, "908")
    probe = _marker_json(probe_status, "A3C_PROBE_JSON=")
    first_videos = _output_latents(entry, "902", args.output_root)
    first_audios = _output_latents(entry, "903", args.output_root)
    refined_videos = _output_latents(entry, "904", args.output_root)
    refined_audios = _output_latents(entry, "905", args.output_root)
    groups = list(probe.get("physical_groups") or [])
    expected_groups = 2 if case.terminal_merge else 3
    contract = (probe.get("output_plan") or {}).get("normalized", {}).get(
        "second_pass_contract",
        {},
    )
    checks = {
        "physical_group_count": len(groups) == expected_groups,
        "latent_counts": len(first_videos)
        == len(first_audios)
        == len(refined_videos)
        == len(refined_audios)
        == expected_groups,
        "target_contract": probe.get("target_mode") == "video_audio"
        and probe.get("seed_namespace") == "h3-continuum-refine-av-v1",
        "contract_version": probe.get("second_pass_contract_version") == 1,
        "schedule_hash": probe.get("schedule_sigma_hash")
        == baseline_probe.get("schedule_sigma_hash"),
        "plan_scope_field_absent": "refine_scope_contract" not in contract,
        "media": int(result["media"]["frames"]) == 360
        and abs(float(result["media"]["format_duration"]) - 15.0) <= 0.05
        and int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
    }
    if case.selected_group_index is None:
        current_plan = (probe.get("output_plan") or {}).get("normalized")
        baseline_plan = (baseline_probe.get("output_plan") or {}).get("normalized")
        checks.update(
            {
                "all_video_input_parity": probe.get("input_video_sha256")
                == baseline_probe.get("input_video_sha256"),
                "all_audio_input_parity": probe.get("input_audio_sha256")
                == baseline_probe.get("input_audio_sha256"),
                "all_video_output_parity": probe.get("output_video_sha256")
                == baseline_probe.get("output_video_sha256"),
                "all_audio_output_parity": probe.get("output_audio_sha256")
                == baseline_probe.get("output_audio_sha256"),
                "all_seed_parity": probe.get("refine_group_seeds")
                == baseline_probe.get("refine_group_seeds"),
                "all_sampling_parity": probe.get("sampling_passes_reported")
                == baseline_probe.get("sampling_passes_reported")
                == expected_groups,
                "all_plan_field_set": _key_paths(current_plan)
                == _key_paths(baseline_plan),
                "all_plan_parity": _cross_run_plan_identity(current_plan)
                == _cross_run_plan_identity(baseline_plan),
                "all_decoded_video_parity": result["decoded"]["video_sha256"]
                == args.a3c_decoded_video_sha,
                "all_decoded_audio_parity": result["decoded"]["audio_pcm_sha256"]
                == args.a3c_decoded_audio_sha,
                "all_scope_status": "scope=All." in probe_status,
            }
        )
    else:
        selected_index = int(case.selected_group_index)
        expected_seed = list(baseline_probe.get("refine_group_seeds") or [])[selected_index]
        checks.update(
            {
                "one_sampling": probe.get("sampling_passes_reported") == 1,
                "one_seed": probe.get("refine_group_seeds") == [expected_seed],
                "scope_status": f"scope={case.scope} {case.scope_index}." in probe_status,
                "resolved_groups_status": f"selected_physical_groups=[{selected_index + 1}]"
                in probe_status,
                "resolved_logical_status": (
                    "selected_logical_chunks="
                    + str(list(case.selected_logical_chunks))
                )
                in probe_status,
                **_identity_checks(
                    probe,
                    selected_group_index=selected_index,
                ),
            }
        )
        if all_result is not None:
            all_plan = (all_result.get("probe") or {}).get("output_plan") or {}
            checks["assembly_field_set_parity"] = _key_paths(
                (probe.get("output_plan") or {}).get("normalized")
            ) == _key_paths(all_plan.get("normalized"))
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
    print(f"A4 end: {case.name}; pass={result['pass']}", flush=True)
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"{case.name} failed: {failed}")
    return result


def _load_resumed_all_result(
    path: Path,
    *,
    baseline_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate one completed G1 without repeating its GPU Sampling."""

    result = json.loads(path.read_text(encoding="utf-8"))
    probe = result.get("probe") or {}
    checks = dict(result.get("checks") or {})
    current_plan = (probe.get("output_plan") or {}).get("normalized")
    baseline_plan = (baseline_probe.get("output_plan") or {}).get("normalized")
    checks["all_plan_field_set"] = _key_paths(current_plan) == _key_paths(
        baseline_plan
    )
    checks["all_plan_parity"] = _cross_run_plan_identity(
        current_plan
    ) == _cross_run_plan_identity(baseline_plan)
    result["checks"] = checks
    result["pass"] = all(checks.values())
    result["resumed_without_resampling"] = True
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"resumed G1 validation failed: {failed}")
    path.with_name("result.revalidated.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _load_resumed_scoped_result(
    path: Path,
    *,
    assembly_reference_probe: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate one completed scoped case without repeating GPU Sampling."""

    result = json.loads(path.read_text(encoding="utf-8"))
    probe = result.get("probe") or {}
    checks = dict(result.get("checks") or {})
    current_plan = (probe.get("output_plan") or {}).get("normalized")
    reference_plan = (assembly_reference_probe.get("output_plan") or {}).get(
        "normalized"
    )
    checks["assembly_field_set_parity"] = _key_paths(current_plan) == _key_paths(
        reference_plan
    )
    result["checks"] = checks
    result["pass"] = all(checks.values())
    result["resumed_without_resampling"] = True
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"resumed scoped validation failed: {failed}")
    path.with_name("result.revalidated.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _cross_case_checks(results: Mapping[str, Mapping[str, Any]]) -> dict[str, bool]:
    g2 = results[CASES[1].name]
    g3 = results[CASES[2].name]
    g4 = results[CASES[3].name]
    g5 = results[CASES[4].name]
    return {
        "physical_vs_logical_nonterminal_video": g2["probe"]["output_video_sha256"]
        == g3["probe"]["output_video_sha256"],
        "physical_vs_logical_nonterminal_audio": g2["probe"]["output_audio_sha256"]
        == g3["probe"]["output_audio_sha256"],
        "physical_vs_logical_nonterminal_seed": g2["probe"]["refine_group_seeds"]
        == g3["probe"]["refine_group_seeds"],
        "terminal_2_vs_3_video": g4["probe"]["output_video_sha256"]
        == g5["probe"]["output_video_sha256"],
        "terminal_2_vs_3_audio": g4["probe"]["output_audio_sha256"]
        == g5["probe"]["output_audio_sha256"],
        "terminal_2_vs_3_seed": g4["probe"]["refine_group_seeds"]
        == g5["probe"]["refine_group_seeds"],
        "terminal_2_vs_3_plan": g4["probe"]["output_plan"]["sha256"]
        == g5["probe"]["output_plan"]["sha256"],
        "terminal_2_vs_3_decoded_video": g4["decoded"]["video_sha256"]
        == g5["decoded"]["video_sha256"],
        "terminal_2_vs_3_decoded_audio": g4["decoded"]["audio_pcm_sha256"]
        == g5["decoded"]["audio_pcm_sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    t2_template = json.loads(args.t2_template.read_text(encoding="utf-8"))
    terminal_template = json.loads(args.terminal_template.read_text(encoding="utf-8"))
    baseline_result = _load_a3c_baseline(
        args.a3c_baseline,
        "g4_video_audio_3x5",
    )
    terminal_baseline_result = _load_a3c_baseline(
        args.a3c_baseline,
        "g5_video_audio_terminal",
    )
    baseline_probe = baseline_result["probe"]
    args.a3c_decoded_video_sha = baseline_result["decoded"]["video_sha256"]
    args.a3c_decoded_audio_sha = baseline_result["decoded"]["audio_pcm_sha256"]
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-a4-chunk-scoped-refine-gpu-matrix-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "a3c_baseline": str(args.a3c_baseline),
        "results": {},
        "pass": False,
        "production_status": "HOLD for audio-targeted modes",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        resume_results: dict[str, Path] = {}
        for path in args.resume_result or ():
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = str((payload.get("case") or {}).get("name") or "")
            if not name or name in resume_results:
                raise GateFailure(f"invalid or duplicate resumed result: {path}")
            resume_results[name] = path
        all_result: Mapping[str, Any] | None = None
        for case in CASES:
            assembly_reference_result = (
                terminal_baseline_result if case.terminal_merge else all_result
            )
            if case.name in resume_results:
                if case.selected_group_index is None:
                    result = _load_resumed_all_result(
                        resume_results[case.name],
                        baseline_probe=baseline_probe,
                    )
                else:
                    if assembly_reference_result is None:
                        raise GateFailure("scoped resume has no Assembly reference")
                    result = _load_resumed_scoped_result(
                        resume_results[case.name],
                        assembly_reference_probe=assembly_reference_result["probe"],
                    )
            else:
                template = terminal_template if case.terminal_merge else t2_template
                result = _run_case(
                    args,
                    case=case,
                    template=template,
                    baseline_probe=(
                        terminal_baseline_result["probe"]
                        if case.terminal_merge
                        else baseline_probe
                    ),
                    all_result=assembly_reference_result,
                )
            summary["results"][case.name] = result
            if case.selected_group_index is None:
                all_result = result
        summary["cross_case_checks"] = _cross_case_checks(summary["results"])
        if not all(summary["cross_case_checks"].values()):
            failed = [
                name
                for name, passed in summary["cross_case_checks"].items()
                if not passed
            ]
            raise GateFailure(f"cross-case parity failed: {failed}")
        summary["pass"] = True
        summary["status"] = "A4 Chunk-scoped Refine GPU Experimental PASS"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "A4 STOP / Production HOLD"
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
    parser.add_argument("--a3c-baseline", type=Path, required=True)
    parser.add_argument("--resume-result", type=Path, action="append")
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
