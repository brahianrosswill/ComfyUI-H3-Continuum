"""Run the A6 Temporal Window Refine GPU Matrix against the public node."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import types
from typing import Any, Mapping, Sequence

from safetensors.torch import load_file
import torch


TOOLS_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = TOOLS_ROOT.parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from v38_a2_audio_only_gpu_canary import _output_latents, _output_text
from v38_a3c_selective_refine_gpu_matrix import (
    GateFailure,
    MatrixCase,
    _configure_g1,
    _configure_selective,
    _execute,
    _marker_json,
)
from v38_a4_chunk_scoped_refine_gpu_matrix import (
    _cross_run_plan_identity,
    _key_paths,
    _load_a3c_baseline,
)


@dataclass(frozen=True)
class WindowCase:
    name: str
    target: str
    chunks: int
    terminal_merge: bool
    start_sec: float
    end_sec: float
    selected_group_indices: tuple[int, ...]
    boundary_seconds: tuple[float, ...]


WINDOW_CASES = (
    WindowCase(
        "g2_video_only_partial",
        "Video Only",
        1,
        False,
        1.0,
        3.0,
        (0,),
        (),
    ),
    WindowCase(
        "g3_video_audio_partial",
        "Video + Audio",
        1,
        False,
        1.0,
        3.0,
        (0,),
        (),
    ),
    WindowCase(
        "g4_video_audio_cross_group",
        "Video + Audio",
        3,
        False,
        4.0,
        6.0,
        (0, 1),
        (5.0, 10.0),
    ),
    WindowCase(
        "g5_terminal_one_logical_side",
        "Video + Audio",
        3,
        True,
        5.5,
        8.0,
        (1,),
        (5.0, 10.0),
    ),
)

VIDEO_ONLY_CONTRACT_FIELDS = {
    "audio_output",
    "audio_sampling",
    "conditioning_sources",
    "execution",
    "physical_groups",
    "refine_execution_contract",
    "refine_group_seeds",
    "refine_schedule",
    "refine_schedule_identity",
    "refine_seed_base",
    "refine_target_contract",
    "target_height",
    "target_width",
    "version",
}


def _load_source_contract():
    """Load only constants/temporal/refine_window under an isolated package."""

    package_name = "h3_a6_gate_source"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SOURCE_ROOT)]
    sys.modules[package_name] = package
    v3_package = types.ModuleType(f"{package_name}.v3")
    v3_package.__path__ = [str(SOURCE_ROOT / "v3")]
    sys.modules[f"{package_name}.v3"] = v3_package

    def load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise GateFailure(f"cannot load source module {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    load(f"{package_name}.constants", SOURCE_ROOT / "constants.py")
    temporal = load(f"{package_name}.temporal", SOURCE_ROOT / "temporal.py")
    refine_window = load(
        f"{package_name}.v3.refine_window",
        SOURCE_ROOT / "v3" / "refine_window.py",
    )
    return temporal, refine_window


TEMPORAL, REFINE_WINDOW = _load_source_contract()


def _load_tensor(record: Mapping[str, Any]) -> torch.Tensor:
    return load_file(str(record["path"]), device="cpu")["latent_tensor"]


def _temporal_selection(
    tensor: torch.Tensor,
    ranges: Sequence[Sequence[int]],
    *,
    time_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dimension = int(time_dim)
    if dimension < 0:
        dimension += tensor.ndim
    mask = torch.zeros(int(tensor.shape[dimension]), dtype=torch.bool)
    for start, stop in ranges:
        mask[int(start) : int(stop)] = True
    moved = tensor.movedim(dimension, 0)
    return moved[mask], moved[~mask]


def _temporal_checks(
    before: torch.Tensor,
    after: torch.Tensor,
    ranges: Sequence[Sequence[int]],
    *,
    time_dim: int,
    expect_inside_change: bool,
) -> dict[str, bool]:
    if tuple(before.shape) != tuple(after.shape) or before.dtype != after.dtype:
        return {
            "shape_dtype": False,
            "inside_expected": False,
            "outside_exact": False,
        }
    before_inside, before_outside = _temporal_selection(
        before,
        ranges,
        time_dim=time_dim,
    )
    after_inside, after_outside = _temporal_selection(
        after,
        ranges,
        time_dim=time_dim,
    )
    inside_equal = torch.equal(before_inside, after_inside)
    return {
        "shape_dtype": True,
        "inside_expected": (not inside_equal) if expect_inside_change else inside_equal,
        "outside_exact": torch.equal(before_outside, after_outside),
    }


def _field_set(value: Any) -> set[str]:
    return set(_key_paths(value))


def _contains_window_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            "window" in str(key).lower() or _contains_window_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_window_key(item) for item in value)
    return False


def _configure_parity(template: Mapping[str, Any], *, output_prefix: str) -> dict[str, Any]:
    prompt = _configure_g1(template, output_prefix=output_prefix)
    prompt["303"]["inputs"].update(
        {
            "refine_scope": "All",
            "refine_scope_index": 1,
            "window_start_sec": 0.0,
            "window_end_sec": 5.0,
        }
    )
    return prompt


def _configure_window(
    template: Mapping[str, Any],
    *,
    case: WindowCase,
    output_prefix: str,
) -> dict[str, Any]:
    matrix_case = MatrixCase(
        name=case.name,
        target=case.target,
        chunks=case.chunks,
        terminal_merge=case.terminal_merge,
        expected_physical_groups=2 if case.terminal_merge else case.chunks,
        boundary_seconds=case.boundary_seconds,
    )
    prompt = _configure_selective(
        template,
        case=matrix_case,
        output_prefix=output_prefix,
    )
    prompt["302"]["inputs"].update(
        {
            "refine_scope": "Time Window",
            "refine_scope_index": 1,
            "window_start_sec": float(case.start_sec),
            "window_end_sec": float(case.end_sec),
        }
    )
    prompt["302"]["_meta"]["title"] = (
        f"Selective {case.target} / Time Window "
        f"[{case.start_sec:g}, {case.end_sec:g})"
    )
    return prompt


def _run_g1(args: argparse.Namespace, template: Mapping[str, Any]) -> dict[str, Any]:
    output_prefix = f"v38_a6/{args.run_tag}/g1_no_window_all_video_only"
    prompt = _configure_parity(template, output_prefix=output_prefix)
    result, entry = _execute(
        args,
        name="g1_no_window_all_video_only",
        prompt=prompt,
        expected_seconds=5.0,
        boundaries=(),
    )
    parity_status = _output_text(entry, "908")
    parity = _marker_json(parity_status, "A3C_PARITY_JSON=")
    checks = {
        "one_physical_group": parity.get("physical_group_count") == 1,
        "video_latent_sha_parity": parity.get("legacy_video_sha256")
        == parity.get("selective_video_sha256"),
        "audio_latent_sha_parity": parity.get("legacy_audio_sha256")
        == parity.get("selective_audio_sha256"),
        "audio_object_passthrough": all(
            parity.get("legacy_audio_object_identity") or []
        )
        and all(parity.get("selective_audio_object_identity") or []),
        "audio_tensor_passthrough": all(
            parity.get("legacy_audio_tensor_identity") or []
        )
        and all(parity.get("selective_audio_tensor_identity") or []),
        "seed_parity": parity.get("legacy_refine_group_seeds")
        == parity.get("selective_refine_group_seeds"),
        "sigmas_parity": parity.get("legacy_schedule_sigma_hash")
        == parity.get("selective_schedule_sigma_hash"),
        "sampling_parity": parity.get("legacy_sampling_passes_reported")
        == parity.get("selective_sampling_passes_reported")
        == 1,
        "assembly_plan_parity": parity.get("legacy_plan_without_target_identity")
        == parity.get("selective_plan_without_target_identity"),
        "target_contract": parity.get("selective_target_mode") == "video_only",
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
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    case_root = args.evidence_root / "g1_no_window_all_video_only"
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"A6 G1 parity failed: {failed}")
    return result


def _run_window_case(
    args: argparse.Namespace,
    *,
    case: WindowCase,
    template: Mapping[str, Any],
    baseline_probe: Mapping[str, Any],
    expected_seeds: Sequence[int],
    expected_contract_fields: set[str],
) -> dict[str, Any]:
    output_prefix = f"v38_a6/{args.run_tag}/{case.name}"
    prompt = _configure_window(template, case=case, output_prefix=output_prefix)
    result, entry = _execute(
        args,
        name=case.name,
        prompt=prompt,
        expected_seconds=case.chunks * 5.0,
        boundaries=case.boundary_seconds,
    )
    status = _output_text(entry, "908")
    probe = _marker_json(status, "A3C_PROBE_JSON=")
    input_video_records = _output_latents(entry, "902", args.output_root)
    input_audio_records = _output_latents(entry, "903", args.output_root)
    output_video_records = _output_latents(entry, "904", args.output_root)
    output_audio_records = _output_latents(entry, "905", args.output_root)
    input_videos = [_load_tensor(record) for record in input_video_records]
    input_audios = [_load_tensor(record) for record in input_audio_records]
    output_videos = [_load_tensor(record) for record in output_video_records]
    output_audios = [_load_tensor(record) for record in output_audio_records]
    normalized_plan = (probe.get("output_plan") or {}).get("normalized") or {}
    window = REFINE_WINDOW.resolve_refine_window(
        case.start_sec,
        case.end_sec,
        normalized_plan,
    )
    selected = tuple(window.selected_group_indices)
    expected_selected = tuple(case.selected_group_indices)
    contract = normalized_plan.get("second_pass_contract") or {}
    baseline_plan = (baseline_probe.get("output_plan") or {}).get("normalized") or {}
    baseline_contract = baseline_plan.get("second_pass_contract") or {}
    checks: dict[str, bool] = {
        "selected_groups": selected == expected_selected,
        "physical_group_count": len(input_videos)
        == len(input_audios)
        == len(output_videos)
        == len(output_audios)
        == (2 if case.terminal_merge else case.chunks),
        "sampling_count": probe.get("sampling_passes_reported")
        == len(expected_selected),
        "absolute_seed_parity": probe.get("refine_group_seeds")
        == list(expected_seeds),
        "sigmas_parity": probe.get("schedule_sigma_hash")
        == baseline_probe.get("schedule_sigma_hash"),
        "contract_version": probe.get("second_pass_contract_version") == 1,
        "window_status": "scope=Time Window" in status
        and "timeline=final_visible_output" in status
        and "continuation_prefix=protected" in status,
        "window_not_in_plan": not _contains_window_key(contract),
        "assembly_field_set": set(contract) == set(expected_contract_fields),
        "refine_schedule_v1_unchanged": contract.get("refine_schedule")
        == baseline_contract.get("refine_schedule"),
        "media": int(result["media"]["frames"]) == case.chunks * 120
        and abs(
            float(result["media"]["format_duration"]) - case.chunks * 5.0
        )
        <= 0.05
        and int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
        "finite": all(probe.get("output_video_finite") or [])
        and all(probe.get("output_audio_finite") or []),
    }
    group_checks: dict[str, dict[str, bool]] = {}
    for group_index, (before_v, before_a, after_v, after_a) in enumerate(
        zip(input_videos, input_audios, output_videos, output_audios, strict=True)
    ):
        group_window = window.group(group_index)
        selected_group = group_index in expected_selected
        current: dict[str, bool] = {}
        if not selected_group:
            current.update(
                {
                    "video_whole_exact": torch.equal(before_v, after_v),
                    "audio_whole_exact": torch.equal(before_a, after_a),
                    "video_object_passthrough": bool(
                        (probe.get("input_video_object_identity") or [])[group_index]
                    ),
                    "video_tensor_passthrough": bool(
                        (probe.get("input_video_tensor_identity") or [])[group_index]
                    ),
                    "audio_object_passthrough": bool(
                        (probe.get("input_audio_object_identity") or [])[group_index]
                    ),
                    "audio_tensor_passthrough": bool(
                        (probe.get("input_audio_tensor_identity") or [])[group_index]
                    ),
                }
            )
        else:
            if group_window is None:
                raise GateFailure(f"selected group {group_index + 1} has no window")
            video_expect_change = case.target in ("Video Only", "Video + Audio")
            audio_expect_change = case.target in ("Audio Only", "Video + Audio")
            for name, passed in _temporal_checks(
                before_v,
                after_v,
                group_window.video_slot_ranges,
                time_dim=2,
                expect_inside_change=video_expect_change,
            ).items():
                current[f"video_{name}"] = passed
            for name, passed in _temporal_checks(
                before_a,
                after_a,
                group_window.audio_tick_ranges,
                time_dim=-1,
                expect_inside_change=audio_expect_change,
            ).items():
                current[f"audio_{name}"] = passed
            physical_group = contract["physical_groups"][group_index]
            trim_frames = int(physical_group.get("trim_prefix_frames", 0))
            offsets = TEMPORAL.latent_slot_offsets(int(before_v.shape[2]))
            first_video_slot = int(group_window.video_slot_ranges[0][0])
            first_audio_tick = int(group_window.audio_tick_ranges[0][0])
            current["video_prefix_excluded"] = offsets[first_video_slot] >= trim_frames
            current["audio_prefix_excluded"] = (
                first_audio_tick >= TEMPORAL.audio_latent_t(trim_frames)
            )
        group_checks[str(group_index + 1)] = current
        checks[f"group_{group_index + 1}"] = all(current.values())
    if case.terminal_merge:
        terminal = list(probe.get("physical_groups") or [])[-1]
        checks["terminal_atomic"] = (
            terminal.get("logical_chunks") == [2, 3]
            and terminal.get("terminal_merged") is True
            and probe.get("sampling_passes_reported") == 1
            and selected == (1,)
        )
    result.update(
        {
            "case": asdict(case),
            "probe_status": status,
            "probe": probe,
            "window_contract": REFINE_WINDOW.serializable_window_contract(window),
            "input_video_latents": input_video_records,
            "input_audio_latents": input_audio_records,
            "output_video_latents": output_video_records,
            "output_audio_latents": output_audio_records,
            "group_checks": group_checks,
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    case_root = args.evidence_root / case.name
    (case_root / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"{case.name} failed: {failed}")
    return result


def _load_resumed_g1(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not bool(result.get("pass")) or not all((result.get("checks") or {}).values()):
        raise GateFailure(f"resumed G1 is not PASS: {path}")
    result["resumed_without_resampling"] = True
    path.with_name("result.revalidated.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def _load_resumed_window_result(
    path: Path,
    *,
    expected_seeds: Sequence[int],
    expected_contract_fields: set[str],
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    probe = result.get("probe") or {}
    contract = (
        ((probe.get("output_plan") or {}).get("normalized") or {}).get(
            "second_pass_contract"
        )
        or {}
    )
    checks = dict(result.get("checks") or {})
    checks["absolute_seed_parity"] = probe.get("refine_group_seeds") == list(
        expected_seeds
    )
    checks["assembly_field_set"] = set(contract) == set(expected_contract_fields)
    result["checks"] = checks
    result["pass"] = all(checks.values())
    result["resumed_without_resampling"] = True
    if not result["pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise GateFailure(f"resumed window validation failed: {failed}")
    path.with_name("result.revalidated.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    t2_template = json.loads(args.t2_template.read_text(encoding="utf-8"))
    terminal_template = json.loads(args.terminal_template.read_text(encoding="utf-8"))
    baseline_summary = json.loads(args.a3c_baseline.read_text(encoding="utf-8"))
    nonterminal_baseline = _load_a3c_baseline(
        args.a3c_baseline,
        "g4_video_audio_3x5",
    )["probe"]
    one_group_av_baseline = _load_a3c_baseline(
        args.a3c_baseline,
        "g3_video_audio_1x5",
    )["probe"]
    terminal_baseline = _load_a3c_baseline(
        args.a3c_baseline,
        "g5_video_audio_terminal",
    )["probe"]
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-a6-temporal-window-refine-gpu-matrix-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "a3c_baseline": str(args.a3c_baseline),
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
        "results": {},
        "pass": False,
        "production_status": "Experimental; audio-targeted modes remain HOLD",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        if args.resume_g1 is not None:
            summary["results"]["g1_no_window_all_video_only"] = _load_resumed_g1(
                args.resume_g1
            )
        else:
            summary["results"]["g1_no_window_all_video_only"] = _run_g1(
                args,
                t2_template,
            )
        video_only_seeds = list(
            (((baseline_summary.get("results") or {}).get("g1_video_only_parity") or {})
            .get("parity", {})
            .get("selective_refine_group_seeds")
            or [])
        )
        resume_results: dict[str, Path] = {}
        for path in args.resume_result or ():
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = str((payload.get("case") or {}).get("name") or "")
            if not name or name in resume_results:
                raise GateFailure(f"invalid or duplicate resumed result: {path}")
            resume_results[name] = path
        for case in WINDOW_CASES:
            template = terminal_template if case.terminal_merge else t2_template
            if case.target == "Video Only":
                baseline = one_group_av_baseline
                expected_seeds = video_only_seeds
                expected_fields = VIDEO_ONLY_CONTRACT_FIELDS
            elif case.terminal_merge:
                baseline = terminal_baseline
                expected_seeds = [
                    list(terminal_baseline.get("refine_group_seeds") or [])[index]
                    for index in case.selected_group_indices
                ]
                expected_fields = set(
                    (
                        (terminal_baseline.get("output_plan") or {})
                        .get("normalized", {})
                        .get("second_pass_contract", {})
                    )
                )
            elif case.chunks == 1:
                baseline = one_group_av_baseline
                expected_seeds = list(
                    one_group_av_baseline.get("refine_group_seeds") or []
                )
                expected_fields = set(
                    (
                        (one_group_av_baseline.get("output_plan") or {})
                        .get("normalized", {})
                        .get("second_pass_contract", {})
                    )
                )
            else:
                baseline = nonterminal_baseline
                expected_seeds = [
                    list(nonterminal_baseline.get("refine_group_seeds") or [])[index]
                    for index in case.selected_group_indices
                ]
                expected_fields = set(
                    (
                        (nonterminal_baseline.get("output_plan") or {})
                        .get("normalized", {})
                        .get("second_pass_contract", {})
                    )
                )
            if case.name in resume_results:
                result = _load_resumed_window_result(
                    resume_results[case.name],
                    expected_seeds=expected_seeds,
                    expected_contract_fields=expected_fields,
                )
            else:
                result = _run_window_case(
                    args,
                    case=case,
                    template=template,
                    baseline_probe=baseline,
                    expected_seeds=expected_seeds,
                    expected_contract_fields=expected_fields,
                )
            summary["results"][case.name] = result
        summary["pass"] = True
        summary["status"] = "A6 Temporal Window Refine GPU Experimental PASS"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "A6 STOP / Production unchanged"
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
    parser.add_argument("--resume-g1", type=Path)
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
