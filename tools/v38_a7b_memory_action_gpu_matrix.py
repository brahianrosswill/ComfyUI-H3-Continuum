"""Run the A7b explicit Reference downgrade GPU Matrix."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from v38_a2_audio_only_gpu_canary import _output_latents, _output_text
from v38_a3c_selective_refine_gpu_matrix import (
    GateFailure,
    PROMPT,
    _execute,
    _find_one,
    _marker_json,
)
from v38_a7a_packed_row_gpu_matrix import (
    _cross_run_plan_identity,
    _parse_packed_rows,
)


ACTION_DISABLED = "Disabled"
ACTION_REDUCE = "Reduce References One Step"
IMAGE_MAX = "Max Identity"
IMAGE_MATCH = "Match Output"
VIDEO_BALANCED = "Balanced - 0.6 MP"
VIDEO_EFFICIENT = "Efficient - 0.4 MP"
POLICY_STATUS = "Memory Action Policy [Experimental A7b v1]"
SEED_PATTERN = re.compile(r"chunk 1/1: seed=(\d+)")
IMAGE_GEOMETRY_PATTERN = re.compile(r"Ref1=(\d+)x(\d+)")


@dataclass(frozen=True)
class A7bCase:
    name: str
    conditioning: str
    reference_count: int = 0
    video_file: str = ""

    @property
    def is_image(self) -> bool:
        return self.conditioning == "image"

    @property
    def is_video(self) -> bool:
        return self.conditioning == "video"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _configure_prompt(
    template: Mapping[str, Any],
    *,
    case: A7bCase,
    action: str,
    output_prefix: str,
    run_name: str,
    reference_image: str,
) -> dict[str, Any]:
    prompt = copy.deepcopy(dict(template))
    sampler_id, sampler = _find_one(prompt, "H3ContinuumSamplerV38")
    sampler["class_type"] = "H3ContinuumSamplerV38MemoryPolicyExperimental"
    inputs = sampler["inputs"]
    inputs["diagnostics"] = "Detailed Report"
    inputs["run_storage"] = "Save + Auto Resume"
    inputs["run_name"] = run_name
    inputs["project_id"] = ""
    inputs["reference_size"] = IMAGE_MAX
    inputs["video_reference_size"] = VIDEO_BALANCED
    for name in (
        "reference_image_1",
        "reference_image_2",
        "reference_image_3",
        "reference_video_1",
    ):
        inputs.pop(name, None)

    prompt["949"] = {
        "class_type": "H3ContinuumMemoryActionPolicyExperimental",
        "inputs": {"reference_action": action},
        "_meta": {"title": "A7b Explicit Memory Action Policy"},
    }
    inputs["memory_action_policy"] = ["949", 0]
    prompt_id = str(inputs["sequence_prompt"][0])
    prompt[prompt_id]["inputs"]["value"] = PROMPT

    if case.is_image:
        labels = []
        for index in range(1, case.reference_count + 1):
            node_id = str(949 + index)
            input_name = f"reference_image_{index}"
            prompt[node_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_image},
                "_meta": {"title": f"A7b Reference Image {index}"},
            }
            inputs[input_name] = [node_id, 0]
            labels.append(f"<Picture {index}>")
        prompt[prompt_id]["inputs"]["value"] = (
            PROMPT
            + " Preserve the identity and natural appearance from "
            + ", ".join(labels)
            + "."
        )
    elif case.is_video:
        prompt["950"] = {
            "class_type": "H3ContinuumLoadVideo",
            "inputs": {
                "enable_video": True,
                "file": case.video_file,
                "force_rate": 24.0,
            },
            "_meta": {"title": "A7b Video Guide"},
        }
        inputs["reference_video_1"] = ["950", 0]
        prompt[prompt_id]["inputs"]["value"] = (
            PROMPT
            + " Follow the appearance and motion guidance from <Video 1> while "
            "preserving continuity."
        )

    _save_video_id, save_video = _find_one(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = output_prefix
    prompt["902"]["inputs"]["filename_prefix"] = output_prefix + "_video"
    prompt["903"]["inputs"]["filename_prefix"] = output_prefix + "_audio"
    # The accepted A7a template already routes sampler outputs through Core
    # decode, assemble, SaveLatent, and the read-only A3c evidence probe.
    for node_id in ("900", "902", "903", "907", "908"):
        if node_id not in prompt:
            raise GateFailure(f"accepted A7a evidence node {node_id} is missing")
    prompt["900"]["inputs"]["source"] = [sampler_id, 3]
    prompt["902"]["inputs"]["samples"] = [sampler_id, 0]
    prompt["903"]["inputs"]["samples"] = [sampler_id, 1]
    for name, slot in (
        ("input_video_latents", 0),
        ("output_video_latents", 0),
        ("input_audio_latents", 1),
        ("output_audio_latents", 1),
        ("input_assembly_plan", 2),
        ("output_assembly_plan", 2),
        ("refine_status", 3),
    ):
        prompt["907"]["inputs"][name] = [sampler_id, slot]
    return prompt


def _latest_manifest(output_root: Path, run_name: str) -> dict[str, Any]:
    revision_root = output_root / "h3_continuum" / "runs" / run_name / "revisions"
    manifests = sorted(
        revision_root.glob("*/manifest.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not manifests:
        raise GateFailure(f"Run Storage manifest is missing for {run_name}")
    path = manifests[-1]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "manifest": manifest}


def _one_seed(status: str) -> int:
    match = SEED_PATTERN.search(status)
    if match is None:
        raise GateFailure("effective chunk seed is missing from sampler status")
    return int(match.group(1))


def _image_geometry(status: str) -> tuple[int, int] | None:
    match = IMAGE_GEOMETRY_PATTERN.search(status)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def _run_variant(
    args: argparse.Namespace,
    *,
    template: Mapping[str, Any],
    case: A7bCase,
    action: str,
) -> dict[str, Any]:
    variant = "disabled" if action == ACTION_DISABLED else "reduce"
    name = f"{case.name}_{variant}"
    run_name = f"v38_a7b_{args.run_tag}_{name}"[:96]
    output_prefix = f"v38_a7b/{args.run_tag}/{name}"
    prompt = _configure_prompt(
        template,
        case=case,
        action=action,
        output_prefix=output_prefix,
        run_name=run_name,
        reference_image=args.reference_image,
    )
    result, entry = _execute(
        args,
        name=name,
        prompt=prompt,
        expected_seconds=5.0,
        boundaries=(),
    )
    status = _output_text(entry, "900")
    probe = _marker_json(_output_text(entry, "908"), "A3C_PROBE_JSON=")
    video_records = _output_latents(entry, "902", args.output_root)
    audio_records = _output_latents(entry, "903", args.output_root)
    packed_rows = _parse_packed_rows(status)
    storage = _latest_manifest(args.output_root, run_name)
    manifest = storage["manifest"]
    contract = manifest.get("contract") or {}
    global_contract = contract.get("global") or {}
    chunk_contracts = contract.get("chunk_contracts") or []
    sampler_id, sampler = _find_one(prompt, "H3ContinuumSamplerV38MemoryPolicyExperimental")
    checks = {
        "one_physical_group": len(video_records) == len(audio_records) == 1,
        "saved_video_sha_matches_probe": [item["tensor_sha256"] for item in video_records]
        == list(probe.get("input_video_sha256") or []),
        "saved_audio_sha_matches_probe": [item["tensor_sha256"] for item in audio_records]
        == list(probe.get("input_audio_sha256") or []),
        "finite": all(probe.get("output_video_finite") or [])
        and all(probe.get("output_audio_finite") or []),
        "media": int(result["media"]["frames"]) == 120
        and abs(float(result["media"]["format_duration"]) - 5.0) <= 0.05
        and int(result["media"]["audio_sample_rate"]) == 32000
        and int(result["media"]["audio_channels"]) == 2,
        "audio_health": bool(result["audio_health"]["pass"]),
        "storage_complete": manifest.get("status") == "complete",
        "one_chunk_contract": len(chunk_contracts) == 1,
        "packed_rows_valid": packed_rows["observed_groups"] == 1
        and len(packed_rows["groups"]) == 1
        and packed_rows["groups"][0]["component_sum"]
        == packed_rows["groups"][0]["total"],
        "experimental_sampler_only": sampler["class_type"]
        == "H3ContinuumSamplerV38MemoryPolicyExperimental",
        "policy_status": (
            POLICY_STATUS not in status
            if action == ACTION_DISABLED
            else POLICY_STATUS in status
            and "automatic MODEL unload was not executed" in status
            and "automatic attention/backend switching was not executed" in status
            and "external MODEL wrapper was preserved" in status
        ),
    }
    result.update(
        {
            "action": action,
            "run_name": run_name,
            "sampler_id": sampler_id,
            "status": status,
            "seed": _one_seed(status),
            "image_geometry": _image_geometry(status),
            "probe": probe,
            "video_latents": video_records,
            "audio_latents": audio_records,
            "packed_rows": packed_rows,
            "run_storage": storage,
            "global_hash": (
                str(chunk_contracts[0].get("global_hash"))
                if chunk_contracts
                else ""
            ),
            "sigmas_hash": str(
                (global_contract.get("sigmas") or {}).get("exact_sha256") or ""
            ),
            "model_contract_hash": _sha256_json(global_contract.get("model")),
            "reference_contract": copy.deepcopy(global_contract.get("reference")),
            "reference_video_contract": copy.deepcopy(
                global_contract.get("reference_video")
            ),
            "assembly_normalized": copy.deepcopy(
                (probe.get("input_plan") or {}).get("normalized")
            ),
            "checks": checks,
            "pass": all(checks.values()),
        }
    )
    result_path = args.evidence_root / name / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not result["pass"]:
        failed = [key for key, value in checks.items() if not value]
        raise GateFailure(f"{name} failed: {failed}")
    return result


def _latent_hashes(result: Mapping[str, Any], stream: str) -> list[str]:
    return list((result.get("probe") or {}).get(f"input_{stream}_sha256") or [])


def _plan_without_runtime(value: Any) -> Any:
    return _cross_run_plan_identity(value)


def _effective_contract_checks(
    case: A7bCase,
    disabled: Mapping[str, Any],
    reduced: Mapping[str, Any],
) -> dict[str, bool]:
    base_rows = disabled["packed_rows"]["groups"][0]
    reduced_rows = reduced["packed_rows"]["groups"][0]
    checks = {
        "global_identity_changed": disabled["global_hash"]
        and reduced["global_hash"]
        and disabled["global_hash"] != reduced["global_hash"],
        "actual_total_rows_decreased": reduced_rows["total"] < base_rows["total"],
        "condition_video_rows_decreased": reduced_rows["condition_video"]
        < base_rows["condition_video"],
        "output_variation_expected": _latent_hashes(disabled, "video")
        != _latent_hashes(reduced, "video")
        or _latent_hashes(disabled, "audio") != _latent_hashes(reduced, "audio"),
        "assembly_semantics_unchanged": _plan_without_runtime(
            disabled["assembly_normalized"]
        )
        == _plan_without_runtime(reduced["assembly_normalized"]),
        "seed_unchanged": disabled["seed"] == reduced["seed"],
        "sigmas_unchanged": disabled["sigmas_hash"]
        and disabled["sigmas_hash"] == reduced["sigmas_hash"],
        "external_model_wrapper_unchanged": disabled["model_contract_hash"]
        == reduced["model_contract_hash"],
    }
    if case.is_image:
        base_contract = disabled.get("reference_contract") or {}
        reduced_contract = reduced.get("reference_contract") or {}
        base_geometry = disabled.get("image_geometry")
        reduced_geometry = reduced.get("image_geometry")
        checks.update(
            {
                "image_mode_one_step": base_contract.get("size_mode") == IMAGE_MAX
                and reduced_contract.get("size_mode") == IMAGE_MATCH,
                "image_contract_hash_changed": base_contract.get("combined_hash")
                != reduced_contract.get("combined_hash"),
                "image_geometry_decreased": base_geometry is not None
                and reduced_geometry is not None
                and reduced_geometry[0] * reduced_geometry[1]
                < base_geometry[0] * base_geometry[1],
            }
        )
    elif case.is_video:
        base_contract = disabled.get("reference_video_contract") or {}
        reduced_contract = reduced.get("reference_video_contract") or {}
        checks.update(
            {
                "video_mode_one_step": base_contract.get("size_mode")
                == VIDEO_BALANCED
                and reduced_contract.get("size_mode") == VIDEO_EFFICIENT,
                "video_contract_hash_changed": base_contract.get("combined_hash")
                != reduced_contract.get("combined_hash"),
                "video_geometry_decreased": int(reduced_contract.get("target_width", 0))
                * int(reduced_contract.get("target_height", 0))
                < int(base_contract.get("target_width", 0))
                * int(base_contract.get("target_height", 0)),
                "video_frame_count_unchanged": base_contract.get("frame_count")
                == reduced_contract.get("frame_count"),
            }
        )
    return checks


def _g1_checks(
    result: Mapping[str, Any],
    *,
    baseline_result: Mapping[str, Any],
    baseline_prompt: Mapping[str, Any],
) -> dict[str, bool]:
    baseline_probe = baseline_result.get("probe") or {}
    current_probe = result.get("probe") or {}
    _baseline_sampler_id, baseline_sampler = _find_one(
        baseline_prompt, "H3ContinuumSamplerV38"
    )
    current_prompt = json.loads(Path(result["prompt"]).read_text(encoding="utf-8"))
    _current_sampler_id, current_sampler = _find_one(
        current_prompt, "H3ContinuumSamplerV38MemoryPolicyExperimental"
    )
    return {
        "video_latent_bit_exact": list(current_probe.get("input_video_sha256") or [])
        == list(baseline_probe.get("input_video_sha256") or []),
        "audio_latent_bit_exact": list(current_probe.get("input_audio_sha256") or [])
        == list(baseline_probe.get("input_audio_sha256") or []),
        "assembly_semantics_bit_exact": _plan_without_runtime(
            (current_probe.get("input_plan") or {}).get("normalized")
        )
        == _plan_without_runtime(
            (baseline_probe.get("input_plan") or {}).get("normalized")
        ),
        "decoded_video_bit_exact": (result.get("decoded") or {}).get("video_sha256")
        == (baseline_result.get("decoded") or {}).get("video_sha256"),
        "decoded_audio_bit_exact": (result.get("decoded") or {}).get(
            "audio_pcm_sha256"
        )
        == (baseline_result.get("decoded") or {}).get("audio_pcm_sha256"),
        "seed_input_unchanged": current_sampler["inputs"].get("base_seed")
        == baseline_sampler["inputs"].get("base_seed"),
        "sampler_link_unchanged": current_sampler["inputs"].get("sampler")
        == baseline_sampler["inputs"].get("sampler"),
        "sigmas_link_unchanged": current_sampler["inputs"].get("sigmas")
        == baseline_sampler["inputs"].get("sigmas"),
        "no_reference_contract": result.get("reference_contract") is None
        and result.get("reference_video_contract") is None,
        "disabled_status_no_action": POLICY_STATUS not in str(result.get("status")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    template = json.loads(args.template.read_text(encoding="utf-8"))
    baseline_result = json.loads(args.a7a_g1_result.read_text(encoding="utf-8"))
    baseline_prompt = json.loads(
        Path(baseline_result["prompt"]).read_text(encoding="utf-8")
    )
    cases = (
        A7bCase("g2_image_1", "image", reference_count=1),
        A7bCase("g3_video_short", "video", video_file=args.short_video),
        A7bCase("g4_images_3", "image", reference_count=3),
        A7bCase("g5_video_high_row", "video", video_file=args.high_video),
    )
    summary: dict[str, Any] = {
        "format": "h3-continuum-v38-a7b-memory-action-gpu-matrix-v1",
        "server": args.server,
        "backend_pid": args.backend_pid,
        "run_tag": args.run_tag,
        "settings": {
            "resolution": "576x576",
            "steps": 8,
            "fixed_seed": True,
            "run_storage": "Save + Auto Resume with unique run names",
            "policy": "explicit opt-in one-step Reference downgrade",
            "auto_unload": "not executed; advisory-only",
            "backend_switch": "not executed; advisory-only",
        },
        "results": {},
        "pass": False,
        "production_status": "unchanged; Experimental path only; A5 HOLD",
    }
    summary_path = args.evidence_root / "summary.json"
    try:
        g1_case = A7bCase("g1_disabled", "none")
        g1 = _run_variant(
            args,
            template=template,
            case=g1_case,
            action=ACTION_DISABLED,
        )
        g1_parity = _g1_checks(
            g1,
            baseline_result=baseline_result,
            baseline_prompt=baseline_prompt,
        )
        summary["results"][g1_case.name] = {
            "baseline": str(args.a7a_g1_result),
            "disabled": g1,
            "checks": g1_parity,
            "pass": all(g1_parity.values()),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if not all(g1_parity.values()):
            failed = [key for key, value in g1_parity.items() if not value]
            raise GateFailure(f"G1 Disabled parity failed: {failed}")
        print("A7b G1 Disabled parity PASS", flush=True)

        for case in cases:
            disabled = _run_variant(
                args,
                template=template,
                case=case,
                action=ACTION_DISABLED,
            )
            reduced = _run_variant(
                args,
                template=template,
                case=case,
                action=ACTION_REDUCE,
            )
            checks = _effective_contract_checks(case, disabled, reduced)
            case_result = {
                "case": {
                    "name": case.name,
                    "conditioning": case.conditioning,
                    "reference_count": case.reference_count,
                    "video_file": case.video_file,
                },
                "disabled": disabled,
                "reduced": reduced,
                "checks": checks,
                "pass": all(checks.values()),
            }
            summary["results"][case.name] = case_result
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                f"A7b {case.name}: pass={case_result['pass']} "
                f"rows={disabled['packed_rows']['peak_total_rows']}->"
                f"{reduced['packed_rows']['peak_total_rows']}",
                flush=True,
            )
            if not case_result["pass"]:
                failed = [key for key, value in checks.items() if not value]
                raise GateFailure(f"{case.name} failed: {failed}")
        summary["pass"] = True
        summary["status"] = "A7b v1 Memory Action Policy GPU Experimental PASS"
        return summary
    except Exception as exc:
        summary["failure_type"] = type(exc).__name__
        summary["failure"] = str(exc)
        summary["status"] = "A7b STOP / Production unchanged / A5 HOLD"
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
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--a7a-g1-result", type=Path, required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--short-video", required=True)
    parser.add_argument("--high-video", required=True)
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
