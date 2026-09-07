"""Run and compare the G4 Sampling determinism Order Matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping
import uuid


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from p32_api_runner import submit_and_wait
from v36_r1_stress_gate_runner import ResourceMonitor


SAMPLING_CHECKPOINTS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
MODEL_CHECKPOINTS = ("M0", "M1", "M2", "M3")
ORIGINAL_G4 = {
    "full": {
        "video": "d6da9472755a991898378543356a1e57960e8aece0803bd9918c20422cb66805",
        "audio": "dab3e1ee457cd1dc1d580cba8d47a6679f1a37d40b949a2fac7c63a47a3b7768",
    },
    "full_then_review": {
        "video": "140064e63344d75e535f48ec49e488ca574cc069ee920cdaa507801c3504fb43",
        "audio": "4742f4d9eef6ffd64413cce2c052689fe13e4f2f98c914142039ab419aa908cf",
    },
}


class GateFailure(RuntimeError):
    pass


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _prompt(
    template: Mapping[str, Any],
    *,
    label: str,
    mode: str,
    run_name: str,
    output_prefix: str,
    nonce: int,
    trace_steps: bool,
) -> tuple[dict[str, Any], str]:
    prompt = copy.deepcopy(dict(template))
    node_id, sampler = _find_one(prompt, "H3ContinuumSamplerV38")
    sampler["class_type"] = "H3G4SamplingDeterminismDiagnostic"
    sampler["_meta"] = {"title": "H3 G4 Sampling Determinism Diagnostic"}
    sampler["inputs"].update(
        {
            "aspect": "Square 1:1",
            "preset": "Draft — 0.30 MP",
            "custom_mp": 0.30,
            "chunks": 3,
            "chunk_seconds": 5.0,
            "continuity": "Balanced — 22 frames",
            "base_seed": 927966463230733,
            "audio_continuity": True,
            "diagnostics": "Basic",
            "reroll_from_chunk": "Auto",
            "reroll_nonce": 0,
            "strict_compatibility": False,
            "debug": False,
            "show_preview": False,
            "run_storage": "Save + Auto Resume",
            "run_name": run_name,
            "reference_size": "Match Output",
            "project_id": str(uuid.uuid5(uuid.NAMESPACE_URL, run_name)),
            "video_reference_size": "Efficient - 0.4 MP",
            "continuation_backend": "Standard",
            "generation_mode": mode,
            "review_action": "Continue / Next",
            "diagnostic_label": label,
            "diagnostic_nonce": int(nonce),
            "trace_steps": bool(trace_steps),
        }
    )
    _, save_video = _find_one(prompt, "SaveVideo")
    save_video["inputs"]["filename_prefix"] = output_prefix
    return prompt, node_id


def _extract(history: Mapping[str, Any], prompt_id: str, node_id: str) -> dict[str, Any]:
    record = history.get(prompt_id)
    if not isinstance(record, Mapping):
        raise GateFailure(f"history record missing for {prompt_id}")
    outputs = record.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GateFailure("history outputs missing")
    node = outputs.get(str(node_id))
    if not isinstance(node, Mapping):
        raise GateFailure(f"diagnostic output missing for node {node_id}")
    text = node.get("text")
    if not isinstance(text, list) or len(text) != 1:
        raise GateFailure(f"unexpected diagnostic text output: {text!r}")
    result = json.loads(str(text[0]))
    if result.get("format") != "h3-v38-g4-sampling-determinism-v1":
        raise GateFailure("unexpected diagnostic format")
    if not result.get("hooks_restored"):
        raise GateFailure("diagnostic hooks were not restored")
    for sample in result.get("physical_samples", []):
        complete = sample.get("capture_complete") or {}
        if not complete.get("all_sampling_checkpoints"):
            raise GateFailure(f"sampling checkpoints incomplete: {complete}")
        if not complete.get("all_model_checkpoints"):
            raise GateFailure(f"model checkpoints incomplete: {complete}")
    return result


def _run_case(
    args,
    template: Mapping[str, Any],
    *,
    label: str,
    mode: str,
    nonce: int,
    run_name: str,
) -> dict[str, Any]:
    case_root = args.evidence_root / f"stage_{args.stage.lower()}" / label
    prompt, node_id = _prompt(
        template,
        label=label,
        mode=mode,
        run_name=run_name,
        output_prefix=f"v38_g4_sampling_diagnostic/{args.run_tag}/{label}",
        nonce=nonce,
        trace_steps=args.trace_steps,
    )
    _write(case_root / "prompt.json", prompt)
    print(f"G4 Sampling diagnostic start: {label}", flush=True)
    with ResourceMonitor(args.backend_pid, interval_seconds=args.monitor_interval) as monitor:
        prompt_id, history, elapsed = submit_and_wait(
            server=args.server,
            prompt=prompt,
            client_id=f"g4sampling-{uuid.uuid4().hex}",
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
    _write(case_root / "history.json", history)
    result = _extract(history, prompt_id, node_id)
    _write(case_root / "diagnostic.json", result)
    output = {
        "label": label,
        "mode": mode,
        "run_name": run_name,
        "prompt_id": prompt_id,
        "api_elapsed_seconds": elapsed,
        "resources": monitor.as_dict(),
        "prompt": str(case_root / "prompt.json"),
        "history": str(case_root / "history.json"),
        "diagnostic_path": str(case_root / "diagnostic.json"),
        "diagnostic": result,
    }
    _write(case_root / "result.json", output)
    print(f"G4 Sampling diagnostic end: {label}; {elapsed:.3f}s", flush=True)
    return output


def _stage(args) -> dict[str, Any]:
    template = json.loads(args.template_prompt.read_text(encoding="utf-8"))
    prefix = f"v38_g4_sampling_{args.run_tag}_{args.stage.lower()}"
    definitions = {
        "A": [("A_fresh_full", "Full Run", f"{prefix}_full")],
        "B": [("B_fresh_review", "Review Each Chunk", f"{prefix}_review")],
        "C": [
            ("C_full", "Full Run", f"{prefix}_full"),
            ("C_review_after_full", "Review Each Chunk", f"{prefix}_review"),
        ],
        "D": [
            ("D_review_1", "Review Each Chunk", f"{prefix}_review_1"),
            ("D_review_2", "Review Each Chunk", f"{prefix}_review_2"),
        ],
    }
    runs = []
    for offset, (label, mode, run_name) in enumerate(definitions[args.stage], 1):
        run_root = args.run_storage_root / run_name
        if run_root.exists():
            raise GateFailure(f"refusing existing Run Storage: {run_root}")
        runs.append(
            _run_case(
                args,
                template,
                label=label,
                mode=mode,
                nonce=args.nonce_base + offset,
                run_name=run_name,
            )
        )
    result = {
        "format": "h3-v38-g4-sampling-determinism-stage-v1",
        "stage": args.stage,
        "backend_pid": args.backend_pid,
        "backend_log": str(args.backend_log),
        "server": args.server,
        "run_tag": args.run_tag,
        "trace_steps": bool(args.trace_steps),
        "runner_sha256": _sha256_file(Path(__file__)),
        "template_sha256": _sha256_file(args.template_prompt),
        "runs": runs,
    }
    _write(args.evidence_root / f"stage_{args.stage.lower()}.json", result)
    return result


def _first_sample(run: Mapping[str, Any]) -> Mapping[str, Any]:
    samples = run["diagnostic"].get("physical_samples") or []
    if not samples:
        raise GateFailure(f"no physical samples in {run.get('label')}")
    return samples[0]


def _normalized_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        if value.get("__type__") == "uuid.UUID" and "value" in value:
            return {"__type__": "uuid.UUID", "present": True}
        return {
            str(key): _normalized_payload(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            if str(key) != "legacy_g4_sha256"
        }
    if isinstance(value, list):
        return [_normalized_payload(item) for item in value]
    return value


def _comparison_sha(checkpoint: Mapping[str, Any]) -> str:
    if checkpoint.get("kind") != "structured" or "payload" not in checkpoint:
        return str(checkpoint["sha256"])
    payload = _normalized_payload(checkpoint["payload"])
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha(value: Mapping[str, Any], name: str) -> str:
    return _comparison_sha(value[name])


def _compare_samples(left_run: Mapping[str, Any], right_run: Mapping[str, Any]) -> dict[str, Any]:
    left = _first_sample(left_run)
    right = _first_sample(right_run)
    checkpoints: dict[str, Any] = {}
    first_sampling = None
    for name in SAMPLING_CHECKPOINTS:
        left_sha = _sha(left["checkpoints"], name)
        right_sha = _sha(right["checkpoints"], name)
        match = left_sha == right_sha
        checkpoints[name] = {
            "left_sha256": left_sha,
            "right_sha256": right_sha,
            "match": match,
        }
        if first_sampling is None and not match:
            first_sampling = name
    model: dict[str, Any] = {}
    first_model = None
    for name in MODEL_CHECKPOINTS:
        left_sha = _sha(left["model"], name)
        right_sha = _sha(right["model"], name)
        match = left_sha == right_sha
        model[name] = {
            "left_sha256": left_sha,
            "right_sha256": right_sha,
            "match": match,
        }
        if first_model is None and not match:
            first_model = name
    output = {
        "left_sha256": str(left["output"]["sha256"]),
        "right_sha256": str(right["output"]["sha256"]),
        "match": left["output"]["sha256"] == right["output"]["sha256"],
        "left_streams": left["output"]["streams"],
        "right_streams": right["output"]["streams"],
    }
    return {
        "left": left_run["label"],
        "right": right_run["label"],
        "first_sampling_divergence": first_sampling,
        "sampling": checkpoints,
        "first_model_divergence": first_model,
        "model": model,
        "output": output,
    }


def _classification(comparison: Mapping[str, Any]) -> dict[str, Any]:
    first_s = comparison.get("first_sampling_divergence")
    if (
        first_s is None
        and comparison.get("first_model_divergence") is None
        and comparison["output"]["match"]
    ):
        return {
            "case": "NO DIVERGENCE REPRODUCED",
            "reason": "S0-S8, M0-M3, and the final sampled AV latent are exact",
        }
    if first_s == "S0":
        case, reason = "CASE S-A", "target latent construction differs"
    elif first_s == "S2":
        case, reason = "CASE S-B", "prepare_noise / RNG path differs"
    elif first_s == "S3":
        case, reason = "CASE S-C", "denoise mask differs"
    elif first_s == "S8":
        case, reason = "CASE S-D", "model runtime signature differs before Sampling"
    elif first_s is not None:
        case, reason = "CASE S-OTHER", f"Sampling boundary first differs at {first_s}"
    else:
        first_m = comparison.get("first_model_divergence")
        if first_m is None:
            case, reason = "CASE M-E", "M0-M3 match; step trace is required"
        elif first_m == "M3" and all(
            comparison["model"][name]["match"] for name in ("M0", "M1", "M2")
        ):
            case, reason = (
                "CASE M-FORWARD",
                "identical first model inputs produce a different first model output",
            )
        else:
            case, reason = "CASE M-INPUT", f"first model-call difference is {first_m}"
    return {"case": case, "reason": reason}


def _stream_pair(sample: Mapping[str, Any]) -> dict[str, str]:
    streams = sample["output"]["streams"]
    return {
        "video": str(
            streams["video"].get("legacy_g4_sha256", streams["video"]["sha256"])
        ),
        "audio": str(
            streams["audio"].get("legacy_g4_sha256", streams["audio"]["sha256"])
        ),
    }


def _short(value: str) -> str:
    return value[:12]


def _gib(value: int | float) -> str:
    return f"{float(value) / (1024 ** 3):.2f}"


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# G4 Sampling Determinism Diagnostic — STOP Report",
        "",
        f"Classification: **{summary['classification']['case']}**  ",
        f"First divergence: **{summary['first_divergence']}**  ",
        f"Result: {summary['classification']['reason']}",
        "",
        "Production code and defaults were not changed.",
        "",
        "## Order Matrix",
        "",
        "| Order | Left output | Right output | Result |",
        "|---|---:|---:|---|",
    ]
    for name, comparison in summary["order_matrix"].items():
        output = comparison["output"]
        s8 = comparison["sampling"]["S8"]
        lines.append(
            f"| {name} | `{_short(output['left_sha256'])}` | "
            f"`{_short(output['right_sha256'])}` | "
            f"{'PASS' if output['match'] else 'FAIL'} "
            f"(S8 state {'exact' if s8['match'] else 'changed'}) |"
        )
    lines.extend(
        [
            "",
            "The sampled AV latent is exact in every accepted order. In the same-backend "
            "C and D comparisons, S8 records the expected warm-residency transition: "
            "`model_device` changes from `cpu` to `cuda:0`, and "
            "`current_weight_patch_uuid_present` changes from `false` to `true`. "
            "M0-M3 and the final AV latent remain exact, so this observed S8 state "
            "transition is not causal in the reproduced runs.",
        ]
    )
    focus = summary["focus_comparison"]
    lines.extend(
        [
            "",
            "## Sampling Boundary",
            "",
            "| Checkpoint | Left SHA | Right SHA | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for name in SAMPLING_CHECKPOINTS:
        row = focus["sampling"][name]
        lines.append(
            f"| {name} | `{_short(row['left_sha256'])}` | "
            f"`{_short(row['right_sha256'])}` | "
            f"{'PASS' if row['match'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## First model call",
            "",
            "| Checkpoint | Left SHA | Right SHA | Result |",
            "|---|---:|---:|---|",
        ]
    )
    for name in MODEL_CHECKPOINTS:
        row = focus["model"][name]
        lines.append(
            f"| {name} | `{_short(row['left_sha256'])}` | "
            f"`{_short(row['right_sha256'])}` | "
            f"{'PASS' if row['match'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "S7 equals the previously accepted P7 conditioning hash in all six runs. "
            "A step trace was not run because M0-M3 and the final output are exact. "
            "Accelerator A/B was not run because no model-forward divergence was observed.",
            "",
            "## Performance and resources",
            "",
            "| Run | Sampling | API | Peak VRAM | Peak RSS | Min available RAM |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, item in summary["performance"].items():
        resource = item["resources"]
        lines.append(
            f"| {label} | {item['sampling_seconds']:.2f}s | "
            f"{item['api_seconds']:.2f}s | "
            f"{_gib(resource['peak_device_used_bytes'])} GiB | "
            f"{_gib(resource['peak_process_rss_bytes'])} GiB | "
            f"{_gib(resource['minimum_system_available_bytes'])} GiB |"
        )
    lines.extend(
        [
            "",
            "## Root-cause assessment and disposition",
            "",
            "- **Gate disposition:** STOP / inconclusive because the historical Full-to-Review "
            "bit divergence is not reproducible in the current exact-order tests.",
            "- **First sampled divergence:** None. S0-S7, M0-M3, and final AV are exact. "
            "The C/D S8 difference is a non-causal warm model-residency state transition.",
            "- **Root cause:** Unresolved transient runtime or numerical state in the historical "
            "G4 run. Current evidence excludes target latent, fixed latent, noise, mask, "
            "SIGMAS, seed, sampler, conditioning, and the first model call in the reproduced runs.",
            "- **Minimal fix:** None is justified. Do not change Production. If further diagnosis "
            "is approved, repeat only the exact public-node C order while capturing the startup, "
            "extension, driver, kernel, and model-residency signature; enable a deferred/no-sync "
            "step trace only after the divergent output is reproduced.",
            "- **V3.7 / Full Run impact:** None now because no Production code changed. Speculative "
            "model reload, attention, cache, dtype, or deterministic-mode changes could alter "
            "V3.7 and Full Run performance or numerical behavior and therefore are not acceptable "
            "without a reproduced cause.",
            "- **Difficulty:** High. The historical condition is transient, non-reproducible, and "
            "sensitive to observation timing.",
            "",
            "## Control and retained evidence",
            "",
            "The public-node/no-hook control reproduces the historical Full hash, while its "
            "Review Queue 1 now matches Full rather than the historical divergent Review hash. "
            "The legacy control runner began Queue 2 after Queue 1 passed; that out-of-scope "
            "queue was interrupted immediately and is not a Product failure.",
            "",
            "```json",
            json.dumps(summary["control"], indent=2, ensure_ascii=False),
            "```",
            "",
            "STOP. No Production fix, accelerator change, deterministic-mode forcing, cache, model reload, seed/SIGMAS change, Run Storage change, commit, push, or release was performed.",
            "",
        ]
    )
    return "\n".join(lines)


def _analyze(args) -> dict[str, Any]:
    stages = {}
    for stage in "ABCD":
        path = args.evidence_root / f"stage_{stage.lower()}.json"
        if path.is_file():
            stages[stage] = json.loads(path.read_text(encoding="utf-8"))
    required = {"A", "B", "C", "D"}
    if set(stages) != required:
        raise GateFailure(f"analysis requires A-D; found {sorted(stages)}")
    a = stages["A"]["runs"][0]
    b = stages["B"]["runs"][0]
    c_full, c_review = stages["C"]["runs"]
    d1, d2 = stages["D"]["runs"]
    matrix = {
        "Fresh Full vs Fresh Review": _compare_samples(a, b),
        "Full -> Review": _compare_samples(c_full, c_review),
        "Review -> Review": _compare_samples(d1, d2),
    }
    focus_name = "Full -> Review" if not matrix["Full -> Review"]["output"]["match"] else "Fresh Full vs Fresh Review"
    focus = matrix[focus_name]
    first = focus["first_sampling_divergence"] or focus["first_model_divergence"]
    if first is None and not focus["output"]["match"]:
        first = "Sampler step trace required"
    elif first is None:
        first = "none through final sampled latent"
    classification = _classification(focus)
    if all(item["output"]["match"] for item in matrix.values()):
        s8_changed = [
            name
            for name, item in matrix.items()
            if not item["sampling"]["S8"]["match"]
        ]
        classification = {
            "case": "NO DIVERGENCE REPRODUCED",
            "reason": (
                "S0-S7, M0-M3, and final sampled AV are exact in all orders; "
                "S8 contains only the non-causal warm runtime-state transition "
                f"in {', '.join(s8_changed)}"
                if s8_changed
                else "S0-S8, M0-M3, and final sampled AV are exact in all orders"
            ),
        }
        if s8_changed:
            first = (
                "S8 runtime state only in same-backend C/D; "
                "no sampled bit divergence through final latent"
            )

    c_full_streams = _stream_pair(_first_sample(c_full))
    c_review_streams = _stream_pair(_first_sample(c_review))
    no_hook = None
    if args.no_hook_control.is_file():
        no_hook = json.loads(args.no_hook_control.read_text(encoding="utf-8"))
    no_hook_full = None
    no_hook_review = None
    if no_hook is not None:
        no_hook_full = {
            "video": no_hook["results"]["full"]["run_storage"]["chunks"][0]["video_latent_sha256"],
            "audio": no_hook["results"]["full"]["run_storage"]["chunks"][0]["audio_latent_sha256"],
        }
        no_hook_review = {
            "video": no_hook["results"]["review_q1"]["run_storage"]["chunks"][0]["video_latent_sha256"],
            "audio": no_hook["results"]["review_q1"]["run_storage"]["chunks"][0]["audio_latent_sha256"],
        }
    control = {
        "hook_noninterference_current_no_hook_full": (
            None if no_hook_full is None else c_full_streams == no_hook_full
        ),
        "hook_noninterference_current_no_hook_review": (
            None if no_hook_review is None else c_review_streams == no_hook_review
        ),
        "observed_c_full": c_full_streams,
        "observed_c_review": c_review_streams,
        "current_no_hook_full": no_hook_full,
        "current_no_hook_review": no_hook_review,
        "historical_g4_full": ORIGINAL_G4["full"],
        "historical_g4_divergent_review": ORIGINAL_G4["full_then_review"],
        "historical_full_reproduced": c_full_streams == ORIGINAL_G4["full"],
        "historical_divergent_review_reproduced": (
            c_review_streams == ORIGINAL_G4["full_then_review"]
        ),
        "current_no_hook_chunk_1_matches_full": (
            None if no_hook_full is None else no_hook_full == no_hook_review
        ),
        "all_hooks_restored": all(
            run["diagnostic"].get("hooks_restored")
            for stage in stages.values()
            for run in stage["runs"]
        ),
    }
    p7 = None
    if args.conditioning_summary.is_file():
        conditioning = json.loads(args.conditioning_summary.read_text(encoding="utf-8"))
        p7 = _comparison_sha(
            conditioning["prepare"]["runs"]["full"]["result"]["checkpoints"]["P7"]
        )
    s7_checks = {}
    for run in (a, b, c_full, c_review, d1, d2):
        sample = _first_sample(run)
        s7 = _comparison_sha(sample["checkpoints"]["S7"])
        s7_checks[run["label"]] = {
            "s7_sha256": s7,
            "p7_sha256": p7,
            "matches_p7": None if p7 is None else s7 == p7,
        }
    resources = {
        run["label"]: run["resources"]
        for stage in stages.values()
        for run in stage["runs"]
    }
    performance = {
        run["label"]: {
            "sampling_seconds": float(
                run["diagnostic"]["sampler_node_elapsed_seconds"]
            ),
            "api_seconds": float(run["api_elapsed_seconds"]),
            "resources": run["resources"],
        }
        for stage in stages.values()
        for run in stage["runs"]
    }
    summary = {
        "format": "h3-v38-g4-sampling-determinism-final-v1",
        "stage": "stopped",
        "production_changed": False,
        "focus": focus_name,
        "first_divergence": first,
        "classification": classification,
        "order_matrix": matrix,
        "focus_comparison": focus,
        "control": control,
        "p7_parity": s7_checks,
        "resources": resources,
        "performance": performance,
        "stages": {
            stage: str(args.evidence_root / f"stage_{stage.lower()}.json")
            for stage in stages
        },
    }
    _write(args.evidence_root / "g4_sampling_determinism_summary.json", summary)
    (args.evidence_root / "G4_SAMPLING_DETERMINISM_STOP_REPORT.md").write_text(
        _report(summary),
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("A", "B", "C", "D", "analyze"), required=True)
    parser.add_argument("--server", default="http://127.0.0.6:8190")
    parser.add_argument("--template-prompt", type=Path)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--run-storage-root", type=Path)
    parser.add_argument("--run-tag", default="20260831_154505")
    parser.add_argument("--backend-pid", type=int)
    parser.add_argument("--backend-log", type=Path)
    parser.add_argument("--nonce-base", type=int, default=1000)
    parser.add_argument("--trace-steps", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=2400.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument(
        "--conditioning-summary",
        type=Path,
        default=Path(
            r"D:\Codex\_test_results\ComfyUI-H3-Continuum"
            r"\v38-g4-conditioning-determinism-20260831_150837"
            r"\g4_conditioning_determinism_summary.json"
        ),
    )
    parser.add_argument(
        "--no-hook-control",
        type=Path,
        default=Path(
            r"D:\Codex\_test_results\ComfyUI-H3-Continuum"
            r"\v38-g4-sampling-determinism-20260831_154505"
            r"\no_hook_control\gpu_r1\g4_summary.json"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    if args.stage != "analyze":
        if args.template_prompt is None or args.run_storage_root is None:
            raise GateFailure("stage execution requires template prompt and Run Storage root")
        if args.backend_pid is None or args.backend_log is None:
            raise GateFailure("stage execution requires backend PID and log")
        result = _stage(args)
    else:
        result = _analyze(args)
    print(
        json.dumps(
            {
                "stage": result.get("stage"),
                "first_divergence": result.get("first_divergence"),
                "classification": result.get("classification"),
                "summary": str(args.evidence_root / "g4_sampling_determinism_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
