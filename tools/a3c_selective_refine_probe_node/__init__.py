"""Temporary read-only probe for the A3c Selective Refine GPU Matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

import torch


def _single(name: str, value: Any) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return value[0]


def _samples(latent: Any, *, name: str) -> torch.Tensor:
    if not isinstance(latent, dict) or not torch.is_tensor(latent.get("samples")):
        raise ValueError(f"{name} must be a LATENT dictionary with Tensor samples")
    return latent["samples"]


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(int(item) for item in tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _plan_identity(plan: Any, *, omit_target_identity: bool) -> dict[str, Any]:
    normalized = copy.deepcopy(_json_safe(plan))
    if omit_target_identity and isinstance(normalized, dict):
        contract = normalized.get("second_pass_contract")
        if isinstance(contract, dict):
            contract.pop("refine_target_contract", None)
            contract.pop("refine_execution_contract", None)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "normalized": normalized,
    }


class H3A3cSelectiveRefineProbe:
    """Compare first-pass and public Second Pass outputs without changing them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_video_latents": ("LATENT",),
                "input_audio_latents": ("LATENT",),
                "output_video_latents": ("LATENT",),
                "output_audio_latents": ("LATENT",),
                "input_assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "output_assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "refine_status": ("STRING", {"forceInput": True}),
                "target_label": (
                    ["Legacy V3.5", "Video Only", "Audio Only", "Video + Audio"],
                    {"default": "Video Only"},
                ),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("probe_report",)
    FUNCTION = "run"
    CATEGORY = "H3 Continuum Test/A3c"

    def run(
        self,
        input_video_latents,
        input_audio_latents,
        output_video_latents,
        output_audio_latents,
        input_assembly_plan,
        output_assembly_plan,
        refine_status,
        target_label,
    ):
        if len(input_video_latents) != len(output_video_latents):
            raise ValueError("Video physical-group count changed")
        if len(input_audio_latents) != len(output_audio_latents):
            raise ValueError("Audio physical-group count changed")
        input_v = [
            _samples(item, name=f"input_video_latents[{index}]")
            for index, item in enumerate(input_video_latents)
        ]
        input_a = [
            _samples(item, name=f"input_audio_latents[{index}]")
            for index, item in enumerate(input_audio_latents)
        ]
        output_v = [
            _samples(item, name=f"output_video_latents[{index}]")
            for index, item in enumerate(output_video_latents)
        ]
        output_a = [
            _samples(item, name=f"output_audio_latents[{index}]")
            for index, item in enumerate(output_audio_latents)
        ]
        output_plan = _single("output_assembly_plan", output_assembly_plan)
        contract = output_plan.get("second_pass_contract") or {}
        status = str(_single("refine_status", refine_status))
        target_contract = contract.get("refine_target_contract") or {}
        schedule = contract.get("refine_schedule") or {}
        physical_groups = [
            {
                "group_id": group.get("group_id"),
                "logical_chunks": group.get("logical_chunks"),
                "terminal_merged": bool(group.get("terminal_merged")),
                "source_latent_t": group.get("source_latent_t"),
                "source_audio_shape": group.get("source_audio_shape"),
            }
            for group in contract.get("physical_groups", [])
        ]
        summary = {
            "format": "h3-continuum-a3c-selective-refine-probe-v1",
            "target_label": str(_single("target_label", target_label)),
            "target_mode": target_contract.get("mode", "legacy_video_only"),
            "seed_namespace": target_contract.get(
                "seed_namespace", "h3-continuum-refine-v1"
            ),
            "second_pass_contract_version": contract.get("version"),
            "physical_groups": physical_groups,
            "refine_group_seeds": contract.get("refine_group_seeds"),
            "schedule_mode": schedule.get("mode"),
            "schedule_evaluations": schedule.get("evaluation_count"),
            "schedule_sigma_hash": schedule.get("sigma_hash"),
            "sampling_passes_reported": status.count("sampling_passes=1"),
            "input_video_object_identity": [
                output is source
                for output, source in zip(
                    output_video_latents,
                    input_video_latents,
                    strict=True,
                )
            ],
            "input_audio_object_identity": [
                output is source
                for output, source in zip(
                    output_audio_latents,
                    input_audio_latents,
                    strict=True,
                )
            ],
            "input_video_tensor_identity": [
                output is source
                for output, source in zip(output_v, input_v, strict=True)
            ],
            "input_audio_tensor_identity": [
                output is source
                for output, source in zip(output_a, input_a, strict=True)
            ],
            "input_video_sha256": [_tensor_sha256(item) for item in input_v],
            "output_video_sha256": [_tensor_sha256(item) for item in output_v],
            "input_audio_sha256": [_tensor_sha256(item) for item in input_a],
            "output_audio_sha256": [_tensor_sha256(item) for item in output_a],
            "input_video_shapes": [list(item.shape) for item in input_v],
            "output_video_shapes": [list(item.shape) for item in output_v],
            "input_audio_shapes": [list(item.shape) for item in input_a],
            "output_audio_shapes": [list(item.shape) for item in output_a],
            "output_video_finite": [
                bool(torch.isfinite(item.float()).all().item()) for item in output_v
            ],
            "output_audio_finite": [
                bool(torch.isfinite(item.float()).all().item()) for item in output_a
            ],
            "input_plan": _plan_identity(
                _single("input_assembly_plan", input_assembly_plan),
                omit_target_identity=False,
            ),
            "output_plan": _plan_identity(
                output_plan,
                omit_target_identity=False,
            ),
            "output_plan_without_target_identity": _plan_identity(
                output_plan,
                omit_target_identity=True,
            ),
        }
        report = status + "\nA3C_PROBE_JSON=" + json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return (report,)


class H3A3cVideoOnlyParityProbe:
    """Compare legacy and Selective Video Only outputs from one First Pass."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_audio_latents": ("LATENT",),
                "legacy_video_latents": ("LATENT",),
                "legacy_audio_latents": ("LATENT",),
                "legacy_assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "legacy_status": ("STRING", {"forceInput": True}),
                "selective_video_latents": ("LATENT",),
                "selective_audio_latents": ("LATENT",),
                "selective_assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "selective_status": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("parity_report",)
    FUNCTION = "run"
    CATEGORY = "H3 Continuum Test/A3c"

    def run(
        self,
        input_audio_latents,
        legacy_video_latents,
        legacy_audio_latents,
        legacy_assembly_plan,
        legacy_status,
        selective_video_latents,
        selective_audio_latents,
        selective_assembly_plan,
        selective_status,
    ):
        counts = {
            len(input_audio_latents),
            len(legacy_video_latents),
            len(legacy_audio_latents),
            len(selective_video_latents),
            len(selective_audio_latents),
        }
        if len(counts) != 1:
            raise ValueError("Video Only parity physical-group counts differ")
        input_a = [
            _samples(item, name=f"input_audio_latents[{index}]")
            for index, item in enumerate(input_audio_latents)
        ]
        legacy_v = [
            _samples(item, name=f"legacy_video_latents[{index}]")
            for index, item in enumerate(legacy_video_latents)
        ]
        legacy_a = [
            _samples(item, name=f"legacy_audio_latents[{index}]")
            for index, item in enumerate(legacy_audio_latents)
        ]
        selective_v = [
            _samples(item, name=f"selective_video_latents[{index}]")
            for index, item in enumerate(selective_video_latents)
        ]
        selective_a = [
            _samples(item, name=f"selective_audio_latents[{index}]")
            for index, item in enumerate(selective_audio_latents)
        ]
        legacy_plan = _single("legacy_assembly_plan", legacy_assembly_plan)
        selective_plan = _single("selective_assembly_plan", selective_assembly_plan)
        legacy_contract = legacy_plan.get("second_pass_contract") or {}
        selective_contract = selective_plan.get("second_pass_contract") or {}
        legacy_text = str(_single("legacy_status", legacy_status))
        selective_text = str(_single("selective_status", selective_status))
        summary = {
            "format": "h3-continuum-a3c-video-only-parity-v1",
            "physical_group_count": len(legacy_v),
            "legacy_video_sha256": [_tensor_sha256(item) for item in legacy_v],
            "selective_video_sha256": [
                _tensor_sha256(item) for item in selective_v
            ],
            "legacy_audio_sha256": [_tensor_sha256(item) for item in legacy_a],
            "selective_audio_sha256": [
                _tensor_sha256(item) for item in selective_a
            ],
            "legacy_video_shapes": [list(item.shape) for item in legacy_v],
            "selective_video_shapes": [list(item.shape) for item in selective_v],
            "legacy_audio_shapes": [list(item.shape) for item in legacy_a],
            "selective_audio_shapes": [list(item.shape) for item in selective_a],
            "legacy_video_finite": [
                bool(torch.isfinite(item.float()).all().item()) for item in legacy_v
            ],
            "selective_video_finite": [
                bool(torch.isfinite(item.float()).all().item())
                for item in selective_v
            ],
            "legacy_audio_finite": [
                bool(torch.isfinite(item.float()).all().item()) for item in legacy_a
            ],
            "selective_audio_finite": [
                bool(torch.isfinite(item.float()).all().item())
                for item in selective_a
            ],
            "legacy_audio_object_identity": [
                output is source
                for output, source in zip(
                    legacy_audio_latents,
                    input_audio_latents,
                    strict=True,
                )
            ],
            "selective_audio_object_identity": [
                output is source
                for output, source in zip(
                    selective_audio_latents,
                    input_audio_latents,
                    strict=True,
                )
            ],
            "legacy_audio_tensor_identity": [
                output is source
                for output, source in zip(legacy_a, input_a, strict=True)
            ],
            "selective_audio_tensor_identity": [
                output is source
                for output, source in zip(selective_a, input_a, strict=True)
            ],
            "legacy_refine_group_seeds": legacy_contract.get(
                "refine_group_seeds"
            ),
            "selective_refine_group_seeds": selective_contract.get(
                "refine_group_seeds"
            ),
            "legacy_schedule_sigma_hash": (
                legacy_contract.get("refine_schedule") or {}
            ).get("sigma_hash"),
            "selective_schedule_sigma_hash": (
                selective_contract.get("refine_schedule") or {}
            ).get("sigma_hash"),
            "legacy_sampling_passes_reported": legacy_text.count(
                "sampling_passes=1"
            ),
            "selective_sampling_passes_reported": selective_text.count(
                "sampling_passes=1"
            ),
            "legacy_plan_without_target_identity": _plan_identity(
                legacy_plan,
                omit_target_identity=True,
            )["sha256"],
            "selective_plan_without_target_identity": _plan_identity(
                selective_plan,
                omit_target_identity=True,
            )["sha256"],
            "selective_target_mode": (
                selective_contract.get("refine_target_contract") or {}
            ).get("mode"),
            "selective_seed_namespace": (
                selective_contract.get("refine_target_contract") or {}
            ).get("seed_namespace"),
        }
        report = "A3C_PARITY_JSON=" + json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return (report,)


NODE_CLASS_MAPPINGS = {
    "H3A3cSelectiveRefineProbe": H3A3cSelectiveRefineProbe,
    "H3A3cVideoOnlyParityProbe": H3A3cVideoOnlyParityProbe,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3A3cSelectiveRefineProbe": "H3 A3c Selective Refine Probe (Temporary)",
    "H3A3cVideoOnlyParityProbe": "H3 A3c Video Only Parity Probe (Temporary)",
}
