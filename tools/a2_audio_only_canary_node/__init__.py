"""Temporary GPU-canary node for the internal A2 Audio Only path."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from types import SimpleNamespace
from typing import Any

import torch


def _single(name: str, value: Any) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return value[0]


def _active_modules() -> SimpleNamespace:
    import nodes as comfy_nodes

    sampler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("H3ContinuumSamplerV38")
    if sampler_class is None:
        raise RuntimeError("H3ContinuumSamplerV38 is not loaded")
    module_name = sampler_class.__module__
    base_name = (
        module_name.split(".v3.", 1)[0]
        if ".v3." in module_name
        else module_name.rsplit(".", 2)[0]
    )
    return SimpleNamespace(
        refine_schedule=importlib.import_module(f"{base_name}.v3.refine_schedule"),
        refine_target=importlib.import_module(f"{base_name}.v3.refine_target"),
        targeted_second_pass=importlib.import_module(
            f"{base_name}.v3.targeted_second_pass"
        ),
    )


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


class H3A2AudioOnlyCanary:
    """Non-public bridge that records the A2 GPU acceptance contract."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "video_latents": ("LATENT",),
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "refine_seed": (
                    "INT",
                    {"default": 3802601, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "schedule_mode": (["Tail", "External"], {"default": "Tail"}),
                "tail_evaluations": (
                    "INT",
                    {"default": 6, "min": 1, "max": 100},
                ),
            },
            "optional": {
                "refine_context": ("H3_CONTINUUM_REFINE_CONTEXT",),
                "video_vae": ("VAE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "STRING")
    RETURN_NAMES = (
        "video_latents",
        "refined_audio_latents",
        "updated_assembly_plan",
        "canary_status",
    )
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "run"
    CATEGORY = "H3 Continuum Test/A2"

    def run(
        self,
        model,
        clip,
        sampler,
        sigmas,
        video_latents,
        audio_latents,
        assembly_plan,
        refine_seed,
        schedule_mode,
        tail_evaluations,
        refine_context=None,
        video_vae=None,
    ):
        modules = _active_modules()
        source_sigmas = _single("sigmas", sigmas)
        mode = str(_single("schedule_mode", schedule_mode))
        if mode == "Tail":
            schedule = modules.refine_schedule.make_tail_schedule(
                source_sigmas,
                evaluation_count=int(
                    _single("tail_evaluations", tail_evaluations)
                ),
            )
        else:
            schedule = None

        input_video_samples = [
            _samples(item, name=f"video_latents[{index}]")
            for index, item in enumerate(video_latents)
        ]
        input_audio_samples = [
            _samples(item, name=f"audio_latents[{index}]")
            for index, item in enumerate(audio_latents)
        ]
        input_video_hashes = [_tensor_sha256(item) for item in input_video_samples]
        input_audio_hashes = [_tensor_sha256(item) for item in input_audio_samples]

        videos, audios, plan, status = (
            modules.targeted_second_pass.run_targeted_second_pass_groups(
                model=_single("model", model),
                clip=_single("clip", clip),
                sampler=_single("sampler", sampler),
                sigmas=source_sigmas,
                video_latents=video_latents,
                audio_latents=audio_latents,
                assembly_plan=_single("assembly_plan", assembly_plan),
                refine_seed=int(_single("refine_seed", refine_seed)),
                refine_target=modules.refine_target.MODE_AUDIO_ONLY,
                refine_context=(
                    None
                    if refine_context is None
                    else _single("refine_context", refine_context)
                ),
                video_vae=(
                    None if video_vae is None else _single("video_vae", video_vae)
                ),
                refine_schedule=schedule,
                enable_preview=False,
            )
        )
        output_video_samples = [
            _samples(item, name=f"output_video_latents[{index}]")
            for index, item in enumerate(videos)
        ]
        output_audio_samples = [
            _samples(item, name=f"output_audio_latents[{index}]")
            for index, item in enumerate(audios)
        ]
        contract = plan["second_pass_contract"]
        summary = {
            "format": "h3-continuum-a2-audio-only-gpu-canary-v1",
            "target": contract["refine_target_contract"],
            "execution": contract["refine_execution_contract"],
            "second_pass_contract_version": contract.get("version"),
            "physical_groups": [
                {
                    "group_id": group.get("group_id"),
                    "logical_chunks": group.get("logical_chunks"),
                    "terminal_merged": bool(group.get("terminal_merged")),
                    "source_latent_t": group.get("source_latent_t"),
                    "source_audio_shape": group.get("source_audio_shape"),
                }
                for group in contract.get("physical_groups", [])
            ],
            "refine_group_seeds": contract.get("refine_group_seeds"),
            "schedule_mode": contract["refine_schedule"]["mode"],
            "schedule_evaluations": contract["refine_schedule"][
                "evaluation_count"
            ],
            "schedule_sigma_hash": contract["refine_schedule"]["sigma_hash"],
            "video_object_identity": [
                output is source
                for output, source in zip(videos, video_latents, strict=True)
            ],
            "video_tensor_identity": [
                output is source
                for output, source in zip(
                    output_video_samples,
                    input_video_samples,
                    strict=True,
                )
            ],
            "video_input_sha256": input_video_hashes,
            "video_output_sha256": [
                _tensor_sha256(item) for item in output_video_samples
            ],
            "audio_input_sha256": input_audio_hashes,
            "audio_output_sha256": [
                _tensor_sha256(item) for item in output_audio_samples
            ],
            "audio_input_shapes": [list(item.shape) for item in input_audio_samples],
            "audio_output_shapes": [list(item.shape) for item in output_audio_samples],
            "video_output_policy": contract.get("video_output"),
            "video_sampling_policy": contract.get("video_sampling"),
            "audio_output_policy": contract.get("audio_output"),
            "audio_sampling_policy": contract.get("audio_sampling"),
        }
        canary_status = status + "\nA2_CANARY_JSON=" + json.dumps(
            summary,
            sort_keys=True,
            separators=(",", ":"),
        )
        return videos, audios, plan, canary_status


NODE_CLASS_MAPPINGS = {"H3A2AudioOnlyCanary": H3A2AudioOnlyCanary}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3A2AudioOnlyCanary": "H3 A2 Audio Only Canary (Temporary)"
}

