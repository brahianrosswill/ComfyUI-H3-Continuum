"""Observation-only Sampling boundary collector for the V3.8 G4 Gate.

The collector is loaded only on the dedicated diagnostic backend.  It wraps the
active Production V3.8 sampler for one node execution, returns the unchanged six
Production outputs, and adds a seventh JSON evidence output.  Every monkeypatch
is restored in ``finally`` blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any
import uuid

import torch


CHECKPOINTS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
MODEL_CHECKPOINTS = ("M0", "M1", "M2", "M3")


def _type_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _callable_name(value: Any) -> str:
    return (
        f"{getattr(value, '__module__', '')}."
        f"{getattr(value, '__qualname__', type(value).__qualname__)}"
    )


def _tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    if bool(getattr(tensor, "is_nested", False)):
        return _nested_fingerprint(tensor)
    device_before = str(tensor.device)
    value = tensor.detach().contiguous().to("cpu")
    shape = [int(part) for part in value.shape]
    dtype = str(value.dtype)
    raw = value.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)
    legacy_digest = hashlib.sha256()
    legacy_digest.update(dtype.encode("ascii"))
    legacy_digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
    legacy_digest.update(raw)
    return {
        "kind": "tensor",
        "shape": shape,
        "dtype": dtype,
        "device_before_hash": device_before,
        "numel": int(value.numel()),
        "raw_bytes": len(raw),
        "sha256": digest.hexdigest(),
        "legacy_g4_sha256": legacy_digest.hexdigest(),
    }


def _nested_fingerprint(value: Any) -> dict[str, Any]:
    parts = list(value.unbind())
    fingerprints = [_tensor_fingerprint(part) for part in parts]
    payload = json.dumps(
        [item["sha256"] for item in fingerprints],
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "kind": "nested_tensor",
        "parts": fingerprints,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _canonicalize(value: Any, unsupported: set[str]) -> Any:
    if torch.is_tensor(value):
        fingerprint = _tensor_fingerprint(value)
        fingerprint.pop("legacy_g4_sha256", None)
        return {"__tensor__": fingerprint}
    if hasattr(value, "unbind") and not isinstance(value, (str, bytes)):
        try:
            parts = list(value.unbind())
        except Exception:
            parts = None
        if parts is not None and parts and all(torch.is_tensor(part) for part in parts):
            fingerprint = _nested_fingerprint(value)
            for part in fingerprint.get("parts", []):
                part.pop("legacy_g4_sha256", None)
            return {"__nested_tensor__": fingerprint}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {
            "__bytes__": {
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        }
    if isinstance(value, (torch.dtype, torch.device, uuid.UUID, Enum)):
        return {"__type__": _type_name(value), "value": str(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": _type_name(value),
            "value": _canonicalize(asdict(value), unsupported),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(value[key], unsupported)
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item, unsupported) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_canonicalize(item, unsupported) for item in value]
        return sorted(
            encoded,
            key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")),
        )
    if callable(value):
        return {"__callable__": _callable_name(value)}
    unsupported.add(_type_name(value))
    return {"__opaque_type__": _type_name(value)}


def _structured_checkpoint(value: Any) -> dict[str, Any]:
    unsupported: set[str] = set()
    canonical = _canonicalize(value, unsupported)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "kind": "structured",
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "canonical_bytes": len(encoded),
        "unsupported_types": sorted(unsupported),
        "payload": canonical,
    }


def _value_checkpoint(value: Any) -> dict[str, Any]:
    if torch.is_tensor(value):
        return _tensor_fingerprint(value)
    if hasattr(value, "unbind") and not isinstance(value, (str, bytes)):
        try:
            parts = list(value.unbind())
        except Exception:
            parts = None
        if parts is not None and parts and all(torch.is_tensor(part) for part in parts):
            return _nested_fingerprint(value)
    return _structured_checkpoint(value)


def _av_checkpoint(value: Any, state_module: Any) -> dict[str, Any]:
    if isinstance(value, Mapping) and "samples" in value:
        video, audio = state_module.extract_av_streams(dict(value))
    else:
        samples = value
        parts = list(samples.unbind()) if hasattr(samples, "unbind") else list(samples)
        if len(parts) < 2:
            raise RuntimeError("expected MiniMax H3 Video/Audio NestedTensor")
        video, audio = parts[0], parts[1]
    streams = {
        "video": _tensor_fingerprint(video),
        "audio": _tensor_fingerprint(audio),
    }
    combined = json.dumps(
        {name: item["sha256"] for name, item in streams.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "kind": "h3_av",
        "streams": streams,
        "sha256": hashlib.sha256(combined).hexdigest(),
    }


def _runtime_descriptor(value: Any, depth: int = 0) -> Any:
    if depth > 7:
        return {"__truncated_type__": _type_name(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device, uuid.UUID, Enum)):
        return {"type": _type_name(value), "value": str(value)}
    if torch.is_tensor(value):
        return {
            "type": "tensor",
            "shape": [int(part) for part in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _runtime_descriptor(item, depth + 1)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_runtime_descriptor(item, depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_runtime_descriptor(item, depth + 1) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if callable(value):
        return {"callable": _callable_name(value)}
    return {"type": _type_name(value)}


def _wrapper_state(patcher: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for wrapper_type, keyed in sorted(
        dict(getattr(patcher, "wrappers", {}) or {}).items(),
        key=lambda item: str(item[0]),
    ):
        output[str(wrapper_type)] = {
            str(key): [_callable_name(item) for item in list(items or [])]
            for key, items in sorted(
                dict(keyed or {}).items(), key=lambda item: str(item[0])
            )
        }
    return output


def _patch_value_summary(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_patch_value_summary(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _patch_value_summary(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if callable(value):
        return {"callable": _callable_name(value)}
    return {"type": _type_name(value)}


def _model_runtime_state(patcher: Any) -> dict[str, Any]:
    model = getattr(patcher, "model", None)
    current_uuid = getattr(model, "current_weight_patches_uuid", None)
    patches = dict(getattr(patcher, "patches", {}) or {})
    model_options = dict(getattr(patcher, "model_options", {}) or {})
    dtype = None
    try:
        dtype = str(patcher.model_dtype())
    except Exception:
        dtype = None
    inference_dtype = None
    try:
        inference_dtype = str(model.get_dtype_inference()) if model is not None else None
    except Exception:
        inference_dtype = None
    observed = {
        "patcher_type": _type_name(patcher),
        "model_type": None if model is None else _type_name(model),
        "diffusion_model_type": (
            None
            if model is None or getattr(model, "diffusion_model", None) is None
            else _type_name(model.diffusion_model)
        ),
        "patch_uuid": str(getattr(patcher, "patches_uuid", "")),
        "current_weight_patch_uuid": (
            None if current_uuid is None else str(current_uuid)
        ),
        "load_device": str(getattr(patcher, "load_device", "")),
        "offload_device": str(getattr(patcher, "offload_device", "")),
        "model_device": None if model is None else str(getattr(model, "device", "")),
        "model_dtype": dtype,
        "inference_dtype": inference_dtype,
        "hook_mode": str(getattr(patcher, "hook_mode", "")),
        "current_hook_group_type": (
            None
            if getattr(patcher, "current_hook_group", None) is None
            else _type_name(patcher.current_hook_group)
        ),
        "wrappers": _wrapper_state(patcher),
        "patches": {
            str(key): _patch_value_summary(value)
            for key, value in sorted(patches.items(), key=lambda item: str(item[0]))
        },
        "object_patches": {
            str(key): _type_name(value)
            for key, value in sorted(
                dict(getattr(patcher, "object_patches", {}) or {}).items(),
                key=lambda item: str(item[0]),
            )
        },
        "model_options": _runtime_descriptor(model_options),
        "torch": {
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
    }
    semantic = dict(observed)
    semantic.pop("patch_uuid", None)
    semantic.pop("current_weight_patch_uuid", None)
    semantic["patch_uuid_present"] = bool(observed["patch_uuid"])
    semantic["current_weight_patch_uuid_present"] = (
        observed["current_weight_patch_uuid"] is not None
    )
    checkpoint = _structured_checkpoint(semantic)
    checkpoint["observed_state"] = observed
    return checkpoint


def _sampler_signature(sampler: Any) -> dict[str, Any]:
    value = {
        "sampler_type": _type_name(sampler),
        "sampler_function": (
            None
            if getattr(sampler, "sampler_function", None) is None
            else _callable_name(sampler.sampler_function)
        ),
        "extra_options": _runtime_descriptor(
            dict(getattr(sampler, "extra_options", {}) or {})
        ),
        "inpaint_options": _runtime_descriptor(
            dict(getattr(sampler, "inpaint_options", {}) or {})
        ),
    }
    return _structured_checkpoint(value)


def _resolve_modules() -> SimpleNamespace:
    import nodes as comfy_nodes

    sampler_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("H3ContinuumSamplerV38")
    if sampler_class is None:
        raise RuntimeError("H3ContinuumSamplerV38 is not loaded")
    owner = sys.modules.get(sampler_class.__module__)
    source = getattr(owner, "__file__", None)
    if source is None:
        raise RuntimeError("cannot resolve active H3 Continuum package")
    module_name = sampler_class.__module__
    base_name = (
        module_name.split(".v3.", 1)[0]
        if ".v3." in module_name
        else module_name.rsplit(".", 2)[0]
    )
    return SimpleNamespace(
        sampler_class=sampler_class,
        production_root=str(Path(source).resolve().parent.parent),
        sequence=importlib.import_module(f"{base_name}.v2.sequence"),
        state=importlib.import_module(f"{base_name}.state"),
    )


class SamplingCollector:
    def __init__(self, modules: SimpleNamespace, *, trace_steps: bool):
        self.modules = modules
        self.trace_steps = bool(trace_steps)
        self.records: list[dict[str, Any]] = []
        self.hooks_restored = True

    def _capture_model_payload(
        self,
        record: dict[str, Any],
        *,
        x: Any,
        t: Any,
        c_concat: Any,
        c_crossattn: Any,
        control: Any,
        transformer_options: Any,
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        record["model"] = {
            "M0": _value_checkpoint(x),
            "M1": _value_checkpoint(t),
            "M2": _structured_checkpoint(
                {
                    "c_concat": c_concat,
                    "c_crossattn": c_crossattn,
                    "control": control,
                    "transformer_options": transformer_options,
                    "kwargs": dict(kwargs),
                }
            ),
            "M3": _value_checkpoint(output),
        }

    def wrap(self, original_sample):
        def observed_sample_chunk(**kwargs):
            import comfy.model_base
            import comfy.sample
            import comfy.samplers

            record: dict[str, Any] = {
                "physical_call_index": len(self.records) + 1,
                "seed": int(kwargs["seed"]),
                "checkpoints": {},
                "model": {},
                "steps": [],
            }
            checkpoints = record["checkpoints"]
            checkpoints["S0"] = _av_checkpoint(kwargs["latent"], self.modules.state)
            checkpoints["S3"] = _value_checkpoint(kwargs["latent"].get("noise_mask"))
            checkpoints["S4"] = _tensor_fingerprint(kwargs["sigmas"])
            checkpoints["S5"] = _structured_checkpoint(int(kwargs["seed"]))
            checkpoints["S6"] = _sampler_signature(kwargs["sampler"])
            checkpoints["S7"] = _structured_checkpoint(kwargs["conditioning"])
            checkpoints["S8"] = _model_runtime_state(kwargs["model"])

            original_fix = comfy.sample.fix_empty_latent_channels
            original_noise = comfy.sample.prepare_noise
            original_apply = comfy.model_base.BaseModel._apply_model
            original_guider_sample = comfy.samplers.CFGGuider.sample
            fixed_seen = False
            noise_seen = False
            model_seen = False
            pending_model_payload = None

            def fix_empty(*args, **inner_kwargs):
                nonlocal fixed_seen
                result = original_fix(*args, **inner_kwargs)
                if not fixed_seen:
                    fixed_seen = True
                    checkpoints["S1"] = _av_checkpoint(result, self.modules.state)
                return result

            def prepare_noise(*args, **inner_kwargs):
                nonlocal noise_seen
                result = original_noise(*args, **inner_kwargs)
                if not noise_seen:
                    noise_seen = True
                    checkpoints["S2"] = _av_checkpoint(result, self.modules.state)
                return result

            def apply_model(
                model_self,
                x,
                t,
                c_concat=None,
                c_crossattn=None,
                control=None,
                transformer_options={},
                **inner_kwargs,
            ):
                nonlocal model_seen, pending_model_payload
                result = original_apply(
                    model_self,
                    x,
                    t,
                    c_concat,
                    c_crossattn,
                    control,
                    transformer_options,
                    **inner_kwargs,
                )
                if not model_seen and model_self.__class__.__name__ == "MiniMaxH3":
                    model_seen = True
                    # Retain references only.  Hashing CUDA tensors here forces a
                    # host synchronization between sampler steps and masks the
                    # ordering defect this Gate is intended to observe.
                    pending_model_payload = {
                        "x": x,
                        "t": t,
                        "c_concat": c_concat,
                        "c_crossattn": c_crossattn,
                        "control": control,
                        "transformer_options": transformer_options,
                        "kwargs": dict(inner_kwargs),
                        "output": result,
                    }
                return result

            def guider_sample(guider_self, *args, **inner_kwargs):
                existing_callback = inner_kwargs.get("callback")

                def chained_callback(step, x0, x, total_steps):
                    if existing_callback is not None:
                        existing_callback(step, x0, x, total_steps)
                    record["steps"].append(
                        {
                            "step": int(step),
                            "total_steps": int(total_steps),
                            "x0": _value_checkpoint(x0),
                            "x": _value_checkpoint(x),
                        }
                    )

                if self.trace_steps:
                    inner_kwargs["callback"] = chained_callback
                    record["step_callback_chained"] = existing_callback is not None
                return original_guider_sample(guider_self, *args, **inner_kwargs)

            comfy.sample.fix_empty_latent_channels = fix_empty
            comfy.sample.prepare_noise = prepare_noise
            comfy.model_base.BaseModel._apply_model = apply_model
            if self.trace_steps:
                comfy.samplers.CFGGuider.sample = guider_sample
            started = time.perf_counter()
            self.records.append(record)
            try:
                result = original_sample(**kwargs)
                if pending_model_payload is not None:
                    self._capture_model_payload(record, **pending_model_payload)
                record["output"] = _av_checkpoint(result, self.modules.state)
                record["sampling_elapsed_seconds"] = time.perf_counter() - started
                record["model_runtime_after"] = _model_runtime_state(kwargs["model"])
                return result
            finally:
                comfy.sample.fix_empty_latent_channels = original_fix
                comfy.sample.prepare_noise = original_noise
                comfy.model_base.BaseModel._apply_model = original_apply
                comfy.samplers.CFGGuider.sample = original_guider_sample
                restored = (
                    comfy.sample.fix_empty_latent_channels is original_fix
                    and comfy.sample.prepare_noise is original_noise
                    and comfy.model_base.BaseModel._apply_model is original_apply
                    and comfy.samplers.CFGGuider.sample is original_guider_sample
                )
                record["hooks_restored"] = bool(restored)
                record["capture_complete"] = {
                    "fixed_latent": fixed_seen,
                    "noise": noise_seen,
                    "first_model_call": model_seen,
                    "all_sampling_checkpoints": all(
                        name in checkpoints for name in CHECKPOINTS
                    ),
                    "all_model_checkpoints": all(
                        name in record["model"] for name in MODEL_CHECKPOINTS
                    ),
                }
                self.hooks_restored = self.hooks_restored and bool(restored)

        return observed_sample_chunk


class H3G4SamplingDeterminismDiagnostic:
    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
        "AUDIO",
        "H3_CONTINUUM_REFINE_CONTEXT",
        "STRING",
    )
    RETURN_NAMES = (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
        "refine_context",
        "diagnostic_json",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False, False, False)
    FUNCTION = "run"
    CATEGORY = "H3 Continuum/Diagnostics"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        modules = _resolve_modules()
        schema = modules.sampler_class.INPUT_TYPES()
        required = dict(schema.get("required", {}))
        required["diagnostic_label"] = (
            "STRING",
            {"default": "sampling_diagnostic"},
        )
        required["diagnostic_nonce"] = (
            "INT",
            {"default": 0, "min": 0, "max": 0x7FFFFFFF},
        )
        required["trace_steps"] = ("BOOLEAN", {"default": False})
        schema["required"] = required
        return schema

    @classmethod
    def IS_CHANGED(cls, generation_mode="Full Run", **kwargs):
        # Preserve the public V3.8 cache contract exactly: Full Run is stable,
        # while Review Each Chunk is intentionally non-cacheable.
        modules = _resolve_modules()
        return modules.sampler_class.IS_CHANGED(generation_mode=generation_mode)

    def run(
        self,
        diagnostic_label,
        diagnostic_nonce,
        trace_steps,
        **kwargs,
    ):
        modules = _resolve_modules()
        sequence = modules.sequence
        collector = SamplingCollector(modules, trace_steps=bool(trace_steps))
        original_sample = sequence.sample_chunk
        model = kwargs.get("model")
        state_before = _model_runtime_state(model)
        sequence.sample_chunk = collector.wrap(original_sample)
        started = time.perf_counter()
        try:
            outputs = modules.sampler_class().run(**kwargs)
            state_after = _model_runtime_state(model)
        finally:
            sequence.sample_chunk = original_sample
        restored = sequence.sample_chunk is original_sample and collector.hooks_restored
        result = {
            "format": "h3-v38-g4-sampling-determinism-v1",
            "diagnostic_label": str(diagnostic_label),
            "diagnostic_nonce": int(diagnostic_nonce),
            "trace_steps": bool(trace_steps),
            "production_root": modules.production_root,
            "sampler_node_elapsed_seconds": time.perf_counter() - started,
            "sampler_model_state_before": state_before,
            "sampler_model_state_after": state_after,
            "physical_samples": collector.records,
            "physical_sample_count": len(collector.records),
            "hooks_restored": bool(restored),
        }
        payload = json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return {
            "ui": {"text": [payload]},
            "result": (*outputs, payload),
        }


NODE_CLASS_MAPPINGS = {
    "H3G4SamplingDeterminismDiagnostic": H3G4SamplingDeterminismDiagnostic,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3G4SamplingDeterminismDiagnostic": "H3 G4 Sampling Determinism Diagnostic",
}
