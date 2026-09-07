"""Versioned stream-target policies for future Continuum refinement modes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


MAGIC = "H3_CONTINUUM_REFINE_TARGET"
SCHEMA_VERSION = 1
EXECUTION_MAGIC = "H3_CONTINUUM_TARGETED_REFINE_EXECUTION"
EXECUTION_SCHEMA_VERSION = 1

MODE_VIDEO_ONLY = "video_only"
MODE_AUDIO_ONLY = "audio_only"
MODE_VIDEO_AUDIO = "video_audio"
MODES = (MODE_VIDEO_ONLY, MODE_AUDIO_ONLY, MODE_VIDEO_AUDIO)

NOISE_SEEDED_RANDOM = "seeded_random"
NOISE_ZERO = "zero"
OUTPUT_SAMPLED = "sampled"
OUTPUT_INPUT_PASSTHROUGH = "input_passthrough_bit_exact"

VIDEO_ONLY_SEED_NAMESPACE = "h3-continuum-refine-v1"
AUDIO_ONLY_SEED_NAMESPACE = "h3-continuum-refine-audio-v1"
VIDEO_AUDIO_SEED_NAMESPACE = "h3-continuum-refine-av-v1"


class RefineTargetError(ValueError):
    """Raised when a target policy or target execution contract is invalid."""


@dataclass(frozen=True)
class RefineTarget:
    """One immutable target policy; it contains no Tensor or runtime object."""

    mode: str
    contract: Mapping[str, Any]


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stream_policy(*, noise: str, mask: int, output: str) -> dict[str, Any]:
    return {
        "noise": str(noise),
        "mask": int(mask),
        "output": str(output),
    }


_POLICIES: dict[str, dict[str, Any]] = {
    MODE_VIDEO_ONLY: {
        "video": _stream_policy(
            noise=NOISE_SEEDED_RANDOM,
            mask=1,
            output=OUTPUT_SAMPLED,
        ),
        "audio": _stream_policy(
            noise=NOISE_ZERO,
            mask=0,
            output=OUTPUT_INPUT_PASSTHROUGH,
        ),
        "seed_namespace": VIDEO_ONLY_SEED_NAMESPACE,
    },
    MODE_AUDIO_ONLY: {
        "video": _stream_policy(
            noise=NOISE_ZERO,
            mask=0,
            output=OUTPUT_INPUT_PASSTHROUGH,
        ),
        "audio": _stream_policy(
            noise=NOISE_SEEDED_RANDOM,
            mask=1,
            output=OUTPUT_SAMPLED,
        ),
        "seed_namespace": AUDIO_ONLY_SEED_NAMESPACE,
    },
    MODE_VIDEO_AUDIO: {
        "video": _stream_policy(
            noise=NOISE_SEEDED_RANDOM,
            mask=1,
            output=OUTPUT_SAMPLED,
        ),
        "audio": _stream_policy(
            noise=NOISE_SEEDED_RANDOM,
            mask=1,
            output=OUTPUT_SAMPLED,
        ),
        "seed_namespace": VIDEO_AUDIO_SEED_NAMESPACE,
    },
}


def make_refine_target(mode: str) -> RefineTarget:
    canonical_mode = str(mode).strip().lower()
    if canonical_mode not in MODES:
        raise RefineTargetError(f"unsupported refine target mode {mode!r}")
    policy = _POLICIES[canonical_mode]
    contract: dict[str, Any] = {
        "magic": MAGIC,
        "schema_version": SCHEMA_VERSION,
        "mode": canonical_mode,
        "video": dict(policy["video"]),
        "audio": dict(policy["audio"]),
        "seed_namespace": str(policy["seed_namespace"]),
    }
    contract["target_hash"] = _identity(contract)
    return RefineTarget(
        mode=canonical_mode,
        contract=MappingProxyType(contract),
    )


def resolve_refine_target(value: RefineTarget | str | None) -> RefineTarget:
    if value is None:
        return make_refine_target(MODE_VIDEO_ONLY)
    if isinstance(value, str):
        return make_refine_target(value)
    if not isinstance(value, RefineTarget):
        raise RefineTargetError("refine_target must be a RefineTarget or mode string")
    rebuilt = make_refine_target(value.mode)
    if dict(rebuilt.contract) != dict(value.contract):
        raise RefineTargetError("refine_target contract identity is inconsistent")
    return value


def serializable_target_contract(target: RefineTarget) -> dict[str, Any]:
    resolved = resolve_refine_target(target)
    return {
        **dict(resolved.contract),
        "video": dict(resolved.contract["video"]),
        "audio": dict(resolved.contract["audio"]),
    }


def derive_target_refine_seed(
    refine_seed: int,
    physical_group_index: int,
    target: RefineTarget | str,
) -> int:
    if type(refine_seed) is not int or refine_seed < 0:
        raise RefineTargetError("refine_seed must be a non-negative integer")
    if type(physical_group_index) is not int or physical_group_index < 0:
        raise RefineTargetError("physical_group_index must be a non-negative integer")
    resolved = resolve_refine_target(target)
    namespace = str(resolved.contract["seed_namespace"])
    payload = f"{namespace}:{refine_seed}:{physical_group_index}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


_SIGMA_RANGE_FIELDS = (
    "schema_version",
    "mode",
    "source_sigma_hash",
    "sigma_hash",
    "start_sigma",
    "end_sigma",
    "evaluation_count",
    "source_evaluation_count",
    "start_index",
    "end_index",
)


def build_target_execution_contract(
    target: RefineTarget,
    refine_schedule_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine target policy with sigma-range identity without mutating V1."""

    resolved = resolve_refine_target(target)
    if not isinstance(refine_schedule_contract, Mapping):
        raise RefineTargetError("refine_schedule_contract must be a mapping")
    missing = [
        name for name in _SIGMA_RANGE_FIELDS if name not in refine_schedule_contract
    ]
    if missing or not refine_schedule_contract.get("schedule_hash"):
        details = ", ".join([*missing, "schedule_hash"] if not refine_schedule_contract.get("schedule_hash") else missing)
        raise RefineTargetError(f"refine schedule range identity is incomplete: {details}")

    sigma_range = {
        name: refine_schedule_contract[name] for name in _SIGMA_RANGE_FIELDS
    }
    sigma_range["source_schedule_hash"] = str(
        refine_schedule_contract["schedule_hash"]
    )
    contract: dict[str, Any] = {
        "magic": EXECUTION_MAGIC,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "policy_authority": "refine_target_contract_v1",
        "target": serializable_target_contract(resolved),
        "sigma_range": sigma_range,
    }
    contract["execution_hash"] = _identity(contract)
    return contract

