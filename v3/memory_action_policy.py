"""Explicit opt-in A7b memory actions that preserve Production ownership.

This module resolves only Reference sizing policy.  Model unloading and
attention/backend selection remain advisory because ComfyUI Core and external
MODEL wrappers own those lifecycles.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from ..reference import REFERENCE_SIZE_MATCH_OUTPUT, REFERENCE_SIZE_MAX_IDENTITY
from ..reference_video import (
    REFERENCE_VIDEO_SIZE_BALANCED,
    REFERENCE_VIDEO_SIZE_EFFICIENT,
    REFERENCE_VIDEO_SIZE_MATCH_OUTPUT,
)


MEMORY_ACTION_POLICY_TYPE = "H3_CONTINUUM_MEMORY_ACTION_POLICY"
MEMORY_ACTION_POLICY_VERSION = 1

ACTION_DISABLED = "Disabled"
ACTION_REDUCE_REFERENCES_ONE_STEP = "Reduce References One Step"
ACTION_OPTIONS = (
    ACTION_DISABLED,
    ACTION_REDUCE_REFERENCES_ONE_STEP,
)

_IMAGE_DOWNGRADE = {
    REFERENCE_SIZE_MAX_IDENTITY: REFERENCE_SIZE_MATCH_OUTPUT,
    REFERENCE_SIZE_MATCH_OUTPUT: REFERENCE_SIZE_MATCH_OUTPUT,
}
_VIDEO_DOWNGRADE = {
    REFERENCE_VIDEO_SIZE_MATCH_OUTPUT: REFERENCE_VIDEO_SIZE_BALANCED,
    REFERENCE_VIDEO_SIZE_BALANCED: REFERENCE_VIDEO_SIZE_EFFICIENT,
    REFERENCE_VIDEO_SIZE_EFFICIENT: REFERENCE_VIDEO_SIZE_EFFICIENT,
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _contract_base(action: str) -> dict[str, Any]:
    return {
        "memory_action_policy_version": MEMORY_ACTION_POLICY_VERSION,
        "action": action,
        "reference_downgrade": (
            "one_step"
            if action == ACTION_REDUCE_REFERENCES_ONE_STEP
            else "disabled"
        ),
        "auto_unload": "advisory_only",
        "backend_switch": "advisory_only",
        "same_run_actual_row_trigger": False,
    }


def make_memory_action_policy(action: str = ACTION_DISABLED) -> dict[str, Any]:
    """Build the versioned policy object emitted by the public helper node."""

    normalized = str(action)
    if normalized not in ACTION_OPTIONS:
        raise ValueError(f"unsupported memory action {action!r}")
    contract = _contract_base(normalized)
    return {**contract, "contract_hash": _canonical_hash(contract)}


@dataclass(frozen=True)
class MemoryActionDecision:
    action: str
    enabled: bool
    requested_image_mode: str
    effective_image_mode: str
    requested_video_mode: str
    effective_video_mode: str
    image_applied: bool
    video_applied: bool
    contract_hash: str | None
    warning: str | None = None

    @property
    def applied(self) -> bool:
        return bool(self.image_applied or self.video_applied)


def _disabled_decision(
    *,
    image_mode: str,
    video_mode: str,
    contract_hash: str | None = None,
    warning: str | None = None,
) -> MemoryActionDecision:
    return MemoryActionDecision(
        action=ACTION_DISABLED,
        enabled=False,
        requested_image_mode=image_mode,
        effective_image_mode=image_mode,
        requested_video_mode=video_mode,
        effective_video_mode=video_mode,
        image_applied=False,
        video_applied=False,
        contract_hash=contract_hash,
        warning=warning,
    )


def resolve_memory_action_policy(
    policy: Any,
    *,
    image_mode: str,
    video_mode: str,
    has_reference_images: bool,
    has_video_guide: bool,
) -> MemoryActionDecision:
    """Resolve an optional policy without changing tensors or runtime state.

    Malformed or unsupported objects fail soft to Disabled.  The caller may
    surface ``warning`` after generation, but Sampling inputs remain unchanged.
    """

    requested_image = str(image_mode)
    requested_video = str(video_mode)
    if policy is None:
        return _disabled_decision(
            image_mode=requested_image,
            video_mode=requested_video,
        )
    if not isinstance(policy, dict):
        return _disabled_decision(
            image_mode=requested_image,
            video_mode=requested_video,
            warning="invalid policy object; no memory action was applied",
        )

    action = str(policy.get("action", ""))
    try:
        version = int(policy.get("memory_action_policy_version"))
    except (TypeError, ValueError):
        version = -1
    supplied_hash = policy.get("contract_hash")
    expected_contract = _contract_base(action) if action in ACTION_OPTIONS else None
    expected_hash = (
        _canonical_hash(expected_contract) if expected_contract is not None else None
    )
    if (
        version != MEMORY_ACTION_POLICY_VERSION
        or action not in ACTION_OPTIONS
        or not isinstance(supplied_hash, str)
        or supplied_hash != expected_hash
    ):
        return _disabled_decision(
            image_mode=requested_image,
            video_mode=requested_video,
            warning="unsupported or modified policy contract; no memory action was applied",
        )
    if action == ACTION_DISABLED:
        return _disabled_decision(
            image_mode=requested_image,
            video_mode=requested_video,
            contract_hash=supplied_hash,
        )

    effective_image = requested_image
    effective_video = requested_video
    if bool(has_reference_images):
        effective_image = _IMAGE_DOWNGRADE.get(requested_image, requested_image)
    if bool(has_video_guide):
        effective_video = _VIDEO_DOWNGRADE.get(requested_video, requested_video)
    return MemoryActionDecision(
        action=action,
        enabled=True,
        requested_image_mode=requested_image,
        effective_image_mode=effective_image,
        requested_video_mode=requested_video,
        effective_video_mode=effective_video,
        image_applied=(
            bool(has_reference_images) and effective_image != requested_image
        ),
        video_applied=(bool(has_video_guide) and effective_video != requested_video),
        contract_hash=supplied_hash,
    )


def format_memory_action_status(decision: MemoryActionDecision) -> str | None:
    """Format post-generation status without claiming VRAM prediction."""

    if decision.warning is not None:
        return "Memory Action Policy [Experimental A7b v1]: " + decision.warning + "."
    if not decision.enabled:
        return None
    changes: list[str] = []
    if decision.image_applied:
        changes.append(
            "Reference Image "
            f"{decision.requested_image_mode} -> {decision.effective_image_mode}"
        )
    if decision.video_applied:
        changes.append(
            "Video Guide "
            f"{decision.requested_video_mode} -> {decision.effective_video_mode}"
        )
    action = "; ".join(changes) if changes else "no applicable Reference downgrade"
    return (
        "Memory Action Policy [Experimental A7b v1]: explicit opt-in; "
        f"{action}.\n"
        "Memory advisory: automatic MODEL unload was not executed; Core retains "
        "lifecycle ownership.\n"
        "Memory advisory: automatic attention/backend switching was not executed; "
        "the external MODEL wrapper was preserved."
    )
