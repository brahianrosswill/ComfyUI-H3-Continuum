"""Versioned physical-group selection for Experimental selective refinement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


MAGIC = "H3_CONTINUUM_REFINE_SCOPE"
SCHEMA_VERSION = 1

MODE_ALL = "all"
MODE_PHYSICAL_GROUP = "physical_group"
MODE_LOGICAL_CHUNK = "logical_chunk"
MODES = (MODE_ALL, MODE_PHYSICAL_GROUP, MODE_LOGICAL_CHUNK)


class RefineScopeError(ValueError):
    """Raised before preparation when a requested scope cannot be resolved."""


@dataclass(frozen=True)
class RefineScope:
    """One immutable scope resolution containing no Tensor or runtime object."""

    mode: str
    selected_group_indices: tuple[int, ...]
    contract: Mapping[str, Any]

    @property
    def is_all(self) -> bool:
        return self.mode == MODE_ALL


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _physical_groups(assembly_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(assembly_plan, Mapping):
        raise RefineScopeError("assembly_plan must be a mapping")
    second_pass = assembly_plan.get("second_pass_contract")
    if not isinstance(second_pass, Mapping) or second_pass.get("version") != 1:
        raise RefineScopeError("assembly_plan has no Second Pass contract version 1")
    groups = second_pass.get("physical_groups")
    if (
        not isinstance(groups, Sequence)
        or isinstance(groups, (str, bytes))
        or not groups
        or not all(isinstance(group, Mapping) for group in groups)
    ):
        raise RefineScopeError("assembly_plan has no physical groups")
    return list(groups)


def _logical_chunks(group: Mapping[str, Any], group_index: int) -> tuple[int, ...]:
    values = group.get("logical_chunks")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(type(value) is not int or value <= 0 for value in values)
    ):
        raise RefineScopeError(
            f"physical group {group_index + 1} has invalid logical chunk identity"
        )
    chunks = tuple(int(value) for value in values)
    if len(set(chunks)) != len(chunks):
        raise RefineScopeError(
            f"physical group {group_index + 1} repeats a logical chunk"
        )
    return chunks


def resolve_refine_scope(
    mode: str | None,
    scope_index: int,
    assembly_plan: Mapping[str, Any],
) -> RefineScope:
    """Resolve one public 1-based selection to complete physical groups.

    Logical chunks never split a physical group. Selecting either logical half of
    a Terminal Merge group therefore selects the complete terminal pair.
    """

    if mode is None:
        canonical_mode = MODE_ALL
    elif not isinstance(mode, str):
        raise RefineScopeError(f"unsupported refine scope mode {mode!r}")
    else:
        canonical_mode = mode.strip().lower()
    if canonical_mode not in MODES:
        raise RefineScopeError(f"unsupported refine scope mode {mode!r}")
    if type(scope_index) is not int or scope_index <= 0:
        raise RefineScopeError("refine scope index must be a positive integer")

    groups = _physical_groups(assembly_plan)
    group_chunks = [
        _logical_chunks(group, group_index)
        for group_index, group in enumerate(groups)
    ]
    terminal_expanded = False
    requested_index: int | None = None

    if canonical_mode == MODE_ALL:
        selected = tuple(range(len(groups)))
    elif canonical_mode == MODE_PHYSICAL_GROUP:
        requested_index = int(scope_index)
        selected_index = requested_index - 1
        if selected_index >= len(groups):
            raise RefineScopeError(
                f"physical group {requested_index} is outside 1..{len(groups)}"
            )
        selected = (selected_index,)
    else:
        requested_index = int(scope_index)
        matches = [
            group_index
            for group_index, chunks in enumerate(group_chunks)
            if requested_index in chunks
        ]
        if len(matches) != 1:
            if matches:
                raise RefineScopeError(
                    f"logical chunk {requested_index} belongs to multiple physical groups"
                )
            raise RefineScopeError(
                f"logical chunk {requested_index} is not present in the assembly plan"
            )
        selected = (matches[0],)
        selected_group = groups[selected[0]]
        selected_chunks = group_chunks[selected[0]]
        terminal_expanded = bool(
            selected_group.get("terminal_merged") and len(selected_chunks) > 1
        )

    selected_chunks = tuple(
        chunk
        for group_index in selected
        for chunk in group_chunks[group_index]
    )
    contract: dict[str, Any] = {
        "magic": MAGIC,
        "schema_version": SCHEMA_VERSION,
        "mode": canonical_mode,
        "requested_index": requested_index,
        "selected_physical_group_indices": tuple(selected),
        "selected_physical_groups": tuple(index + 1 for index in selected),
        "selected_logical_chunks": tuple(selected_chunks),
        "terminal_expanded": terminal_expanded,
    }
    contract["scope_hash"] = _identity(contract)
    return RefineScope(
        mode=canonical_mode,
        selected_group_indices=selected,
        contract=MappingProxyType(contract),
    )


def serializable_scope_contract(scope: RefineScope) -> dict[str, Any]:
    if not isinstance(scope, RefineScope):
        raise RefineScopeError("refine_scope must be a RefineScope")
    return {
        **dict(scope.contract),
        "selected_physical_group_indices": list(
            scope.contract["selected_physical_group_indices"]
        ),
        "selected_physical_groups": list(scope.contract["selected_physical_groups"]),
        "selected_logical_chunks": list(scope.contract["selected_logical_chunks"]),
    }
