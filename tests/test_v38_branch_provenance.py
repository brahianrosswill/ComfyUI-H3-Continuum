from __future__ import annotations

import pytest

from ComfyUI_H3_Continuum_Join.branch_provenance import (
    BranchProvenanceError,
    PhysicalGroup,
    active_revision_map,
    make_group_revision,
    physical_groups,
    resolve_chain,
    validate_group_revision,
)


def _record(position: int, source: str, seed: int):
    return {
        "sequence_index": position,
        "storage_revision_id": source,
        "filename": f"chunk_{position + 1:04d}.safetensors",
        "file_size": 100 + position,
        "file_sha256": f"sha-{source}-{position}",
        "entry": {"seed": seed, "prompt_hash": f"prompt-{position}"},
    }


def _revision(group, *, parent=None, source="a", nonce=0, branch_cut=None):
    return make_group_revision(
        parent_revision_id=parent,
        group=group,
        variation_nonce=nonce,
        created_utc="2026-09-03T00:00:00+00:00",
        storage_revision_id=source,
        generation_contract_sha256=f"contract-{source}",
        lineage_sha256="lineage",
        records=[
            _record(position, source, 1000 + position + nonce)
            for position in range(group.start - 1, group.end)
        ],
        branch_cut=branch_cut,
    )


def test_normal_and_terminal_physical_groups_are_atomic():
    assert [(g.start, g.end) for g in physical_groups(chunks=3, terminal_merge_enabled=False)] == [
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert [(g.start, g.end) for g in physical_groups(chunks=3, terminal_merge_enabled=True)] == [
        (1, 1),
        (2, 3),
    ]


def test_group_revision_identity_is_immutable_and_display_name_free():
    group = PhysicalGroup(3, 3, 3)
    first = _revision(group, source="take-b", nonce=2)
    second = _revision(group, source="take-b", nonce=2)
    assert first["revision_id"] == second["revision_id"]
    assert "take" not in first
    assert validate_group_revision(first)["revision_id"] == first["revision_id"]
    changed = dict(first)
    changed["variation_nonce"] = 3
    with pytest.raises(BranchProvenanceError, match="identity"):
        validate_group_revision(changed)


def test_branch_chain_keeps_both_takes_and_resolves_selected_parent():
    r1 = _revision(PhysicalGroup(1, 1, 1), source="root")
    r2 = _revision(PhysicalGroup(2, 2, 2), parent=r1["revision_id"], source="root")
    r3a = _revision(PhysicalGroup(3, 3, 3), parent=r2["revision_id"], source="a", nonce=1)
    r3b = _revision(PhysicalGroup(3, 3, 3), parent=r2["revision_id"], source="b", nonce=2)
    r4b = _revision(
        PhysicalGroup(4, 4, 4),
        parent=r3b["revision_id"],
        source="b2",
        nonce=3,
        branch_cut={
            "selected_revision_id": r3b["revision_id"],
            "after_physical_group": 3,
        },
    )
    catalog = {item["revision_id"]: item for item in (r1, r2, r3a, r3b, r4b)}
    chain = resolve_chain(catalog, r4b["revision_id"])
    assert [item["revision_id"] for item in chain] == [
        r1["revision_id"],
        r2["revision_id"],
        r3b["revision_id"],
        r4b["revision_id"],
    ]
    assert r3a["revision_id"] in catalog
    assert active_revision_map(chain)["3"] == r3b["revision_id"]


def test_wrong_parent_and_terminal_half_are_rejected():
    r1 = _revision(PhysicalGroup(1, 1, 1), source="root")
    orphan = _revision(PhysicalGroup(3, 3, 3), parent=r1["revision_id"], source="bad")
    with pytest.raises(BranchProvenanceError, match="contiguous"):
        resolve_chain(
            {r1["revision_id"]: r1, orphan["revision_id"]: orphan},
            orphan["revision_id"],
        )
    with pytest.raises(BranchProvenanceError, match="non-atomic"):
        make_group_revision(
            parent_revision_id=r1["revision_id"],
            group=PhysicalGroup(2, 3, 2),
            variation_nonce=1,
            created_utc="2026-09-03T00:00:00+00:00",
            storage_revision_id="terminal",
            generation_contract_sha256="contract",
            lineage_sha256="lineage",
            records=[_record(1, "terminal", 1)],
        )
