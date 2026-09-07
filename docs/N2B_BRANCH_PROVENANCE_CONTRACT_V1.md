# Branch Provenance Contract v1

Status: V3.8 N2b implementation contract.

## Purpose

Branch Provenance records which immutable physical-group revision is active at
each point in a Run Storage chain. It allows a user to select an older Take and
generate only the physical groups after that Take without deleting the former
branch or confusing equal nonces with equal ancestry.

Take labels are presentation only. Internal identity is always `revision_id`.

## Physical-group revision

Every completed physical group records:

- `revision_id`: immutable SHA-256-derived identity;
- `parent_revision_id`: the immediately preceding physical-group revision;
- `group`: one-based `start`, `end`, and `physical_group` values;
- `variation_nonce` and `seed_identity`;
- `storage_revision_id` and `generation_contract_sha256`;
- `lineage_sha256` and immutable chunk record identities;
- `created_utc` plus deterministic `revision_order`;
- optional `branch_cut` metadata.

Terminal Merge `[N-1,N]` is one physical group. A chain that contains only one
logical half is invalid.

## Canonical chain

`project.json` has exactly one `canonical_chain`, one
`canonical_head_revision_id`, and an `active_revisions` map. Browsing Takes does
not change them. `Use This Take` or `Continue From Here`, followed by the normal
Queue action, creates a finalized Schema v3 storage revision and atomically
switches the canonical chain.

`Continue From Here` keeps the selected prefix exactly and starts a new existing
sampling nonce boundary at the next physical group. The Sampling Contract itself
remains version 5 and is not extended with provenance fields.

## Run Storage Schema v3

Schema v3 separates `revision_id` (storage execution identity) from
`sampling_revision_id` (the unchanged Sampling Contract identity). A finalized
manifest contains Branch Provenance v1 metadata with its complete active chain,
branch selection/cut when applicable, and a monotonic `canonical_sequence`.

The write order is:

1. save and fsync new chunk payloads;
2. validate all logical records in the completed physical group;
3. atomically write the finalized manifest and its active chain;
4. atomically replace `project.json` with the new canonical head.

A failure before step 4 leaves the preceding canonical project head intact.
If `project.json` is corrupt, the highest validated `canonical_sequence` in a
finalized manifest is the recovery source. In-progress and interrupted revisions
are not Render History Take candidates.

## Schema v2 compatibility

Schema v2 manifests remain readable and are never rewritten merely by loading
them. Their contiguous records are interpreted in memory as a legacy chain with
one canonical physical-group revision per group. The first Schema v3 save uses a
separate storage revision directory, so the legacy manifest and payloads remain
unchanged.

No automatic Take garbage collection is part of N2b.
