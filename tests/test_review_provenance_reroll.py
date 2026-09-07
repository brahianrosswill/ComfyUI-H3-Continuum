"""Real on-disk provenance must not bypass a newly requested reroll."""
import pytest
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "review_storage_fixtures", Path(__file__).with_name("test_v38_review_run_storage.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
_contract, _controller, _persist = _fixtures._contract, _fixtures._controller, _fixtures._persist


@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5, 6])
def test_new_reroll_only_reuses_chunks_before_boundary(tmp_path, boundary):
    old = _persist(tmp_path, _contract(chunks=6), prefix=6, review_unit=(6, 6),
                   updated_utc="2026-09-07T00:00:00+00:00")
    requested = _contract(chunks=6, boundary=boundary, nonce=1)
    current = _controller(tmp_path, requested)
    entries, records = current._valid_provenance_prefix(
        old.manifest, current_contract=requested, enforce_sampling_contract=True)
    assert len(entries) == len(records) == boundary - 1
    assert len(old.manifest["chunks"]) == 6


def test_same_nonce_interrupted_prefix_can_resume(tmp_path):
    contract = _contract(chunks=6, boundary=2, nonce=1)
    old = _persist(tmp_path, contract, prefix=3, review_unit=(3, 3),
                   updated_utc="2026-09-07T00:00:00+00:00")
    current = _controller(tmp_path, contract)
    entries, records = current._valid_provenance_prefix(
        old.manifest, current_contract=contract, enforce_sampling_contract=True)
    assert len(entries) == len(records) == 3


def test_completed_same_contract_still_reuses_all(tmp_path):
    contract = _contract(chunks=2)
    old = _persist(tmp_path, contract, prefix=2, review_unit=(2, 2),
                   updated_utc="2026-09-07T00:00:00+00:00")
    current = _controller(tmp_path, contract)
    assert len(current._valid_provenance_prefix(
        old.manifest, current_contract=contract, enforce_sampling_contract=True)[0]) == 2


def test_smart_retry_new_nonce_does_not_reuse_previous_take(tmp_path):
    old = _persist(tmp_path, _contract(chunks=3, boundary=2, nonce=1),
                   prefix=2, review_unit=(2, 2), updated_utc="2026-09-07T00:00:00+00:00")
    contract = _contract(chunks=3, boundary=2, nonce=2)
    current = _controller(tmp_path, contract)
    assert len(current._valid_provenance_prefix(
        old.manifest, current_contract=contract, enforce_sampling_contract=True)[0]) == 1
