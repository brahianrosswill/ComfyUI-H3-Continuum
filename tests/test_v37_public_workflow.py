from __future__ import annotations

import copy
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / "examples" / "workflows"
V36_PATH = WORKFLOW_DIR / "MiniMax_H3_Continuum_V36.json"
V37_PATH = WORKFLOW_DIR / "MiniMax_H3_Continuum_V37.json"
V37_ZIP_PATH = WORKFLOW_DIR / "MiniMax_H3_Continuum_V37.zip"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v37_template_is_the_v36_graph_with_only_the_sampler_class_replaced():
    v36 = _load(V36_PATH)
    v37 = _load(V37_PATH)
    sampler = next(node for node in v37["nodes"] if node["id"] == 305)

    assert sampler["type"] == "H3ContinuumSamplerV37"
    assert sampler["properties"]["Node name for S&R"] == "H3ContinuumSamplerV37"
    assert sampler["widgets_values_named"]["continuation_backend"] == "Standard"
    assert all(input_["name"] != "guide" for input_ in sampler["inputs"])

    normalized = copy.deepcopy(v37)
    normalized_sampler = next(node for node in normalized["nodes"] if node["id"] == 305)
    normalized_sampler["type"] = "H3ContinuumSamplerV36"
    normalized_sampler["properties"]["Node name for S&R"] = "H3ContinuumSamplerV36"
    assert normalized == v36


def test_v37_zip_contains_the_exact_public_workflow_json():
    with zipfile.ZipFile(V37_ZIP_PATH) as archive:
        assert archive.namelist() == [V37_PATH.name]
        assert archive.read(V37_PATH.name) == V37_PATH.read_bytes()
