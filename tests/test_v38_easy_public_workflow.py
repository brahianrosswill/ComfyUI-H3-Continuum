from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = (
    ROOT / "examples" / "workflows" / "MiniMax_H3_Continuum_Easy_V38.json"
)


def _workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _one_node(workflow: dict, node_type: str) -> dict:
    values = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(values) == 1
    return values[0]


def _link_map(workflow: dict) -> dict[int, list]:
    return {int(link[0]): link for link in workflow["links"]}


FORBIDDEN_MINIMAL_MARKERS = (
    "power lora loader",
    "rgthree",
    "easy cleangpuused",
    "spectrum",
    "hires",
    "hi-res",
    "second pass",
    "secondpass",
    "upscale",
    "minimaxchunkfeedforward",
    "sageattention",
)


def test_v38_easy_public_workflow_uses_the_approved_default_contract():
    workflow = _workflow()
    easy = _one_node(workflow, "H3ContinuumEasyV38")
    helper = _one_node(workflow, "H3ContinuumEasyReferences")

    assert easy["widgets_values"][:10] == [
        "",
        10,
        0,
        "randomize",
        "Randomize",
        "Auto from First Image",
        "Draft — 0.30 MP",
        0.3,
        "Off",
        "",
    ]
    easy_widget_names = [
        "prompt_text",
        "duration",
        "seed",
        "control_after_generate",
        "seed_mode",
        "aspect",
        "preset",
        "custom_mp",
        "run_storage",
        "run_name",
        "project_id",
    ]
    assert easy["widgets_values"] == [
        easy["widgets_values_named"][name] for name in easy_widget_names
    ]
    assert [item["name"] for item in easy["inputs"]] == [
        "model",
        "clip",
        "video_vae",
        "sampler",
        "sigmas",
        "first_frame",
        "last_frame",
        "references",
        "driving_audio",
        "audio_vae",
        "prompt_text",
    ]
    assert helper["outputs"][0]["type"] == "H3_CONTINUUM_EASY_REFERENCES"
    assert [item["name"] for item in helper["inputs"]] == [
        "reference_image_1",
        "reference_image_2",
        "reference_image_3",
    ]
    assert not {
        "reference_audio_1",
        "reference_video_1",
        "timeline_video",
        "guide",
    } & {item["name"] for item in easy["inputs"]}
    assert len(workflow["nodes"]) == 20
    assert len([node for node in workflow["nodes"] if node["type"] == "H3EasyLoadImage"]) == 5
    audio_loader = _one_node(workflow, "H3EasyLoadAudio")
    assert int(audio_loader["mode"]) == 4
    assert audio_loader.get("title", "") == ""
    assert [output["type"] for output in audio_loader["outputs"]] == ["AUDIO"]
    assert not [node for node in workflow["nodes"] if node["type"] == "LoadImage"]
    assert workflow["extra"] == {
        "frontendVersion": "1.49.6",
        "ds": workflow["extra"]["ds"],
    }


def test_v38_easy_public_workflow_is_strict_minimal():
    workflow = _workflow()
    node_types = [str(node["type"]) for node in workflow["nodes"]]
    searchable = "\n".join(node_types).lower()
    assert not [marker for marker in FORBIDDEN_MINIMAL_MARKERS if marker in searchable]
    assert not [
        node
        for node in workflow["nodes"]
        if node.get("properties", {}).get("cnr_id")
        not in {None, "", "comfy-core"}
    ]
    assert "PreviewAny" not in node_types
    assert "MarkdownNote" not in node_types
    scheduler = _one_node(workflow, "BasicScheduler")
    assert scheduler["widgets_values_named"] == {
        "scheduler": "simple",
        "steps": 20,
        "denoise": 1,
    }


def test_v38_easy_public_workflow_defaults_to_t2va_with_optional_hybrids():
    workflow = _workflow()
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    links = _link_map(workflow)
    easy = _one_node(workflow, "H3ContinuumEasyV38")
    helper = _one_node(workflow, "H3ContinuumEasyReferences")

    helper_inputs = {item["name"]: item for item in helper["inputs"]}
    source_modes = {}
    for name, input_ in helper_inputs.items():
        link = links[int(input_["link"])]
        source_modes[name] = int(nodes[int(link[1])]["mode"])
    assert source_modes == {
        "reference_image_1": 4,
        "reference_image_2": 4,
        "reference_image_3": 4,
    }

    easy_inputs = {item["name"]: item for item in easy["inputs"]}
    first_source = nodes[int(links[int(easy_inputs["first_frame"]["link"])][1])]
    last_source = nodes[int(links[int(easy_inputs["last_frame"]["link"])][1])]
    assert int(first_source["mode"]) == 4
    assert int(last_source["mode"]) == 4
    assert links[int(easy_inputs["references"]["link"])][1] == helper["id"]

    prompt_link = links[int(easy_inputs["prompt_text"]["link"])]
    prompt_node = nodes[int(prompt_link[1])]
    assert "<Picture" not in prompt_node["widgets_values"][0]


def test_v38_easy_public_workflow_keeps_core_display_titles_standard():
    workflow = _workflow()
    core_nodes = [
        node
        for node in workflow["nodes"]
        if node.get("properties", {}).get("cnr_id") == "comfy-core"
    ]
    assert core_nodes
    assert all(not node.get("title") for node in core_nodes)


def test_v38_easy_public_workflow_links_are_closed_and_unique():
    workflow = _workflow()
    nodes = {int(node["id"]) for node in workflow["nodes"]}
    links = _link_map(workflow)
    assert len(links) == len(workflow["links"])
    for link_id, link in links.items():
        assert int(link[0]) == link_id
        assert int(link[1]) in nodes
        assert int(link[3]) in nodes
    for node in workflow["nodes"]:
        for input_ in node.get("inputs", []):
            if input_.get("link") is not None:
                assert int(input_["link"]) in links
        for output in node.get("outputs", []):
            for link_id in output.get("links") or []:
                assert int(link_id) in links
    assert json.loads(json.dumps(workflow, ensure_ascii=False)) == workflow
    assert len({int(node["id"]) for node in workflow["nodes"]}) == len(
        workflow["nodes"]
    )


def test_easy_image_loaders_use_native_modes_without_custom_titles():
    workflow = _workflow()
    loaders = {
        int(node["id"]): node
        for node in workflow["nodes"]
        if node["type"] == "H3EasyLoadImage"
    }
    assert {node_id: int(node["mode"]) for node_id, node in loaders.items()} == {
        114: 4,
        290: 4,
        258: 4,
        257: 4,
        264: 4,
    }
    for node in loaders.values():
        assert node.get("title", "") == ""
        assert node["properties"] == {"Node name for S&R": "H3EasyLoadImage"}
        assert [output["type"] for output in node["outputs"]] == ["IMAGE", "MASK"]
