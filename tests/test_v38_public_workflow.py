from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / "examples" / "workflows" / "MiniMax_H3_Continuum_V38.json"

# User-approved final distribution. Do not sanitize or reserialize this graph.
WORKFLOW_SHA256 = "e490c87e4995930f95c8fffc4c678111d7897e74003e1962b9b4a247ba43f4ac"
ZIP_SHA256 = "83d28cc0cd549066962dc21ceae05d2b189fe22391b556887eab2cc1800e1e80"
EXTERNAL_PACKAGES = {
    "comfyui-spectrum-minimax-h3",
    "rgthree-comfy",
    "comfyui-kjnodes",
    "comfyui-easy-use",
}


def _workflow() -> dict:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _one_node(workflow: dict, node_type: str) -> dict:
    values = [node for node in workflow["nodes"] if node["type"] == node_type]
    assert len(values) == 1
    return values[0]


def _graph(workflow: dict) -> tuple[dict[int, dict], dict[int, list]]:
    nodes = {int(node["id"]): node for node in workflow["nodes"]}
    links = {int(link[0]): link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])
    for link_id, link in links.items():
        assert int(link[0]) == link_id
        source = nodes[int(link[1])]
        target = nodes[int(link[3])]
        assert link_id in (source["outputs"][int(link[2])].get("links") or [])
        assert int(target["inputs"][int(link[4])]["link"]) == link_id
    for node_id, node in nodes.items():
        for slot, item in enumerate(node.get("inputs", [])):
            if item.get("link") is not None:
                assert links[int(item["link"])][3:5] == [node_id, slot]
        for slot, item in enumerate(node.get("outputs", [])):
            for link_id in item.get("links") or []:
                assert links[int(link_id)][1:3] == [node_id, slot]
    return nodes, links


def test_v38_public_workflow_declares_external_dependencies_and_is_reloadable():
    workflow = _workflow()
    assert len(workflow["nodes"]) == 32
    assert len(workflow["links"]) == 38
    packages = {
        node.get("properties", {}).get("cnr_id") for node in workflow["nodes"]
    } - {None, "", "comfy-core"}
    assert packages == EXTERNAL_PACKAGES
    node_types = {node["type"] for node in workflow["nodes"]}
    assert {
        "SpectrumApplyMiniMaxH3", "Power Lora Loader (rgthree)",
        "Fast Groups Bypasser (rgthree)", "easy cleanGpuUsed",
        "MiniMaxH3MemoryEfficientSageAttentionPatch",
    } <= node_types
    assert not any("hires" in name.lower() for name in node_types)
    _graph(workflow)
    assert json.loads(json.dumps(workflow, ensure_ascii=False)) == workflow
    assert int(workflow["last_node_id"]) >= max(int(node["id"]) for node in workflow["nodes"])
    assert int(workflow["last_link_id"]) >= max(int(link[0]) for link in workflow["links"])


def test_v38_public_workflow_and_zip_preserve_supplied_bytes():
    payload = WORKFLOW_PATH.read_bytes()
    archive_path = WORKFLOW_PATH.with_suffix(".zip")
    assert hashlib.sha256(payload).hexdigest() == WORKFLOW_SHA256
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == ZIP_SHA256
    with ZipFile(archive_path) as archive:
        assert archive.namelist() == [WORKFLOW_PATH.name]
        assert archive.testzip() is None
        assert archive.read(WORKFLOW_PATH.name) == payload


def test_v38_public_workflow_preserves_sampler_widget_positions():
    workflow = _workflow()
    sampler = _one_node(workflow, "H3ContinuumSamplerV38")
    names = [
        "prompt_mode",
        "chunks",
        "chunk_seconds",
        "aspect",
        "preset",
        "custom_mp",
        "continuity",
        "base_seed",
        "control_after_generate",
        "audio_continuity",
        "diagnostics",
        "reroll_from_chunk",
        "reroll_nonce",
        "strict_compatibility",
        "debug",
        "show_preview",
        "run_storage",
        "run_name",
        "reference_size",
        "project_id",
        "video_reference_size",
        "continuation_backend",
        "generation_mode",
        "review_action",
        "take_group",
        "take_revision_id",
        "take_action",
        "size_source",
        "width",
        "height",
    ]
    assert len(sampler["widgets_values"]) == len(names) == 30
    assert sampler["widgets_values"] == [
        sampler["widgets_values_named"][name] for name in names
    ]
    assert sampler["widgets_values_named"]["generation_mode"] == "Full Run"
    assert sampler["widgets_values_named"]["review_action"] == "Continue / Next"
    assert sampler["widgets_values_named"]["run_storage"] == "Off"
    assert sampler["widgets_values_named"]["aspect"] == "Auto from First Image"
    assert sampler["widgets_values_named"]["preset"] == "Draft — 0.30 MP"
    assert sampler["widgets_values_named"]["size_source"] == "First Image"
    assert sampler["widgets_values_named"]["width"] == 640
    assert sampler["widgets_values_named"]["height"] == 640
    assert sampler["widgets_values_named"]["chunks"] == 2
    assert sampler["widgets_values_named"]["chunk_seconds"] == 5
    assert sampler["widgets_values_named"]["control_after_generate"] == "fixed"
    assert sampler["widgets_values_named"]["continuation_backend"] == "Standard"


def test_v38_public_workflow_keeps_required_and_optional_connections():
    workflow = _workflow()
    nodes, links = _graph(workflow)
    sampler = _one_node(workflow, "H3ContinuumSamplerV38")
    inputs = {item["name"]: item for item in sampler["inputs"]}
    required = {"model", "clip", "video_vae", "sampler", "sigmas", "sequence_prompt"}
    optional = {
        "first_frame",
        "last_frame",
        "reference_image_1",
        "reference_image_2",
        "reference_image_3",
        "driving_audio",
        "audio_vae",
        "reference_video_1",
        "audio_references",
    }
    assert required | optional <= set(inputs)
    assert all(inputs[name]["link"] is not None for name in required | optional)

    source_types = {}
    source_modes = {}
    for name in required | optional:
        link = links[int(inputs[name]["link"])]
        source = nodes[int(link[1])]
        source_types[name] = source["type"]
        source_modes[name] = int(source["mode"])
    assert source_types["model"] == "SpectrumApplyMiniMaxH3"
    assert source_types["sampler"] == "KSamplerSelect"
    assert source_types["sigmas"] == "BasicScheduler"
    assert source_types["sequence_prompt"] == "PrimitiveStringMultiline"
    assert source_types["first_frame"] == "easy cleanGpuUsed"
    assert source_modes["first_frame"] == 0
    assert source_modes["last_frame"] == 4
    assert source_modes["reference_image_1"] == 0
    assert source_modes["reference_image_2"] == 0
    assert source_modes["reference_image_3"] == 0
    assert source_modes["driving_audio"] == 4
    assert source_modes["reference_video_1"] == 4
    assert source_types["audio_references"] == "H3ContinuumReferenceAudios"


def test_v38_public_workflow_preserves_spectrum_default_and_model_chain():
    workflow = _workflow()
    nodes, links = _graph(workflow)

    def source(node, name):
        item = next(item for item in node["inputs"] if item["name"] == name)
        return nodes[links[item["link"]][1]]

    sampler = _one_node(workflow, "H3ContinuumSamplerV38")
    spectrum = source(sampler, "model")
    assert spectrum["type"] == "SpectrumApplyMiniMaxH3"
    assert spectrum["widgets_values"][0] is True
    lora = source(spectrum, "model")
    assert lora["type"] == "Power Lora Loader (rgthree)"
    loras = [item for item in lora["widgets_values"] if isinstance(item, dict) and "lora" in item]
    assert len(loras) == 4
    assert all(item["on"] is False for item in loras)
    sage = source(lora, "model")
    assert sage["type"] == "MiniMaxH3MemoryEfficientSageAttentionPatch"
    assert source(sage, "model")["type"] == "UNETLoader"
    assert _one_node(workflow, "KSamplerSelect")["widgets_values"] == ["res_multistep"]
    scheduler = _one_node(workflow, "BasicScheduler")
    assert scheduler["widgets_values"] == ["simple", 20, 1]
    assert source(scheduler, "model")["id"] == spectrum["id"]


def test_v38_public_workflow_keeps_decode_finalize_create_save_chain():
    workflow = _workflow()
    nodes, links = _graph(workflow)
    save = _one_node(workflow, "SaveVideo")
    cleanup = nodes[links[int(save["inputs"][0]["link"])][1]]
    assert cleanup["type"] == "easy cleanGpuUsed"
    create_link = links[int(cleanup["inputs"][0]["link"])]
    assert nodes[int(create_link[1])]["type"] == "CreateVideo"
    create = nodes[int(create_link[1])]
    assert all(item["link"] is not None for item in create["inputs"][:2])
    finalize = _one_node(workflow, "H3ContinuumAssembleSeamV35")
    assert [links[item["link"]][1] for item in create["inputs"][:2]] == [finalize["id"]] * 2
    incoming = [nodes[links[item["link"]][1]]["type"] for item in finalize["inputs"][:3]]
    assert incoming == ["VAEDecode", "VAEDecodeAudio", "H3ContinuumSamplerV38"]
    for node_type in ("VAEDecode", "VAEDecodeAudio"):
        decode = _one_node(workflow, node_type)
        assert nodes[links[decode["inputs"][0]["link"]][1]]["type"] == "H3ContinuumSamplerV38"
