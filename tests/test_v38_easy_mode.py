from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.reference import (
    REFERENCE_SIZE_MATCH_OUTPUT,
    prepare_reference_assets,
    validate_reference_prompts,
)
from ComfyUI_H3_Continuum_Join.v2.sequence import _terminal_flf_merge_enabled
from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV37
from ComfyUI_H3_Continuum_Join.v3.easy_nodes import (
    EASY_ASPECT_AUTO,
    EASY_ASPECT_LANDSCAPE,
    EASY_ASPECT_PORTRAIT,
    EASY_ASPECT_SQUARE,
    EASY_PRESET_BALANCED,
    EASY_PRESET_CUSTOM,
    EASY_PRESET_DRAFT,
    EASY_PRESET_NATIVE,
    EASY_REFERENCES_TYPE,
    EASY_SEED_MODE_RANDOMIZE,
    H3ContinuumEasyReferences,
    H3ContinuumEasyV38,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    make_easy_references,
    resolve_easy_duration,
    resolve_easy_references,
    resolve_easy_resolution,
)


ROOT = Path(__file__).resolve().parents[1]
EASY_JS = ROOT / "web" / "easy_v38.js"


class _Image:
    def __init__(self, height: int, width: int):
        self.shape = (1, height, width, 3)


@pytest.mark.parametrize(
    ("duration", "chunks", "chunk_seconds"),
    (
        (4, 1, 4.0),
        (10, 1, 10.0),
        (15, 2, 7.5),
        (20, 2, 10.0),
        (30, 3, 10.0),
        (160, 16, 10.0),
        (480, 16, 30.0),
    ),
)
def test_duration_resolver_uses_approved_ten_second_policy(
    duration,
    chunks,
    chunk_seconds,
):
    plan = resolve_easy_duration(duration)
    assert plan.duration_seconds == duration
    assert plan.chunks == chunks
    assert plan.chunk_seconds == chunk_seconds
    assert plan.chunks * plan.chunk_seconds == duration


@pytest.mark.parametrize("duration", (3, 481, True, 10.5))
def test_duration_resolver_rejects_values_outside_integer_production_contract(duration):
    with pytest.raises((TypeError, ValueError)):
        resolve_easy_duration(duration)


def test_decimal_mp_presets_follow_first_image_aspect_and_32_pixel_grid():
    first = _Image(1200, 900)
    draft = resolve_easy_resolution(
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_DRAFT,
        first_frame=first,
    )
    balanced = resolve_easy_resolution(
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_BALANCED,
        first_frame=first,
    )

    assert (draft.width, draft.height) == (480, 640)
    assert (balanced.width, balanced.height) == (672, 896)
    assert draft.aspect_source == "First Image"
    assert draft.width % 32 == draft.height % 32 == 0
    assert balanced.width % 32 == balanced.height % 32 == 0


@pytest.mark.parametrize(
    ("aspect", "expected"),
    (
        (EASY_ASPECT_SQUARE, (768, 768)),
        (EASY_ASPECT_LANDSCAPE, (1344, 768)),
        (EASY_ASPECT_PORTRAIT, (768, 1344)),
    ),
)
def test_native_768_representative_dimensions(aspect, expected):
    plan = resolve_easy_resolution(aspect=aspect, preset=EASY_PRESET_NATIVE)
    assert (plan.width, plan.height) == expected


def test_native_768_preserves_extreme_aspect_by_prioritizing_long_edge_cap():
    landscape = resolve_easy_resolution(
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_NATIVE,
        first_frame=_Image(900, 2100),
    )
    portrait = resolve_easy_resolution(
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_NATIVE,
        first_frame=_Image(2100, 900),
    )
    assert (landscape.width, landscape.height) == (1344, 576)
    assert (portrait.width, portrait.height) == (576, 1344)


def test_auto_without_first_image_falls_back_to_landscape():
    auto = resolve_easy_resolution(
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_DRAFT,
    )
    explicit = resolve_easy_resolution(
        aspect=EASY_ASPECT_LANDSCAPE,
        preset=EASY_PRESET_DRAFT,
    )
    assert (auto.width, auto.height) == (explicit.width, explicit.height)
    assert auto.aspect_source == "Landscape 16:9 fallback"


def test_custom_mp_is_not_clamped_to_native_and_reports_diagnostics():
    plan = resolve_easy_resolution(
        aspect=EASY_ASPECT_SQUARE,
        preset=EASY_PRESET_CUSTOM,
        custom_mp=3.0,
    )
    assert plan.width > 768
    assert plan.height > 768
    assert len(plan.warnings) == 2
    assert "Above Native 768" in plan.warnings[0]
    assert "High Custom MP" in plan.warnings[1]


def test_easy_schema_keeps_driving_audio_optional_and_scheduler_out():
    schema = H3ContinuumEasyV38.INPUT_TYPES()
    required = schema["required"]
    optional = schema["optional"]

    assert tuple(required)[:5] == ("model", "clip", "video_vae", "sampler", "sigmas")
    assert "scheduler" not in required
    assert "audio_vae" not in required
    assert optional["driving_audio"][0] == "AUDIO"
    assert optional["driving_audio"][1]["display_name"] == "Driving Audio"
    assert optional["audio_vae"][0] == "VAE"
    assert optional["audio_vae"][1]["display_name"] == "Driving Audio VAE"
    assert "first_frame" in optional
    assert "last_frame" in optional
    assert optional["references"][0] == EASY_REFERENCES_TYPE
    assert not {
        "driving_audio_vae",
        "reference_audio_1",
        "reference_audio_vae",
        "reference_video_1",
        "still_image_guide",
        "timeline_video",
    } & set(optional)
    assert required["duration"][1]["default"] == 10
    assert required["seed_mode"][1]["default"] == EASY_SEED_MODE_RANDOMIZE
    assert required["run_storage"][1]["default"] == "Off"
    assert required["prompt_text"][1]["multiline"] is True


def test_easy_node_is_registered_without_replacing_v37():
    assert NODE_CLASS_MAPPINGS["H3ContinuumEasyReferences"] is (
        H3ContinuumEasyReferences
    )
    assert NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumEasyReferences"] == (
        "H3 Continuum Easy References"
    )
    assert NODE_CLASS_MAPPINGS["H3ContinuumEasyV38"] is H3ContinuumEasyV38
    assert NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumEasyV38"] == (
        "H3 Continuum Easy V3.8"
    )
    assert "H3ContinuumEasyV38" not in root_nodes.NODE_CLASS_MAPPINGS
    assert "H3ContinuumSamplerV37" not in root_nodes.NODE_CLASS_MAPPINGS
    assert "H3ContinuumEasyReferences" not in root_nodes.NODE_CLASS_MAPPINGS
    assert H3ContinuumEasyReferences.DEPRECATED is False
    assert H3ContinuumEasyV38.DEPRECATED is False


def test_easy_references_schema_has_three_fixed_optional_image_sockets_only():
    schema = H3ContinuumEasyReferences.INPUT_TYPES()
    assert "required" not in schema
    assert tuple(schema["optional"]) == (
        "reference_image_1",
        "reference_image_2",
        "reference_image_3",
    )
    assert all(value[0] == "IMAGE" for value in schema["optional"].values())
    assert H3ContinuumEasyReferences.RETURN_TYPES == (EASY_REFERENCES_TYPE,)


@pytest.mark.parametrize(
    "active_slots",
    ((), (1,), (1, 2), (1, 2, 3), (2,)),
)
def test_easy_references_preserve_fixed_slots_for_v37_mapping(active_slots):
    images = {index: object() for index in range(1, 4)}
    kwargs = {
        f"reference_image_{index}": images[index]
        for index in active_slots
    }
    bundle = H3ContinuumEasyReferences().pack(**kwargs)[0]
    resolved = resolve_easy_references(bundle)

    assert resolved == tuple(
        images[index] if index in active_slots else None
        for index in range(1, 4)
    )


def test_easy_references_none_and_invalid_payload_contract():
    assert resolve_easy_references(None) == (None, None, None)
    with pytest.raises(TypeError, match="H3 Continuum Easy References"):
        resolve_easy_references("not-a-reference-bundle")
    with pytest.raises(ValueError, match="schema version"):
        resolve_easy_references({"schema_version": 999})


@pytest.mark.parametrize(
    ("duration", "expected_chunks", "expected_seconds"),
    ((10, 1, 10.0), (15, 2, 7.5), (20, 2, 10.0), (30, 3, 10.0)),
)
def test_easy_facade_maps_to_v37_without_copying_sampling(
    monkeypatch,
    duration,
    expected_chunks,
    expected_seconds,
):
    captured = {}

    def fake_v37_run(_self, **kwargs):
        captured.update(kwargs)
        return "video", "audio", {"plan": True}, "V3.7 status", None, "refine"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    outputs = H3ContinuumEasyV38().run(
        model="model",
        clip="clip",
        video_vae="video-vae",
        sampler="sampler",
        sigmas="sigmas",
        prompt_text="<|caption|> literal embedding:example",
        duration=duration,
        seed=42,
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_DRAFT,
        first_frame=_Image(1200, 900),
        last_frame=_Image(1200, 900),
    )

    assert captured["chunks"] == expected_chunks
    assert captured["chunk_seconds"] == expected_seconds
    assert captured["sequence_prompt"] == "<|caption|> literal embedding:example"
    assert captured["prompt_mode"] == "Auto"
    assert (captured["width"], captured["height"]) == (480, 640)
    assert captured["first_frame"] is not None
    assert captured["last_frame"] is not None
    assert captured["base_seed"] == 42
    assert captured["run_storage"] == "Off"
    assert captured["driving_audio"] is None
    assert captured["audio_vae"] is None
    assert len(outputs) == 5
    assert "FL2VA + Continuation" in outputs[3]
    assert "V3.7 status" in outputs[3]


def test_easy_driving_audio_and_vae_map_directly_to_v37(monkeypatch):
    captured = {}
    driving_audio = {"waveform": object(), "sample_rate": 32000}
    audio_vae = object()

    def fake_v37_run(_self, **kwargs):
        captured.update(kwargs)
        return "video", "audio", {"plan": True}, "V3.7 status", driving_audio

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    outputs = H3ContinuumEasyV38().run(
        model="model",
        clip="clip",
        video_vae="video-vae",
        sampler="sampler",
        sigmas="sigmas",
        prompt_text="prompt",
        duration=10,
        seed=42,
        driving_audio=driving_audio,
        audio_vae=audio_vae,
    )

    assert captured["driving_audio"] is driving_audio
    assert captured["audio_vae"] is audio_vae
    assert outputs[4] is driving_audio


def test_easy_audio_vae_only_remains_normal_generation(monkeypatch):
    captured = {}
    audio_vae = object()

    def fake_v37_run(_self, **kwargs):
        captured.update(kwargs)
        return "video", "generated-audio", {"plan": True}, "V3.7 status", None

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    outputs = H3ContinuumEasyV38().run(
        model="model",
        clip="clip",
        video_vae="video-vae",
        sampler="sampler",
        sigmas="sigmas",
        prompt_text="prompt",
        duration=10,
        seed=42,
        audio_vae=audio_vae,
    )

    assert captured["driving_audio"] is None
    assert captured["audio_vae"] is audio_vae
    assert outputs[1] == "generated-audio"
    assert outputs[4] is None


def test_easy_audio_without_vae_keeps_v37_validation(monkeypatch):
    def fake_v37_run(_self, **kwargs):
        if kwargs["driving_audio"] is not None and kwargs["audio_vae"] is None:
            raise ValueError("anchoring guide audio needs the audio_vae input")
        raise AssertionError("expected the existing V3.7 validation path")

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    with pytest.raises(ValueError, match="audio_vae input"):
        H3ContinuumEasyV38().run(
            model="model",
            clip="clip",
            video_vae="video-vae",
            sampler="sampler",
            sigmas="sigmas",
            prompt_text="prompt",
            duration=10,
            seed=42,
            driving_audio={"waveform": object(), "sample_rate": 32000},
        )


def test_easy_reference_mapping_matches_direct_v37_slots_and_output_geometry(
    monkeypatch,
):
    captured = {}
    first_ref, second_ref, third_ref = object(), object(), object()

    def fake_v37_run(_self, **kwargs):
        captured.update(kwargs)
        return "video", "audio", {"plan": True}, "V3.7 status", None, "refine"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    references = make_easy_references(first_ref, second_ref, third_ref)
    H3ContinuumEasyV38().run(
        model="model",
        clip="clip",
        video_vae="video-vae",
        sampler="sampler",
        sigmas="sigmas",
        prompt_text="Use <Picture 1>, <Picture 2>, and <Picture 3>.",
        duration=10,
        seed=42,
        aspect=EASY_ASPECT_PORTRAIT,
        preset=EASY_PRESET_DRAFT,
        references=references,
    )

    direct_v37_reference_kwargs = {
        "reference_image_1": first_ref,
        "reference_image_2": second_ref,
        "reference_image_3": third_ref,
    }
    assert {
        key: captured[key] for key in direct_v37_reference_kwargs
    } == direct_v37_reference_kwargs
    assert captured["reference_size"] == REFERENCE_SIZE_MATCH_OUTPUT
    assert (captured["width"], captured["height"]) == (416, 736)


@pytest.mark.parametrize(
    ("first_frame", "last_frame", "expected_mode"),
    (
        (_Image(640, 480), None, "Hybrid I2VA + Reference + Continuation"),
        (
            _Image(640, 480),
            _Image(640, 480),
            "Hybrid FL2VA + Reference + Continuation",
        ),
    ),
)
def test_easy_status_reports_first_last_with_reference(
    monkeypatch,
    first_frame,
    last_frame,
    expected_mode,
):
    def fake_v37_run(_self, **_kwargs):
        return "video", "audio", {"plan": True}, "V3.7 status", None, "refine"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    outputs = H3ContinuumEasyV38().run(
        model="model",
        clip="clip",
        video_vae="video-vae",
        sampler="sampler",
        sigmas="sigmas",
        prompt_text="Use <Picture 1>.",
        duration=10,
        seed=42,
        first_frame=first_frame,
        last_frame=last_frame,
        references=make_easy_references(object()),
    )
    assert expected_mode in outputs[3]


def test_easy_reference_gap_compacts_to_picture_one_in_existing_v37_preprocessing():
    second = torch.ones((1, 32, 32, 3))
    refs = resolve_easy_references(
        make_easy_references(reference_image_2=second)
    )
    assets = prepare_reference_assets(
        reference_image_1=refs[0],
        reference_image_2=refs[1],
        reference_image_3=refs[2],
        output_width=480,
        output_height=640,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )

    assert assets is not None
    assert assets.count == 1
    assert torch.equal(assets.images[0], second)
    assert validate_reference_prompts(["Use <Picture 1>."], assets.count) == ""
    assert "unavailable <Picture 2>" in validate_reference_prompts(
        ["Use <Picture 2>."], assets.count
    )


def test_easy_reference_size_uses_existing_v37_output_geometry_policy(monkeypatch):
    comfy_module = ModuleType("comfy")
    comfy_utils_module = ModuleType("comfy.utils")

    def common_upscale(image, width, height, _method, _crop):
        return torch.nn.functional.interpolate(
            image,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

    comfy_utils_module.common_upscale = common_upscale
    comfy_module.utils = comfy_utils_module
    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy_utils_module)
    source = torch.zeros((1, 512, 1024, 3))
    refs = resolve_easy_references(make_easy_references(source))
    assets = prepare_reference_assets(
        reference_image_1=refs[0],
        reference_image_2=refs[1],
        reference_image_3=refs[2],
        output_width=320,
        output_height=320,
        size_mode=REFERENCE_SIZE_MATCH_OUTPUT,
    )

    assert assets is not None
    height, width = assets.images[0].shape[1:3]
    assert width * height <= 320 * 320
    assert width % 32 == height % 32 == 0
    assert width == 2 * height


@pytest.mark.parametrize("duration", (10, 15, 20, 30))
def test_approved_easy_fl2va_durations_do_not_enter_five_second_terminal_merge(
    duration,
):
    plan = resolve_easy_duration(duration)
    enabled = _terminal_flf_merge_enabled(
        multi_chunk_flf=plan.chunks >= 2,
        chunks=plan.chunks,
        chunk_seconds=plan.chunk_seconds,
        prompt_hashes=["p"] * plan.chunks,
        timeline_video_source=None,
    )
    assert enabled is False


def _function_source(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


def test_easy_frontend_maps_only_randomize_fixed_and_hides_conditional_widgets(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")
    source = EASY_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setWidgetVisible", "attachRefresh"),
            _function_source(source, "attachRefresh", "createProjectId"),
            _function_source(source, "createProjectId", "seedControlWidget"),
            _function_source(source, "seedControlWidget", "configureEasyNode"),
            _function_source(source, "configureEasyNode", "configureEasyNodeAfterSetup"),
        )
    )
    script = f"""
const EASY_NODE_CLASS = "H3ContinuumEasyV38";
const CUSTOM_PRESET = "Custom";
const STORAGE_ENABLED = "Save + Auto Resume";
{functions}
function widget(name, value) {{
    return {{ name, value, type: "combo", options: {{}}, computeSize: () => [120, 20] }};
}}
const node = {{
    comfyClass: EASY_NODE_CLASS,
    widgets: [
        widget("preset", "Draft — 0.30 MP"),
        widget("custom_mp", 0.3),
        widget("run_storage", "Off"),
        widget("run_name", ""),
        widget("seed_mode", "Randomize"),
        widget("control_after_generate", "fixed"),
        widget("project_id", "test-project"),
    ],
    setDirtyCanvas() {{}},
}};
configureEasyNode(node);
const observed = {{
    initialControl: findWidget(node, "control_after_generate").value,
    controlHidden: findWidget(node, "control_after_generate").hidden,
    customHidden: findWidget(node, "custom_mp").hidden,
    runNameHidden: findWidget(node, "run_name").hidden,
    projectHidden: findWidget(node, "project_id").hidden,
}};
findWidget(node, "seed_mode").value = "Fixed";
findWidget(node, "seed_mode").callback("Fixed");
findWidget(node, "preset").value = "Custom";
findWidget(node, "preset").callback("Custom");
findWidget(node, "run_storage").value = "Save + Auto Resume";
findWidget(node, "run_storage").callback("Save + Auto Resume");
observed.fixedControl = findWidget(node, "control_after_generate").value;
observed.customVisible = !findWidget(node, "custom_mp").hidden;
observed.runNameVisible = !findWidget(node, "run_name").hidden;
console.log(JSON.stringify(observed));
"""
    script_path = tmp_path / "easy-v38-frontend.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "initialControl": "randomize",
        "controlHidden": True,
        "customHidden": True,
        "runNameHidden": True,
        "projectHidden": True,
        "fixedControl": "fixed",
        "customVisible": True,
        "runNameVisible": True,
    }
