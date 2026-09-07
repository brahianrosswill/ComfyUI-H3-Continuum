from __future__ import annotations

from fractions import Fraction
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3 import easy_video_nodes as video_nodes
from ComfyUI_H3_Continuum_Join.v3.easy_video_nodes import (
    ExecutionBlocker,
    H3ContinuumLoadVideo,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    _resample_video_frames,
    load_video_components,
)


ROOT = Path(__file__).resolve().parents[1]
COMMON_FRONTEND_PATH = ROOT / "web" / "easy_bypass_toggle.js"
FRONTEND_PATH = ROOT / "web" / "easy_load_video.js"


def _frames(count: int) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.float32).view(count, 1, 1, 1)
    return values.expand(-1, 2, 2, 3).clone()


def test_schema_registration_and_minimal_core_contract():
    schema = H3ContinuumLoadVideo.define_schema()
    assert NODE_CLASS_MAPPINGS["H3ContinuumLoadVideo"] is H3ContinuumLoadVideo
    assert root_nodes.NODE_CLASS_MAPPINGS["H3ContinuumLoadVideo"] is (
        H3ContinuumLoadVideo
    )
    assert NODE_DISPLAY_NAME_MAPPINGS["H3ContinuumLoadVideo"] == (
        "H3 Continuum Load Video"
    )
    assert schema.node_id == "H3ContinuumLoadVideo"
    assert schema.display_name == "H3 Continuum Load Video"
    assert schema.category == "MiniMax H3/Continuum"
    assert [item.id for item in schema.inputs] == [
        "enable_video",
        "file",
        "force_rate",
    ]
    assert schema.inputs[0].default is True
    assert schema.inputs[1].upload.value == "video_upload"
    assert schema.inputs[2].default == 24.0
    assert schema.inputs[2].min == 0.0
    assert schema.inputs[2].max == 120.0
    assert schema.inputs[2].step == 0.01
    assert len(schema.outputs) == 2
    source = inspect.getsource(video_nodes)
    assert "InputImpl.VideoFromFile" in source
    assert "import av" not in source
    assert "subprocess" not in source


def test_24_to_24_returns_the_identical_image_batch():
    images = _frames(240)
    result = _resample_video_frames(images, Fraction(24, 1), 24.0)
    assert result is images


def test_30_to_24_preserves_ten_second_duration():
    result = _resample_video_frames(_frames(300), Fraction(30, 1), 24.0)
    assert result.shape[0] == 240
    assert result.shape[0] / 24.0 == 10.0


def test_24_to_30_preserves_duration_and_duplicates_frames():
    result = _resample_video_frames(_frames(240), Fraction(24, 1), 30.0)
    assert result.shape[0] == 300
    assert result.shape[0] / 30.0 == 10.0
    source_values = result[:, 0, 0, 0]
    assert torch.unique(source_values).numel() == 240


def test_2997_to_24_uses_fraction_without_off_by_one():
    result = _resample_video_frames(
        _frames(300),
        Fraction(30_000, 1_001),
        24.0,
    )
    assert result.shape[0] == 240


def test_force_rate_zero_returns_the_identical_image_batch():
    images = _frames(123)
    result = _resample_video_frames(images, Fraction(30_000, 1_001), 0.0)
    assert result is images


def test_audio_is_bit_exact_and_core_components_are_used(monkeypatch):
    images = _frames(300)
    waveform = torch.linspace(-1.0, 1.0, 64).reshape(1, 1, 64)
    audio = {"waveform": waveform, "sample_rate": 32_000}
    calls = []

    class FakeVideo:
        def get_components(self):
            calls.append("get_components")
            return SimpleNamespace(
                images=images,
                audio=audio,
                frame_rate=Fraction(30, 1),
            )

    def fake_video_from_file(path):
        calls.append(("VideoFromFile", path))
        return FakeVideo()

    monkeypatch.setattr(
        video_nodes.folder_paths,
        "get_annotated_filepath",
        lambda file: f"D:/input/{file}",
    )
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", fake_video_from_file)
    output_images, output_audio = load_video_components(
        file="source.mp4",
        enable_video=True,
        force_rate=24.0,
    )

    assert output_images.shape[0] == 240
    assert output_audio is audio
    assert output_audio["sample_rate"] == 32_000
    assert output_audio["waveform"].shape == waveform.shape
    assert torch.equal(output_audio["waveform"], waveform)
    assert calls == [("VideoFromFile", "D:/input/source.mp4"), "get_components"]


def test_enable_video_off_never_resolves_or_decodes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled video must not touch Core decode or the file")

    monkeypatch.setattr(video_nodes.folder_paths, "get_annotated_filepath", forbidden)
    monkeypatch.setattr(video_nodes.InputImpl, "VideoFromFile", forbidden)
    image, audio = load_video_components(
        file="missing-old-file.mp4",
        enable_video=False,
        force_rate=24.0,
    )
    assert isinstance(image, ExecutionBlocker)
    assert isinstance(audio, ExecutionBlocker)
    assert image.message is None
    assert audio.message is None


def test_audio_missing_blocks_only_audio_branch(monkeypatch):
    images = _frames(24)

    class FakeVideo:
        def get_components(self):
            return SimpleNamespace(
                images=images,
                audio=None,
                frame_rate=Fraction(24, 1),
            )

    monkeypatch.setattr(
        video_nodes.folder_paths,
        "get_annotated_filepath",
        lambda file: f"D:/input/{file}",
    )
    monkeypatch.setattr(
        video_nodes.InputImpl,
        "VideoFromFile",
        lambda path: FakeVideo(),
    )
    output_images, output_audio = load_video_components(
        file="silent.mp4",
        enable_video=True,
        force_rate=24.0,
    )
    assert output_images is images
    assert isinstance(output_audio, ExecutionBlocker)
    assert output_audio.message is None


def test_validation_and_fingerprint_skip_stale_file_when_disabled(
    monkeypatch,
    tmp_path: Path,
):
    calls = []

    def exists(file):
        calls.append(("exists", file))
        return file == "valid.mp4"

    monkeypatch.setattr(video_nodes.folder_paths, "exists_annotated_filepath", exists)
    assert H3ContinuumLoadVideo.validate_inputs(False, "missing.mp4", 24.0) is True
    assert calls == []
    disabled_fingerprint = H3ContinuumLoadVideo.fingerprint_inputs(
        False,
        "missing.mp4",
        24.0,
    )
    assert disabled_fingerprint == (False, "missing.mp4", 24.0, None)
    assert calls == []

    invalid = H3ContinuumLoadVideo.validate_inputs(True, "missing.mp4", 24.0)
    assert invalid == "Invalid video file: missing.mp4"
    assert H3ContinuumLoadVideo.validate_inputs(True, "valid.mp4", 24.0) is True

    video_path = tmp_path / "valid.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(
        video_nodes.folder_paths,
        "get_annotated_filepath",
        lambda file: str(video_path),
    )
    fingerprint = H3ContinuumLoadVideo.fingerprint_inputs(
        True,
        "valid.mp4",
        30.0,
    )
    assert fingerprint[:3] == (True, "valid.mp4", 30.0)
    assert fingerprint[3] == os.path.getmtime(video_path)


def test_enable_video_false_uses_native_bypass_appearance(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")
    functions = COMMON_FRONTEND_PATH.read_text(encoding="utf-8").replace(
        "export ", ""
    )
    script = f"""
{functions}
function plainWidget(name, value) {{
    return {{ name, value, options: {{}}, serialize: true }};
}}
const enableWidget = plainWidget("enable_video", false);
const fileWidget = plainWidget("file", "clip.mp4");
const rateWidget = plainWidget("force_rate", 24.0);
const uploadWidget = plainWidget("upload", "video");
const node = {{
    comfyClass: "H3ContinuumLoadVideo",
    mode: 0,
    widgets: [enableWidget, fileWidget, rateWidget, uploadWidget],
    dirtyCalls: 0,
    setDirtyCanvas() {{ this.dirtyCalls += 1; }},
    serialize() {{
        return {{
            mode: this.mode,
            widgets_values: this.widgets.map((widget) => widget.value),
        }};
    }},
    configure(info) {{
        this.mode = info.mode;
        for (let index = 0; index < info.widgets_values.length; index++) {{
            this.widgets[index].value = info.widgets_values[index];
        }}
    }},
}};
configureExistingEasyBypassWidgetNode(node, {{
    nodeClass: "H3ContinuumLoadVideo",
    widgetNames: ["enable_video", "Enable Video"],
}});
const initial = {{
    mode: node.mode,
    enabled: enableWidget.value,
    names: node.widgets.map((widget) => widget.name),
    saved: node.serialize(),
}};
enableWidget.callback(true);
const enabled = {{ mode: node.mode, enabled: enableWidget.value }};
node.mode = 4;
const externalBypass = {{ mode: node.mode, enabled: enableWidget.value }};
node.mode = 0;
const externalUnbypass = {{ mode: node.mode, enabled: enableWidget.value }};
node.configure({{
    mode: 0,
    widgets_values: [false, "reloaded.mp4", 30.0, "video"],
}});
configureExistingEasyBypassWidgetNode(node, {{
    nodeClass: "H3ContinuumLoadVideo",
    widgetNames: ["enable_video", "Enable Video"],
}});
const reloaded = {{
    mode: node.mode,
    enabled: enableWidget.value,
    file: fileWidget.value,
    rate: rateWidget.value,
    names: node.widgets.map((widget) => widget.name),
    saved: node.serialize(),
}};
console.log(JSON.stringify({{ initial, enabled, externalBypass, externalUnbypass, reloaded }}));
"""
    script_path = tmp_path / "easy-load-video-frontend.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "initial": {
            "mode": 4,
            "enabled": False,
            "names": ["enable_video", "file", "force_rate", "upload"],
            "saved": {
                "mode": 4,
                "widgets_values": [False, "clip.mp4", 24, "video"],
            },
        },
        "enabled": {"mode": 0, "enabled": True},
        "externalBypass": {"mode": 4, "enabled": False},
        "externalUnbypass": {"mode": 0, "enabled": True},
        "reloaded": {
            "mode": 4,
            "enabled": False,
            "file": "reloaded.mp4",
            "rate": 30,
            "names": ["enable_video", "file", "force_rate", "upload"],
            "saved": {
                "mode": 4,
                "widgets_values": [False, "reloaded.mp4", 30, "video"],
            },
        },
    }


def test_video_bypass_appearance_uses_native_mode_without_hardcoded_color():
    source = FRONTEND_PATH.read_text(encoding="utf-8")
    common = COMMON_FRONTEND_PATH.read_text(encoding="utf-8")
    assert 'from "./easy_bypass_toggle.js"' in source
    assert "configureExistingEasyBypassWidgetNode" in source
    assert 'widgetNames: ["enable_video", "Enable Video"]' in source
    assert "NODE_MODE_BYPASS" in common
    assert "node.color" not in source
    assert "node.bgcolor" not in source
    assert "setInterval" not in common
    assert "setTimeout" not in common
