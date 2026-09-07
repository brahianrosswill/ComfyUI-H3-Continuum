from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3.easy_audio_nodes import (
    H3EasyLoadAudio,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "v3" / "easy_audio_nodes.py"
COMMON_FRONTEND_PATH = ROOT / "web" / "easy_bypass_toggle.js"
FRONTEND_PATH = ROOT / "web" / "easy_load_audio.js"


def test_easy_load_audio_is_public_and_keeps_core_audio_contract():
    schema = H3EasyLoadAudio.define_schema()
    assert NODE_CLASS_MAPPINGS["H3EasyLoadAudio"] is H3EasyLoadAudio
    assert root_nodes.NODE_CLASS_MAPPINGS["H3EasyLoadAudio"] is H3EasyLoadAudio
    assert NODE_DISPLAY_NAME_MAPPINGS["H3EasyLoadAudio"] == "H3 Continuum Load Audio"
    assert root_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3EasyLoadAudio"] == (
        "H3 Continuum Load Audio"
    )
    assert schema.node_id == "H3EasyLoadAudio"
    assert schema.display_name == "H3 Continuum Load Audio"
    source = inspect.getsource(H3EasyLoadAudio)
    assert "def execute" not in source
    assert "def fingerprint_inputs" not in source
    assert "def validate_inputs" not in source


def test_easy_load_audio_inherits_and_delegates_to_runtime_core(monkeypatch):
    fake_package = ModuleType("comfy_extras")
    fake_audio_module = ModuleType("comfy_extras.nodes_audio")

    class FakeCoreLoadAudio:
        @classmethod
        def define_schema(cls):
            return SimpleNamespace(
                node_id="LoadAudio",
                display_name="Load Audio",
                category="audio",
                essentials_category="Audio",
                description="core",
                search_aliases=["audio file"],
            )

        @classmethod
        def execute(cls, audio):
            return f"loaded:{audio}"

        @classmethod
        def fingerprint_inputs(cls, audio):
            return f"fingerprint:{audio}"

        @classmethod
        def validate_inputs(cls, audio):
            return audio == "valid.wav"

    fake_audio_module.LoadAudio = FakeCoreLoadAudio
    monkeypatch.setitem(sys.modules, "comfy_extras", fake_package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", fake_audio_module)
    spec = importlib.util.spec_from_file_location("h3_easy_audio_runtime_probe", BACKEND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert issubclass(module.H3EasyLoadAudio, FakeCoreLoadAudio)
    schema = module.H3EasyLoadAudio.define_schema()
    assert schema.node_id == "H3EasyLoadAudio"
    assert schema.display_name == "H3 Continuum Load Audio"
    assert module.H3EasyLoadAudio.execute("voice.wav") == "loaded:voice.wav"
    assert module.H3EasyLoadAudio.fingerprint_inputs("voice.wav") == (
        "fingerprint:voice.wav"
    )
    assert module.H3EasyLoadAudio.validate_inputs("valid.wav") is True


def test_easy_load_audio_frontend_preserves_filename_player_and_mode(tmp_path):
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
const audioWidget = plainWidget("audio", "voice.wav");
const playerWidget = plainWidget("audioUI", {{ position: 1.25, volume: 0.8 }});
const node = {{
    comfyClass: "H3EasyLoadAudio",
    mode: 0,
    widgets: [audioWidget, playerWidget],
    addWidget(type, name, value, callback, options) {{
        const widget = {{ type, name, value, callback, options, serialize: true }};
        this.widgets.push(widget);
        return widget;
    }},
    setDirtyCanvas() {{}},
    serialize() {{
        return {{
            mode: this.mode,
            widgets_values: this.widgets
                .filter((widget) => widget.serialize !== false)
                .map((widget) => widget.value),
        }};
    }},
    configure(info) {{
        this.mode = info.mode;
        for (let index = 0; index < info.widgets_values.length; index++) {{
            this.widgets[index].value = info.widgets_values[index];
        }}
    }},
}};
configureEasyBypassToggleNode(node, {{
    nodeClass: "H3EasyLoadAudio",
    widgetName: "Enable Audio",
    tooltip: "audio tooltip",
}});
const toggle = findEasyBypassWidget(node);
const initial = {{ names: node.widgets.map((w) => w.name), saved: node.serialize() }};
toggle.callback(false);
const off = {{ mode: node.mode, toggle: toggle.value, saved: node.serialize() }};
toggle.callback(true);
const onAgain = {{ mode: node.mode, toggle: toggle.value, saved: node.serialize() }};
node.mode = 4;
const externalBypass = {{ mode: node.mode, toggle: toggle.value }};
node.configure({{
    mode: 4,
    widgets_values: ["reloaded.wav", {{ position: 2.5, volume: 0.6 }}],
}});
const reloaded = {{
    mode: node.mode,
    toggle: toggle.value,
    audio: audioWidget.value,
    player: playerWidget.value,
    names: node.widgets.map((w) => w.name),
}};
console.log(JSON.stringify({{ initial, off, onAgain, externalBypass, reloaded }}));
"""
    script_path = tmp_path / "easy-load-audio-frontend.js"
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
            "names": ["Enable Audio", "audio", "audioUI"],
            "saved": {
                "mode": 0,
                "widgets_values": [
                    "voice.wav",
                    {"position": 1.25, "volume": 0.8},
                ],
            },
        },
        "off": {
            "mode": 4,
            "toggle": False,
            "saved": {
                "mode": 4,
                "widgets_values": [
                    "voice.wav",
                    {"position": 1.25, "volume": 0.8},
                ],
            },
        },
        "onAgain": {
            "mode": 0,
            "toggle": True,
            "saved": {
                "mode": 0,
                "widgets_values": [
                    "voice.wav",
                    {"position": 1.25, "volume": 0.8},
                ],
            },
        },
        "externalBypass": {"mode": 4, "toggle": False},
        "reloaded": {
            "mode": 4,
            "toggle": False,
            "audio": "reloaded.wav",
            "player": {"position": 2.5, "volume": 0.6},
            "names": ["Enable Audio", "audio", "audioUI"],
        },
    }


def test_easy_load_audio_uses_shared_non_polling_bypass_helper():
    source = FRONTEND_PATH.read_text(encoding="utf-8")
    common = COMMON_FRONTEND_PATH.read_text(encoding="utf-8")
    assert 'from "./easy_bypass_toggle.js"' in source
    assert 'widgetName: "Enable Audio"' in source
    assert "setInterval" not in common
    assert "setTimeout" not in common
    assert "node.mode = enabled ? NODE_MODE_ALWAYS : NODE_MODE_BYPASS" in common
