from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType

import pytest

from ComfyUI_H3_Continuum_Join import nodes as root_nodes
from ComfyUI_H3_Continuum_Join.v3.easy_image_nodes import (
    H3EasyLoadImage,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)


ROOT = Path(__file__).resolve().parents[1]
BACKEND_PATH = ROOT / "v3" / "easy_image_nodes.py"
FRONTEND_PATH = ROOT / "web" / "easy_load_image.js"
COMMON_FRONTEND_PATH = ROOT / "web" / "easy_bypass_toggle.js"


def test_easy_load_image_is_public_and_keeps_core_image_mask_contract():
    assert NODE_CLASS_MAPPINGS["H3EasyLoadImage"] is H3EasyLoadImage
    assert root_nodes.NODE_CLASS_MAPPINGS["H3EasyLoadImage"] is H3EasyLoadImage
    assert NODE_DISPLAY_NAME_MAPPINGS["H3EasyLoadImage"] == "H3 Continuum Load Image"
    assert root_nodes.NODE_DISPLAY_NAME_MAPPINGS["H3EasyLoadImage"] == (
        "H3 Continuum Load Image"
    )
    assert H3EasyLoadImage.RETURN_TYPES == ("IMAGE", "MASK")
    assert H3EasyLoadImage.FUNCTION == "load_image"
    assert H3EasyLoadImage.DEPRECATED is False
    assert "def load_image" not in inspect.getsource(H3EasyLoadImage)


def test_easy_load_image_inherits_the_runtime_core_loader(monkeypatch):
    fake_core = ModuleType("nodes")

    class FakeCoreLoadImage:
        RETURN_TYPES = ("IMAGE", "MASK")
        FUNCTION = "load_image"

        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": (("core.png",), {"image_upload": True})}}

        def load_image(self, image):
            return f"image:{image}", f"mask:{image}"

    fake_core.LoadImage = FakeCoreLoadImage
    monkeypatch.setitem(sys.modules, "nodes", fake_core)
    spec = importlib.util.spec_from_file_location("h3_easy_image_runtime_probe", BACKEND_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert issubclass(module.H3EasyLoadImage, FakeCoreLoadImage)
    assert module.H3EasyLoadImage.INPUT_TYPES() == FakeCoreLoadImage.INPUT_TYPES()
    assert module.H3EasyLoadImage().load_image("core.png") == (
        "image:core.png",
        "mask:core.png",
    )


def _frontend_functions(source: str) -> str:
    return source.replace("export ", "")


def test_easy_load_image_frontend_uses_mode_as_single_source_of_truth(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")
    source = COMMON_FRONTEND_PATH.read_text(encoding="utf-8")
    functions = _frontend_functions(source)
    script = f"""
const EASY_LOAD_IMAGE_CLASS = "H3EasyLoadImage";
{functions}
function plainWidget(name, value) {{
    return {{ name, value, options: {{}}, serialize: true }};
}}
const imageWidget = plainWidget("image", "first.png");
const uploadWidget = plainWidget("upload", "image");
const node = {{
    comfyClass: EASY_LOAD_IMAGE_CLASS,
    mode: NODE_MODE_ALWAYS,
    widgets: [imageWidget, uploadWidget],
    dirtyCalls: 0,
    addWidget(type, name, value, callback, options) {{
        const widget = {{ type, name, value, callback, options, serialize: true }};
        this.widgets.push(widget);
        return widget;
    }},
    setDirtyCanvas() {{ this.dirtyCalls += 1; }},
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
    nodeClass: EASY_LOAD_IMAGE_CLASS,
    widgetName: "Enable Image",
    tooltip: "image tooltip",
}});
const toggle = findEasyBypassWidget(node);
const initial = {{
    names: node.widgets.map((widget) => widget.name),
    toggle: toggle.value,
    saved: node.serialize(),
}};
toggle.value = false;
toggle.callback(false);
const toggledOff = {{ mode: node.mode, toggle: toggle.value, saved: node.serialize() }};
node.mode = NODE_MODE_ALWAYS;
const externalUnbypass = {{ mode: node.mode, toggle: toggle.value }};
node.mode = NODE_MODE_BYPASS;
const externalBypass = {{ mode: node.mode, toggle: toggle.value }};
node.configure({{ mode: NODE_MODE_BYPASS, widgets_values: ["reloaded.png", "image"] }});
const reloaded = {{
    mode: node.mode,
    toggle: toggle.value,
    image: imageWidget.value,
    upload: uploadWidget.value,
    names: node.widgets.map((widget) => widget.name),
}};
console.log(JSON.stringify({{ initial, toggledOff, externalUnbypass, externalBypass, reloaded }}));
"""
    script_path = tmp_path / "easy-load-image-frontend.js"
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
            "names": ["Enable Image", "image", "upload"],
            "toggle": True,
            "saved": {"mode": 0, "widgets_values": ["first.png", "image"]},
        },
        "toggledOff": {
            "mode": 4,
            "toggle": False,
            "saved": {"mode": 4, "widgets_values": ["first.png", "image"]},
        },
        "externalUnbypass": {"mode": 0, "toggle": True},
        "externalBypass": {"mode": 4, "toggle": False},
        "reloaded": {
            "mode": 4,
            "toggle": False,
            "image": "reloaded.png",
            "upload": "image",
            "names": ["Enable Image", "image", "upload"],
        },
    }


def test_shared_bypass_helper_redraws_only_enable_row_at_full_opacity(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend drawing regression")
    functions = _frontend_functions(
        COMMON_FRONTEND_PATH.read_text(encoding="utf-8")
    )
    script = f"""
{functions}
function widget(name, value) {{
    return {{ name, value, options: {{}}, serialize: true }};
}}
function nodeFor(nodeClass, enableName, existing) {{
    const enable = existing ? widget(enableName, false) : undefined;
    const node = {{
        comfyClass: nodeClass,
        mode: NODE_MODE_ALWAYS,
        widgets: existing ? [enable, widget("file", "media.bin")] : [widget("file", "media.bin")],
        drawCalls: [],
        addWidget(type, name, value, callback, options) {{
            const created = {{ type, name, value, callback, options, serialize: true }};
            this.widgets.push(created);
            return created;
        }},
        drawWidgets(ctx, options) {{
            this.drawCalls.push({{
                names: this.widgets.map((item) => item.name),
                alpha: options.editorAlpha,
            }});
        }},
        setDirtyCanvas() {{}},
    }};
    if (existing) {{
        configureExistingEasyBypassWidgetNode(node, {{
            nodeClass,
            widgetNames: [enableName],
        }});
    }} else {{
        configureEasyBypassToggleNode(node, {{
            nodeClass,
            widgetName: enableName,
            tooltip: "enable tooltip",
        }});
        node.mode = NODE_MODE_BYPASS;
    }}
    const toggle = findEasyBypassWidget(node);
    node.drawWidgets({{}}, {{ editorAlpha: 0.2, lowQuality: false }});
    const bypassCalls = node.drawCalls.slice();
    toggle.callback(true);
    node.drawCalls = [];
    node.drawWidgets({{}}, {{ editorAlpha: 1, lowQuality: false }});
    return {{
        bypassCalls,
        enabledCalls: node.drawCalls,
        mode: node.mode,
        toggle: toggle.value,
        clickable: typeof toggle.callback === "function",
        visible: toggle.hidden !== true,
    }};
}}
console.log(JSON.stringify({{
    image: nodeFor("H3EasyLoadImage", "Enable Image", false),
    audio: nodeFor("H3EasyLoadAudio", "Enable Audio", false),
    video: nodeFor("H3ContinuumLoadVideo", "enable_video", true),
}}));
"""
    script_path = tmp_path / "easy-loader-bypass-foreground.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "image": {
            "bypassCalls": [
                {"names": ["Enable Image", "file"], "alpha": 0.2},
                {"names": ["Enable Image"], "alpha": 1},
            ],
            "enabledCalls": [
                {"names": ["Enable Image", "file"], "alpha": 1}
            ],
            "mode": 0,
            "toggle": True,
            "clickable": True,
            "visible": True,
        },
        "audio": {
            "bypassCalls": [
                {"names": ["Enable Audio", "file"], "alpha": 0.2},
                {"names": ["Enable Audio"], "alpha": 1},
            ],
            "enabledCalls": [
                {"names": ["Enable Audio", "file"], "alpha": 1}
            ],
            "mode": 0,
            "toggle": True,
            "clickable": True,
            "visible": True,
        },
        "video": {
            "bypassCalls": [
                {"names": ["enable_video", "file"], "alpha": 0.2},
                {"names": ["enable_video"], "alpha": 1},
            ],
            "enabledCalls": [
                {"names": ["enable_video", "file"], "alpha": 1}
            ],
            "mode": 0,
            "toggle": True,
            "clickable": True,
            "visible": True,
        },
    }


def test_easy_load_image_frontend_has_no_backend_boolean_or_polling():
    source = FRONTEND_PATH.read_text(encoding="utf-8")
    common = COMMON_FRONTEND_PATH.read_text(encoding="utf-8")
    assert "setInterval" not in common
    assert "setTimeout" not in common
    assert "node.mode = enabled ? NODE_MODE_ALWAYS : NODE_MODE_BYPASS" in common
    assert "widget.options.serialize = false" in common
    assert 'from "./easy_bypass_toggle.js"' in source
    assert "configureEasyBypassToggleNode" in source
