import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import H3ContinuumSamplerV38


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID_JS = ROOT / "web" / "project_id.js"


def _function_source(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


def test_v38_public_sampler_inputs_all_have_hover_help():
    schema = H3ContinuumSamplerV38.INPUT_TYPES()
    missing = []
    for group in ("required", "optional"):
        for name, definition in schema[group].items():
            metadata = definition[1] if len(definition) > 1 else {}
            if not str(metadata.get("tooltip", "")).strip():
                missing.append(f"{group}.{name}")
    assert missing == []


def test_v38_main_facade_help_covers_every_visible_control():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    facade_names = (
        "FACADE_PROMPT_FORMAT_WIDGET",
        "FACADE_CONTINUITY_WIDGET",
        "FACADE_BASE_SEED_WIDGET",
        "FACADE_CONTROL_AFTER_WIDGET",
        "FACADE_AUDIO_CONTINUITY_WIDGET",
        "FACADE_CHUNKS_WIDGET",
        "FACADE_SECONDS_WIDGET",
        "FACADE_TOTAL_LENGTH_WIDGET",
        "FACADE_GENERATE_WIDGET",
        "FACADE_SIZE_SOURCE_WIDGET",
        "FACADE_RESOLUTION_WIDGET",
        "FACADE_CUSTOM_MP_WIDGET",
        "FACADE_WIDTH_WIDGET",
        "FACADE_HEIGHT_WIDGET",
        "FACADE_SAVE_WIDGET",
        "FACADE_READY_WIDGET",
        "FACADE_REFERENCE_SIZE_WIDGET",
        "FACADE_VIDEO_SIZE_WIDGET",
        "FACADE_ADVANCED_WIDGET",
    )
    for name in facade_names:
        assert f"[{name}]" in source

    assert "function setWidgetTooltip(widget, tooltip)" in source
    assert "widget.tooltip = tooltip;" in source
    assert "widget.options.tooltip = tooltip;" in source
    assert "function facadeWidgetTooltip(node, name)" in source
    assert "function applyV38WidgetHelp(node)" in source
    assert "applyV38WidgetHelp(node);" in source


def test_v38_size_help_explains_t2va_i2va_and_resolution_choices():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    for marker in (
        "Use this for T2VA",
        "Use this for I2VA or FL2VA",
        "Draft — 0.30 MP",
        "Balanced — 0.60 MP",
        "Native 768",
        "Efficient - 0.4 MP",
        "Match Output",
        "Review Each Chunk",
        "Render History",
    ):
        assert marker in source


def test_v38_hover_help_is_attached_and_updates_with_current_selection(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend tooltip regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    constants = source[
        source.index("const PROJECT_WIDGET") : source.index("const V38_VIEW_PROPERTY")
    ]
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "linkedInput", "attachRefresh"),
            _function_source(
                source, "facadeProductionWidgets", "addFacadeWidget"
            ),
            _function_source(source, "setWidgetTooltip", "facadeWidgetTooltip"),
            _function_source(source, "facadeWidgetTooltip", "applyV38WidgetHelp"),
            _function_source(
                source, "applyV38WidgetHelp", "installCompactNumberDisplay"
            ),
        )
    )
    script = f"""
{constants}
{functions}

const facadeNames = [
  FACADE_PROMPT_FORMAT_WIDGET, FACADE_CONTINUITY_WIDGET, FACADE_BASE_SEED_WIDGET,
  FACADE_CONTROL_AFTER_WIDGET, FACADE_AUDIO_CONTINUITY_WIDGET,
  FACADE_CHUNKS_WIDGET, FACADE_SECONDS_WIDGET, FACADE_TOTAL_LENGTH_WIDGET,
  FACADE_SIZE_SOURCE_WIDGET, FACADE_RESOLUTION_WIDGET, FACADE_CUSTOM_MP_WIDGET,
  FACADE_WIDTH_WIDGET, FACADE_HEIGHT_WIDGET, FACADE_SAVE_WIDGET,
  FACADE_GENERATE_WIDGET, FACADE_READY_WIDGET, FACADE_REFERENCE_SIZE_WIDGET,
  FACADE_VIDEO_SIZE_WIDGET, FACADE_ADVANCED_WIDGET,
];
const sourceValues = {{
  [GENERATION_MODE_WIDGET]: GENERATION_MODE_REVIEW,
  [SIZE_SOURCE_WIDGET]: SIZE_SOURCE_FIRST_IMAGE,
  [RESOLUTION_PRESET_WIDGET]: RESOLUTION_PRESET_DRAFT,
  [RUN_STORAGE_WIDGET]: "Save + Auto Resume",
  [REFERENCE_SIZE_WIDGET]: "Match Output",
  [VIDEO_REFERENCE_SIZE_WIDGET]: "Efficient - 0.4 MP",
}};
const node = {{
  widgets: [],
  inputs: [{{ name: "first_frame", link: 1 }}],
}};
for (const [name, value] of Object.entries(sourceValues)) {{
  node.widgets.push({{ name, value, options: {{}} }});
}}
for (const name of Object.keys(ADVANCED_WIDGET_HELP)) {{
  if (!node.widgets.some((widget) => widget.name === name)) {{
    node.widgets.push({{ name, value: null, options: {{}} }});
  }}
}}
for (const name of facadeNames) {{
  node.widgets.push({{ name, value: null, options: {{}}, [FACADE_TRANSIENT_WIDGET]: true }});
}}

applyV38WidgetHelp(node);
const firstImageHelp = findWidget(node, FACADE_SIZE_SOURCE_WIDGET).tooltip;
const firstPass = {{
  missingFacade: facadeProductionWidgets(node).filter((widget) => !widget.tooltip).map((widget) => widget.name),
  missingFacadeOptions: facadeProductionWidgets(node).filter((widget) => !widget.options.tooltip).map((widget) => widget.name),
  missingAdvanced: Object.keys(ADVANCED_WIDGET_HELP).filter((name) => !findWidget(node, name)?.tooltip),
  firstImageHelp,
  reviewHelp: findWidget(node, FACADE_GENERATE_WIDGET).tooltip,
}};

findWidget(node, SIZE_SOURCE_WIDGET).value = SIZE_SOURCE_MANUAL;
findWidget(node, RESOLUTION_PRESET_WIDGET).value = RESOLUTION_PRESET_NATIVE;
findWidget(node, RUN_STORAGE_WIDGET).value = "Off";
applyV38WidgetHelp(node);
console.log(JSON.stringify({{
  firstPass,
  manualHelp: findWidget(node, FACADE_SIZE_SOURCE_WIDGET).tooltip,
  nativeHelp: findWidget(node, FACADE_RESOLUTION_WIDGET).tooltip,
  storageOffHelp: findWidget(node, FACADE_SAVE_WIDGET).tooltip,
}}));
"""
    script_path = tmp_path / "v38-hover-help.mjs"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed["firstPass"]["missingFacade"] == []
    assert observed["firstPass"]["missingFacadeOptions"] == []
    assert observed["firstPass"]["missingAdvanced"] == []
    assert "I2VA or FL2VA" in observed["firstPass"]["firstImageHelp"]
    assert "Review Each Chunk" in observed["firstPass"]["reviewHelp"]
    assert "Set Chunks to 2 or more" in observed["firstPass"]["reviewHelp"]
    assert "T2VA" in observed["manualHelp"]
    assert "Native 768" in observed["nativeHelp"]
    assert "does not keep resumable chunk history" in observed["storageOffHelp"]
