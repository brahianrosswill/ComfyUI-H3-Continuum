from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ComfyUI_H3_Continuum_Join.v3.driving_nodes import (
    H3ContinuumSamplerV37,
    H3ContinuumSamplerV38,
)
from ComfyUI_H3_Continuum_Join.v3.nodes import (
    _format_review_status,
    _partial_review_warning,
)
from ComfyUI_H3_Continuum_Join.v3.review_control import (
    EXECUTION_MODE_FULL_RUN,
    EXECUTION_MODE_REVIEW_CONTINUE,
    EXECUTION_MODE_REVIEW_FINISH,
    EXECUTION_MODE_REVIEW_REGENERATE,
    GENERATION_MODE_FULL_RUN,
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_CONTINUE,
    REVIEW_ACTION_FINISH_REMAINING,
    REVIEW_ACTION_REGENERATE_CURRENT,
    REVISION_STATUS_REVIEW_READY,
    RUN_STORAGE_OFF,
    RUN_STORAGE_SAVE_AUTO_RESUME,
    ReviewControlError,
    resolve_review_execution,
)


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID_JS = ROOT / "web" / "project_id.js"
README = ROOT / "README.md"
README_JA = ROOT / "README_JA.md"


def _resolved(*, mode, action, storage=RUN_STORAGE_SAVE_AUTO_RESUME, manual=0):
    return resolve_review_execution(
        generation_mode=mode,
        review_action=action,
        configured_chunks=4,
        validated_prefix_count=1,
        terminal_merge_enabled=False,
        terminal_pair_start=None,
        manual_regenerate_from=manual,
        run_storage_mode=storage,
        latest_review_unit={"start": 1, "end": 1, "physical_group": 1},
        latest_revision_status=REVISION_STATUS_REVIEW_READY,
        latest_effective_nonce=2,
        latest_branch_regenerate_from=1,
    )


def test_e0_e6_backend_queue_intents_keep_phase_b_contract():
    full = _resolved(
        mode=GENERATION_MODE_FULL_RUN,
        action=REVIEW_ACTION_REGENERATE_CURRENT,
        storage=RUN_STORAGE_OFF,
    )
    assert full.execution_mode == EXECUTION_MODE_FULL_RUN
    assert full.max_new_physical_groups is None
    assert full.smart_regenerate is False

    continued = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_CONTINUE,
    )
    assert continued.execution_mode == EXECUTION_MODE_REVIEW_CONTINUE
    assert continued.max_new_physical_groups == 1

    regenerated = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_REGENERATE_CURRENT,
    )
    assert regenerated.execution_mode == EXECUTION_MODE_REVIEW_REGENERATE
    assert regenerated.effective_regenerate_from == 1
    assert regenerated.max_new_physical_groups == 1
    assert regenerated.requested_effective_nonce == 3

    finished = _resolved(
        mode=GENERATION_MODE_REVIEW,
        action=REVIEW_ACTION_FINISH_REMAINING,
    )
    assert finished.execution_mode == EXECUTION_MODE_REVIEW_FINISH
    assert finished.max_new_physical_groups is None

    with pytest.raises(ReviewControlError, match="requires Run Storage"):
        _resolved(
            mode=GENERATION_MODE_REVIEW,
            action=REVIEW_ACTION_CONTINUE,
            storage=RUN_STORAGE_OFF,
        )
    with pytest.raises(ReviewControlError, match="cannot be combined"):
        _resolved(
            mode=GENERATION_MODE_REVIEW,
            action=REVIEW_ACTION_REGENERATE_CURRENT,
            manual=2,
        )

    stale_finish = _resolved(
        mode=GENERATION_MODE_FULL_RUN,
        action=REVIEW_ACTION_FINISH_REMAINING,
        storage=RUN_STORAGE_OFF,
    )
    assert stale_finish.execution_mode == EXECUTION_MODE_FULL_RUN
    assert stale_finish.max_new_physical_groups is None
    assert stale_finish.finish_remaining is False


def test_v38_schema_appends_review_widgets_without_changing_v37():
    v37 = H3ContinuumSamplerV37.INPUT_TYPES()["required"]
    v38 = H3ContinuumSamplerV38.INPUT_TYPES()["required"]
    assert "generation_mode" not in v37
    assert "review_action" not in v37
    assert list(v38)[-8:] == [
        "generation_mode",
        "review_action",
        "take_group",
        "take_revision_id",
        "take_action",
        "size_source",
        "width",
        "height",
    ]
    assert v38["generation_mode"][1]["default"] == GENERATION_MODE_FULL_RUN
    assert v38["review_action"][1]["default"] == REVIEW_ACTION_CONTINUE
    assert v38["take_action"][1]["default"] == "Automatic"


def test_v38_facade_forwards_only_public_review_intent(monkeypatch):
    captured = {}

    def fake_v37_run(self, **kwargs):
        captured.update(kwargs)
        return "ok"

    monkeypatch.setattr(H3ContinuumSamplerV37, "run", fake_v37_run)
    result = H3ContinuumSamplerV38().run(
        generation_mode=GENERATION_MODE_REVIEW,
        review_action=REVIEW_ACTION_FINISH_REMAINING,
    )
    assert result == "ok"
    assert captured["generation_mode"] == GENERATION_MODE_REVIEW
    assert captured["review_action"] == REVIEW_ACTION_FINISH_REMAINING
    assert "max_new_physical_groups" not in captured


def _storage(*, execution, chunks, total, unit=None, reused=0, generated=0):
    manifest = {"chunks": [{} for _ in range(chunks)]}
    if unit is not None:
        manifest["review_unit"] = unit
    return SimpleNamespace(
        review_generation_mode=GENERATION_MODE_REVIEW,
        review_execution=execution,
        manifest=manifest,
        contract={"chunk_count": total},
        reused_count=reused,
        generated_count=generated,
    )


def test_review_status_covers_ready_terminal_regenerate_and_finish():
    ready = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=False,
            partial_review=True,
        ),
        chunks=2,
        total=6,
        unit={"start": 2, "end": 2, "physical_group": 2},
    )
    ready_status = _format_review_status(ready)
    assert "Chunk 2 / 6 ready" in ready_status
    assert "Completed: Chunks 1-2" in ready_status
    assert "Queue again = Accept + Continue" in ready_status

    terminal = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=False,
            partial_review=False,
        ),
        chunks=3,
        total=3,
        unit={"start": 2, "end": 3, "physical_group": 2},
    )
    terminal_status = _format_review_status(terminal)
    assert "Chunks 2-3 / 3 ready" in terminal_status
    assert "Terminal Merge: 1 physical review unit" in terminal_status
    assert terminal_status.endswith("Sequence complete")

    regenerated = _storage(
        execution=SimpleNamespace(
            finish_remaining=False,
            smart_regenerate=True,
            requested_effective_nonce=3,
            partial_review=True,
        ),
        chunks=3,
        total=6,
        unit={"start": 3, "end": 3, "physical_group": 3},
    )
    regenerated_status = _format_review_status(regenerated, detailed=True)
    assert "Smart Regenerate" in regenerated_status
    assert "Chunk 3 regenerated" in regenerated_status
    assert "Preserved: Chunks 1-2" in regenerated_status
    assert "Variation: 3" in regenerated_status

    finished = _storage(
        execution=SimpleNamespace(
            finish_remaining=True,
            smart_regenerate=False,
            partial_review=False,
        ),
        chunks=6,
        total=6,
        reused=3,
        generated=3,
    )
    finished_status = _format_review_status(finished)
    assert finished_status == (
        "Review completed\n3 reused; 3 generated; 6 total\nSequence complete"
    )


def test_partial_review_second_pass_warning_is_explicit():
    partial = SimpleNamespace(
        review_execution=SimpleNamespace(partial_review=True)
    )
    assert "Second Pass refinement" in _partial_review_warning(
        partial,
        capture_refine_context=True,
    )
    assert _partial_review_warning(
        partial,
        capture_refine_context=False,
    ) == ""


def _function_source(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start)
    return source[start:end]


def test_f0_f7_frontend_review_lifecycle(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setWidgetVisible", "hidePersistentWidget"),
            _function_source(source, "setExistingWidgetValue", "facadeProductionWidgets"),
            _function_source(source, "baseSeedControlWidget", "setReviewSeedControlFixed"),
            _function_source(source, "setReviewSeedControlFixed", "requireFixedSeedForReview"),
            _function_source(source, "isOneShotReviewAction", "normalizeReviewActionOnLoad"),
            _function_source(source, "normalizeReviewActionOnLoad", "resetReviewActionAfterQueued"),
            _function_source(source, "resetReviewActionAfterQueued", "prepareReviewQueueIntent"),
            _function_source(source, "prepareReviewQueueIntent", "configureReviewControls"),
            _function_source(source, "configureReviewControls", "configureAssembler"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const PROJECT_WIDGET = "project_id";
const LEGACY_RUN_NAME_WIDGET = "run_name";
const RUN_STORAGE_WIDGET = "run_storage";
const GENERATION_MODE_WIDGET = "generation_mode";
const REVIEW_ACTION_WIDGET = "review_action";
const GENERATION_MODE_FULL_RUN = "Full Run";
const REGENERATE_WIDGET = "reroll_from_chunk";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
const REVIEW_ACTION_CONTINUE = "Continue / Next";
const REVIEW_ACTION_REGENERATE = "Regenerate Current";
const REVIEW_ACTION_FINISH = "Finish Remaining";
const TAKE_GROUP_WIDGET = "take_group";
const TAKE_REVISION_WIDGET = "take_revision_id";
const TAKE_ACTION_WIDGET = "take_action";
const TAKE_ACTION_AUTOMATIC = "Automatic";
const TAKE_ACTION_USE = "Use This Take";
const TAKE_ACTION_CONTINUE = "Continue From Here";
const REVIEW_PENDING_ACTION = "__h3ContinuumPendingReviewAction";
const REVIEW_UI_SELECTION = "__h3ContinuumReviewUiSelection";
const TAKE_PENDING_ACTION = "__h3ContinuumPendingTakeAction";
{functions}

function widget(name, value) {{
    return {{ name, value, type: "combo", options: {{}}, computeSize: () => [120, 20] }};
}}
function makeNode(mode, action, storage, comfyClass = V38_NODE_CLASS) {{
    return {{
        comfyClass,
        widgets: [
            widget("existing_1", 11),
            widget(RUN_STORAGE_WIDGET, storage),
            widget("existing_2", 22),
            widget(GENERATION_MODE_WIDGET, mode),
            widget(REVIEW_ACTION_WIDGET, action),
        ],
        setDirtyCanvas() {{}},
    }};
}}

const defaults = makeNode(GENERATION_MODE_FULL_RUN, REVIEW_ACTION_CONTINUE, "Off");
configureReviewControls(defaults);
const defaultAction = findWidget(defaults, REVIEW_ACTION_WIDGET);

const switching = makeNode(GENERATION_MODE_FULL_RUN, REVIEW_ACTION_CONTINUE, "Off");
configureReviewControls(switching);
const switchingMode = findWidget(switching, GENERATION_MODE_WIDGET);
const switchingStorage = findWidget(switching, RUN_STORAGE_WIDGET);
switchingMode.value = GENERATION_MODE_REVIEW;
switchingMode.callback(GENERATION_MODE_REVIEW);
const storageAfterReview = switchingStorage.value;
switchingMode.value = GENERATION_MODE_FULL_RUN;
switchingMode.callback(GENERATION_MODE_FULL_RUN);
const storageAfterFull = switchingStorage.value;

function acceptedSnapshot(action) {{
    const node = makeNode(GENERATION_MODE_REVIEW, action, "Save + Auto Resume");
    configureReviewControls(node);
    const inputs = {{ generation_mode: GENERATION_MODE_REVIEW, review_action: action }};
    prepareReviewQueueIntent(node, inputs);
    const submitted = JSON.parse(JSON.stringify(inputs));
    findWidget(node, REVIEW_ACTION_WIDGET).afterQueued({{ isPartialExecution: false }});
    return {{
        submitted,
        promptAfterReset: inputs.review_action,
        widgetAfterReset: findWidget(node, REVIEW_ACTION_WIDGET).value,
    }};
}}

const regenerate = acceptedSnapshot(REVIEW_ACTION_REGENERATE);
const finish = acceptedSnapshot(REVIEW_ACTION_FINISH);

const failed = makeNode(
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_REGENERATE,
    "Save + Auto Resume",
);
configureReviewControls(failed);
const failedInputs = {{}};
prepareReviewQueueIntent(failed, failedInputs);

const saved = makeNode(
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_FINISH,
    "Save + Auto Resume",
);
const savedValues = saved.widgets.map((item) => item.value);
const reloaded = makeNode(
    savedValues[3],
    savedValues[4],
    savedValues[1],
);
normalizeReviewActionOnLoad(reloaded);

const v37 = makeNode(
    GENERATION_MODE_REVIEW,
    REVIEW_ACTION_REGENERATE,
    "Off",
    "H3ContinuumSamplerV37",
);
const v37Inputs = {{ untouched: true }};
configureReviewControls(v37);
const v37Prepared = prepareReviewQueueIntent(v37, v37Inputs);

console.log(JSON.stringify({{
    defaults: {{
        mode: findWidget(defaults, GENERATION_MODE_WIDGET).value,
        action: defaultAction.value,
        actionHidden: defaultAction.hidden,
    }},
    switching: {{ storageAfterReview, storageAfterFull }},
    regenerate,
    finish,
    failed: {{
        prompt: failedInputs.review_action,
        widget: findWidget(failed, REVIEW_ACTION_WIDGET).value,
    }},
    reload: {{
        savedValues,
        reloadedValues: reloaded.widgets.map((item) => item.value),
    }},
    v37: {{ prepared: v37Prepared, inputs: v37Inputs }},
}}));
"""
    script_path = tmp_path / "v38-review-ui-regression.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed["defaults"] == {
        "mode": GENERATION_MODE_FULL_RUN,
        "action": REVIEW_ACTION_CONTINUE,
        "actionHidden": True,
    }
    assert observed["switching"] == {
        "storageAfterReview": RUN_STORAGE_SAVE_AUTO_RESUME,
        "storageAfterFull": RUN_STORAGE_SAVE_AUTO_RESUME,
    }
    for key, action in (
        ("regenerate", REVIEW_ACTION_REGENERATE_CURRENT),
        ("finish", REVIEW_ACTION_FINISH_REMAINING),
    ):
        assert observed[key]["submitted"]["review_action"] == action
        assert observed[key]["promptAfterReset"] == action
        assert observed[key]["widgetAfterReset"] == REVIEW_ACTION_CONTINUE
    assert observed["failed"] == {
        "prompt": REVIEW_ACTION_REGENERATE_CURRENT,
        "widget": REVIEW_ACTION_REGENERATE_CURRENT,
    }
    assert observed["reload"]["reloadedValues"][:-1] == observed["reload"][
        "savedValues"
    ][:-1]
    assert observed["reload"]["reloadedValues"][-1] == REVIEW_ACTION_CONTINUE
    assert observed["v37"] == {"prepared": False, "inputs": {"untouched": True}}


def test_frontend_uses_formal_after_queued_callback_without_queue_monkey_patch():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    assert "actionWidget.afterQueued = function" in source
    assert "prepareReviewQueueIntent(node, apiNode.inputs);" in source
    assert "normalizeReviewActionOnLoad(node);" in source
    assert "app.queuePrompt =" not in source
    assert "api.queuePrompt =" not in source


def test_execution_success_reloads_v38_history_without_auto_queue():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    refresh = _function_source(
        source,
        "refreshV38TakeHistoryAfterExecution",
        "attachTakeHistoryReload",
    )
    setup = source[source.index("setup() {") : source.index("nodeCreated(node)")]
    for event_name in (
        "execution_success",
        "execution_error",
        "execution_interrupted",
    ):
        assert f'"{event_name}"' in setup
    assert "refreshV38TakeHistoryAfterExecution" in setup
    assert "node.comfyClass !== V38_NODE_CLASS" in refresh
    assert "void loadTakeHistory(node);" in refresh
    assert "setTimeout(() => void loadTakeHistory(node), 250);" in refresh
    assert "queuePrompt" not in refresh


def test_review_requires_fixed_control_after_generate(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "baseSeedControlWidget", "setReviewSeedControlFixed"),
            _function_source(source, "requireFixedSeedForReview", "readySummary"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const GENERATION_MODE_WIDGET = "generation_mode";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
{functions}
function node(mode, control, comfyClass = V38_NODE_CLASS) {{
    const linkedControl = {{ name: "control_after_generate", value: control }};
    return {{
        comfyClass,
        widgets: [
            {{ name: GENERATION_MODE_WIDGET, value: mode }},
            {{ name: "base_seed", value: 123, linkedWidgets: [linkedControl] }},
            {{ name: "control_after_generate", value: control === "fixed" ? "randomize" : "fixed" }},
        ],
    }};
}}
function result(value) {{
    try {{ requireFixedSeedForReview(value); return "allowed"; }}
    catch (error) {{ return String(error.message); }}
}}
console.log(JSON.stringify({{
    fullRandom: result(node("Full Run", "randomize")),
    reviewFixed: result(node(GENERATION_MODE_REVIEW, "fixed")),
    reviewRandom: result(node(GENERATION_MODE_REVIEW, "randomize")),
    legacyReviewRandom: result(node(GENERATION_MODE_REVIEW, "randomize", "H3ContinuumSamplerV37")),
}}));
"""
    script_path = tmp_path / "v38-review-fixed-seed.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["fullRandom"] == "allowed"
    assert observed["reviewFixed"] == "allowed"
    assert observed["legacyReviewRandom"] == "allowed"
    assert "Control After Generate = fixed" in observed["reviewRandom"]


def test_review_mode_switch_fixes_the_linked_seed_control(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setWidgetVisible", "hidePersistentWidget"),
            _function_source(source, "setExistingWidgetValue", "facadeProductionWidgets"),
            _function_source(source, "baseSeedControlWidget", "setReviewSeedControlFixed"),
            _function_source(source, "setReviewSeedControlFixed", "requireFixedSeedForReview"),
            _function_source(source, "configureReviewControls", "configureAssembler"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const GENERATION_MODE_WIDGET = "generation_mode";
const REVIEW_ACTION_WIDGET = "review_action";
const RUN_STORAGE_WIDGET = "run_storage";
const TAKE_ACTION_WIDGET = "take_action";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
{functions}
const linkedControl = {{ name: "control_after_generate", value: "randomize" }};
const decoyControl = {{ name: "control_after_generate", value: "fixed" }};
const generation = {{ name: GENERATION_MODE_WIDGET, value: "Full Run" }};
const storage = {{ name: RUN_STORAGE_WIDGET, value: "Off" }};
const node = {{
    comfyClass: V38_NODE_CLASS,
    widgets: [
        generation,
        {{ name: REVIEW_ACTION_WIDGET, value: "Continue / Next" }},
        storage,
        {{ name: TAKE_ACTION_WIDGET, value: "Automatic" }},
        {{ name: "base_seed", value: 123, linkedWidgets: [linkedControl] }},
        decoyControl,
    ],
    setDirtyCanvas() {{}},
}};
configureReviewControls(node);
generation.value = GENERATION_MODE_REVIEW;
generation.callback(GENERATION_MODE_REVIEW);
console.log(JSON.stringify({{
    linked: linkedControl.value,
    decoy: decoyControl.value,
    storage: storage.value,
}}));
"""
    script_path = tmp_path / "v38-review-linked-seed-control.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed == {
        "linked": "fixed",
        "decoy": "fixed",
        "storage": "Save + Auto Resume",
    }


def test_readmes_document_exact_two_by_five_review_workflow():
    for path in (README, README_JA):
        text = path.read_text(encoding="utf-8")
        for label in (
            "`Chunks`",
            "`Seconds per Chunk`",
            "`Total Length`",
            "`Run`",
            "`Review Each Chunk`",
            "`Progress`",
            "`On — Resume and Takes available`",
            "`Ready to Queue`",
            "`Control After Generate`",
            "`fixed`",
            "`Use it and continue`",
            "`Try this chunk again`",
            "`Use it and finish the rest`",
            "`Back to Settings`",
            "`Return to Review`",
            "`Queue`",
        ):
            assert label in text
        assert "5" in text
        assert "10" in text


def test_n1_production_shortcuts_are_transient_and_map_to_existing_intent(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setWidgetVisible", "hidePersistentWidget"),
            _function_source(source, "attachRefresh", "normalizedV38View"),
            _function_source(source, "normalizedV38View", "applyV38View"),
            _function_source(source, "isOneShotReviewAction", "normalizeReviewActionOnLoad"),
            _function_source(source, "prepareReviewQueueIntent", "configureReviewControls"),
        )
    )
    script = f"""
const V38_NODE_CLASS = "H3ContinuumSamplerV38";
const PROJECT_WIDGET = "project_id";
const LEGACY_RUN_NAME_WIDGET = "run_name";
const RUN_STORAGE_WIDGET = "run_storage";
const GENERATION_MODE_WIDGET = "generation_mode";
const REVIEW_ACTION_WIDGET = "review_action";
const REGENERATE_WIDGET = "reroll_from_chunk";
const REROLL_NONCE_WIDGET = "reroll_nonce";
const GENERATION_MODE_FULL_RUN = "Full Run";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
const REVIEW_ACTION_CONTINUE = "Continue / Next";
const REVIEW_ACTION_REGENERATE = "Regenerate Current";
const REVIEW_ACTION_FINISH = "Finish Remaining";
const TAKE_GROUP_WIDGET = "take_group";
const TAKE_REVISION_WIDGET = "take_revision_id";
const TAKE_ACTION_WIDGET = "take_action";
const TAKE_ACTION_AUTOMATIC = "Automatic";
const TAKE_ACTION_USE = "Use This Take";
const TAKE_ACTION_CONTINUE = "Continue From Here";
const REVIEW_PENDING_ACTION = "__h3ContinuumPendingReviewAction";
const REVIEW_UI_SELECTION = "__h3ContinuumReviewUiSelection";
const TAKE_PENDING_ACTION = "__h3ContinuumPendingTakeAction";
const PRODUCTION_STATUS_WIDGET = "Review Ready";
const PRODUCTION_CONTINUE_WIDGET = "Use it and continue";
const PRODUCTION_REGENERATE_WIDGET = "Try this chunk again";
const PRODUCTION_FINISH_WIDGET = "Use it and finish the rest";
const PRODUCTION_BACK_TO_SETTINGS_WIDGET = "Back to Settings";
const PRODUCTION_RETURN_TO_REVIEW_WIDGET = "Return to Review";
const PRODUCTION_RESTART_WIDGET = "Start again from Chunk 1";
const FACADE_READY_WIDGET = "Ready to Queue";
const FACADE_ADVANCED_WIDGET = "Advanced Settings";
const FACADE_REFERENCE_SIZE_WIDGET = "Reference Image Size";
const FACADE_VIDEO_SIZE_WIDGET = "Video Guide Size";
const TAKE_STATUS_WIDGET = "Render History / Takes";
const TAKE_TOGGLE_WIDGET = "Render History";
const TAKE_PREVIOUS_WIDGET = "Previous Take";
const TAKE_NEXT_WIDGET = "Next Take";
const TAKE_USE_WIDGET = "Use This Take";
const TAKE_CONTINUE_WIDGET = "Continue From Here";
const PRODUCTION_TRANSIENT_WIDGET = "__h3ContinuumProductionTransient";
const FACADE_TRANSIENT_WIDGET = "__h3ContinuumFacadeTransient";
const V38_VIEW_PROPERTY = "H3 Continuum View";
const V38_VIEW_BASIC = "Basic";
const V38_VIEW_PRODUCTION = "Production";
{functions}

function widget(name, value) {{
    return {{ name, value, type: "combo", options: {{}}, computeSize: () => [120, 20] }};
}}
function makeNode() {{
    const node = {{
        comfyClass: V38_NODE_CLASS,
        properties: {{ [V38_VIEW_PROPERTY]: V38_VIEW_BASIC }},
        widgets: [
            widget("unrelated", 73),
            widget(RUN_STORAGE_WIDGET, "Off"),
            widget(REGENERATE_WIDGET, "Auto"),
            widget(REROLL_NONCE_WIDGET, 7),
            widget(GENERATION_MODE_WIDGET, GENERATION_MODE_FULL_RUN),
            widget(REVIEW_ACTION_WIDGET, REVIEW_ACTION_CONTINUE),
            widget(TAKE_GROUP_WIDGET, 0),
            widget(TAKE_REVISION_WIDGET, ""),
            widget(TAKE_ACTION_WIDGET, TAKE_ACTION_AUTOMATIC),
        ],
        addWidget(type, name, value, callback, options) {{
            const item = {{ name, value, callback, type, options: options || {{}}, computeSize: () => [120, 20] }};
            this.widgets.push(item);
            return item;
        }},
        serialize() {{
            return {{ widgets_values: this.widgets.map((item) => item.value) }};
        }},
        configure(info) {{
            info.widgets_values.forEach((value, index) => {{ this.widgets[index].value = value; }});
        }},
        setDirtyCanvas() {{}},
    }};
    return node;
}}

const node = makeNode();
const originalValues = node.widgets.map((item) => item.value);
configureProductionReviewUx(node);
const basic = {{
    totalWidgets: node.widgets.length,
    transientHidden: transientProductionWidgets(node).every((item) => item.hidden),
    saved: node.serialize().widgets_values,
}};

node.properties[V38_VIEW_PROPERTY] = V38_VIEW_PRODUCTION;
node.__h3ContinuumProductionUxRefresh();
const fullRunReviewUiHidden = transientProductionWidgets(node).every((item) => item.hidden);

findWidget(node, PRODUCTION_CONTINUE_WIDGET).callback();
const continued = {{
    storage: findWidget(node, RUN_STORAGE_WIDGET).value,
    from: findWidget(node, REGENERATE_WIDGET).value,
    nonce: findWidget(node, REROLL_NONCE_WIDGET).value,
    mode: findWidget(node, GENERATION_MODE_WIDGET).value,
    action: findWidget(node, REVIEW_ACTION_WIDGET).value,
    plan: findWidget(node, PRODUCTION_STATUS_WIDGET).value,
}};

setExistingWidgetValue(findWidget(node, REGENERATE_WIDGET), "Auto");
findWidget(node, PRODUCTION_REGENERATE_WIDGET).callback();
const regenerateInputs = {{ unrelated: "preserved" }};
prepareReviewQueueIntent(node, regenerateInputs);
const regenerated = {{
    from: findWidget(node, REGENERATE_WIDGET).value,
    action: findWidget(node, REVIEW_ACTION_WIDGET).value,
    inputs: regenerateInputs,
    plan: findWidget(node, PRODUCTION_STATUS_WIDGET).value,
}};

findWidget(node, PRODUCTION_FINISH_WIDGET).callback();
const finished = {{
    action: findWidget(node, REVIEW_ACTION_WIDGET).value,
    plan: findWidget(node, PRODUCTION_STATUS_WIDGET).value,
}};

node.__h3ContinuumTakeProject = {{
    branch_provenance_version: 1,
    active_revisions: {{ "3": "r3b" }},
    group_revisions: [
        {{ revision_id: "r3a", revision_order: "1", group: {{ physical_group: 3 }} }},
        {{ revision_id: "r3b", revision_order: "2", group: {{ physical_group: 3 }} }},
    ],
}};
node.__h3ContinuumProductionUxRefresh();
findWidget(node, TAKE_PREVIOUS_WIDGET).callback();
findWidget(node, TAKE_USE_WIDGET).callback();
const takeInputs = {{}};
prepareReviewQueueIntent(node, takeInputs);
const take = {{
    group: findWidget(node, TAKE_GROUP_WIDGET).value,
    revision: findWidget(node, TAKE_REVISION_WIDGET).value,
    action: findWidget(node, TAKE_ACTION_WIDGET).value,
    status: findWidget(node, TAKE_STATUS_WIDGET).value,
    inputs: takeInputs,
}};

const saved = node.serialize();
const reloaded = makeNode();
configureProductionReviewUx(reloaded);
reloaded.configure(saved);
const reload = {{
    values: reloaded.serialize().widgets_values,
    totalWidgets: reloaded.widgets.length,
    transientCount: transientProductionWidgets(reloaded).length,
}};

console.log(JSON.stringify({{
    originalValues,
    basic,
    fullRunReviewUiHidden,
    continued,
    regenerated,
    finished,
    take,
    saved,
    reload,
}}));
"""
    script_path = tmp_path / "v38-n1-production-ux.js"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert observed["basic"] == {
        "totalWidgets": 22,
        "transientHidden": True,
        "saved": observed["originalValues"],
    }
    assert observed["fullRunReviewUiHidden"] is True
    assert observed["continued"] == {
        "storage": RUN_STORAGE_SAVE_AUTO_RESUME,
        "from": "Auto",
        "nonce": 7,
        "mode": GENERATION_MODE_REVIEW,
        "action": REVIEW_ACTION_CONTINUE,
        "plan": "Queue the workflow to create the first chunk.",
    }
    assert observed["regenerated"]["from"] == "Auto"
    assert observed["regenerated"]["action"] == REVIEW_ACTION_REGENERATE_CURRENT
    assert observed["regenerated"]["inputs"] == {
        "unrelated": "preserved",
        "generation_mode": GENERATION_MODE_REVIEW,
        "review_action": REVIEW_ACTION_REGENERATE_CURRENT,
        "take_group": 0,
        "take_revision_id": "",
        "take_action": "Automatic",
    }
    assert observed["regenerated"]["plan"] == (
        "Queue the workflow to create the first chunk."
    )
    assert observed["finished"]["action"] == REVIEW_ACTION_FINISH_REMAINING
    assert observed["finished"]["plan"] == (
        "Queue the workflow to create the first chunk."
    )
    assert {
        key: observed["take"][key]
        for key in ("group", "revision", "action", "inputs")
    } == {
        "group": 3,
        "revision": "r3a",
        "action": "Use This Take",
        "inputs": {
            "generation_mode": GENERATION_MODE_REVIEW,
            "review_action": REVIEW_ACTION_CONTINUE,
            "take_group": 3,
            "take_revision_id": "r3a",
            "take_action": "Use This Take",
        },
    }
    assert "Selected: Group 3 / Take 1 | r3a" in observed["take"]["status"]
    assert "Canonical: Group 3 / Take 2 | r3b" in observed["take"]["status"]
    assert "Next Queue: Use This Take (normal Queue required)" in observed["take"]["status"]
    assert len(observed["saved"]["widgets_values"]) == len(observed["originalValues"])
    assert observed["reload"] == {
        "values": observed["saved"]["widgets_values"],
        "totalWidgets": 22,
        "transientCount": 13,
    }


def test_n2c_render_history_take_ux_is_complete_and_backend_derived(tmp_path):
    node_executable = shutil.which("node")
    if node_executable is None:
        pytest.skip("Node.js is required for the frontend behavior regression")

    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    functions = "\n".join(
        (
            _function_source(source, "findWidget", "setWidgetVisible"),
            _function_source(source, "setExistingWidgetValue", "productionQueueSummary"),
            _function_source(source, "takeRunName", "configureProductionReviewUx"),
        )
    )
    script = f"""
const PROJECT_WIDGET = "project_id";
const LEGACY_RUN_NAME_WIDGET = "run_name";
const RUN_STORAGE_WIDGET = "run_storage";
const GENERATION_MODE_WIDGET = "generation_mode";
const REVIEW_ACTION_WIDGET = "review_action";
const REGENERATE_WIDGET = "reroll_from_chunk";
const TAKE_GROUP_WIDGET = "take_group";
const TAKE_REVISION_WIDGET = "take_revision_id";
const TAKE_ACTION_WIDGET = "take_action";
const TAKE_ACTION_AUTOMATIC = "Automatic";
const TAKE_ACTION_USE = "Use This Take";
const TAKE_ACTION_CONTINUE = "Continue From Here";
const GENERATION_MODE_REVIEW = "Review Each Chunk";
const REVIEW_ACTION_CONTINUE = "Continue / Next";
{functions}

function widget(name, value) {{
    return {{ name, value, callback: null }};
}}

function makeNode(project = null, storage = "Off") {{
    return {{
        widgets: [
            widget(PROJECT_WIDGET, "123e4567-e89b-42d3-a456-426614174000"),
            widget(LEGACY_RUN_NAME_WIDGET, ""),
            widget(RUN_STORAGE_WIDGET, storage),
            widget(REGENERATE_WIDGET, "Auto"),
            widget(GENERATION_MODE_WIDGET, "Full Run"),
            widget(REVIEW_ACTION_WIDGET, REVIEW_ACTION_CONTINUE),
            widget(TAKE_GROUP_WIDGET, 0),
            widget(TAKE_REVISION_WIDGET, ""),
            widget(TAKE_ACTION_WIDGET, TAKE_ACTION_AUTOMATIC),
        ],
        __h3ContinuumTakeProject: project,
        refreshes: 0,
        __h3ContinuumProductionUxRefresh() {{ this.refreshes += 1; }},
        setDirtyCanvas() {{}},
    }};
}}

const project = {{
    run_storage_schema_version: 3,
    branch_provenance_version: 1,
    canonical_head_revision_id: "r56a-canonical-head",
    active_revisions: {{
        "1": "r1-root",
        "2": "r2-main",
        "3": "r3a-canonical",
        "4": "r4a-canonical",
        "5": "r56a-canonical-head",
    }},
    group_revisions: [
        {{ revision_id: "r1-root", parent_revision_id: null, variation_nonce: 0, revision_order: "01", group: {{ physical_group: 1, start: 1, end: 1 }} }},
        {{ revision_id: "r2-main", parent_revision_id: "r1-root", variation_nonce: 0, revision_order: "02", group: {{ physical_group: 2, start: 2, end: 2 }} }},
        {{ revision_id: "r3a-canonical", parent_revision_id: "r2-main", variation_nonce: 1, revision_order: "03a", group: {{ physical_group: 3, start: 3, end: 3 }} }},
        {{ revision_id: "r3b-selected", parent_revision_id: "r2-main", variation_nonce: 2, revision_order: "03b", group: {{ physical_group: 3, start: 3, end: 3 }} }},
        {{ revision_id: "r4a-canonical", parent_revision_id: "r3a-canonical", variation_nonce: 1, revision_order: "04a", group: {{ physical_group: 4, start: 4, end: 4 }} }},
        {{ revision_id: "r4b-branch", parent_revision_id: "r3b-selected", variation_nonce: 2, revision_order: "04b", group: {{ physical_group: 4, start: 4, end: 4 }} }},
        {{ revision_id: "r56a-canonical-head", parent_revision_id: "r4a-canonical", variation_nonce: 1, revision_order: "05", group: {{ physical_group: 5, start: 5, end: 6 }} }},
    ],
}};

const one = makeNode({{
    run_storage_schema_version: 3,
    branch_provenance_version: 1,
    canonical_head_revision_id: "only-take",
    active_revisions: {{ "1": "only-take" }},
    group_revisions: [
        {{ revision_id: "only-take", parent_revision_id: null, variation_nonce: 0, revision_order: "01", group: {{ physical_group: 1, start: 1, end: 1 }} }},
    ],
}});
const oneText = takeStatus(one);

const node = makeNode(project);
findWidget(node, TAKE_GROUP_WIDGET).value = 3;
findWidget(node, TAKE_REVISION_WIDGET).value = "r3b-selected";
const selectedText = takeStatus(node);
const canonicalBefore = JSON.stringify(project.active_revisions);
selectTakeOffset(node, -1);
const previous = {{
    group: findWidget(node, TAKE_GROUP_WIDGET).value,
    revision: findWidget(node, TAKE_REVISION_WIDGET).value,
    canonical: JSON.stringify(project.active_revisions),
}};
selectTakeOffset(node, 1);
const next = {{
    group: findWidget(node, TAKE_GROUP_WIDGET).value,
    revision: findWidget(node, TAKE_REVISION_WIDGET).value,
    canonical: JSON.stringify(project.active_revisions),
}};
selectTakeAction(node, TAKE_ACTION_USE);
const use = {{
    storage: findWidget(node, RUN_STORAGE_WIDGET).value,
    mode: findWidget(node, GENERATION_MODE_WIDGET).value,
    action: findWidget(node, TAKE_ACTION_WIDGET).value,
    canonical: JSON.stringify(project.active_revisions),
    text: takeStatus(node),
}};
findWidget(node, TAKE_ACTION_WIDGET).value = TAKE_ACTION_AUTOMATIC;
selectTakeAction(node, TAKE_ACTION_CONTINUE);
const continued = {{
    action: findWidget(node, TAKE_ACTION_WIDGET).value,
    revision: findWidget(node, TAKE_REVISION_WIDGET).value,
    text: takeStatus(node),
}};

const legacy = makeNode({{
    run_storage_schema_version: 2,
    revisions: [{{ revision_id: "legacy" }}],
}});
const legacyText = takeStatus(legacy);

let fetchCount = 0;
globalThis.fetch = async () => {{
    fetchCount += 1;
    return {{ ok: true, status: 200, json: async () => project }};
}};
const restarted = makeNode(null, "Save + Auto Resume");
const loaded = await loadTakeHistory(restarted);
const restart = {{
    loaded,
    fetchCount,
    group: findWidget(restarted, TAKE_GROUP_WIDGET).value,
    revision: findWidget(restarted, TAKE_REVISION_WIDGET).value,
    text: takeStatus(restarted),
}};

globalThis.fetch = async () => ({{ ok: false, status: 404 }});
const missing = makeNode(project, "Save + Auto Resume");
findWidget(missing, TAKE_GROUP_WIDGET).value = 3;
findWidget(missing, TAKE_REVISION_WIDGET).value = "r3b-selected";
findWidget(missing, TAKE_ACTION_WIDGET).value = TAKE_ACTION_CONTINUE;
await loadTakeHistory(missing);
const missingResult = {{
    group: findWidget(missing, TAKE_GROUP_WIDGET).value,
    revision: findWidget(missing, TAKE_REVISION_WIDGET).value,
    action: findWidget(missing, TAKE_ACTION_WIDGET).value,
    text: takeStatus(missing),
}};

console.log(JSON.stringify({{
    oneText,
    selectedText,
    canonicalBefore,
    previous,
    next,
    use,
    continued,
    legacyText,
    restart,
    missingResult,
}}));
"""
    script_path = tmp_path / "v38-n2c-render-history-ux.mjs"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [node_executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)

    assert "Group 1\n  Take 1/1" in observed["oneText"]
    assert "✓ Canonical | Head" in observed["oneText"]
    assert "Run Storage: Off" in observed["oneText"]

    selected = observed["selectedText"]
    assert "Selected: Group 3 / Take 2 | r3b-selected" in selected
    assert "Canonical: Group 3 / Take 1 | r3a-canonic" in selected
    assert "Canonical head: Group 5-6 (atomic) / Take 1 | r56a-canonic" in selected
    assert "Continue From Here: Reuse Groups 1-3" in selected
    assert "Regenerate Groups 4, 5-6 (atomic)" in selected
    assert "Take 2/2 | r3b-selected | nonce 2 | parent r2-main | ← Selected" in selected
    assert "Take 1/2 | r3a-canonica | nonce 1" in selected

    assert observed["previous"] == {
        "group": 3,
        "revision": "r3a-canonical",
        "canonical": observed["canonicalBefore"],
    }
    assert observed["next"] == {
        "group": 3,
        "revision": "r3b-selected",
        "canonical": observed["canonicalBefore"],
    }
    assert observed["use"]["storage"] == RUN_STORAGE_SAVE_AUTO_RESUME
    assert observed["use"]["mode"] == GENERATION_MODE_REVIEW
    assert observed["use"]["action"] == "Use This Take"
    assert observed["use"]["canonical"] == observed["canonicalBefore"]
    assert "Next Queue: Use This Take (normal Queue required)" in observed["use"]["text"]
    assert observed["continued"]["action"] == "Continue From Here"
    assert observed["continued"]["revision"] == "r3b-selected"
    assert "Next Queue: Continue From Here (normal Queue required)" in observed["continued"]["text"]

    assert "Legacy Run Storage v2 detected" in observed["legacyText"]
    assert "non-destructively" in observed["legacyText"]
    assert observed["restart"]["loaded"] is True
    assert observed["restart"]["fetchCount"] == 1
    assert observed["restart"]["group"] == 5
    assert observed["restart"]["revision"] == "r56a-canonical-head"
    assert "Canonical head: Group 5-6 (atomic)" in observed["restart"]["text"]
    assert observed["missingResult"] == {
        "group": 0,
        "revision": "",
        "action": "Automatic",
        "text": "No Render History yet",
    }


def test_n2c_history_refresh_hooks_and_multiline_surface_are_present():
    source = PROJECT_ID_JS.read_text(encoding="utf-8")
    assert '"Render History / Takes"' in source
    assert '{ multiline: true }' in source
    assert "void loadTakeHistory(this);" in source
    assert "attachTakeHistoryReload(node);" in source
    assert "node.__h3ContinuumTakeInitialLoad" in source
    assert 'const TAKE_TOGGLE_WIDGET = "Render History";' in source
    assert "app.queuePrompt =" not in source
    assert "api.queuePrompt =" not in source
