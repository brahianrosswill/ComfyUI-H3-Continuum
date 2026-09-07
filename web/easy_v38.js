import { app } from "../../scripts/app.js";

const EASY_NODE_CLASS = "H3ContinuumEasyV38";
const CUSTOM_PRESET = "Custom";
const STORAGE_ENABLED = "Save + Auto Resume";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget.__h3EasyOriginal) {
        widget.__h3EasyOriginal = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
        };
    }
    widget.options ||= {};
    if (visible) {
        const original = widget.__h3EasyOriginal;
        widget.type = original.type;
        widget.computeSize = original.computeSize;
        widget.hidden = original.hidden;
        if (original.optionsHidden === undefined) {
            delete widget.options.hidden;
        } else {
            widget.options.hidden = original.optionsHidden;
        }
        return;
    }
    widget.hidden = true;
    widget.type = "converted-widget";
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
}

function attachRefresh(widget, key, refresh) {
    if (!widget || widget[key]) return;
    const previous = widget.callback;
    widget.callback = function(value, ...args) {
        const result = previous?.call(this, value, ...args);
        refresh();
        return result;
    };
    widget[key] = true;
}

function createProjectId() {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = Array.from(bytes, (value) => value.toString(16).padStart(2, "0")).join("");
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

function seedControlWidget(node) {
    return node.widgets?.find((widget) => (
        widget.name === "control_after_generate"
        || widget.name === "seed_control_after_generate"
    ));
}

function configureEasyNode(node) {
    if (node.comfyClass !== EASY_NODE_CLASS) return null;
    const projectWidget = findWidget(node, "project_id");
    const presetWidget = findWidget(node, "preset");
    const customWidget = findWidget(node, "custom_mp");
    const storageWidget = findWidget(node, "run_storage");
    const runNameWidget = findWidget(node, "run_name");
    const seedModeWidget = findWidget(node, "seed_mode");
    const controlWidget = seedControlWidget(node);

    if (projectWidget && !String(projectWidget.value || "").trim()) {
        projectWidget.value = createProjectId();
    }
    const refresh = () => {
        setWidgetVisible(customWidget, presetWidget?.value === CUSTOM_PRESET);
        setWidgetVisible(runNameWidget, storageWidget?.value === STORAGE_ENABLED);
        if (controlWidget) {
            controlWidget.value = seedModeWidget?.value === "Fixed" ? "fixed" : "randomize";
            controlWidget.callback?.(controlWidget.value);
        }
        setWidgetVisible(controlWidget, false);
        setWidgetVisible(projectWidget, false);
        node.setDirtyCanvas?.(true, true);
    };
    attachRefresh(presetWidget, "__h3EasyPresetCallback", refresh);
    attachRefresh(storageWidget, "__h3EasyStorageCallback", refresh);
    attachRefresh(seedModeWidget, "__h3EasySeedModeCallback", refresh);
    refresh();
    return projectWidget;
}

function configureEasyNodeAfterSetup(node) {
    configureEasyNode(node);
    setTimeout(() => configureEasyNode(node), 0);
    setTimeout(() => configureEasyNode(node), 100);
}

app.registerExtension({
    name: "H3Continuum.EasyV38",

    nodeCreated(node) {
        configureEasyNodeAfterSetup(node);
    },

    loadedGraphNode(node) {
        configureEasyNodeAfterSetup(node);
    },

    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) configureEasyNodeAfterSetup(node);
    },

    async beforeQueuePrompt(prompt) {
        const seen = new Set();
        for (const node of app.graph?._nodes || []) {
            const projectWidget = configureEasyNode(node);
            if (!projectWidget) continue;
            let projectId = String(projectWidget.value || "").trim();
            if (!projectId || seen.has(projectId)) {
                projectId = createProjectId();
                projectWidget.value = projectId;
            }
            seen.add(projectId);
            const apiNode = prompt.output?.[String(node.id)];
            if (apiNode?.inputs) apiNode.inputs.project_id = projectId;
        }
    },
});
