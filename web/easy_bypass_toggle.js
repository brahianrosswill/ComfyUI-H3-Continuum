// Shared native-Bypass control for Easy workflow loader nodes.

// ComfyUI Frontend 1.49.6 defines LGraphEventMode.ALWAYS=0 and BYPASS=4.
// Prefer exported LiteGraph constants when available and keep the audited
// numeric values only as compatibility fallbacks.
const NODE_MODE_ALWAYS =
    typeof LiteGraph !== "undefined" && Number.isInteger(LiteGraph.ALWAYS)
        ? LiteGraph.ALWAYS
        : 0;
const NODE_MODE_BYPASS =
    typeof LiteGraph !== "undefined" && Number.isInteger(LiteGraph.BYPASS)
        ? LiteGraph.BYPASS
        : 4;

function isConfiguredNode(node, nodeClass) {
    return node?.comfyClass === nodeClass || node?.type === nodeClass;
}

export function findEasyBypassWidget(node) {
    const name = node.__h3EasyBypassToggle?.widgetName;
    return name ? node.widgets?.find((widget) => widget.name === name) : undefined;
}

function syncWidgetFromMode(node) {
    const widget = findEasyBypassWidget(node);
    if (!widget) return;
    widget.value = node.mode === NODE_MODE_ALWAYS;
}

function installModeSourceOfTruth(node) {
    if (node.__h3EasyBypassModeObserved) return true;
    const descriptor = Object.getOwnPropertyDescriptor(node, "mode");
    if (descriptor && descriptor.configurable === false) return false;

    let currentMode = Number(node.mode ?? NODE_MODE_ALWAYS);
    Object.defineProperty(node, "mode", {
        configurable: true,
        enumerable: descriptor?.enumerable ?? true,
        get() {
            return descriptor?.get ? descriptor.get.call(this) : currentMode;
        },
        set(value) {
            if (descriptor?.set) {
                descriptor.set.call(this, value);
            } else {
                currentMode = Number(value);
            }
            syncWidgetFromMode(this);
        },
    });
    node.__h3EasyBypassModeObserved = true;
    return true;
}

function installWidgetSerializationGuard(node, widget) {
    if (node.__h3EasyBypassSerializationGuard) return;

    const originalSerialize = node.serialize;
    if (typeof originalSerialize === "function") {
        node.serialize = function (...args) {
            const index = this.widgets?.indexOf(widget) ?? -1;
            if (index >= 0) this.widgets.splice(index, 1);
            try {
                return originalSerialize.apply(this, args);
            } finally {
                if (index >= 0) this.widgets.splice(index, 0, widget);
            }
        };
    }

    const originalConfigure = node.configure;
    if (typeof originalConfigure === "function") {
        node.configure = function (...args) {
            const index = this.widgets?.indexOf(widget) ?? -1;
            if (index >= 0) this.widgets.splice(index, 1);
            try {
                return originalConfigure.apply(this, args);
            } finally {
                if (index >= 0) this.widgets.splice(index, 0, widget);
                syncWidgetFromMode(this);
            }
        };
    }

    node.__h3EasyBypassSerializationGuard = true;
}

function installBypassToggleForeground(node) {
    if (node.__h3EasyBypassForegroundInstalled) return;
    const originalDrawWidgets = node.drawWidgets;
    if (typeof originalDrawWidgets !== "function") return;

    node.drawWidgets = function (ctx, options = {}) {
        const result = originalDrawWidgets.call(this, ctx, options);
        if (this.mode !== NODE_MODE_BYPASS) return result;

        const widget = findEasyBypassWidget(this);
        const widgets = this.widgets;
        if (!widget || widget.hidden || !Array.isArray(widgets)) return result;
        if (!widgets.includes(widget)) return result;

        // ComfyUI draws every widget at the node's Bypass alpha. Redraw only
        // the Enable row afterwards at full opacity so it remains readable and
        // uses the unchanged native widget hit area/click handling.
        this.widgets = [widget];
        try {
            originalDrawWidgets.call(this, ctx, {
                ...(options ?? {}),
                editorAlpha: 1,
            });
        } finally {
            this.widgets = widgets;
        }
        return result;
    };
    node.__h3EasyBypassForegroundInstalled = true;
}

function setModeFromToggle(node, enabled) {
    node.mode = enabled ? NODE_MODE_ALWAYS : NODE_MODE_BYPASS;
    syncWidgetFromMode(node);
    node.setDirtyCanvas?.(true, true);
}

export function configureEasyBypassToggleNode(
    node,
    { nodeClass, widgetName, tooltip },
) {
    if (!isConfiguredNode(node, nodeClass)) return;
    node.__h3EasyBypassToggle = { nodeClass, widgetName };

    let widget = findEasyBypassWidget(node);
    if (!widget) {
        widget = node.addWidget(
            "toggle",
            widgetName,
            node.mode === NODE_MODE_ALWAYS,
            (value) => setModeFromToggle(node, Boolean(value)),
            {
                on: "ON",
                off: "OFF",
                serialize: false,
            },
        );
        widget.serialize = false;
        widget.options ||= {};
        widget.options.serialize = false;
        widget.tooltip = tooltip;
        const currentIndex = node.widgets?.indexOf(widget) ?? -1;
        if (currentIndex > 0) {
            node.widgets.splice(currentIndex, 1);
            node.widgets.unshift(widget);
        }
        installWidgetSerializationGuard(node, widget);
    }

    if (!installModeSourceOfTruth(node)) {
        console.warn(
            `${nodeClass} could not observe node.mode; external Bypass changes may require a workflow reload.`,
        );
    }
    installBypassToggleForeground(node);
    syncWidgetFromMode(node);
    node.setDirtyCanvas?.(true, true);
}

export function configureExistingEasyBypassWidgetNode(
    node,
    { nodeClass, widgetNames },
) {
    if (!isConfiguredNode(node, nodeClass)) return;
    const names = Array.isArray(widgetNames) ? widgetNames : [widgetNames];
    const widget = node.widgets?.find((item) => names.includes(item.name));
    if (!widget) return;

    node.__h3EasyBypassToggle = {
        nodeClass,
        widgetName: widget.name,
    };

    if (!widget.__h3EasyBypassCallbackObserved) {
        const originalCallback = widget.callback;
        widget.callback = function (value, ...args) {
            const result = originalCallback?.call(this, value, ...args);
            setModeFromToggle(node, Boolean(value));
            return result;
        };
        widget.__h3EasyBypassCallbackObserved = true;
    }

    if (!installModeSourceOfTruth(node)) {
        console.warn(
            `${nodeClass} could not observe node.mode; external Bypass changes may require a workflow reload.`,
        );
    }
    installBypassToggleForeground(node);

    if (node.mode === NODE_MODE_BYPASS) {
        syncWidgetFromMode(node);
    } else {
        setModeFromToggle(node, Boolean(widget.value));
    }
    node.setDirtyCanvas?.(true, true);
}
