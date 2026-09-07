import { app } from "../../scripts/app.js";
import { configureExistingEasyBypassWidgetNode } from "./easy_bypass_toggle.js";

const EASY_LOAD_VIDEO_CLASS = "H3ContinuumLoadVideo";

function configureEasyLoadVideoNode(node) {
    if (
        node?.comfyClass !== EASY_LOAD_VIDEO_CLASS &&
        node?.type !== EASY_LOAD_VIDEO_CLASS
    ) {
        return;
    }

    try {
        configureExistingEasyBypassWidgetNode(node, {
            nodeClass: EASY_LOAD_VIDEO_CLASS,
            widgetNames: ["enable_video", "Enable Video"],
        });
    } catch (error) {
        console.warn(
            "H3 Continuum Load Video bypass appearance synchronization was skipped; backend blocking remains available.",
            error,
        );
    }
}

app.registerExtension({
    name: "H3Continuum.EasyLoadVideo",

    nodeCreated(node) {
        configureEasyLoadVideoNode(node);
    },

    loadedGraphNode(node) {
        configureEasyLoadVideoNode(node);
    },

    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) configureEasyLoadVideoNode(node);
    },
});
