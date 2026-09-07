import { app } from "../../scripts/app.js";
import { configureEasyBypassToggleNode } from "./easy_bypass_toggle.js";

const EASY_LOAD_IMAGE_CLASS = "H3EasyLoadImage";

function configureEasyLoadImageNode(node) {
    configureEasyBypassToggleNode(node, {
        nodeClass: EASY_LOAD_IMAGE_CLASS,
        widgetName: "Enable Image",
        tooltip:
            "ON runs Core Load Image. OFF uses ComfyUI native Bypass and behaves like an unconnected optional image input.",
    });
}

app.registerExtension({
    name: "H3Continuum.EasyLoadImage",

    nodeCreated(node) {
        configureEasyLoadImageNode(node);
    },

    loadedGraphNode(node) {
        configureEasyLoadImageNode(node);
    },

    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) configureEasyLoadImageNode(node);
    },
});
