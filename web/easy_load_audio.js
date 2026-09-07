import { app } from "../../scripts/app.js";
import { configureEasyBypassToggleNode } from "./easy_bypass_toggle.js";

const EASY_LOAD_AUDIO_CLASS = "H3EasyLoadAudio";

function configureEasyLoadAudioNode(node) {
    if (
        node?.comfyClass !== EASY_LOAD_AUDIO_CLASS &&
        node?.type !== EASY_LOAD_AUDIO_CLASS
    ) {
        return;
    }

    // A frontend-only convenience control must never make the Core loader
    // itself unplaceable. Some ComfyUI frontend paths initialize upload
    // widgets later than nodeCreated; fail open and keep the native node.
    try {
        configureEasyBypassToggleNode(node, {
            nodeClass: EASY_LOAD_AUDIO_CLASS,
            widgetName: "Enable Audio",
            tooltip:
                "ON runs Core Load Audio. OFF uses ComfyUI native Bypass and behaves like an unconnected optional audio input.",
        });
    } catch (error) {
        console.warn(
            "H3 Easy Load Audio bypass control was skipped; the Core audio loader remains available.",
            error,
        );
    }
}

app.registerExtension({
    name: "H3Continuum.EasyLoadAudio",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeType?.prototype?.comfyClass !== EASY_LOAD_AUDIO_CLASS &&
            nodeData?.name !== EASY_LOAD_AUDIO_CLASS
        ) {
            return;
        }

        const required = nodeData?.input?.required;
        if (!required?.audio?.[1]?.audio_upload || required.audioUI) return;

        // ComfyUI Frontend 1.49.x adds AUDIOUPLOAD to every audio_upload
        // schema, but adds its required AUDIO_UI preview only to a hard-coded
        // list containing Core LoadAudio. Insert the same Core preview widget
        // before AUDIOUPLOAD so the upload callback always receives it.
        const { upload, ...inputs } = required;
        nodeData.input.required = {
            ...inputs,
            audioUI: ["AUDIO_UI", {}],
            ...(upload ? { upload } : {}),
        };
    },

    nodeCreated(node) {
        configureEasyLoadAudioNode(node);
    },

    loadedGraphNode(node) {
        configureEasyLoadAudioNode(node);
    },

    afterConfigureGraph() {
        for (const node of app.graph?._nodes || []) configureEasyLoadAudioNode(node);
    },
});
