"""Generic Easy workflow audio loader backed by ComfyUI Core LoadAudio."""

from __future__ import annotations

from types import SimpleNamespace


try:
    from comfy_extras.nodes_audio import LoadAudio as CoreLoadAudio
except (ImportError, AttributeError):  # pragma: no cover - standalone unit tests
    class CoreLoadAudio:
        """Non-runtime shim; real ComfyUI always supplies Core LoadAudio."""

        @classmethod
        def define_schema(cls):
            return SimpleNamespace(
                node_id="LoadAudio",
                display_name="Load Audio",
                category="audio",
                essentials_category="Audio",
                description="",
                search_aliases=["import audio", "open audio", "audio file"],
                is_deprecated=False,
            )

        @classmethod
        def execute(cls, audio):
            return audio

        load = execute


class H3EasyLoadAudio(CoreLoadAudio):
    """Core LoadAudio with an Easy frontend control for native node bypass."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3EasyLoadAudio"
        schema.display_name = "H3 Continuum Load Audio"
        schema.category = "MiniMax H3/Continuum"
        schema.description = (
            "Loads audio through ComfyUI Core. Enable Audio controls the node's "
            "native Bypass mode without changing the AUDIO contract."
        )
        schema.search_aliases = list(
            dict.fromkeys(
                [
                    *getattr(schema, "search_aliases", []),
                    "H3 Continuum Load Audio",
                    "H3 Easy Load Audio",
                    "H3 Easy Audio Loader",
                    "Load Audio with Bypass",
                ]
            )
        )
        return schema


NODE_CLASS_MAPPINGS = {
    "H3EasyLoadAudio": H3EasyLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EasyLoadAudio": "H3 Continuum Load Audio",
}
