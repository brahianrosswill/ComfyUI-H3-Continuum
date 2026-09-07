"""Generic Easy workflow image loader backed by ComfyUI Core LoadImage."""

from __future__ import annotations


try:
    from nodes import LoadImage as CoreLoadImage
except (ImportError, AttributeError):  # pragma: no cover - standalone unit tests
    class CoreLoadImage:
        """Non-runtime shim; real ComfyUI always supplies Core LoadImage."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "image": (("",), {"image_upload": True}),
                }
            }

        RETURN_TYPES = ("IMAGE", "MASK")
        FUNCTION = "load_image"


class H3EasyLoadImage(CoreLoadImage):
    """Core LoadImage with an Easy frontend control for native node bypass."""

    DEPRECATED = False
    CATEGORY = "MiniMax H3/Continuum"
    DESCRIPTION = (
        "Loads an image through ComfyUI Core. Enable Image controls the node's "
        "native Bypass mode, so the same node can serve First, Last, or Reference "
        "image inputs without changing the IMAGE/MASK contract."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Load Image",
        "H3 Easy Load Image",
        "H3 Easy Image Loader",
        "Load Image with Bypass",
    ]


NODE_CLASS_MAPPINGS = {
    "H3EasyLoadImage": H3EasyLoadImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3EasyLoadImage": "H3 Continuum Load Image",
}
