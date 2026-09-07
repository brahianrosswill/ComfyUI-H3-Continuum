"""Simple H3 video loader backed by the current ComfyUI Core video API."""

from __future__ import annotations

from fractions import Fraction
import os

import torch


try:
    import folder_paths
except ImportError:  # pragma: no cover - standalone unit tests
    class _FolderPathsShim:
        @staticmethod
        def get_input_directory():
            return "."

        @staticmethod
        def filter_files_content_types(files, content_types):
            del files, content_types
            return []

        @staticmethod
        def get_annotated_filepath(file):
            raise RuntimeError(f"ComfyUI folder_paths is unavailable for {file}")

        @staticmethod
        def exists_annotated_filepath(file):
            del file
            return False

    folder_paths = _FolderPathsShim()

try:
    from comfy_api.latest import InputImpl, io
except ImportError:  # pragma: no cover - standalone unit tests
    class _SchemaItem:
        def __init__(self, item_id=None, **kwargs):
            self.id = item_id
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _InputFactory:
        Input = _SchemaItem

    class _OutputFactory:
        @staticmethod
        def Output(**kwargs):
            return _SchemaItem(**kwargs)

    class _UploadVideo:
        value = "video_upload"

    class _UploadType:
        video = _UploadVideo()

    class _Schema:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _NodeOutput:
        def __init__(self, *args):
            self.args = args

    class _ComfyNode:
        pass

    class _IOShim:
        ComfyNode = _ComfyNode
        Schema = _Schema
        NodeOutput = _NodeOutput
        Boolean = _InputFactory
        Combo = _InputFactory
        Float = _InputFactory
        Image = _OutputFactory
        Audio = _OutputFactory
        UploadType = _UploadType

    class _InputImplShim:
        @staticmethod
        def VideoFromFile(path):
            raise RuntimeError(f"ComfyUI VideoFromFile is unavailable for {path}")

    InputImpl = _InputImplShim
    io = _IOShim()

try:
    from comfy_execution.graph_utils import ExecutionBlocker
except ImportError:  # pragma: no cover - standalone unit tests
    class ExecutionBlocker:
        def __init__(self, message):
            self.message = message


CATEGORY = "MiniMax H3/Continuum"


def _fraction_from_rate(value: float) -> Fraction:
    """Convert a UI frame-rate value without introducing binary-float drift."""

    return Fraction(str(float(value))).limit_denominator(1_000_000)


def _resample_video_frames(
    images: torch.Tensor,
    source_rate: Fraction,
    target_rate: float,
) -> torch.Tensor:
    """Drop or duplicate IMAGE frames while preserving nominal duration."""

    if images.ndim != 4:
        raise ValueError(
            f"Expected IMAGE batch [B,H,W,C], got shape {tuple(images.shape)}"
        )

    frame_count = int(images.shape[0])
    if frame_count <= 0:
        raise ValueError("Video contains no decodable frames")

    source = Fraction(source_rate)
    if source <= 0:
        raise ValueError(f"Invalid source frame rate: {source_rate}")

    requested_rate = float(target_rate)
    if requested_rate <= 0:
        return images

    target = _fraction_from_rate(requested_rate)
    if target == source:
        return images

    target_count = max(1, round(Fraction(frame_count, 1) * target / source))
    source_numerator = source.numerator
    source_denominator = source.denominator
    target_numerator = target.numerator
    target_denominator = target.denominator
    indices = [
        min(
            frame_count - 1,
            (
                index
                * source_numerator
                * target_denominator
                // (source_denominator * target_numerator)
            ),
        )
        for index in range(target_count)
    ]
    index_tensor = torch.tensor(indices, dtype=torch.long, device=images.device)
    return images.index_select(0, index_tensor)


def load_video_components(
    *,
    file: str,
    enable_video: bool,
    force_rate: float,
):
    """Decode through Core and return standard IMAGE/AUDIO outputs."""

    if not enable_video:
        blocker = ExecutionBlocker(None)
        return blocker, blocker

    video_path = folder_paths.get_annotated_filepath(file)
    components = InputImpl.VideoFromFile(video_path).get_components()
    images = _resample_video_frames(
        components.images,
        Fraction(components.frame_rate),
        float(force_rate),
    )
    audio = components.audio
    if audio is None:
        audio = ExecutionBlocker(None)
    return images, audio


class H3ContinuumLoadVideo(io.ComfyNode):
    """Minimal Core-backed video loader for H3 Continuum workflows."""

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            name
            for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
        ]
        files = folder_paths.filter_files_content_types(files, ["video"])
        return io.Schema(
            node_id="H3ContinuumLoadVideo",
            display_name="H3 Continuum Load Video",
            category=CATEGORY,
            description=(
                "Loads IMAGE and AUDIO through ComfyUI Core VideoFromFile. "
                "Force Rate drops or duplicates frames without changing duration or audio."
            ),
            search_aliases=[
                "H3 Continuum Load Video",
                "H3 Easy Load Video",
                "MiniMax H3 Video Loader",
            ],
            inputs=[
                io.Boolean.Input(
                    "enable_video",
                    default=True,
                    display_name="Enable Video",
                ),
                io.Combo.Input(
                    "file",
                    options=sorted(files),
                    upload=io.UploadType.video,
                    display_name="Video",
                ),
                io.Float.Input(
                    "force_rate",
                    default=24.0,
                    min=0.0,
                    max=120.0,
                    step=0.01,
                    display_name="Force Rate",
                    tooltip=(
                        "0 uses the source FPS. A positive value resamples frames "
                        "while preserving nominal duration."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, enable_video: bool, file: str, force_rate: float):
        return io.NodeOutput(
            *load_video_components(
                file=file,
                enable_video=bool(enable_video),
                force_rate=float(force_rate),
            )
        )

    @classmethod
    def fingerprint_inputs(cls, enable_video: bool, file: str, force_rate: float):
        enabled = bool(enable_video)
        file_name = str(file)
        modification_time = None
        if enabled and folder_paths.exists_annotated_filepath(file_name):
            video_path = folder_paths.get_annotated_filepath(file_name)
            modification_time = os.path.getmtime(video_path)
        return enabled, file_name, float(force_rate), modification_time

    @classmethod
    def validate_inputs(cls, enable_video: bool, file: str, force_rate: float):
        del force_rate
        if not bool(enable_video):
            return True
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


NODE_CLASS_MAPPINGS = {
    "H3ContinuumLoadVideo": H3ContinuumLoadVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumLoadVideo": "H3 Continuum Load Video",
}
