"""V3.8 Easy Mode facade and deterministic UI resolvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from ..conditioning import conditioning_display_label
from ..constants import (
    CONTINUITY_OPTIONS,
    DIAGNOSTICS_BASIC,
    PROMPT_FORMAT_AUTO,
)
from ..reference import REFERENCE_SIZE_MATCH_OUTPUT
from ..reference_video import REFERENCE_VIDEO_SIZE_EFFICIENT
from .driving_nodes import (
    CONTINUATION_BACKEND_STANDARD,
    H3ContinuumSamplerV37,
)
from .nodes import CATEGORY as CONTINUUM_CATEGORY, REGENERATE_AUTO
from .resolution import (
    H3_ASPECT_AUTO,
    H3_ASPECT_LANDSCAPE,
    H3_ASPECT_OPTIONS,
    H3_ASPECT_PORTRAIT,
    H3_ASPECT_SQUARE,
    H3_BALANCED_MP,
    H3_CANVAS_MULTIPLE,
    H3_CUSTOM_MP_DEFAULT,
    H3_CUSTOM_MP_MAX,
    H3_CUSTOM_MP_MIN,
    H3_CUSTOM_MP_STRONG_WARNING,
    H3_DRAFT_MP,
    H3_MP_PIXELS,
    H3_NATIVE_LONG_EDGE_CAP,
    H3_NATIVE_SHORT_EDGE,
    H3_PRESET_BALANCED,
    H3_PRESET_CUSTOM,
    H3_PRESET_DRAFT,
    H3_PRESET_NATIVE,
    H3_PRESET_OPTIONS,
    H3ResolutionPlan,
    resolve_h3_aspect,
    resolve_h3_resolution,
)


EASY_DURATION_DEFAULT = 10
EASY_DURATION_MIN = 4
EASY_DURATION_MAX = 480
EASY_TARGET_CHUNK_SECONDS = 10.0
EASY_MAX_CHUNKS = 16

EASY_SEED_MODE_RANDOMIZE = "Randomize"
EASY_SEED_MODE_FIXED = "Fixed"
EASY_SEED_MODE_OPTIONS = (
    EASY_SEED_MODE_RANDOMIZE,
    EASY_SEED_MODE_FIXED,
)

EASY_ASPECT_AUTO = H3_ASPECT_AUTO
EASY_ASPECT_LANDSCAPE = H3_ASPECT_LANDSCAPE
EASY_ASPECT_PORTRAIT = H3_ASPECT_PORTRAIT
EASY_ASPECT_SQUARE = H3_ASPECT_SQUARE
EASY_ASPECT_OPTIONS = H3_ASPECT_OPTIONS

EASY_PRESET_DRAFT = H3_PRESET_DRAFT
EASY_PRESET_BALANCED = H3_PRESET_BALANCED
EASY_PRESET_NATIVE = H3_PRESET_NATIVE
EASY_PRESET_CUSTOM = H3_PRESET_CUSTOM
EASY_PRESET_OPTIONS = H3_PRESET_OPTIONS

EASY_DRAFT_MP = H3_DRAFT_MP
EASY_BALANCED_MP = H3_BALANCED_MP
EASY_CUSTOM_MP_DEFAULT = H3_CUSTOM_MP_DEFAULT
EASY_CUSTOM_MP_MIN = H3_CUSTOM_MP_MIN
EASY_CUSTOM_MP_MAX = H3_CUSTOM_MP_MAX
EASY_CUSTOM_MP_STRONG_WARNING = H3_CUSTOM_MP_STRONG_WARNING
EASY_MP_PIXELS = H3_MP_PIXELS
EASY_CANVAS_MULTIPLE = H3_CANVAS_MULTIPLE
EASY_NATIVE_SHORT_EDGE = H3_NATIVE_SHORT_EDGE
EASY_NATIVE_LONG_EDGE_CAP = H3_NATIVE_LONG_EDGE_CAP
EasyResolutionPlan = H3ResolutionPlan
resolve_easy_aspect = resolve_h3_aspect
resolve_easy_resolution = resolve_h3_resolution
EASY_REFERENCES_TYPE = "H3_CONTINUUM_EASY_REFERENCES"
EASY_REFERENCES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EasyDurationPlan:
    duration_seconds: int
    chunks: int
    chunk_seconds: float


def make_easy_references(
    reference_image_1: Any = None,
    reference_image_2: Any = None,
    reference_image_3: Any = None,
) -> dict[str, Any]:
    """Pack fixed Easy reference sockets without preprocessing their images."""

    return {
        "schema_version": EASY_REFERENCES_SCHEMA_VERSION,
        "reference_image_1": reference_image_1,
        "reference_image_2": reference_image_2,
        "reference_image_3": reference_image_3,
    }


def resolve_easy_references(references: Any = None) -> tuple[Any, Any, Any]:
    """Expand the private Easy bundle back into the mature V3.7 input contract."""

    if references is None:
        return None, None, None
    if not isinstance(references, dict):
        raise TypeError("Easy References must come from H3 Continuum Easy References")
    if references.get("schema_version") != EASY_REFERENCES_SCHEMA_VERSION:
        raise ValueError("Unsupported H3 Continuum Easy References schema version")
    return tuple(
        references.get(f"reference_image_{index}") for index in range(1, 4)
    )


def resolve_easy_duration(duration_seconds: int) -> EasyDurationPlan:
    """Resolve one total duration around ten-second logical chunks."""

    if isinstance(duration_seconds, bool):
        raise TypeError("Duration must be an integer number of seconds")
    duration = int(duration_seconds)
    if duration != duration_seconds:
        raise ValueError("Duration must be an integer number of seconds")
    if not EASY_DURATION_MIN <= duration <= EASY_DURATION_MAX:
        raise ValueError(
            f"Duration must be between {EASY_DURATION_MIN} and {EASY_DURATION_MAX} seconds"
        )
    chunks = min(
        EASY_MAX_CHUNKS,
        max(1, int(math.ceil(float(duration) / EASY_TARGET_CHUNK_SECONDS))),
    )
    chunk_seconds = float(duration) / float(chunks)
    if not 4.0 <= chunk_seconds <= 30.0:
        raise ValueError("Duration cannot be represented by the Production chunk contract")
    return EasyDurationPlan(
        duration_seconds=duration,
        chunks=chunks,
        chunk_seconds=chunk_seconds,
    )


class H3ContinuumEasyReferences:
    """Fixed three-image facade for the existing V3.7 reference contract."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "Collects up to three optional still-image references for H3 Continuum "
        "Easy V3.8. Images are passed unchanged to the existing V3.7 reference "
        "preprocessing and conditioning path."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Easy References",
        "MiniMax H3 Easy Reference Images",
    ]
    RETURN_TYPES = (EASY_REFERENCES_TYPE,)
    RETURN_NAMES = ("references",)
    FUNCTION = "pack"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "reference_image_1": (
                    "IMAGE",
                    {"display_name": "Reference Image 1 (Optional)"},
                ),
                "reference_image_2": (
                    "IMAGE",
                    {"display_name": "Reference Image 2 (Optional)"},
                ),
                "reference_image_3": (
                    "IMAGE",
                    {"display_name": "Reference Image 3 (Optional)"},
                ),
            }
        }

    def pack(
        self,
        reference_image_1=None,
        reference_image_2=None,
        reference_image_3=None,
    ):
        return (
            make_easy_references(
                reference_image_1=reference_image_1,
                reference_image_2=reference_image_2,
                reference_image_3=reference_image_3,
            ),
        )


class H3ContinuumEasyV38(H3ContinuumSamplerV37):
    """Beginner-facing resolver over the unchanged V3.7 Production engine."""

    DEPRECATED = False
    CATEGORY = CONTINUUM_CATEGORY
    DESCRIPTION = (
        "Easy Mode facade over H3 Continuum V3.7 Production. It resolves total "
        "duration, decimal-MP resolution, seed UI, and image mode without copying "
        "the sampler or scheduler implementation."
    )
    SEARCH_ALIASES = [
        "H3 Continuum Easy V3.8",
        "MiniMax H3 Easy Mode",
    ]
    RETURN_TYPES = (
        "LATENT",
        "LATENT",
        "H3_CONTINUUM_ASSEMBLY_PLAN",
        "STRING",
        "AUDIO",
    )
    RETURN_NAMES = (
        "video_latents",
        "audio_latents",
        "assembly_plan",
        "status",
        "driving_audio",
    )
    OUTPUT_IS_LIST = (True, True, False, False, False)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "video_vae": (
                    "VAE",
                    {
                        "display_name": "Video VAE",
                        "tooltip": "Used only for connected image conditioning.",
                    },
                ),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "display_name": "Prompt",
                    },
                ),
                "duration": (
                    "INT",
                    {
                        "default": EASY_DURATION_DEFAULT,
                        "min": EASY_DURATION_MIN,
                        "max": EASY_DURATION_MAX,
                        "step": 1,
                        "display_name": "Duration (sec)",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "display_name": "Seed",
                    },
                ),
                "seed_mode": (
                    EASY_SEED_MODE_OPTIONS,
                    {
                        "default": EASY_SEED_MODE_RANDOMIZE,
                        "display_name": "Seed Mode",
                    },
                ),
                "aspect": (
                    EASY_ASPECT_OPTIONS,
                    {
                        "default": EASY_ASPECT_AUTO,
                        "display_name": "Aspect",
                    },
                ),
                "preset": (
                    EASY_PRESET_OPTIONS,
                    {
                        "default": EASY_PRESET_DRAFT,
                        "display_name": "Resolution Preset",
                    },
                ),
                "custom_mp": (
                    "FLOAT",
                    {
                        "default": EASY_CUSTOM_MP_DEFAULT,
                        "min": EASY_CUSTOM_MP_MIN,
                        "max": EASY_CUSTOM_MP_MAX,
                        "step": 0.01,
                        "display_name": "Custom MP",
                    },
                ),
                "run_storage": (
                    ("Off", "Save + Auto Resume"),
                    {
                        "default": "Off",
                        "display_name": "Run Storage",
                        "advanced": True,
                    },
                ),
                "run_name": (
                    "STRING",
                    {
                        "default": "",
                        "display_name": "Run Name (Optional Override)",
                        "advanced": True,
                    },
                ),
                "project_id": (
                    "STRING",
                    {
                        "default": "",
                        "display_name": "Auto Resume ID Override",
                        "advanced": True,
                    },
                ),
            },
            "optional": {
                "first_frame": (
                    "IMAGE",
                    {"display_name": "First Image (Optional)"},
                ),
                "last_frame": (
                    "IMAGE",
                    {"display_name": "Last Image (Optional)"},
                ),
                "references": (
                    EASY_REFERENCES_TYPE,
                    {"display_name": "Easy References (Optional)"},
                ),
                "driving_audio": (
                    "AUDIO",
                    {
                        "display_name": "Driving Audio",
                        "tooltip": (
                            "Optional original audio timeline. Uses the unchanged "
                            "V3.7 Driving Audio conditioning and final-output contract."
                        ),
                    },
                ),
                "audio_vae": (
                    "VAE",
                    {
                        "display_name": "Driving Audio VAE",
                        "tooltip": (
                            "Required only when Driving Audio is connected. It may be "
                            "shared with the external Core audio decode path."
                        ),
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    def run(
        self,
        model,
        clip,
        video_vae,
        sampler,
        sigmas,
        prompt_text,
        duration,
        seed,
        seed_mode=EASY_SEED_MODE_RANDOMIZE,
        aspect=EASY_ASPECT_AUTO,
        preset=EASY_PRESET_DRAFT,
        custom_mp=EASY_CUSTOM_MP_DEFAULT,
        run_storage="Off",
        run_name="",
        project_id="",
        first_frame=None,
        last_frame=None,
        references=None,
        driving_audio=None,
        audio_vae=None,
        prompt=None,
        unique_id=None,
    ):
        del seed_mode  # Queue-time seed mutation remains owned by ComfyUI's frontend.
        duration_plan = resolve_easy_duration(duration)
        resolution = resolve_easy_resolution(
            aspect=aspect,
            preset=preset,
            custom_mp=custom_mp,
            first_frame=first_frame,
        )
        reference_image_1, reference_image_2, reference_image_3 = (
            resolve_easy_references(references)
        )
        outputs = super().run(
            model=model,
            clip=clip,
            video_vae=video_vae,
            sampler=sampler,
            sigmas=sigmas,
            sequence_prompt=prompt_text,
            prompt_mode=PROMPT_FORMAT_AUTO,
            chunks=duration_plan.chunks,
            chunk_seconds=duration_plan.chunk_seconds,
            width=resolution.width,
            height=resolution.height,
            continuity=CONTINUITY_OPTIONS[0],
            base_seed=int(seed),
            audio_continuity=True,
            diagnostics=DIAGNOSTICS_BASIC,
            reroll_from_chunk=REGENERATE_AUTO,
            reroll_nonce=0,
            strict_compatibility=False,
            debug=False,
            show_preview=True,
            run_storage=run_storage,
            run_name=run_name,
            reference_size=REFERENCE_SIZE_MATCH_OUTPUT,
            reference_image_1=reference_image_1,
            reference_image_2=reference_image_2,
            reference_image_3=reference_image_3,
            driving_audio=driving_audio,
            audio_vae=audio_vae,
            project_id=project_id,
            first_frame=first_frame,
            last_frame=last_frame,
            prompt=prompt,
            unique_id=unique_id,
            video_reference_size=REFERENCE_VIDEO_SIZE_EFFICIENT,
            continuation_backend=CONTINUATION_BACKEND_STANDARD,
        )
        video, audio, assembly_plan, status, driving_audio = outputs[:5]
        mode = conditioning_display_label(
            has_first=first_frame is not None,
            has_last=last_frame is not None,
            has_reference=any(
                image is not None
                for image in (
                    reference_image_1,
                    reference_image_2,
                    reference_image_3,
                )
            ),
        )
        warning_lines = "\n".join(resolution.warnings)
        summary = (
            f"H3 Continuum Easy V3.8\n"
            f"{mode} / {duration_plan.duration_seconds} sec\n"
            f"{resolution.preset}: {resolution.width} x {resolution.height} "
            f"({resolution.actual_mp:.2f} MP)"
        )
        if warning_lines:
            summary += f"\n{warning_lines}"
        return video, audio, assembly_plan, f"{summary}\n\n{status}", driving_audio


NODE_CLASS_MAPPINGS = {
    "H3ContinuumEasyReferences": H3ContinuumEasyReferences,
    "H3ContinuumEasyV38": H3ContinuumEasyV38,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumEasyReferences": "H3 Continuum Easy References",
    "H3ContinuumEasyV38": "H3 Continuum Easy V3.8",
}
