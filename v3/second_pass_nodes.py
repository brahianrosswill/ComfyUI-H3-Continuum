"""V3.5 Second Pass public node."""

from __future__ import annotations

from typing import Any

from .refine_target import MODE_AUDIO_ONLY, MODE_VIDEO_AUDIO, MODE_VIDEO_ONLY
from .refine_scope import MODE_ALL, MODE_LOGICAL_CHUNK, MODE_PHYSICAL_GROUP
from .refine_window import MODE_TIME_WINDOW
from .second_pass import run_second_pass_groups
from .targeted_second_pass import run_targeted_second_pass_groups


REFINE_TARGET_VIDEO_ONLY = "Video Only"
REFINE_TARGET_AUDIO_ONLY = "Audio Only"
REFINE_TARGET_VIDEO_AUDIO = "Video + Audio"
REFINE_TARGET_OPTIONS = (
    REFINE_TARGET_VIDEO_ONLY,
    REFINE_TARGET_AUDIO_ONLY,
    REFINE_TARGET_VIDEO_AUDIO,
)
_REFINE_TARGET_MODES = {
    REFINE_TARGET_VIDEO_ONLY: MODE_VIDEO_ONLY,
    REFINE_TARGET_AUDIO_ONLY: MODE_AUDIO_ONLY,
    REFINE_TARGET_VIDEO_AUDIO: MODE_VIDEO_AUDIO,
}

REFINE_SCOPE_ALL = "All"
REFINE_SCOPE_PHYSICAL_GROUP = "Physical Group"
REFINE_SCOPE_LOGICAL_CHUNK = "Logical Chunk"
REFINE_SCOPE_TIME_WINDOW = "Time Window"
REFINE_SCOPE_OPTIONS = (
    REFINE_SCOPE_ALL,
    REFINE_SCOPE_PHYSICAL_GROUP,
    REFINE_SCOPE_LOGICAL_CHUNK,
    REFINE_SCOPE_TIME_WINDOW,
)
_REFINE_SCOPE_MODES = {
    REFINE_SCOPE_ALL: MODE_ALL,
    REFINE_SCOPE_PHYSICAL_GROUP: MODE_PHYSICAL_GROUP,
    REFINE_SCOPE_LOGICAL_CHUNK: MODE_LOGICAL_CHUNK,
    REFINE_SCOPE_TIME_WINDOW: MODE_TIME_WINDOW,
}


def _single_list_input(name: str, value: Any) -> Any:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{name} must contain exactly one value")
    return value[0]


class H3ContinuumSecondPassV35:
    """Low-sigma refinement over complete physical decode groups."""

    DEPRECATED = False
    DESCRIPTION = (
        "V3.5 context-aware Second Pass. Connect externally upscaled H3 video "
        "latents, the matching first-pass audio latents, and the original assembly plan. "
        "Connect the optional Continuum refine context to preserve the physical group's "
        "First/Last/Reference conditioning. "
        "SIGMAS controls the low-denoise refine schedule; audio output is preserved "
        "bit-exact from the first pass."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "video_latents": ("LATENT",),
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "refine_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Base seed independently derived once per physical group.",
                    },
                ),
            },
            "optional": {
                "refine_context": ("H3_CONTINUUM_REFINE_CONTEXT",),
                "video_vae": ("VAE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "STRING")
    RETURN_NAMES = (
        "refined_video_latents",
        "audio_latents",
        "updated_assembly_plan",
        "status",
    )
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "refine"
    CATEGORY = "MiniMax H3/Continuum/Advanced"

    def refine(
        self,
        model,
        clip,
        sampler,
        sigmas,
        video_latents,
        audio_latents,
        assembly_plan,
        refine_seed,
        refine_context=None,
        video_vae=None,
        refine_schedule=None,
    ):
        return run_second_pass_groups(
            model=_single_list_input("model", model),
            clip=_single_list_input("clip", clip),
            sampler=_single_list_input("sampler", sampler),
            sigmas=_single_list_input("sigmas", sigmas),
            video_latents=video_latents,
            audio_latents=audio_latents,
            assembly_plan=_single_list_input("assembly_plan", assembly_plan),
            refine_seed=int(_single_list_input("refine_seed", refine_seed)),
            refine_context=(
                None
                if refine_context is None
                else _single_list_input("refine_context", refine_context)
            ),
            video_vae=(
                None
                if video_vae is None
                else _single_list_input("video_vae", video_vae)
            ),
            refine_schedule=refine_schedule,
        )


def resolve_selective_refine_target(value: str) -> str:
    """Resolve one public UI label to the versioned internal target mode."""

    try:
        return _REFINE_TARGET_MODES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported selective refine target {value!r}") from exc


def resolve_selective_refine_scope(value: str) -> str:
    """Resolve one public UI label to the versioned internal scope mode."""

    try:
        return _REFINE_SCOPE_MODES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unsupported selective refine scope {value!r}") from exc


def _selective_status_prefix(
    label: str,
    physical_group_count: int,
    scope_label: str,
    scope_index: int,
    window_start_sec: float,
    window_end_sec: float,
) -> str:
    if label == REFINE_TARGET_VIDEO_ONLY:
        detail = "Video=refined; Audio=bit-exact passthrough"
    elif label == REFINE_TARGET_AUDIO_ONLY:
        detail = "Experimental / Production HOLD; Video=locked; Audio=refined"
    elif label == REFINE_TARGET_VIDEO_AUDIO:
        detail = (
            "GPU Experimental PASS / Production HOLD; "
            "Video=refined; Audio=refined"
        )
    else:
        raise ValueError(f"unsupported selective refine target {label!r}")
    if scope_label == REFINE_SCOPE_ALL:
        scope_detail = "."
    elif scope_label == REFINE_SCOPE_TIME_WINDOW:
        scope_detail = f" [{window_start_sec:g}, {window_end_sec:g}) sec."
    else:
        scope_detail = f" {scope_index}."
    return (
        f"Selective Refine: {label}; {detail}; "
        f"physical groups={physical_group_count}; scope={scope_label}"
        + scope_detail
    )


class H3ContinuumSelectiveSecondPassExperimental:
    """Experimental public adapter for versioned selective-refine targets."""

    DEPRECATED = False
    DESCRIPTION = (
        "Experimental Selective Second Pass for MiniMax H3. Video Only preserves "
        "compatibility with the existing Continuum Second Pass. Audio Only keeps "
        "the input video locked and refines audio. Video + Audio refines both AV "
        "streams. Audio-targeted modes remain Experimental while upstream H3 "
        "denoise-mask correctness is under review. Chunk scope skips complete "
        "physical groups without conditioning, model clone, Sampling, or seed use; "
        "a logical chunk inside Terminal Merge expands to the complete pair. Time "
        "Window uses the final visible output timeline, protects continuation "
        "prefixes, and restores every out-of-window latent position bit-exactly."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "video_latents": ("LATENT",),
                "audio_latents": ("LATENT",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
                "refine_seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Base seed independently derived once per physical group.",
                    },
                ),
                "refine_target": (
                    REFINE_TARGET_OPTIONS,
                    {"default": REFINE_TARGET_VIDEO_ONLY},
                ),
                "refine_scope": (
                    REFINE_SCOPE_OPTIONS,
                    {"default": REFINE_SCOPE_ALL},
                ),
                "refine_scope_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 0x7FFFFFFF,
                        "tooltip": (
                            "1-based physical-group or logical-chunk number. "
                            "Ignored when Refine Scope is All or Time Window."
                        ),
                    },
                ),
                "window_start_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 86400.0,
                        "step": 0.01,
                        "tooltip": (
                            "Time Window start on the final visible output timeline. "
                            "Ignored by other Refine Scope modes."
                        ),
                    },
                ),
                "window_end_sec": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.01,
                        "max": 86400.0,
                        "step": 0.01,
                        "tooltip": (
                            "Exclusive Time Window end on the final visible output "
                            "timeline. Ignored by other Refine Scope modes."
                        ),
                    },
                ),
            },
            "optional": {
                "refine_context": ("H3_CONTINUUM_REFINE_CONTEXT",),
                "video_vae": ("VAE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("LATENT", "LATENT", "H3_CONTINUUM_ASSEMBLY_PLAN", "STRING")
    RETURN_NAMES = (
        "refined_video_latents",
        "audio_latents",
        "updated_assembly_plan",
        "status",
    )
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "refine"
    CATEGORY = "MiniMax H3/Continuum/Advanced"

    def refine(
        self,
        model,
        clip,
        sampler,
        sigmas,
        video_latents,
        audio_latents,
        assembly_plan,
        refine_seed,
        refine_target,
        refine_scope=None,
        refine_scope_index=None,
        window_start_sec=None,
        window_end_sec=None,
        refine_context=None,
        video_vae=None,
        refine_schedule=None,
    ):
        label = _single_list_input("refine_target", refine_target)
        target_mode = resolve_selective_refine_target(label)
        scope_label = (
            REFINE_SCOPE_ALL
            if refine_scope is None
            else _single_list_input("refine_scope", refine_scope)
        )
        scope_mode = resolve_selective_refine_scope(scope_label)
        resolved_scope_index = (
            1
            if refine_scope_index is None
            else int(_single_list_input("refine_scope_index", refine_scope_index))
        )
        resolved_window_start = (
            0.0
            if window_start_sec is None
            else float(_single_list_input("window_start_sec", window_start_sec))
        )
        resolved_window_end = (
            5.0
            if window_end_sec is None
            else float(_single_list_input("window_end_sec", window_end_sec))
        )
        videos, audios, updated_plan, status = run_targeted_second_pass_groups(
            model=_single_list_input("model", model),
            clip=_single_list_input("clip", clip),
            sampler=_single_list_input("sampler", sampler),
            sigmas=_single_list_input("sigmas", sigmas),
            video_latents=video_latents,
            audio_latents=audio_latents,
            assembly_plan=_single_list_input("assembly_plan", assembly_plan),
            refine_seed=int(_single_list_input("refine_seed", refine_seed)),
            refine_target=target_mode,
            refine_scope=scope_mode,
            refine_scope_index=resolved_scope_index,
            window_start_sec=resolved_window_start,
            window_end_sec=resolved_window_end,
            refine_context=(
                None
                if refine_context is None
                else _single_list_input("refine_context", refine_context)
            ),
            video_vae=(
                None
                if video_vae is None
                else _single_list_input("video_vae", video_vae)
            ),
            refine_schedule=refine_schedule,
        )
        prefix = _selective_status_prefix(
            label,
            len(video_latents),
            scope_label,
            resolved_scope_index,
            resolved_window_start,
            resolved_window_end,
        )
        return videos, audios, updated_plan, f"{prefix}\n{status}"


NODE_CLASS_MAPPINGS = {
    "H3ContinuumSecondPassV35": H3ContinuumSecondPassV35,
    "H3ContinuumSelectiveSecondPassExperimental": (
        H3ContinuumSelectiveSecondPassExperimental
    ),
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumSecondPassV35": "H3 Continuum Second Pass",
    "H3ContinuumSelectiveSecondPassExperimental": (
        "H3 Continuum Selective Second Pass (Experimental)"
    ),
}
