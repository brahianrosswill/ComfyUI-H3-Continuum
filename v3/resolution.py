"""Shared H3 resolution presets for Easy and V3.8 sampler facades."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


H3_ASPECT_AUTO = "Auto from First Image"
H3_ASPECT_LANDSCAPE = "Landscape 16:9"
H3_ASPECT_PORTRAIT = "Portrait 9:16"
H3_ASPECT_SQUARE = "Square 1:1"
H3_ASPECT_OPTIONS = (
    H3_ASPECT_AUTO,
    H3_ASPECT_LANDSCAPE,
    H3_ASPECT_PORTRAIT,
    H3_ASPECT_SQUARE,
)

H3_SIZE_SOURCE_FIRST_IMAGE = "First Image"
H3_SIZE_SOURCE_MANUAL = "Manual"
H3_SIZE_SOURCE_LEGACY = "Legacy Aspect"
H3_SIZE_SOURCE_OPTIONS = (
    H3_SIZE_SOURCE_FIRST_IMAGE,
    H3_SIZE_SOURCE_MANUAL,
    H3_SIZE_SOURCE_LEGACY,
)

H3_PRESET_DRAFT = "Draft — 0.30 MP"
H3_PRESET_BALANCED = "Balanced — 0.60 MP"
H3_PRESET_NATIVE = "Native 768"
H3_PRESET_CUSTOM = "Custom"
H3_PRESET_OPTIONS = (
    H3_PRESET_DRAFT,
    H3_PRESET_BALANCED,
    H3_PRESET_NATIVE,
    H3_PRESET_CUSTOM,
)

H3_DRAFT_MP = 0.30
H3_BALANCED_MP = 0.60
H3_CUSTOM_MP_DEFAULT = 0.30
H3_CUSTOM_MP_MIN = 0.10
H3_CUSTOM_MP_MAX = 16.0
H3_CUSTOM_MP_STRONG_WARNING = 2.50
H3_MP_PIXELS = 1_000_000.0
H3_CANVAS_MULTIPLE = 32
H3_NATIVE_SHORT_EDGE = 768
H3_NATIVE_LONG_EDGE_CAP = 1344
H3_MANUAL_WIDTH_DEFAULT = 736
H3_MANUAL_HEIGHT_DEFAULT = 416
H3_CANVAS_MIN = 32
H3_CANVAS_MAX = 16384


@dataclass(frozen=True)
class H3ResolutionPlan:
    width: int
    height: int
    aspect_ratio: float
    aspect_source: str
    preset: str
    target_mp: float | None
    warnings: tuple[str, ...]

    @property
    def actual_mp(self) -> float:
        return float(self.width * self.height) / H3_MP_PIXELS


def _round_canvas(value: float) -> int:
    units = math.floor(float(value) / H3_CANVAS_MULTIPLE + 0.5)
    return max(H3_CANVAS_MULTIPLE, int(units) * H3_CANVAS_MULTIPLE)


def _first_image_aspect(first_frame: Any) -> float | None:
    shape = getattr(first_frame, "shape", None)
    if shape is None or len(shape) != 4:
        return None
    height = int(shape[1])
    width = int(shape[2])
    if width < 1 or height < 1:
        return None
    return float(width) / float(height)


def _manual_dimension(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"Manual {name} must be an integer")
    resolved = int(value)
    if float(value) != float(resolved):
        raise ValueError(f"Manual {name} must be an integer")
    if not H3_CANVAS_MIN <= resolved <= H3_CANVAS_MAX:
        raise ValueError(
            f"Manual {name} must be between {H3_CANVAS_MIN} and {H3_CANVAS_MAX}"
        )
    if resolved % H3_CANVAS_MULTIPLE:
        raise ValueError(
            f"Manual {name} must be aligned to {H3_CANVAS_MULTIPLE} pixels"
        )
    return resolved


def _manual_resolution(width: Any, height: Any, *, warning: str | None = None) -> H3ResolutionPlan:
    resolved_width = _manual_dimension(width, "Width")
    resolved_height = _manual_dimension(height, "Height")
    return H3ResolutionPlan(
        width=resolved_width,
        height=resolved_height,
        aspect_ratio=float(resolved_width) / float(resolved_height),
        aspect_source="Manual Width / Height",
        preset="Manual",
        target_mp=None,
        warnings=(warning,) if warning else (),
    )


def resolve_h3_aspect(aspect: str, first_frame: Any = None) -> tuple[float, str]:
    """Resolve the existing Easy aspect policy without creating a second fallback."""

    if aspect == H3_ASPECT_AUTO:
        image_aspect = _first_image_aspect(first_frame)
        if image_aspect is not None:
            return image_aspect, "First Image"
        return 16.0 / 9.0, "Landscape 16:9 fallback"
    if aspect == H3_ASPECT_LANDSCAPE:
        return 16.0 / 9.0, H3_ASPECT_LANDSCAPE
    if aspect == H3_ASPECT_PORTRAIT:
        return 9.0 / 16.0, H3_ASPECT_PORTRAIT
    if aspect == H3_ASPECT_SQUARE:
        return 1.0, H3_ASPECT_SQUARE
    return 16.0 / 9.0, "Landscape 16:9 fallback"


def _area_resolution(aspect_ratio: float, megapixels: float) -> tuple[int, int]:
    target_pixels = float(megapixels) * H3_MP_PIXELS
    raw_width = math.sqrt(target_pixels * float(aspect_ratio))
    raw_height = math.sqrt(target_pixels / float(aspect_ratio))
    return _round_canvas(raw_width), _round_canvas(raw_height)


def _native_resolution(aspect_ratio: float) -> tuple[int, int]:
    landscape = float(aspect_ratio) >= 1.0
    major_ratio = float(aspect_ratio) if landscape else 1.0 / float(aspect_ratio)
    short_edge = float(H3_NATIVE_SHORT_EDGE)
    long_edge = short_edge * major_ratio
    if long_edge > H3_NATIVE_LONG_EDGE_CAP:
        long_edge = float(H3_NATIVE_LONG_EDGE_CAP)
        short_edge = long_edge / major_ratio
    long_px = min(H3_NATIVE_LONG_EDGE_CAP, _round_canvas(long_edge))
    short_px = min(H3_NATIVE_SHORT_EDGE, _round_canvas(short_edge))
    if landscape:
        return long_px, short_px
    return short_px, long_px


def resolve_h3_resolution(
    *,
    aspect: str,
    preset: str,
    custom_mp: float = H3_CUSTOM_MP_DEFAULT,
    first_frame: Any = None,
) -> H3ResolutionPlan:
    """Resolve decimal-MP or Native 768 dimensions on the H3 32-pixel grid."""

    aspect_ratio, aspect_source = resolve_h3_aspect(aspect, first_frame)
    warnings: list[str] = []
    target_mp: float | None
    if preset == H3_PRESET_DRAFT:
        target_mp = H3_DRAFT_MP
        width, height = _area_resolution(aspect_ratio, target_mp)
    elif preset == H3_PRESET_BALANCED:
        target_mp = H3_BALANCED_MP
        width, height = _area_resolution(aspect_ratio, target_mp)
    elif preset == H3_PRESET_NATIVE:
        target_mp = None
        width, height = _native_resolution(aspect_ratio)
    else:
        preset = H3_PRESET_CUSTOM
        if isinstance(custom_mp, bool):
            raise TypeError("Custom MP must be numeric")
        target_mp = float(custom_mp)
        if not H3_CUSTOM_MP_MIN <= target_mp <= H3_CUSTOM_MP_MAX:
            raise ValueError(
                f"Custom MP must be between {H3_CUSTOM_MP_MIN:.2f} and "
                f"{H3_CUSTOM_MP_MAX:.2f}"
            )
        width, height = _area_resolution(aspect_ratio, target_mp)
        native_width, native_height = _native_resolution(aspect_ratio)
        native_pixels = native_width * native_height
        if width * height > native_pixels:
            warnings.append(
                "Above Native 768: processing time and memory usage may increase."
            )
        if target_mp > H3_CUSTOM_MP_STRONG_WARNING:
            warnings.append(
                "High Custom MP: substantial VRAM use and generation time are possible."
            )
    return H3ResolutionPlan(
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        aspect_source=aspect_source,
        preset=preset,
        target_mp=target_mp,
        warnings=tuple(warnings),
    )


def resolve_h3_size_source(
    *,
    size_source: str,
    width: Any = H3_MANUAL_WIDTH_DEFAULT,
    height: Any = H3_MANUAL_HEIGHT_DEFAULT,
    aspect: str = H3_ASPECT_AUTO,
    preset: str = H3_PRESET_DRAFT,
    custom_mp: float = H3_CUSTOM_MP_DEFAULT,
    first_frame: Any = None,
) -> H3ResolutionPlan:
    """Resolve the V3.8 two-choice Size Source while retaining legacy API input."""

    if size_source == H3_SIZE_SOURCE_MANUAL:
        return _manual_resolution(width, height)
    if size_source == H3_SIZE_SOURCE_FIRST_IMAGE:
        if _first_image_aspect(first_frame) is None:
            return _manual_resolution(
                width,
                height,
                warning=(
                    "First Image is unavailable for Size Source = First Image; "
                    "the displayed Manual Width / Height were used."
                ),
            )
        return resolve_h3_resolution(
            aspect=H3_ASPECT_AUTO,
            preset=preset,
            custom_mp=custom_mp,
            first_frame=first_frame,
        )
    # Existing API prompts and saved workflows omit size_source. Preserve their
    # exact V3.8 aspect resolver until the frontend migrates the visible widgets.
    return resolve_h3_resolution(
        aspect=aspect,
        preset=preset,
        custom_mp=custom_mp,
        first_frame=first_frame,
    )
