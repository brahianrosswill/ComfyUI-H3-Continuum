"""Analyze Issue #13 R2 decoded video and continuation-latent trends."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


METRIC_KEYS = (
    "luma_mean",
    "luma_std",
    "saturation_mean",
    "saturation_p95",
    "contrast_p95_p05",
    "sharpness_laplacian_variance",
    "gradient_mean",
    "highlight_clip_fraction",
    "skin_highlight_proxy_fraction",
    "shadow_clip_fraction",
)

LATENT_KEYS = (
    "mean",
    "std",
    "rms",
    "minimum",
    "maximum",
    "temporal_delta_rms",
)


def _video_path(record: dict, video_root: Path) -> Path:
    item = record.get("output_video") or {}
    return video_root / str(item.get("subfolder", "")) / str(item.get("filename", ""))


def _read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"video contains no frames: {path}")
    return frames, fps


def _frame_metrics(frame: np.ndarray) -> dict[str, float]:
    rgb = frame.astype(np.float32) / 255.0
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[..., 1] / 255.0
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y) / 255.0
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    skin_mask = (
        (ycrcb[..., 1] >= 133)
        & (ycrcb[..., 1] <= 173)
        & (ycrcb[..., 2] >= 77)
        & (ycrcb[..., 2] <= 127)
    )
    skin_count = int(np.count_nonzero(skin_mask))
    skin_highlight = (
        float(np.count_nonzero(skin_mask & (y >= (220.0 / 255.0))))
        / float(skin_count)
        if skin_count
        else 0.0
    )
    return {
        "luma_mean": float(np.mean(y)),
        "luma_std": float(np.std(y)),
        "red_mean": float(np.mean(rgb[..., 0])),
        "green_mean": float(np.mean(rgb[..., 1])),
        "blue_mean": float(np.mean(rgb[..., 2])),
        "saturation_mean": float(np.mean(saturation)),
        "saturation_p95": float(np.percentile(saturation, 95)),
        "contrast_p95_p05": float(np.percentile(y, 95) - np.percentile(y, 5)),
        "sharpness_laplacian_variance": float(np.var(laplacian)),
        "gradient_mean": float(np.mean(gradient)),
        "highlight_clip_fraction": float(np.mean(y >= (250.0 / 255.0))),
        "skin_highlight_proxy_fraction": skin_highlight,
        "shadow_clip_fraction": float(np.mean(y <= (5.0 / 255.0))),
    }


def _chunk_metrics(
    frames: list[np.ndarray], *, chunks: int, radius: int
) -> tuple[list[dict], list[np.ndarray]]:
    frame_count = len(frames)
    rows = []
    representatives = []
    for index in range(chunks):
        center = int(round((index + 0.5) * frame_count / chunks))
        center = min(max(center, 0), frame_count - 1)
        start = max(0, center - radius)
        end = min(frame_count, center + radius + 1)
        values = [_frame_metrics(frame) for frame in frames[start:end]]
        merged = {
            key: float(np.mean([item[key] for item in values])) for key in values[0]
        }
        merged.update(
            {
                "chunk": index + 1,
                "center_frame": center,
                "sample_start": start,
                "sample_end_exclusive": end,
                "sample_count": end - start,
            }
        )
        rows.append(merged)
        representatives.append(frames[center])
    return rows, representatives


def _normalized_histogram(values: np.ndarray, *, bins: int = 16) -> list[float]:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = int(counts.sum())
    if total <= 0:
        return [0.0] * int(bins)
    return [float(value) / float(total) for value in counts]


def _continuation_input_rgb_proxy(
    frames: list[np.ndarray], *, chunks: int, context_frames: int
) -> dict:
    """Measure decoded source frames immediately before each chunk boundary.

    The R2 diagnostic separately proves that each source latent tail is copied
    bit-exactly into the following target prefix. This read-only decoded window
    is therefore a visual proxy; it does not add a VAE decode to Sampling.
    """

    boundaries = []
    frame_count = len(frames)
    for target_index in range(1, int(chunks)):
        boundary = int(round(target_index * frame_count / int(chunks)))
        boundary = min(max(boundary, 1), frame_count)
        start = max(0, boundary - int(context_frames))
        window = frames[start:boundary]
        values = [_frame_metrics(frame) for frame in window]
        merged = {
            key: float(np.mean([item[key] for item in values])) for key in values[0]
        }
        rgb = np.concatenate(
            [frame.astype(np.float32).reshape(-1, 3) / 255.0 for frame in window],
            axis=0,
        )
        luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
        saturation = np.concatenate(
            [
                cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[..., 1]
                .reshape(-1)
                .astype(np.float32)
                / 255.0
                for frame in window
            ]
        )
        merged.update(
            {
                "source_group": target_index,
                "target_group": target_index + 1,
                "boundary_frame": boundary,
                "window_start": start,
                "window_end_exclusive": boundary,
                "window_frames": len(window),
                "histogram_bins": 16,
                "red_histogram": _normalized_histogram(rgb[:, 0]),
                "green_histogram": _normalized_histogram(rgb[:, 1]),
                "blue_histogram": _normalized_histogram(rgb[:, 2]),
                "luma_histogram": _normalized_histogram(luma),
                "saturation_histogram": _normalized_histogram(saturation),
            }
        )
        boundaries.append(merged)
    return {
        "contract": "decoded source-tail proxy; no extra VAE decode during Sampling",
        "context_frames": int(context_frames),
        "boundaries": boundaries,
        "trend": _trend(boundaries, METRIC_KEYS) if boundaries else {},
    }


def _trend(rows: list[dict], keys: tuple[str, ...]) -> dict:
    x = np.arange(1, len(rows) + 1, dtype=np.float64)
    result = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        result[key] = {
            "first": float(values[0]),
            "last": float(values[-1]),
            "last_minus_first": float(values[-1] - values[0]),
            "linear_slope_per_chunk": float(np.polyfit(x, values, 1)[0])
            if len(values) > 1
            else 0.0,
            "increase_steps": int(np.count_nonzero(np.diff(values) > 0.0)),
            "decrease_steps": int(np.count_nonzero(np.diff(values) < 0.0)),
        }
    return result


def _flow_mean(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    return float(np.linalg.norm(flow, axis=-1).mean())


def _boundary_metrics(frames: list[np.ndarray], *, chunks: int) -> list[dict]:
    rows = []
    frame_count = len(frames)
    for target_index in range(1, int(chunks)):
        boundary = int(round(target_index * frame_count / int(chunks)))
        boundary = min(max(boundary, 2), frame_count - 2)
        before2, before, after, after2 = frames[
            boundary - 2
        ], frames[boundary - 1], frames[boundary], frames[boundary + 1]
        before_float = before.astype(np.float32) / 255.0
        after_float = after.astype(np.float32) / 255.0
        before_metrics = _frame_metrics(before)
        after_metrics = _frame_metrics(after)
        flow_before = _flow_mean(before2, before)
        flow_boundary = _flow_mean(before, after)
        flow_after = _flow_mean(after, after2)
        local_flow = 0.5 * (flow_before + flow_after)
        rows.append(
            {
                "source_group": target_index,
                "target_group": target_index + 1,
                "boundary_frame": boundary,
                "frame_mae": float(np.abs(after_float - before_float).mean()),
                "luma_jump_abs": abs(
                    after_metrics["luma_mean"] - before_metrics["luma_mean"]
                ),
                "saturation_jump_abs": abs(
                    after_metrics["saturation_mean"]
                    - before_metrics["saturation_mean"]
                ),
                "gradient_jump_abs": abs(
                    after_metrics["gradient_mean"]
                    - before_metrics["gradient_mean"]
                ),
                "flow_before": flow_before,
                "flow_boundary": flow_boundary,
                "flow_after": flow_after,
                "trajectory_flow_discontinuity": abs(flow_boundary - local_flow)
                / max(local_flow, 1.0e-6),
            }
        )
    return rows


def _latent_trend(diagnostic: dict) -> dict:
    calls = list(diagnostic.get("sample_calls") or [])

    def rows_for(name: str) -> list[dict]:
        rows = []
        for call in calls:
            value = call.get(name)
            if not isinstance(value, dict):
                continue
            row = {"sample_number": int(call.get("sample_number", len(rows) + 1))}
            for key in LATENT_KEYS:
                if value.get(key) is not None:
                    row[key] = float(value[key])
            rows.append(row)
        return rows

    streams = {}
    for name in ("input_video", "output_video", "output_tail", "source_tail"):
        rows = rows_for(name)
        available = tuple(key for key in LATENT_KEYS if rows and key in rows[0])
        streams[name] = {
            "rows": rows,
            "trend": _trend(rows, available) if rows and available else {},
        }
    prefix_pairs = list(diagnostic.get("final_prefix_pairs") or [])
    return {
        "streams": streams,
        "prefix_pair_count": len(prefix_pairs),
        "all_video_prefix_pairs_bit_exact": bool(prefix_pairs)
        and all(bool(pair.get("bit_exact")) for pair in prefix_pairs),
        "all_audio_prefix_pairs_bit_exact": bool(prefix_pairs)
        and all(bool(pair.get("audio_bit_exact")) for pair in prefix_pairs),
        "maximum_video_prefix_abs_diff": max(
            (float(pair.get("max_abs_diff", 0.0)) for pair in prefix_pairs),
            default=0.0,
        ),
        "maximum_audio_prefix_abs_diff": max(
            (float(pair.get("audio_max_abs_diff", 0.0)) for pair in prefix_pairs),
            default=0.0,
        ),
    }


def _contact_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    output: Path,
    *,
    column_labels: list[str] | None = None,
) -> None:
    thumb_w, thumb_h = 224, 224
    label_w, top_h = 240, 42
    columns = max((len(frames) for _, frames in rows), default=0)
    width = label_w + thumb_w * columns
    height = top_h + thumb_h * len(rows)
    sheet = Image.new("RGB", (width, height), (9, 15, 25))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column in range(columns):
        draw.text(
            (label_w + column * thumb_w + 8, 14),
            column_labels[column]
            if column_labels is not None and column < len(column_labels)
            else f"Chunk {column + 1}",
            fill=(210, 225, 245),
            font=font,
        )
    for row_index, (label, frames) in enumerate(rows):
        y = top_h + row_index * thumb_h
        draw.text((10, y + 16), label, fill=(115, 210, 255), font=font)
        for column, frame in enumerate(frames):
            image = Image.fromarray(frame).resize(
                (thumb_w, thumb_h), Image.Resampling.LANCZOS
            )
            sheet.paste(image, (label_w + column * thumb_w, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _milestone_frames(
    frames: list[np.ndarray], *, chunks: int
) -> tuple[list[str], list[np.ndarray]]:
    frame_count = len(frames)
    chunk_frames = frame_count / float(chunks)
    indices = [
        min(frame_count - 1, 12),
        min(frame_count - 1, max(0, int(round(3 * chunk_frames)) - 1)),
        min(frame_count - 1, int(round(3 * chunk_frames)) + 12),
        min(frame_count - 1, int(round(4.5 * chunk_frames))),
        frame_count - 1,
    ]
    return (
        ["Near 0s", "G3 end", "G4 post", "G5", "G6 end"],
        [frames[index] for index in indices],
    )


def run(args: argparse.Namespace) -> dict:
    summary = json.loads((args.output_root / "summary.json").read_text(encoding="utf-8"))
    all_rows = []
    sheets = []
    milestone_sheets = []
    milestone_labels = None
    videos = {}
    for record in summary["cases"]:
        case = str(record["case"])
        chunks = int(record["chunks"])
        path = _video_path(record, args.video_root)
        frames, fps = _read_frames(path)
        metrics, representatives = _chunk_metrics(
            frames, chunks=chunks, radius=args.radius
        )
        for row in metrics:
            row["case"] = case
            row["video"] = str(path)
            row["fps"] = fps
            row["frame_count"] = len(frames)
            all_rows.append(row)
        sheets.append((case, representatives))
        labels, milestones = _milestone_frames(frames, chunks=chunks)
        milestone_labels = labels
        milestone_sheets.append((case, milestones))
        videos[case] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fps": fps,
            "frame_count": len(frames),
            "duration_seconds": len(frames) / fps,
            "chunks": chunks,
            "chunk_seconds": float(record["chunk_seconds"]),
            "metrics": metrics,
            "decoded_trend": _trend(metrics, METRIC_KEYS),
            "continuation_input_rgb_proxy": _continuation_input_rgb_proxy(
                frames,
                chunks=chunks,
                context_frames=int(
                    (record.get("diagnostic") or {}).get("context_frames", 22)
                ),
            ),
            "boundary_metrics": _boundary_metrics(frames, chunks=chunks),
            "latent_trend": _latent_trend(record.get("diagnostic") or {}),
        }

    if not all_rows:
        raise RuntimeError("Issue #13 R2 summary contains no completed cases")
    csv_path = args.output_root / "chunk_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    contact_path = args.output_root / "contact_sheet.png"
    _contact_sheet(sheets, contact_path)
    milestone_contact_path = args.output_root / "milestone_contact_sheet.png"
    _contact_sheet(
        milestone_sheets,
        milestone_contact_path,
        column_labels=milestone_labels,
    )
    result = {
        "format": "h3-continuum-issue13-r2-analysis-v2",
        "sample_radius_frames": args.radius,
        "videos": videos,
        "csv": str(csv_path),
        "contact_sheet": str(contact_path),
        "milestone_contact_sheet": str(milestone_contact_path),
    }
    (args.output_root / "analysis.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--video-root", type=Path, default=Path(r"D:\output\video\comfy_video")
    )
    parser.add_argument("--radius", type=int, default=12)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
