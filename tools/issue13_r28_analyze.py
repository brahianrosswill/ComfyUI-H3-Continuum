"""Compare Issue #13 R2.8 candidates with the accepted recursive 22f baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np


TOOLS_ROOT = Path(__file__).resolve().parent
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

import issue13_r2_analyze as base


DRIFT_KEYS = (
    "saturation_mean",
    "contrast_p95_p05",
    "sharpness_laplacian_variance",
    "gradient_mean",
    "highlight_clip_fraction",
    "skin_highlight_proxy_fraction",
)


def _case_video(record: dict, video_root: Path) -> Path:
    value = record.get("output_video") or {}
    return video_root / str(value.get("subfolder", "")) / str(value.get("filename", ""))


def _end_metrics(frames: list[np.ndarray], *, chunks: int, radius: int) -> list[dict]:
    rows = []
    for index in range(int(chunks)):
        stop = min(len(frames), int(round((index + 1) * len(frames) / chunks)))
        start = max(0, stop - int(radius))
        values = [base._frame_metrics(frame) for frame in frames[start:stop]]
        merged = {
            key: float(np.mean([item[key] for item in values])) for key in values[0]
        }
        merged.update(
            {
                "group": index + 1,
                "sample_start": start,
                "sample_end_exclusive": stop,
                "sample_count": stop - start,
            }
        )
        rows.append(merged)
    return rows


def _face_crop(frame: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5)
    if len(faces):
        x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
        margin_x = int(round(width * 0.12))
        margin_y = int(round(height * 0.12))
        x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
        x1 = min(frame.shape[1], x + width + margin_x)
        y1 = min(frame.shape[0], y + height + margin_y)
        return frame[y0:y1, x0:x1], "haar"
    height, width = frame.shape[:2]
    return frame[
        int(height * 0.08) : int(height * 0.68),
        int(width * 0.18) : int(width * 0.82),
    ], "center_fallback"


def _face_descriptor(frame: np.ndarray) -> tuple[np.ndarray, str]:
    crop, method = _face_crop(frame)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray = (gray - gray.mean()) / max(float(gray.std()), 1.0e-6)
    descriptor = cv2.dct(gray)[:16, :16].reshape(-1)[1:]
    descriptor /= max(float(np.linalg.norm(descriptor)), 1.0e-6)
    return descriptor, method


def _identity_and_frame_difference(representatives: list[np.ndarray]) -> dict:
    descriptors = []
    methods = []
    crops = []
    for frame in representatives:
        descriptor, method = _face_descriptor(frame)
        crop, _ = _face_crop(frame)
        descriptors.append(descriptor)
        methods.append(method)
        crops.append(
            cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
            / 255.0
        )
    reference = descriptors[0]
    similarities = [float((np.dot(reference, value) + 1.0) * 0.5) for value in descriptors]
    adjacent_mae = [
        float(np.abs(crops[index] - crops[index - 1]).mean())
        for index in range(1, len(crops))
    ]
    return {
        "contract": "face-structure DCT similarity proxy; not a biometric identity score",
        "crop_methods": methods,
        "group1_similarity": similarities,
        "group1_to_group6_similarity": similarities[-1],
        "adjacent_face_frame_mae": adjacent_mae,
        "mean_adjacent_face_frame_mae": float(np.mean(adjacent_mae)),
    }


def _metric_comparison(baseline: dict, candidate: dict) -> dict:
    metrics = {}
    reductions = []
    for key in DRIFT_KEYS:
        baseline_trend = baseline["end_trend"][key]
        candidate_trend = candidate["end_trend"][key]
        baseline_delta = float(baseline_trend["last_minus_first"])
        candidate_delta = float(candidate_trend["last_minus_first"])
        baseline_slope = float(baseline_trend["linear_slope_per_chunk"])
        candidate_slope = float(candidate_trend["linear_slope_per_chunk"])
        reduction = None
        slope_reduction = None
        if baseline_delta > 0.0:
            reduction = (baseline_delta - candidate_delta) / baseline_delta * 100.0
            reductions.append(reduction)
        if baseline_slope > 0.0:
            slope_reduction = (
                (baseline_slope - candidate_slope) / baseline_slope * 100.0
            )
        metrics[key] = {
            "baseline_delta": baseline_delta,
            "candidate_delta": candidate_delta,
            "reduction_percent": reduction,
            "baseline_slope_per_continuation": baseline_slope,
            "candidate_slope_per_continuation": candidate_slope,
            "slope_reduction_percent": slope_reduction,
            "scored": baseline_delta > 0.0,
        }
    baseline_flow = float(
        np.mean(
            [
                row["trajectory_flow_discontinuity"]
                for row in baseline["boundary_metrics"]
            ]
        )
    )
    candidate_flow = float(
        np.mean(
            [
                row["trajectory_flow_discontinuity"]
                for row in candidate["boundary_metrics"]
            ]
        )
    )
    mean_reduction = float(np.mean(reductions)) if reductions else None
    sharpness_ok = (
        metrics["sharpness_laplacian_variance"]["candidate_delta"]
        <= metrics["sharpness_laplacian_variance"]["baseline_delta"]
    )
    gradient_ok = (
        metrics["gradient_mean"]["candidate_delta"]
        <= metrics["gradient_mean"]["baseline_delta"]
    )
    flow_ratio = candidate_flow / max(baseline_flow, 1.0e-9)
    numeric_promising = bool(
        mean_reduction is not None
        and mean_reduction >= 30.0
        and sharpness_ok
        and gradient_ok
        and flow_ratio <= 1.15
    )
    return {
        "metrics": metrics,
        "screening_mean_reduction_percent": mean_reduction,
        "screening_mean_contract": (
            "mean of baseline-positive drift indicators only; visual review remains required"
        ),
        "sharpness_not_worse": sharpness_ok,
        "gradient_not_worse": gradient_ok,
        "baseline_mean_flow_discontinuity": baseline_flow,
        "candidate_mean_flow_discontinuity": candidate_flow,
        "flow_ratio": flow_ratio,
        "numeric_promising": numeric_promising,
    }


def run(args: argparse.Namespace) -> dict:
    summary = json.loads((args.output_root / "summary.json").read_text(encoding="utf-8"))
    baseline_analysis = json.loads(args.baseline_analysis.read_text(encoding="utf-8"))
    baseline_key = args.baseline_case
    baseline_record = baseline_analysis["videos"][baseline_key]
    baseline_path = Path(baseline_record["path"])
    baseline_frames, baseline_fps = base._read_frames(baseline_path)
    baseline_center, baseline_representatives = base._chunk_metrics(
        baseline_frames, chunks=6, radius=args.radius
    )
    baseline_end = _end_metrics(baseline_frames, chunks=6, radius=args.end_radius)
    cases = {
        "baseline_22f_accepted": {
            "path": str(baseline_path),
            "fps": baseline_fps,
            "frame_count": len(baseline_frames),
            "center_metrics": baseline_center,
            "end_metrics": baseline_end,
            "end_trend": base._trend(baseline_end, base.METRIC_KEYS),
            "boundary_metrics": base._boundary_metrics(baseline_frames, chunks=6),
            "identity_and_frame_difference": _identity_and_frame_difference(
                baseline_representatives
            ),
        }
    }
    sheet_rows = [("A accepted 22f", baseline_representatives)]
    end_csv_rows = []
    for row in baseline_end:
        end_csv_rows.append({"case": "baseline_22f_accepted", **row})

    for record in summary["cases"]:
        case = str(record["case"])
        path = _case_video(record, args.video_root)
        frames, fps = base._read_frames(path)
        center, representatives = base._chunk_metrics(
            frames, chunks=6, radius=args.radius
        )
        end = _end_metrics(frames, chunks=6, radius=args.end_radius)
        cases[case] = {
            "path": str(path),
            "fps": fps,
            "frame_count": len(frames),
            "center_metrics": center,
            "end_metrics": end,
            "end_trend": base._trend(end, base.METRIC_KEYS),
            "boundary_metrics": base._boundary_metrics(frames, chunks=6),
            "identity_and_frame_difference": _identity_and_frame_difference(
                representatives
            ),
        }
        sheet_rows.append((case, representatives))
        for row in end:
            end_csv_rows.append({"case": case, **row})

    baseline = cases["baseline_22f_accepted"]
    comparisons = {
        case: _metric_comparison(baseline, value)
        for case, value in cases.items()
        if case != "baseline_22f_accepted"
    }
    csv_path = args.output_root / "group_end_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(end_csv_rows[0]))
        writer.writeheader()
        writer.writerows(end_csv_rows)
    contact_path = args.output_root / "contact_sheet.png"
    base._contact_sheet(sheet_rows, contact_path)
    result = {
        "format": "h3-continuum-issue13-r28-analysis-v1",
        "baseline_analysis": str(args.baseline_analysis),
        "baseline_case": baseline_key,
        "cases": cases,
        "comparisons": comparisons,
        "group_end_metrics_csv": str(csv_path),
        "contact_sheet": str(contact_path),
    }
    output = args.output_root / "analysis.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-analysis", type=Path, required=True)
    parser.add_argument(
        "--baseline-case", default="screen_6x5_turbo_fixed_prompt"
    )
    parser.add_argument(
        "--video-root", type=Path, default=Path(r"D:\output\video\comfy_video")
    )
    parser.add_argument("--radius", type=int, default=12)
    parser.add_argument("--end-radius", type=int, default=12)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
