from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = PROJECT_ROOT / "dataset_original"

OUTPUT_ROOT = PROJECT_ROOT / "pipeline_v2_ratio_inputs"
RESULTS_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "pipeline_v2_ratio_screening"
)

STAGING_OUTPUT_ROOT = PROJECT_ROOT / "_pipeline_v2_ratio_inputs_staging"
STAGING_RESULTS_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "_pipeline_v2_ratio_screening_staging"
)

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

RATIOS = (0.70, 0.80, 0.90)

CANVAS_SIZE = 512
BACKGROUND_VALUE = 128
ALPHA_THRESHOLD = 16
MIN_COMPONENT_FRACTION = 0.0005
MIN_COMPONENT_PIXELS = 64

MANIFEST_CSV = "ratio_candidate_manifest.csv"
SESSION_TXT = "u2net_session_initialization.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pipeline V2 adaptive-scaling candidates for ratio "
            "screening. Background removal is fixed; no CLAHE or sharpening "
            "is applied in this phase."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the 15 source images without generating candidates.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate 45 Pipeline V2 ratio-candidate input images.",
    )

    return parser.parse_args()


def normalize_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        value.lower(),
    ).strip("_")


def find_original(
    object_name: str,
    view_name: str,
) -> Path:
    folder = ORIGINAL_ROOT / object_name

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Object folder was not found: {folder}"
        )

    expected = normalize_name(
        f"{object_name} {view_name}"
    )

    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed
        and normalize_name(path.stem) == expected
    ]

    if not matches:
        available = ", ".join(
            sorted(
                path.name
                for path in folder.iterdir()
                if path.is_file()
            )
        )

        raise FileNotFoundError(
            f"Original image was not found for "
            f"{object_name}_{view_name}. "
            f"Available files: {available}"
        )

    return matches[0]


def validate_destination_is_new() -> None:
    existing = [
        path
        for path in (
            OUTPUT_ROOT,
            RESULTS_ROOT,
            STAGING_OUTPUT_ROOT,
            STAGING_RESULTS_ROOT,
        )
        if path.exists()
    ]

    if existing:
        details = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "Pipeline V2 ratio-screening output already exists and "
            f"will not be overwritten:\n{details}"
        )


def preflight() -> dict[str, object]:
    originals = [
        find_original(object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    validate_destination_is_new()

    return {
        "originals": originals,
        "expected_candidates": (
            len(OBJECTS)
            * len(VIEWS)
            * len(RATIOS)
        ),
        "expected_previews": (
            len(OBJECTS)
            * len(VIEWS)
        ),
    }


def remove_background_rgba(
    input_path: Path,
    session: object,
) -> np.ndarray:
    output_bytes = remove(
        input_path.read_bytes(),
        session=session,
    )

    rgba = np.asarray(
        Image.open(
            BytesIO(output_bytes)
        ).convert("RGBA")
    )

    return rgba


def clean_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    mask = alpha > ALPHA_THRESHOLD

    height, width = mask.shape
    mask_u8 = (
        mask.astype(np.uint8)
        * 255
    )

    kernel_size = max(
        3,
        round(min(height, width) * 0.004),
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            kernel_size,
            kernel_size,
        ),
    )

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        kernel,
    )

    component_count, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    minimum_area = max(
        MIN_COMPONENT_PIXELS,
        round(
            height
            * width
            * MIN_COMPONENT_FRACTION
        ),
    )

    cleaned = np.zeros_like(
        mask_u8,
        dtype=np.uint8,
    )

    for label in range(
        1,
        component_count,
    ):
        area = int(
            stats[
                label,
                cv2.CC_STAT_AREA,
            ]
        )

        if area >= minimum_area:
            cleaned[
                labels == label
            ] = 255

    if not np.any(cleaned):
        raise ValueError(
            "No foreground remained after alpha-mask cleanup."
        )

    return cleaned


def create_candidate(
    rgba: np.ndarray,
    cleaned_alpha: np.ndarray,
    target_ratio: float,
) -> tuple[np.ndarray, dict[str, float]]:
    rgb = rgba[:, :, :3]

    y_coordinates, x_coordinates = np.where(
        cleaned_alpha > 0
    )

    x_min = int(
        x_coordinates.min()
    )
    x_max = int(
        x_coordinates.max()
    )
    y_min = int(
        y_coordinates.min()
    )
    y_max = int(
        y_coordinates.max()
    )

    foreground_rgb = rgb[
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    foreground_alpha = cleaned_alpha[
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    source_height, source_width = foreground_rgb.shape[:2]

    target_maximum = round(
        CANVAS_SIZE
        * target_ratio
    )

    scale = (
        target_maximum
        / max(
            source_width,
            source_height,
        )
    )

    target_width = max(
        1,
        round(
            source_width
            * scale
        ),
    )

    target_height = max(
        1,
        round(
            source_height
            * scale
        ),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )

    resized_rgb = cv2.resize(
        foreground_rgb,
        (
            target_width,
            target_height,
        ),
        interpolation=interpolation,
    )

    resized_alpha = cv2.resize(
        foreground_alpha,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    resized_alpha = cv2.GaussianBlur(
        resized_alpha,
        (0, 0),
        sigmaX=0.6,
        sigmaY=0.6,
    )

    alpha_float = (
        resized_alpha.astype(np.float32)
        / 255.0
    )[:, :, None]

    canvas = np.full(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
            3,
        ),
        BACKGROUND_VALUE,
        dtype=np.uint8,
    )

    x_offset = (
        CANVAS_SIZE
        - target_width
    ) // 2

    y_offset = (
        CANVAS_SIZE
        - target_height
    ) // 2

    destination = canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ].astype(np.float32)

    blended = (
        resized_rgb.astype(np.float32)
        * alpha_float
        + destination
        * (1.0 - alpha_float)
    )

    canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    final_mask = np.zeros(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
        ),
        dtype=np.uint8,
    )

    final_mask[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = resized_alpha

    binary_mask = final_mask > ALPHA_THRESHOLD
    y_final, x_final = np.where(binary_mask)

    left_margin = int(
        x_final.min()
    )
    right_margin = int(
        CANVAS_SIZE
        - 1
        - x_final.max()
    )
    top_margin = int(
        y_final.min()
    )
    bottom_margin = int(
        CANVAS_SIZE
        - 1
        - y_final.max()
    )

    object_center_x = (
        x_final.min()
        + x_final.max()
    ) / 2.0

    object_center_y = (
        y_final.min()
        + y_final.max()
    ) / 2.0

    image_center = (
        CANVAS_SIZE - 1
    ) / 2.0

    center_offset_ratio = (
        math.hypot(
            object_center_x
            - image_center,
            object_center_y
            - image_center,
        )
        / math.hypot(
            CANVAS_SIZE,
            CANVAS_SIZE,
        )
    )

    metrics = {
        "requested_max_side_ratio": target_ratio,
        "actual_foreground_occupancy_ratio": float(
            np.mean(binary_mask)
        ),
        "actual_bbox_width_ratio": float(
            (
                x_final.max()
                - x_final.min()
                + 1
            )
            / CANVAS_SIZE
        ),
        "actual_bbox_height_ratio": float(
            (
                y_final.max()
                - y_final.min()
                + 1
            )
            / CANVAS_SIZE
        ),
        "center_offset_ratio": center_offset_ratio,
        "minimum_margin_ratio": float(
            min(
                left_margin,
                right_margin,
                top_margin,
                bottom_margin,
            )
            / CANVAS_SIZE
        ),
        "margin_imbalance_ratio": float(
            (
                max(
                    left_margin,
                    right_margin,
                    top_margin,
                    bottom_margin,
                )
                - min(
                    left_margin,
                    right_margin,
                    top_margin,
                    bottom_margin,
                )
            )
            / CANVAS_SIZE
        ),
    }

    return canvas, metrics


def add_label(
    image: np.ndarray,
    label: str,
) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image,
        52,
        0,
        0,
        0,
        borderType=cv2.BORDER_CONSTANT,
        value=(32, 32, 32),
    )

    cv2.putText(
        canvas,
        label,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return canvas


def make_preview(
    original_path: Path,
    candidates: list[
        tuple[float, np.ndarray]
    ],
) -> np.ndarray:
    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_COLOR,
    )

    if original is None:
        raise ValueError(
            f"OpenCV could not read: {original_path}"
        )

    original = cv2.resize(
        original,
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    panels = [
        add_label(
            original,
            "Original",
        )
    ]

    for ratio, image in candidates:
        panels.append(
            add_label(
                image,
                f"V2 ratio {int(ratio * 100)}%",
            )
        )

    return np.hstack(
        panels
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError(
            f"No rows were available for: {path}"
        )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def publish() -> None:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_ROOT}"
        )

    if RESULTS_ROOT.exists():
        raise FileExistsError(
            f"Results appeared during execution: {RESULTS_ROOT}"
        )

    STAGING_OUTPUT_ROOT.rename(
        OUTPUT_ROOT
    )

    STAGING_RESULTS_ROOT.rename(
        RESULTS_ROOT
    )


def run_generation(
    summary: dict[str, object],
) -> None:
    STAGING_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    STAGING_RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    preview_root = (
        STAGING_RESULTS_ROOT
        / "previews"
    )

    preview_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nInitializing U2-Net session ..."
    )

    session_start = time.perf_counter()
    session = new_session(
        "u2net"
    )
    session_seconds = (
        time.perf_counter()
        - session_start
    )

    print(
        f"U2-Net session initialized in "
        f"{session_seconds:.3f}s"
    )

    manifest_rows: list[dict] = []

    try:
        for object_name in OBJECTS:
            print(f"\n[{object_name}]")

            for view_name in VIEWS:
                pair_id = (
                    f"{object_name}_{view_name}"
                )

                original_path = find_original(
                    object_name,
                    view_name,
                )

                rgba = remove_background_rgba(
                    original_path,
                    session,
                )

                cleaned_alpha = clean_alpha_mask(
                    rgba[:, :, 3]
                )

                preview_candidates: list[
                    tuple[float, np.ndarray]
                ] = []

                for ratio in RATIOS:
                    candidate, metrics = (
                        create_candidate(
                            rgba,
                            cleaned_alpha,
                            ratio,
                        )
                    )

                    ratio_name = (
                        f"ratio_{int(ratio * 100)}"
                    )

                    output_directory = (
                        STAGING_OUTPUT_ROOT
                        / ratio_name
                        / pair_id
                        / "0"
                    )

                    output_directory.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    output_path = (
                        output_directory
                        / "input.png"
                    )

                    if not cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(
                            candidate,
                            cv2.COLOR_RGB2BGR,
                        ),
                    ):
                        raise RuntimeError(
                            f"Could not save: {output_path}"
                        )

                    manifest_rows.append(
                        {
                            "ratio_name": ratio_name,
                            "ratio": ratio,
                            "object": object_name,
                            "view": view_name,
                            "pair_id": pair_id,
                            "original_path": str(
                                original_path
                            ),
                            "output_path": str(
                                output_path
                            ),
                            **metrics,
                        }
                    )

                    preview_candidates.append(
                        (
                            ratio,
                            cv2.cvtColor(
                                candidate,
                                cv2.COLOR_RGB2BGR,
                            ),
                        )
                    )

                    print(
                        f"OK  {pair_id:<16} "
                        f"ratio={ratio:.2f} | "
                        f"occupancy="
                        f"{metrics['actual_foreground_occupancy_ratio']:.3f} | "
                        f"center="
                        f"{metrics['center_offset_ratio']:.5f}"
                    )

                preview = make_preview(
                    original_path,
                    preview_candidates,
                )

                preview_path = (
                    preview_root
                    / f"{pair_id}.png"
                )

                if not cv2.imwrite(
                    str(preview_path),
                    preview,
                ):
                    raise RuntimeError(
                        f"Could not save preview: {preview_path}"
                    )

        write_csv(
            STAGING_RESULTS_ROOT
            / MANIFEST_CSV,
            manifest_rows,
        )

        (
            STAGING_RESULTS_ROOT
            / SESSION_TXT
        ).write_text(
            (
                "U2-Net session initialization seconds: "
                f"{session_seconds:.9f}\n"
                "Pipeline V2 ratio screening uses foreground-only "
                "background removal, alpha-mask cleanup, adaptive scaling, "
                "and a uniform 50% grey background. No CLAHE or sharpening "
                "is applied in this phase.\n"
            ),
            encoding="utf-8",
        )

        expected_candidates = int(
            summary[
                "expected_candidates"
            ]
        )

        if len(
            manifest_rows
        ) != expected_candidates:
            raise RuntimeError(
                f"Generated {len(manifest_rows)}/"
                f"{expected_candidates} candidates."
            )

        publish()

    except Exception:
        if STAGING_OUTPUT_ROOT.exists():
            shutil.rmtree(
                STAGING_OUTPUT_ROOT,
                ignore_errors=True,
            )

        if STAGING_RESULTS_ROOT.exists():
            shutil.rmtree(
                STAGING_RESULTS_ROOT,
                ignore_errors=True,
            )

        raise

    print("\n" + "=" * 96)
    print("PIPELINE V2 RATIO-CANDIDATE GENERATION")
    print("=" * 96)
    print(
        f"Generated candidates: "
        f"{len(manifest_rows)}/"
        f"{summary['expected_candidates']}"
    )
    print(
        f"Generated previews: "
        f"{summary['expected_previews']}/"
        f"{summary['expected_previews']}"
    )
    print(
        f"Candidate inputs: {OUTPUT_ROOT}"
    )
    print(
        f"Screening records: {RESULTS_ROOT}"
    )
    print(
        "PIPELINE V2 RATIO-CANDIDATE GENERATION PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print("=" * 96)
    print("Pipeline V2 Adaptive Ratio Screening - Input Generation")
    print("=" * 96)
    print(
        f"Original images: "
        f"{len(summary['originals'])}/15"
    )
    print(
        f"Ratios: "
        f"{', '.join(str(int(ratio * 100)) + '%' for ratio in RATIOS)}"
    )
    print(
        f"Planned candidates: "
        f"{summary['expected_candidates']}"
    )
    print(
        f"Planned previews: "
        f"{summary['expected_previews']}"
    )
    print(
        "Enhancement: disabled in this phase"
    )
    print(
        f"Output: {OUTPUT_ROOT}"
    )
    print(
        f"Results: {RESULTS_ROOT}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no Pipeline V2 candidates were generated."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_generation(
        summary
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        raise
