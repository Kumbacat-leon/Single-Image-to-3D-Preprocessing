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
from scipy.ndimage import distance_transform_edt


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORIGINAL_ROOT = PROJECT_ROOT / "dataset_original"

CURRENT_RATIO_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_ratio_inputs"
    / "ratio_80"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_edge_clean_inputs"
)

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

RESULTS_ROOT = (
    CORRECTED_ROOT
    / "pipeline_v2_edge_clean_screening"
)

STAGING_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "_pipeline_v2_edge_clean_inputs_staging"
)

STAGING_RESULTS_ROOT = (
    CORRECTED_ROOT
    / "_pipeline_v2_edge_clean_screening_staging"
)

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

TARGET_RATIO = 0.80
CANVAS_SIZE = 512
BACKGROUND_VALUE = 128
ALPHA_THRESHOLD = 16

MIN_COMPONENT_FRACTION = 0.0005
MIN_COMPONENT_PIXELS = 64

MANIFEST_CSV = "edge_clean_candidate_manifest.csv"
SESSION_TXT = "u2net_session_initialization.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pipeline V2 ratio-80 inputs with foreground-colour "
            "propagation to remove dark alpha-edge halos."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source and current ratio-80 inputs only.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate 15 edge-cleaned ratio-80 inputs and previews.",
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
        raise FileNotFoundError(
            f"Original image was not found for "
            f"{object_name}_{view_name}."
        )

    return matches[0]


def current_ratio_path(
    object_name: str,
    view_name: str,
) -> Path:
    return (
        CURRENT_RATIO_ROOT
        / f"{object_name}_{view_name}"
        / "0"
        / "input.png"
    )


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
            "Pipeline V2 edge-clean output already exists and will not "
            f"be overwritten:\n{details}"
        )


def preflight() -> dict[str, object]:
    originals = []
    current_inputs = []

    for object_name in OBJECTS:
        for view_name in VIEWS:
            original_path = find_original(
                object_name,
                view_name,
            )
            current_path = current_ratio_path(
                object_name,
                view_name,
            )

            if not current_path.is_file():
                raise FileNotFoundError(
                    f"Current ratio-80 input was not found: {current_path}"
                )

            originals.append(original_path)
            current_inputs.append(current_path)

    validate_destination_is_new()

    return {
        "originals": originals,
        "current_inputs": current_inputs,
        "expected_images": (
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

    return np.asarray(
        Image.open(
            BytesIO(output_bytes)
        ).convert("RGBA")
    )


def clean_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    mask_u8 = (
        (alpha > ALPHA_THRESHOLD)
        .astype(np.uint8)
        * 255
    )

    height, width = mask_u8.shape

    kernel_size = max(
        3,
        round(
            min(height, width)
            * 0.004
        ),
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
            "No foreground remained after alpha cleanup."
        )

    return cleaned


def propagate_foreground_colours(
    foreground_rgb: np.ndarray,
    foreground_mask: np.ndarray,
) -> np.ndarray:
    inside = foreground_mask > 0

    if not np.any(inside):
        raise ValueError(
            "Cannot propagate colours from an empty foreground mask."
        )

    # distance_transform_edt finds the nearest zero pixel. By applying it to
    # the inverse mask, every outside pixel receives the nearest inside index.
    _, nearest_indices = distance_transform_edt(
        ~inside,
        return_indices=True,
    )

    nearest_y = nearest_indices[0]
    nearest_x = nearest_indices[1]

    filled = foreground_rgb.copy()
    outside = ~inside

    filled[outside] = foreground_rgb[
        nearest_y[outside],
        nearest_x[outside],
    ]

    return filled


def create_edge_clean_candidate(
    rgba: np.ndarray,
    cleaned_alpha: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, float],
]:
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

    edge_clean_rgb = propagate_foreground_colours(
        foreground_rgb,
        foreground_alpha,
    )

    source_height, source_width = (
        edge_clean_rgb.shape[:2]
    )

    target_maximum = round(
        CANVAS_SIZE
        * TARGET_RATIO
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
        edge_clean_rgb,
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
        resized_alpha.astype(
            np.float32
        )
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

    final_alpha = np.zeros(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
        ),
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
    ].astype(
        np.float32
    )

    blended = (
        resized_rgb.astype(
            np.float32
        )
        * alpha_float
        + destination
        * (
            1.0
            - alpha_float
        )
    )

    canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(
        np.uint8
    )

    final_alpha[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = resized_alpha

    binary_mask = (
        final_alpha
        > ALPHA_THRESHOLD
    )

    y_final, x_final = np.where(
        binary_mask
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
        CANVAS_SIZE
        - 1
    ) / 2.0

    metrics = {
        "requested_max_side_ratio": TARGET_RATIO,
        "foreground_occupancy_ratio": float(
            np.mean(
                binary_mask
            )
        ),
        "center_offset_ratio": float(
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
        ),
    }

    return (
        canvas,
        final_alpha,
        metrics,
    )


def create_edge_band(
    alpha: np.ndarray,
) -> np.ndarray:
    mask = (
        alpha
        > ALPHA_THRESHOLD
    ).astype(
        np.uint8
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    dilated = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    eroded = cv2.erode(
        mask,
        kernel,
        iterations=1,
    )

    return (
        dilated
        - eroded
    ) > 0


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
        0.85,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return canvas


def create_preview(
    current_bgr: np.ndarray,
    cleaned_bgr: np.ndarray,
) -> np.ndarray:
    return np.hstack(
        [
            add_label(
                current_bgr,
                "Current ratio 80%",
            ),
            add_label(
                cleaned_bgr,
                "Edge-cleaned ratio 80%",
            ),
        ]
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
            print(
                f"\n[{object_name}]"
            )

            for view_name in VIEWS:
                pair_id = (
                    f"{object_name}_{view_name}"
                )

                original_path = find_original(
                    object_name,
                    view_name,
                )

                current_path = current_ratio_path(
                    object_name,
                    view_name,
                )

                current_bgr = cv2.imread(
                    str(
                        current_path
                    ),
                    cv2.IMREAD_COLOR,
                )

                if current_bgr is None:
                    raise ValueError(
                        f"OpenCV could not read: {current_path}"
                    )

                rgba = remove_background_rgba(
                    original_path,
                    session,
                )

                cleaned_alpha = clean_alpha_mask(
                    rgba[:, :, 3]
                )

                (
                    cleaned_rgb,
                    final_alpha,
                    metrics,
                ) = create_edge_clean_candidate(
                    rgba,
                    cleaned_alpha,
                )

                cleaned_bgr = cv2.cvtColor(
                    cleaned_rgb,
                    cv2.COLOR_RGB2BGR,
                )

                output_directory = (
                    STAGING_OUTPUT_ROOT
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
                    str(
                        output_path
                    ),
                    cleaned_bgr,
                ):
                    raise RuntimeError(
                        f"Could not save: {output_path}"
                    )

                preview_path = (
                    preview_root
                    / f"{pair_id}.png"
                )

                preview = create_preview(
                    current_bgr,
                    cleaned_bgr,
                )

                if not cv2.imwrite(
                    str(
                        preview_path
                    ),
                    preview,
                ):
                    raise RuntimeError(
                        f"Could not save preview: {preview_path}"
                    )

                difference = np.abs(
                    current_bgr.astype(
                        np.float32
                    )
                    - cleaned_bgr.astype(
                        np.float32
                    )
                )

                edge_band = create_edge_band(
                    final_alpha
                )

                edge_difference = (
                    float(
                        np.mean(
                            difference[
                                edge_band
                            ]
                        )
                    )
                    if np.any(
                        edge_band
                    )
                    else math.nan
                )

                manifest_rows.append(
                    {
                        "object": object_name,
                        "view": view_name,
                        "pair_id": pair_id,
                        "original_path": str(
                            original_path
                        ),
                        "current_ratio_80_path": str(
                            current_path
                        ),
                        "edge_clean_output_path": str(
                            output_path
                        ),
                        "mean_absolute_pixel_difference": float(
                            np.mean(
                                difference
                            )
                        ),
                        "edge_band_mean_absolute_difference": (
                            edge_difference
                        ),
                        **metrics,
                    }
                )

                print(
                    f"OK  {pair_id:<16} "
                    f"occupancy="
                    f"{metrics['foreground_occupancy_ratio']:.3f} | "
                    f"center="
                    f"{metrics['center_offset_ratio']:.5f} | "
                    f"edge_diff="
                    f"{edge_difference:.3f}"
                )

        expected_images = int(
            summary[
                "expected_images"
            ]
        )

        if len(
            manifest_rows
        ) != expected_images:
            raise RuntimeError(
                f"Generated {len(manifest_rows)}/"
                f"{expected_images} edge-clean inputs."
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
                "Edge cleanup propagates nearest foreground RGB colours "
                "into transparent pixels before resizing and alpha "
                "compositing. This prevents dark transparent RGB values "
                "from creating a visible boundary halo.\n"
            ),
            encoding="utf-8",
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

    edge_differences = [
        float(
            row[
                "edge_band_mean_absolute_difference"
            ]
        )
        for row in manifest_rows
        if math.isfinite(
            float(
                row[
                    "edge_band_mean_absolute_difference"
                ]
            )
        )
    ]

    print(
        "\n"
        + "=" * 96
    )

    print(
        "PIPELINE V2 EDGE-CLEAN INPUT GENERATION"
    )

    print(
        "=" * 96
    )

    print(
        f"Generated inputs: "
        f"{len(manifest_rows)}/"
        f"{summary['expected_images']}"
    )

    print(
        f"Generated previews: "
        f"{len(manifest_rows)}/"
        f"{summary['expected_images']}"
    )

    print(
        f"Mean edge-band pixel change: "
        f"{np.mean(edge_differences):.3f}"
    )

    print(
        f"Inputs: {OUTPUT_ROOT}"
    )

    print(
        f"Screening records: {RESULTS_ROOT}"
    )

    print(
        "PIPELINE V2 EDGE-CLEAN INPUT GENERATION PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print(
        "=" * 96
    )

    print(
        "Pipeline V2 Ratio-80 Edge-Cleanup Screening"
    )

    print(
        "=" * 96
    )

    print(
        f"Original images: "
        f"{len(summary['originals'])}/15"
    )

    print(
        f"Current ratio-80 inputs: "
        f"{len(summary['current_inputs'])}/15"
    )

    print(
        f"Planned edge-clean inputs: "
        f"{summary['expected_images']}"
    )

    print(
        "Target ratio: 80%"
    )

    print(
        "Enhancement: disabled"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        f"Results: {RESULTS_ROOT}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no edge-clean inputs were generated."
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
