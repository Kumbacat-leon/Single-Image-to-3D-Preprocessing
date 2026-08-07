from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_ROOT = PROJECT_ROOT / "dataset_original"

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "preprocessing_runtime"
STAGING_DIR = CORRECTED_ROOT / "_preprocessing_runtime_staging"
TEMP_DIR = CORRECTED_ROOT / "_preprocessing_runtime_temp"

RAW_CSV = "preprocessing_runtime_raw.csv"
BY_IMAGE_CSV = "preprocessing_runtime_by_image.csv"
SUMMARY_CSV = "preprocessing_runtime_summary.csv"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

CROP_PADDING = 20
PADDING_RATIO = 0.2
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
SHARPEN_SIGMA = 3.0
READY_CANVAS_SIZE = 512
READY_FOREGROUND_RATIO = 0.85
READY_GRAY_VALUE = 128

STAGE_FIELDS = (
    "background_removal_seconds",
    "cropping_seconds",
    "padding_seconds",
    "enhancement_seconds",
    "triposr_ready_seconds",
    "total_preprocessing_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the corrected external image-preprocessing pipeline: "
            "U2-Net background removal, alpha crop, transparent padding, "
            "CLAHE, sharpening, and TripoSR-ready conversion."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the 15 source images without running the benchmark.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the preprocessing benchmark.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated preprocessing runs per image. Default: 3.",
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
    for path in (
        OUTPUT_DIR,
        STAGING_DIR,
        TEMP_DIR,
    ):
        if path.exists():
            raise FileExistsError(
                "A previous preprocessing-runtime output exists. "
                f"Inspect or remove it before rerunning: {path}"
            )


def preflight(repeats: int) -> dict[str, object]:
    if repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )

    inputs = [
        find_original(object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    validate_destination_is_new()

    return {
        "inputs": inputs,
        "repeats": repeats,
        "expected_runs": len(inputs) * repeats,
    }


def stage_background_removal(
    input_path: Path,
    output_path: Path,
    session: object,
) -> None:
    input_data = input_path.read_bytes()

    output_data = remove(
        input_data,
        session=session,
    )

    output_path.write_bytes(
        output_data
    )


def stage_crop(
    input_path: Path,
    output_path: Path,
) -> None:
    image = Image.open(
        input_path
    ).convert("RGBA")

    alpha = np.asarray(image)[:, :, 3]
    coordinates = np.where(alpha > 0)

    if len(coordinates[0]) == 0:
        raise ValueError(
            f"No object was detected in: {input_path}"
        )

    y_min = int(coordinates[0].min())
    y_max = int(coordinates[0].max())
    x_min = int(coordinates[1].min())
    x_max = int(coordinates[1].max())

    x_min = max(
        0,
        x_min - CROP_PADDING,
    )
    y_min = max(
        0,
        y_min - CROP_PADDING,
    )

    # Preserve the original project script's crop-bound calculation.
    x_max = min(
        image.width,
        x_max + CROP_PADDING,
    )
    y_max = min(
        image.height,
        y_max + CROP_PADDING,
    )

    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max,
            y_max,
        )
    )

    cropped.save(
        output_path
    )


def stage_padding(
    input_path: Path,
    output_path: Path,
) -> None:
    image = Image.open(
        input_path
    ).convert("RGBA")

    width, height = image.size

    pad_width = int(
        width * PADDING_RATIO
    )
    pad_height = int(
        height * PADDING_RATIO
    )

    padded = Image.new(
        "RGBA",
        (
            width + pad_width * 2,
            height + pad_height * 2,
        ),
        (0, 0, 0, 0),
    )

    padded.paste(
        image,
        (
            pad_width,
            pad_height,
        ),
        image,
    )

    padded.save(
        output_path
    )


def stage_enhancement(
    input_path: Path,
    output_path: Path,
) -> None:
    # Match the original project script: read as BGR colour and discard alpha.
    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read: {input_path}"
        )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )

    lightness, channel_a, channel_b = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    enhanced_lightness = clahe.apply(
        lightness
    )

    enhanced_lab = cv2.merge(
        [
            enhanced_lightness,
            channel_a,
            channel_b,
        ]
    )

    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR,
    )

    blurred = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        SHARPEN_SIGMA,
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        blurred,
        -0.5,
        0,
    )

    if not cv2.imwrite(
        str(output_path),
        sharpened,
    ):
        raise RuntimeError(
            f"OpenCV could not save: {output_path}"
        )


def keep_largest_component(
    mask: np.ndarray,
) -> np.ndarray:
    mask_u8 = (
        mask.astype(np.uint8) * 255
    )

    count, labels, statistics_array, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    if count <= 1:
        return mask

    areas = statistics_array[
        1:,
        cv2.CC_STAT_AREA,
    ]

    largest_label = (
        int(np.argmax(areas)) + 1
    )

    return labels == largest_label


def clean_mask(
    mask: np.ndarray,
) -> np.ndarray:
    mask_u8 = (
        mask.astype(np.uint8) * 255
    )

    kernel_size = max(
        3,
        round(min(mask.shape) * 0.008),
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

    mask_u8 = cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_OPEN,
        kernel,
    )

    return keep_largest_component(
        mask_u8 > 0
    )


def estimate_foreground_mask(
    image: np.ndarray,
) -> np.ndarray:
    height, width = image.shape[:2]

    patch_size = max(
        2,
        round(min(height, width) * 0.04),
    )

    corner_pixels = np.concatenate(
        [
            image[
                :patch_size,
                :patch_size,
            ].reshape(-1, 3),
            image[
                :patch_size,
                -patch_size:,
            ].reshape(-1, 3),
            image[
                -patch_size:,
                :patch_size,
            ].reshape(-1, 3),
            image[
                -patch_size:,
                -patch_size:,
            ].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)

    background_colour = np.median(
        corner_pixels,
        axis=0,
    )

    distance = np.linalg.norm(
        image.astype(np.float32)
        - background_colour,
        axis=2,
    )

    return clean_mask(
        distance > 12.0
    )


def stage_triposr_ready(
    input_path: Path,
    output_path: Path,
) -> float:
    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read: {input_path}"
        )

    mask = estimate_foreground_mask(
        image
    )

    y_coordinates, x_coordinates = np.where(
        mask
    )

    if x_coordinates.size == 0:
        raise ValueError(
            f"No foreground was detected in: {input_path}"
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

    foreground = image[
        y_min:y_max + 1,
        x_min:x_max + 1,
    ]

    foreground_mask = (
        mask[
            y_min:y_max + 1,
            x_min:x_max + 1,
        ].astype(np.uint8)
        * 255
    )

    source_height, source_width = foreground.shape[:2]

    target_maximum = round(
        READY_CANVAS_SIZE
        * READY_FOREGROUND_RATIO
    )

    resize_scale = (
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
            * resize_scale
        ),
    )

    target_height = max(
        1,
        round(
            source_height
            * resize_scale
        ),
    )

    resized_foreground = cv2.resize(
        foreground,
        (
            target_width,
            target_height,
        ),
        interpolation=(
            cv2.INTER_AREA
            if resize_scale < 1
            else cv2.INTER_CUBIC
        ),
    )

    resized_mask = cv2.resize(
        foreground_mask,
        (
            target_width,
            target_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    resized_mask = cv2.GaussianBlur(
        resized_mask,
        (0, 0),
        sigmaX=0.7,
        sigmaY=0.7,
    )

    alpha = (
        resized_mask.astype(np.float32)
        / 255.0
    )[:, :, None]

    canvas = np.full(
        (
            READY_CANVAS_SIZE,
            READY_CANVAS_SIZE,
            3,
        ),
        READY_GRAY_VALUE,
        dtype=np.uint8,
    )

    x_offset = (
        READY_CANVAS_SIZE
        - target_width
    ) // 2

    y_offset = (
        READY_CANVAS_SIZE
        - target_height
    ) // 2

    destination = canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ].astype(np.float32)

    blended = (
        resized_foreground.astype(np.float32)
        * alpha
        + destination
        * (1.0 - alpha)
    )

    canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    if not cv2.imwrite(
        str(output_path),
        canvas,
    ):
        raise RuntimeError(
            f"OpenCV could not save: {output_path}"
        )

    occupancy = (
        np.count_nonzero(
            resized_mask
        )
        / (
            READY_CANVAS_SIZE
            * READY_CANVAS_SIZE
        )
    )

    return float(
        occupancy
    )


def run_one(
    session: object,
    object_name: str,
    view_name: str,
    repeat_number: int,
) -> dict:
    pair_id = (
        f"{object_name}_{view_name}"
    )

    input_path = find_original(
        object_name,
        view_name,
    )

    run_directory = (
        TEMP_DIR
        / pair_id
        / f"repeat_{repeat_number}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    no_background_path = (
        run_directory / "01_nobg.png"
    )

    cropped_path = (
        run_directory / "02_crop.png"
    )

    padded_path = (
        run_directory / "03_pad.png"
    )

    enhanced_path = (
        run_directory / "04_enhanced.png"
    )

    ready_path = (
        run_directory / "05_triposr_ready.png"
    )

    stage_times: dict[str, float] = {}

    total_start = time.perf_counter()

    start = time.perf_counter()
    stage_background_removal(
        input_path,
        no_background_path,
        session,
    )
    stage_times[
        "background_removal_seconds"
    ] = time.perf_counter() - start

    start = time.perf_counter()
    stage_crop(
        no_background_path,
        cropped_path,
    )
    stage_times[
        "cropping_seconds"
    ] = time.perf_counter() - start

    start = time.perf_counter()
    stage_padding(
        cropped_path,
        padded_path,
    )
    stage_times[
        "padding_seconds"
    ] = time.perf_counter() - start

    start = time.perf_counter()
    stage_enhancement(
        padded_path,
        enhanced_path,
    )
    stage_times[
        "enhancement_seconds"
    ] = time.perf_counter() - start

    start = time.perf_counter()
    occupancy = stage_triposr_ready(
        enhanced_path,
        ready_path,
    )
    stage_times[
        "triposr_ready_seconds"
    ] = time.perf_counter() - start

    stage_times[
        "total_preprocessing_seconds"
    ] = (
        time.perf_counter()
        - total_start
    )

    verification_path = ""

    if repeat_number == 1:
        verification_directory = (
            STAGING_DIR
            / "verification_inputs"
            / object_name
        )

        verification_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination = (
            verification_directory
            / (
                f"{object_name.upper()} "
                f"{view_name.upper()}"
                "_triposr_ready.png"
            )
        )

        shutil.copy2(
            ready_path,
            destination,
        )

        verification_path = str(
            destination
        )

    row = {
        "object": object_name,
        "view": view_name,
        "pair_id": pair_id,
        "repeat": repeat_number,
        "input_path": str(input_path),
        **stage_times,
        "foreground_occupancy_ratio": occupancy,
        "verification_output": verification_path,
    }

    shutil.rmtree(
        run_directory,
        ignore_errors=True,
    )

    return row


def create_by_image_rows(
    raw_rows: list[dict],
) -> list[dict]:
    grouped: dict[
        tuple[str, str],
        list[dict],
    ] = defaultdict(list)

    for row in raw_rows:
        grouped[
            (
                str(row["object"]),
                str(row["view"]),
            )
        ].append(row)

    output_rows: list[dict] = []

    for object_name in OBJECTS:
        for view_name in VIEWS:
            selected = grouped[
                (
                    object_name,
                    view_name,
                )
            ]

            output_row: dict[str, object] = {
                "object": object_name,
                "view": view_name,
                "pair_id": (
                    f"{object_name}_{view_name}"
                ),
                "repeat_count": len(selected),
            }

            for field in STAGE_FIELDS:
                values = [
                    float(row[field])
                    for row in selected
                ]

                output_row[
                    f"mean_{field}"
                ] = statistics.mean(
                    values
                )

                output_row[
                    f"median_{field}"
                ] = statistics.median(
                    values
                )

                output_row[
                    f"stdev_{field}"
                ] = (
                    statistics.stdev(
                        values
                    )
                    if len(values) >= 2
                    else 0.0
                )

            output_row[
                "mean_foreground_occupancy_ratio"
            ] = statistics.mean(
                float(
                    row[
                        "foreground_occupancy_ratio"
                    ]
                )
                for row in selected
            )

            output_rows.append(
                output_row
            )

    return output_rows


def summarize_group(
    rows: list[dict],
    group_name: str,
) -> list[dict]:
    output_rows: list[dict] = []

    for field in STAGE_FIELDS:
        values = [
            float(row[field])
            for row in rows
        ]

        output_rows.append(
            {
                "group": group_name,
                "stage": field,
                "run_count": len(values),
                "mean_seconds": statistics.mean(
                    values
                ),
                "median_seconds": statistics.median(
                    values
                ),
                "stdev_seconds": (
                    statistics.stdev(
                        values
                    )
                    if len(values) >= 2
                    else 0.0
                ),
                "minimum_seconds": min(
                    values
                ),
                "maximum_seconds": max(
                    values
                ),
            }
        )

    return output_rows


def create_summary_rows(
    raw_rows: list[dict],
) -> list[dict]:
    rows = summarize_group(
        raw_rows,
        "overall",
    )

    for object_name in OBJECTS:
        rows.extend(
            summarize_group(
                [
                    row
                    for row in raw_rows
                    if row["object"] == object_name
                ],
                object_name,
            )
        )

    return rows


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
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def publish() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(
        OUTPUT_DIR
    )


def run_benchmark(
    summary: dict[str, object],
) -> None:
    CORRECTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    STAGING_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    TEMP_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    raw_rows: list[dict] = []

    try:
        print(
            "\nInitializing U2-Net session ..."
        )

        initialization_start = (
            time.perf_counter()
        )

        session = new_session(
            "u2net"
        )

        session_initialization_seconds = (
            time.perf_counter()
            - initialization_start
        )

        print(
            "U2-Net session initialized in "
            f"{session_initialization_seconds:.3f}s"
        )

        repeats = int(
            summary["repeats"]
        )

        for object_name in OBJECTS:
            print(f"\n[{object_name}]")

            for view_name in VIEWS:
                for repeat_number in range(
                    1,
                    repeats + 1,
                ):
                    print(
                        f"Running {object_name}_{view_name} "
                        f"repeat {repeat_number}/{repeats} ..."
                    )

                    row = run_one(
                        session,
                        object_name,
                        view_name,
                        repeat_number,
                    )

                    raw_rows.append(
                        row
                    )

                    print(
                        "OK  total="
                        f"{row['total_preprocessing_seconds']:.3f}s | "
                        "U2-Net="
                        f"{row['background_removal_seconds']:.3f}s | "
                        "enhance="
                        f"{row['enhancement_seconds']:.3f}s | "
                        "ready="
                        f"{row['triposr_ready_seconds']:.3f}s"
                    )

        by_image_rows = create_by_image_rows(
            raw_rows
        )

        summary_rows = create_summary_rows(
            raw_rows
        )

        write_csv(
            STAGING_DIR / RAW_CSV,
            raw_rows,
        )

        write_csv(
            STAGING_DIR / BY_IMAGE_CSV,
            by_image_rows,
        )

        write_csv(
            STAGING_DIR / SUMMARY_CSV,
            summary_rows,
        )

        (
            STAGING_DIR
            / "session_initialization.txt"
        ).write_text(
            (
                "U2-Net session initialization seconds: "
                f"{session_initialization_seconds:.9f}\n"
                "The one-time session initialization is reported "
                "separately and is not included in each image's "
                "preprocessing runtime.\n"
            ),
            encoding="utf-8",
        )

        expected_runs = int(
            summary["expected_runs"]
        )

        if len(raw_rows) != expected_runs:
            raise RuntimeError(
                f"Only {len(raw_rows)}/{expected_runs} "
                "preprocessing runs completed."
            )

        publish()

    except Exception:
        if TEMP_DIR.exists():
            shutil.rmtree(
                TEMP_DIR,
                ignore_errors=True,
            )

        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )

        raise

    if TEMP_DIR.exists():
        shutil.rmtree(
            TEMP_DIR,
            ignore_errors=True,
        )

    overall = {
        row["stage"]: row
        for row in summary_rows
        if row["group"] == "overall"
    }

    total_mean = float(
        overall[
            "total_preprocessing_seconds"
        ]["mean_seconds"]
    )

    print("\n" + "=" * 90)
    print("CORRECTED PREPROCESSING-RUNTIME RESULTS")
    print("=" * 90)
    print(
        f"Completed runs: "
        f"{len(raw_rows)}/"
        f"{summary['expected_runs']}"
    )
    print(
        "U2-Net session initialization: "
        f"{session_initialization_seconds:.3f}s "
        "(one-time, reported separately)"
    )
    print(
        "Mean external preprocessing time per image: "
        f"{total_mean:.4f}s"
    )

    for stage in (
        "background_removal_seconds",
        "cropping_seconds",
        "padding_seconds",
        "enhancement_seconds",
        "triposr_ready_seconds",
    ):
        mean_value = float(
            overall[stage][
                "mean_seconds"
            ]
        )

        contribution = (
            mean_value
            / total_mean
            * 100.0
        )

        print(
            f"{stage}: "
            f"{mean_value:.4f}s "
            f"({contribution:.1f}% of total)"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print(
        "CORRECTED PREPROCESSING-RUNTIME BENCHMARK PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight(
        args.repeats
    )

    print("=" * 90)
    print("Corrected External Preprocessing-Runtime Benchmark")
    print("=" * 90)
    print(
        f"Original images: "
        f"{len(summary['inputs'])}/15"
    )
    print(
        f"Repeats per image: "
        f"{summary['repeats']}"
    )
    print(
        f"Expected runs: "
        f"{summary['expected_runs']}"
    )
    print(f"Output: {OUTPUT_DIR}")

    if args.check_only:
        print(
            "\nCHECK PASSED: no preprocessing benchmark "
            "runs were executed."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_benchmark(
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
