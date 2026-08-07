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

CURRENT_RATIO_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_ratio_inputs"
    / "ratio_80"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_enhancement_inputs"
)

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

RESULTS_ROOT = (
    CORRECTED_ROOT
    / "pipeline_v2_enhancement_screening"
)

STAGING_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "_pipeline_v2_enhancement_inputs_staging"
)

STAGING_RESULTS_ROOT = (
    CORRECTED_ROOT
    / "_pipeline_v2_enhancement_screening_staging"
)

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

VARIANTS = (
    "clahe_mild",
    "sharpen_mild",
    "clahe_sharpen_mild",
)

TARGET_RATIO = 0.80
CANVAS_SIZE = 512
BACKGROUND_VALUE = 128
ALPHA_THRESHOLD = 16

MIN_COMPONENT_FRACTION = 0.0005
MIN_COMPONENT_PIXELS = 64

CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = (8, 8)

SHARPEN_SIGMA = 1.0
SHARPEN_AMOUNT = 0.20

COMBINED_SHARPEN_AMOUNT = 0.15

MANIFEST_CSV = "foreground_enhancement_candidate_manifest.csv"
SESSION_TXT = "u2net_session_initialization.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mild foreground-only enhancement candidates for the "
            "locked Pipeline V2 ratio-80 input configuration."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate source images and current ratio-80 inputs only.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate 45 foreground-enhancement candidate inputs.",
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
            "Pipeline V2 enhancement-screening output already exists "
            f"and will not be overwritten:\n{details}"
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
        "expected_candidates": (
            len(OBJECTS)
            * len(VIEWS)
            * len(VARIANTS)
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
            "No foreground remained after alpha-mask cleanup."
        )

    return cleaned


def create_ratio_80_foreground(
    rgba: np.ndarray,
    cleaned_alpha: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
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

    source_height, source_width = foreground_rgb.shape[:2]

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

    rgb_canvas = np.full(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
            3,
        ),
        BACKGROUND_VALUE,
        dtype=np.uint8,
    )

    alpha_canvas = np.zeros(
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

    rgb_canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = resized_rgb

    alpha_canvas[
        y_offset:y_offset + target_height,
        x_offset:x_offset + target_width,
    ] = resized_alpha

    return (
        rgb_canvas,
        alpha_canvas,
    )


def apply_clahe_mild(
    rgb: np.ndarray,
) -> np.ndarray:
    lab = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2LAB,
    )

    l_channel, a_channel, b_channel = cv2.split(
        lab
    )

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    enhanced_l = clahe.apply(
        l_channel
    )

    enhanced_lab = cv2.merge(
        (
            enhanced_l,
            a_channel,
            b_channel,
        )
    )

    return cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2RGB,
    )


def apply_unsharp(
    rgb: np.ndarray,
    amount: float,
) -> np.ndarray:
    blurred = cv2.GaussianBlur(
        rgb,
        (0, 0),
        sigmaX=SHARPEN_SIGMA,
        sigmaY=SHARPEN_SIGMA,
    )

    sharpened = cv2.addWeighted(
        rgb,
        1.0 + amount,
        blurred,
        -amount,
        0.0,
    )

    return np.clip(
        sharpened,
        0,
        255,
    ).astype(
        np.uint8
    )


def apply_variant(
    rgb: np.ndarray,
    variant: str,
) -> np.ndarray:
    if variant == "clahe_mild":
        return apply_clahe_mild(
            rgb
        )

    if variant == "sharpen_mild":
        return apply_unsharp(
            rgb,
            SHARPEN_AMOUNT,
        )

    if variant == "clahe_sharpen_mild":
        clahe_rgb = apply_clahe_mild(
            rgb
        )

        return apply_unsharp(
            clahe_rgb,
            COMBINED_SHARPEN_AMOUNT,
        )

    raise ValueError(
        f"Unknown enhancement variant: {variant}"
    )


def composite_foreground_only(
    base_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    alpha_float = (
        alpha.astype(
            np.float32
        )
        / 255.0
    )[:, :, None]

    output = (
        enhanced_rgb.astype(
            np.float32
        )
        * alpha_float
        + base_rgb.astype(
            np.float32
        )
        * (
            1.0
            - alpha_float
        )
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(
        np.uint8
    )


def calculate_region_metrics(
    image_rgb: np.ndarray,
    alpha: np.ndarray,
) -> dict[str, float]:
    mask = (
        alpha
        > ALPHA_THRESHOLD
    )

    background_mask = ~mask

    gray = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2GRAY,
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    return {
        "foreground_brightness": float(
            np.mean(
                gray[
                    mask
                ]
            )
        ),
        "foreground_contrast": float(
            np.std(
                gray[
                    mask
                ]
            )
        ),
        "foreground_sharpness": float(
            np.var(
                laplacian[
                    mask
                ]
            )
        ),
        "foreground_edge_density": float(
            np.mean(
                edges[
                    mask
                ] > 0
            )
        ),
        "background_mean": float(
            np.mean(
                gray[
                    background_mask
                ]
            )
        ),
        "background_std": float(
            np.std(
                gray[
                    background_mask
                ]
            )
        ),
    }


def add_label(
    image_bgr: np.ndarray,
    label: str,
) -> np.ndarray:
    canvas = cv2.copyMakeBorder(
        image_bgr,
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
        (16, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.76,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return canvas


def create_preview(
    current_bgr: np.ndarray,
    variant_images_bgr: dict[str, np.ndarray],
) -> np.ndarray:
    panels = [
        add_label(
            current_bgr,
            "Current ratio 80%",
        ),
        add_label(
            variant_images_bgr[
                "clahe_mild"
            ],
            "Mild CLAHE",
        ),
        add_label(
            variant_images_bgr[
                "sharpen_mild"
            ],
            "Mild sharpening",
        ),
        add_label(
            variant_images_bgr[
                "clahe_sharpen_mild"
            ],
            "CLAHE + mild sharpening",
        ),
    ]

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
    reproduction_differences: list[float] = []

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

                current_rgb = cv2.cvtColor(
                    current_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                rgba = remove_background_rgba(
                    original_path,
                    session,
                )

                cleaned_alpha = clean_alpha_mask(
                    rgba[:, :, 3]
                )

                (
                    foreground_rgb_canvas,
                    alpha_canvas,
                ) = create_ratio_80_foreground(
                    rgba,
                    cleaned_alpha,
                )

                base_alpha_float = (
                    alpha_canvas.astype(
                        np.float32
                    )
                    / 255.0
                )[:, :, None]

                reproduced_base = (
                    foreground_rgb_canvas.astype(
                        np.float32
                    )
                    * base_alpha_float
                    + np.full_like(
                        foreground_rgb_canvas,
                        BACKGROUND_VALUE,
                        dtype=np.uint8,
                    ).astype(
                        np.float32
                    )
                    * (
                        1.0
                        - base_alpha_float
                    )
                )

                reproduced_base = np.clip(
                    reproduced_base,
                    0,
                    255,
                ).astype(
                    np.uint8
                )

                reproduction_difference = float(
                    np.mean(
                        np.abs(
                            reproduced_base.astype(
                                np.float32
                            )
                            - current_rgb.astype(
                                np.float32
                            )
                        )
                    )
                )

                reproduction_differences.append(
                    reproduction_difference
                )

                current_metrics = calculate_region_metrics(
                    current_rgb,
                    alpha_canvas,
                )

                variant_images_bgr: dict[
                    str,
                    np.ndarray,
                ] = {}

                for variant in VARIANTS:
                    enhanced_foreground = apply_variant(
                        foreground_rgb_canvas,
                        variant,
                    )

                    candidate_rgb = composite_foreground_only(
                        current_rgb,
                        enhanced_foreground,
                        alpha_canvas,
                    )

                    candidate_bgr = cv2.cvtColor(
                        candidate_rgb,
                        cv2.COLOR_RGB2BGR,
                    )

                    output_directory = (
                        STAGING_OUTPUT_ROOT
                        / variant
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
                        candidate_bgr,
                    ):
                        raise RuntimeError(
                            f"Could not save: {output_path}"
                        )

                    candidate_metrics = calculate_region_metrics(
                        candidate_rgb,
                        alpha_canvas,
                    )

                    background_mask = (
                        alpha_canvas
                        <= ALPHA_THRESHOLD
                    )

                    background_difference = float(
                        np.mean(
                            np.abs(
                                candidate_rgb[
                                    background_mask
                                ].astype(
                                    np.float32
                                )
                                - current_rgb[
                                    background_mask
                                ].astype(
                                    np.float32
                                )
                            )
                        )
                    )

                    manifest_rows.append(
                        {
                            "variant": variant,
                            "object": object_name,
                            "view": view_name,
                            "pair_id": pair_id,
                            "original_path": str(
                                original_path
                            ),
                            "current_ratio_80_path": str(
                                current_path
                            ),
                            "candidate_path": str(
                                output_path
                            ),
                            "base_reproduction_mean_absolute_difference": (
                                reproduction_difference
                            ),
                            "background_mean_absolute_difference": (
                                background_difference
                            ),
                            "current_foreground_brightness": current_metrics[
                                "foreground_brightness"
                            ],
                            "candidate_foreground_brightness": candidate_metrics[
                                "foreground_brightness"
                            ],
                            "current_foreground_contrast": current_metrics[
                                "foreground_contrast"
                            ],
                            "candidate_foreground_contrast": candidate_metrics[
                                "foreground_contrast"
                            ],
                            "current_foreground_sharpness": current_metrics[
                                "foreground_sharpness"
                            ],
                            "candidate_foreground_sharpness": candidate_metrics[
                                "foreground_sharpness"
                            ],
                            "current_foreground_edge_density": current_metrics[
                                "foreground_edge_density"
                            ],
                            "candidate_foreground_edge_density": candidate_metrics[
                                "foreground_edge_density"
                            ],
                            "candidate_background_mean": candidate_metrics[
                                "background_mean"
                            ],
                            "candidate_background_std": candidate_metrics[
                                "background_std"
                            ],
                        }
                    )

                    variant_images_bgr[
                        variant
                    ] = candidate_bgr

                    print(
                        f"OK  {variant:<22} "
                        f"{pair_id:<16} "
                        f"sharpness="
                        f"{candidate_metrics['foreground_sharpness']:.2f} | "
                        f"bg_diff="
                        f"{background_difference:.6f}"
                    )

                preview = create_preview(
                    current_bgr,
                    variant_images_bgr,
                )

                preview_path = (
                    preview_root
                    / f"{pair_id}.png"
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
                "Locked geometric preprocessing: U2-Net background removal, "
                "alpha crop, adaptive scaling to 80%, centered 512x512 grey "
                "canvas. Enhancement is applied to foreground RGB only; the "
                "saved background is preserved from the current ratio-80 "
                "input.\n"
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

    background_differences = [
        float(
            row[
                "background_mean_absolute_difference"
            ]
        )
        for row in manifest_rows
    ]

    print(
        "\n"
        + "=" * 100
    )

    print(
        "PIPELINE V2 FOREGROUND-ENHANCEMENT INPUT GENERATION"
    )

    print(
        "=" * 100
    )

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
        f"Mean base reproduction difference: "
        f"{np.mean(reproduction_differences):.6f}"
    )

    print(
        f"Maximum background pixel difference: "
        f"{max(background_differences):.6f}"
    )

    print(
        f"Inputs: {OUTPUT_ROOT}"
    )

    print(
        f"Screening records: {RESULTS_ROOT}"
    )

    print(
        "PIPELINE V2 FOREGROUND-ENHANCEMENT INPUT GENERATION PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print(
        "=" * 100
    )

    print(
        "Pipeline V2 Ratio-80 Foreground-Only Enhancement Screening"
    )

    print(
        "=" * 100
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
        "Locked geometric configuration: adaptive ratio 80%"
    )

    print(
        "Candidates: mild CLAHE, mild sharpening, "
        "mild CLAHE + mild sharpening"
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
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        f"Results: {RESULTS_ROOT}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no foreground-enhancement candidates "
            "were generated."
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
