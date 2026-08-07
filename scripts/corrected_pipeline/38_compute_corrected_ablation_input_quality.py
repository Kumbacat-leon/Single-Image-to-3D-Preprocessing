from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_ROOTS = {
    "baseline": PROJECT_ROOT / "baseline_outputs",
    "nobg_only": (
        PROJECT_ROOT
        / "ablation_nobg_outputs_corrected_20260804_final"
    ),
    "nobg_crop_pad": (
        PROJECT_ROOT
        / "ablation_nobg_crop_pad_outputs_corrected_20260804_final"
    ),
    "final_proposed": (
        PROJECT_ROOT
        / "final_proposed_outputs_corrected_20260804_final"
    ),
}

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

TRANSITIONS = (
    ("baseline", "nobg_only", "Background removal"),
    ("nobg_only", "nobg_crop_pad", "Crop and padding"),
    ("nobg_crop_pad", "final_proposed", "Enhancement"),
    ("baseline", "final_proposed", "Full pipeline"),
)

# Directional labels are only applied where the interpretation is reasonably
# clear. Other metrics remain descriptive because "more" is not always better.
LOWER_IS_BETTER = (
    "center_offset_ratio",
    "margin_imbalance_ratio",
    "background_std",
    "background_edge_density",
)

HIGHER_IS_BETTER = (
    "foreground_background_difference",
)

DESCRIPTIVE_METRICS = (
    "occupancy_ratio",
    "bounding_box_ratio",
    "minimum_margin_ratio",
    "foreground_brightness",
    "foreground_contrast",
    "foreground_sharpness",
    "foreground_edge_density",
)

ALL_METRICS = (
    *LOWER_IS_BETTER,
    *HIGHER_IS_BETTER,
    *DESCRIPTIVE_METRICS,
)

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "ablation_input_quality"
STAGING_DIR = CORRECTED_ROOT / "_ablation_input_quality_staging"

METRICS_CSV = "ablation_input_quality_metrics.csv"
METHOD_SUMMARY_CSV = "ablation_input_quality_method_summary.csv"
STAGE_CHANGES_CSV = "ablation_input_quality_stage_changes.csv"
STATISTICS_CSV = "ablation_input_quality_stage_statistics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure corrected input-image ablation metrics using the exact "
            "saved input.png files entering TripoSR."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the 60 input.png files without measuring metrics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Measure and publish corrected input-quality ablation metrics.",
    )

    return parser.parse_args()


def expected_input_paths(root: Path) -> list[Path]:
    return [
        root
        / f"{object_name}_{view_name}"
        / "0"
        / "input.png"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def validate_input_set(
    root: Path,
    method: str,
) -> list[Path]:
    paths = expected_input_paths(root)
    missing = [path for path in paths if not path.is_file()]

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{method} is missing {len(missing)} input.png file(s):\n"
            f"{details}"
        )

    return paths


def validate_destination_is_new() -> None:
    for path in (OUTPUT_DIR, STAGING_DIR):
        if path.exists():
            raise FileExistsError(
                "Corrected input-quality output already exists and will not "
                f"be overwritten: {path}"
            )


def preflight() -> dict[str, object]:
    method_paths: dict[str, list[Path]] = {}

    for method, root in METHOD_ROOTS.items():
        method_paths[method] = validate_input_set(root, method)

    validate_destination_is_new()

    return {
        "method_paths": method_paths,
        "expected_images": (
            len(METHOD_ROOTS)
            * len(OBJECTS)
            * len(VIEWS)
        ),
        "expected_stage_rows": (
            len(TRANSITIONS)
            * len(OBJECTS)
            * len(VIEWS)
        ),
    }


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = (mask.astype(np.uint8) * 255)

    count, labels, statistics_array, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    if count <= 1:
        return mask

    areas = statistics_array[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1

    return labels == largest_label


def clean_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    kernel_size = max(3, round(min(height, width) * 0.006))

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )

    mask_u8 = (mask.astype(np.uint8) * 255)

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

    return keep_largest_component(mask_u8 > 0)


def estimate_rgb_mask(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    patch = max(3, round(min(height, width) * 0.04))

    corners = np.concatenate(
        [
            bgr[:patch, :patch].reshape(-1, 3),
            bgr[:patch, -patch:].reshape(-1, 3),
            bgr[-patch:, :patch].reshape(-1, 3),
            bgr[-patch:, -patch:].reshape(-1, 3),
        ],
        axis=0,
    )

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    corner_lab = cv2.cvtColor(
        corners.reshape(-1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).reshape(-1, 3).astype(np.float32)

    background_colour = np.median(corner_lab, axis=0)
    distance = np.linalg.norm(lab - background_colour, axis=2)

    distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        distance_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    threshold = max(10.0, float(otsu_threshold))
    mask = clean_mask(distance > threshold)

    occupancy = float(np.mean(mask))

    # A conservative fallback for an implausibly empty/full mask.
    if occupancy < 0.005 or occupancy > 0.97:
        threshold = max(6.0, float(np.percentile(distance, 70)))
        mask = clean_mask(distance > threshold)

    return mask


def read_image_and_mask(
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise ValueError(f"OpenCV could not read: {image_path}")

    has_alpha = image.ndim == 3 and image.shape[2] == 4

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
    else:
        bgr = image[:, :, :3]

    if has_alpha:
        alpha = image[:, :, 3]
        transparent_fraction = float(np.mean(alpha < 250))

        if transparent_fraction > 0.001:
            mask = clean_mask(alpha > 16)
            mask_source = "alpha"
        else:
            mask = estimate_rgb_mask(bgr)
            mask_source = "corner_colour"
    else:
        mask = estimate_rgb_mask(bgr)
        mask_source = "corner_colour"

    occupancy = float(np.mean(mask))

    if occupancy <= 0.0 or occupancy >= 1.0:
        raise ValueError(
            f"Invalid foreground mask occupancy {occupancy:.6f}: {image_path}"
        )

    return bgr, mask, mask_source, has_alpha


def calculate_edge_density(
    gray: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    edges = cv2.Canny(gray, 80, 160)
    selected = region_mask.astype(bool)

    if not np.any(selected):
        return math.nan

    return float(np.mean(edges[selected] > 0))


def extract_metrics(
    image_path: Path,
    method: str,
    object_name: str,
    view_name: str,
) -> dict:
    bgr, mask, mask_source, has_alpha = read_image_and_mask(image_path)

    height, width = mask.shape
    y_coordinates, x_coordinates = np.where(mask)

    x_min = int(x_coordinates.min())
    x_max = int(x_coordinates.max())
    y_min = int(y_coordinates.min())
    y_max = int(y_coordinates.max())

    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    object_center_x = (x_min + x_max) / 2.0
    object_center_y = (y_min + y_max) / 2.0
    image_center_x = (width - 1) / 2.0
    image_center_y = (height - 1) / 2.0

    diagonal = math.hypot(width, height)

    center_offset_ratio = (
        math.hypot(
            object_center_x - image_center_x,
            object_center_y - image_center_y,
        )
        / diagonal
    )

    left_margin = x_min
    right_margin = width - 1 - x_max
    top_margin = y_min
    bottom_margin = height - 1 - y_max

    margins = np.asarray(
        [
            left_margin / width,
            right_margin / width,
            top_margin / height,
            bottom_margin / height,
        ],
        dtype=float,
    )

    minimum_margin_ratio = float(np.min(margins))
    margin_imbalance_ratio = float(np.max(margins) - np.min(margins))

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    foreground_pixels = gray[mask]
    background_mask = ~mask
    background_pixels = gray[background_mask]

    foreground_brightness = float(np.mean(foreground_pixels))
    foreground_contrast = float(np.std(foreground_pixels))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    foreground_sharpness = float(np.var(laplacian[mask]))

    foreground_edge_density = calculate_edge_density(
        gray,
        mask,
    )
    background_edge_density = calculate_edge_density(
        gray,
        background_mask,
    )

    background_std = float(np.std(background_pixels))

    foreground_mean_colour = np.mean(
        bgr[mask].astype(np.float32),
        axis=0,
    )
    background_mean_colour = np.mean(
        bgr[background_mask].astype(np.float32),
        axis=0,
    )

    foreground_background_difference = float(
        np.linalg.norm(
            foreground_mean_colour
            - background_mean_colour
        )
    )

    return {
        "method": method,
        "object": object_name,
        "view": view_name,
        "pair_id": f"{object_name}_{view_name}",
        "input_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "has_alpha": has_alpha,
        "mask_source": mask_source,
        "foreground_pixel_count": int(np.count_nonzero(mask)),
        "occupancy_ratio": float(np.mean(mask)),
        "bounding_box_ratio": float(
            (bbox_width * bbox_height)
            / (width * height)
        ),
        "center_offset_ratio": center_offset_ratio,
        "minimum_margin_ratio": minimum_margin_ratio,
        "margin_imbalance_ratio": margin_imbalance_ratio,
        "foreground_brightness": foreground_brightness,
        "foreground_contrast": foreground_contrast,
        "foreground_sharpness": foreground_sharpness,
        "foreground_edge_density": foreground_edge_density,
        "background_std": background_std,
        "background_edge_density": background_edge_density,
        "foreground_background_difference": (
            foreground_background_difference
        ),
    }


def percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    if (
        not math.isfinite(old_value)
        or not math.isfinite(new_value)
        or abs(old_value) <= 1e-12
    ):
        return math.nan

    return (new_value - old_value) / old_value * 100.0


def classify_change(
    metric: str,
    old_value: float,
    new_value: float,
) -> str:
    if metric in LOWER_IS_BETTER:
        if new_value < old_value:
            return "Improved"
        if new_value > old_value:
            return "Worsened"
        return "Unchanged"

    if metric in HIGHER_IS_BETTER:
        if new_value > old_value:
            return "Improved"
        if new_value < old_value:
            return "Worsened"
        return "Unchanged"

    return "Descriptive"


def create_method_summary_rows(
    metric_rows: list[dict],
) -> list[dict]:
    output_rows: list[dict] = []
    groups = ("overall", *OBJECTS)

    for group_name in groups:
        for method in METHOD_ROOTS:
            selected = [
                row
                for row in metric_rows
                if row["method"] == method
                and (
                    group_name == "overall"
                    or row["object"] == group_name
                )
            ]

            output_row: dict[str, object] = {
                "group": group_name,
                "method": method,
                "image_count": len(selected),
            }

            for metric in ALL_METRICS:
                values = np.asarray(
                    [float(row[metric]) for row in selected],
                    dtype=float,
                )
                values = values[np.isfinite(values)]

                output_row[f"mean_{metric}"] = (
                    float(np.mean(values))
                    if values.size
                    else math.nan
                )
                output_row[f"median_{metric}"] = (
                    float(np.median(values))
                    if values.size
                    else math.nan
                )
                output_row[f"stdev_{metric}"] = (
                    float(np.std(values, ddof=1))
                    if values.size >= 2
                    else 0.0
                )

            output_rows.append(output_row)

    return output_rows


def create_stage_change_rows(
    metric_rows: list[dict],
) -> list[dict]:
    indexed = {
        (
            str(row["method"]),
            str(row["object"]),
            str(row["view"]),
        ): row
        for row in metric_rows
    }

    output_rows: list[dict] = []

    for from_method, to_method, stage_name in TRANSITIONS:
        for object_name in OBJECTS:
            for view_name in VIEWS:
                old_row = indexed[(from_method, object_name, view_name)]
                new_row = indexed[(to_method, object_name, view_name)]

                output_row: dict[str, object] = {
                    "stage": stage_name,
                    "from_method": from_method,
                    "to_method": to_method,
                    "object": object_name,
                    "view": view_name,
                    "pair_id": f"{object_name}_{view_name}",
                }

                for metric in ALL_METRICS:
                    old_value = float(old_row[metric])
                    new_value = float(new_row[metric])

                    output_row[f"from_{metric}"] = old_value
                    output_row[f"to_{metric}"] = new_value
                    output_row[f"{metric}_difference"] = (
                        new_value - old_value
                    )
                    output_row[f"{metric}_change_percent"] = (
                        percentage_change(old_value, new_value)
                    )
                    output_row[f"{metric}_result"] = classify_change(
                        metric,
                        old_value,
                        new_value,
                    )

                output_rows.append(output_row)

    return output_rows


def run_wilcoxon(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> tuple[float, float]:
    differences = new_values - old_values

    if np.all(np.abs(differences) <= 1e-12):
        return 0.0, 1.0

    result = wilcoxon(
        new_values,
        old_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def exact_sign_test(
    improved: int,
    worsened: int,
) -> float:
    non_tied = improved + worsened

    if non_tied == 0:
        return 1.0

    return float(
        binomtest(
            improved,
            n=non_tied,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def create_statistics_rows(
    stage_rows: list[dict],
) -> list[dict]:
    output_rows: list[dict] = []
    groups = ("overall", *OBJECTS)

    for from_method, to_method, stage_name in TRANSITIONS:
        for group_name in groups:
            selected = [
                row
                for row in stage_rows
                if row["stage"] == stage_name
                and (
                    group_name == "overall"
                    or row["object"] == group_name
                )
            ]

            for metric in ALL_METRICS:
                old_values = np.asarray(
                    [float(row[f"from_{metric}"]) for row in selected],
                    dtype=float,
                )
                new_values = np.asarray(
                    [float(row[f"to_{metric}"]) for row in selected],
                    dtype=float,
                )

                valid = np.isfinite(old_values) & np.isfinite(new_values)
                old_values = old_values[valid]
                new_values = new_values[valid]

                statistic, p_value = run_wilcoxon(
                    old_values,
                    new_values,
                )

                results = [
                    str(row[f"{metric}_result"])
                    for row in selected
                ]

                improved = results.count("Improved")
                worsened = results.count("Worsened")
                unchanged = results.count("Unchanged")

                old_mean = float(np.mean(old_values))
                new_mean = float(np.mean(new_values))

                output_rows.append(
                    {
                        "stage": stage_name,
                        "from_method": from_method,
                        "to_method": to_method,
                        "group": group_name,
                        "metric": metric,
                        "interpretation": (
                            "lower_is_better"
                            if metric in LOWER_IS_BETTER
                            else (
                                "higher_is_better"
                                if metric in HIGHER_IS_BETTER
                                else "descriptive_only"
                            )
                        ),
                        "pair_count": len(old_values),
                        "from_mean": old_mean,
                        "to_mean": new_mean,
                        "mean_difference": float(
                            np.mean(new_values - old_values)
                        ),
                        "mean_change_percent": percentage_change(
                            old_mean,
                            new_mean,
                        ),
                        "improved_pairs": improved,
                        "worsened_pairs": worsened,
                        "unchanged_pairs": unchanged,
                        "wilcoxon_statistic": statistic,
                        "wilcoxon_p_value": p_value,
                        "exact_sign_test_p_value": (
                            exact_sign_test(improved, worsened)
                            if metric in (
                                *LOWER_IS_BETTER,
                                *HIGHER_IS_BETTER,
                            )
                            else math.nan
                        ),
                    }
                )

    return output_rows


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError(f"No rows were available for: {path}")

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

    STAGING_DIR.rename(OUTPUT_DIR)


def run_analysis(
    summary: dict[str, object],
) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=False)

    try:
        metric_rows: list[dict] = []

        for method, root in METHOD_ROOTS.items():
            print(f"\n[{method}]")

            for object_name in OBJECTS:
                for view_name in VIEWS:
                    input_path = (
                        root
                        / f"{object_name}_{view_name}"
                        / "0"
                        / "input.png"
                    )

                    row = extract_metrics(
                        input_path,
                        method,
                        object_name,
                        view_name,
                    )
                    metric_rows.append(row)

                    print(
                        f"OK  {object_name}_{view_name:<12} "
                        f"occupancy={row['occupancy_ratio']:.3f} | "
                        f"center={row['center_offset_ratio']:.4f} | "
                        f"bg_std={row['background_std']:.3f}"
                    )

        method_summary_rows = create_method_summary_rows(metric_rows)
        stage_rows = create_stage_change_rows(metric_rows)
        statistics_rows = create_statistics_rows(stage_rows)

        expected_images = int(summary["expected_images"])
        expected_stage_rows = int(summary["expected_stage_rows"])
        expected_method_summary = (
            (1 + len(OBJECTS)) * len(METHOD_ROOTS)
        )
        expected_statistics = (
            len(TRANSITIONS)
            * (1 + len(OBJECTS))
            * len(ALL_METRICS)
        )

        checks = (
            (len(metric_rows), expected_images, "metric rows"),
            (
                len(method_summary_rows),
                expected_method_summary,
                "method summary rows",
            ),
            (len(stage_rows), expected_stage_rows, "stage rows"),
            (
                len(statistics_rows),
                expected_statistics,
                "statistical rows",
            ),
        )

        for actual, expected, label in checks:
            if actual != expected:
                raise ValueError(
                    f"Created {actual} {label}; expected {expected}."
                )

        write_csv(
            STAGING_DIR / METRICS_CSV,
            metric_rows,
        )
        write_csv(
            STAGING_DIR / METHOD_SUMMARY_CSV,
            method_summary_rows,
        )
        write_csv(
            STAGING_DIR / STAGE_CHANGES_CSV,
            stage_rows,
        )
        write_csv(
            STAGING_DIR / STATISTICS_CSV,
            statistics_rows,
        )

        publish()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR, ignore_errors=True)
        raise

    overall_directional = [
        row
        for row in statistics_rows
        if row["group"] == "overall"
        and row["metric"] in (
            *LOWER_IS_BETTER,
            *HIGHER_IS_BETTER,
        )
    ]

    print("\n" + "=" * 100)
    print("CORRECTED ABLATION INPUT-QUALITY RESULTS")
    print("=" * 100)
    print(
        f"Metric rows: {len(metric_rows)}/{summary['expected_images']}"
    )
    print(
        f"Stage rows: {len(stage_rows)}/{summary['expected_stage_rows']}"
    )
    print(
        f"Statistical rows: {len(statistics_rows)}/"
        f"{len(TRANSITIONS) * (1 + len(OBJECTS)) * len(ALL_METRICS)}"
    )

    print("\nOverall directional metrics:")

    for row in overall_directional:
        print(
            f"{row['stage']:<20} "
            f"{row['metric']:<34} "
            f"{float(row['from_mean']):.4f} -> "
            f"{float(row['to_mean']):.4f} | "
            f"improved={row['improved_pairs']}, "
            f"worsened={row['worsened_pairs']}, "
            f"unchanged={row['unchanged_pairs']} | "
            f"p={float(row['wilcoxon_p_value']):.4f}"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print("CORRECTED ABLATION INPUT-QUALITY EVALUATION PASSED.")
    print(
        "Note: sharpness, contrast, occupancy, and edge-density values are "
        "descriptive; larger values are not automatically better."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    method_paths = summary["method_paths"]

    if not isinstance(method_paths, dict):
        raise TypeError("Invalid method-path preflight summary.")

    print("=" * 100)
    print("Corrected Ablation Input-Quality Evaluation")
    print("=" * 100)

    for method in METHOD_ROOTS:
        print(
            f"{method:<16} "
            f"{len(method_paths[method])}/15"
        )

    print(f"Expected images: {summary['expected_images']}")
    print(f"Expected stage rows: {summary['expected_stage_rows']}")
    print(f"Output: {OUTPUT_DIR}")

    if args.check_only:
        print(
            "\nCHECK PASSED: no input-quality metrics were computed."
        )
        print("Run again with --run after reviewing this plan.")
        return

    run_analysis(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
