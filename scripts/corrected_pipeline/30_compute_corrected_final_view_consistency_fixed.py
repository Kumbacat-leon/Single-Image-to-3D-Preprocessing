from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import platform
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import rankdata, wilcoxon


# =============================================================================
# Project configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_ROOT = PROJECT_ROOT / "baseline_outputs"
PROPOSED_ROOT = (
    PROJECT_ROOT / "final_proposed_outputs_corrected_20260804_final"
)

RESULTS_ROOT = PROJECT_ROOT / "comparison_results"
CORRECTED_ROOT = RESULTS_ROOT / "corrected_20260804_final"
OUTPUT_DIR = CORRECTED_ROOT / "view_consistency"
STAGING_DIR = CORRECTED_ROOT / "_view_consistency_staging"

REFERENCE_PAIRWISE_CSV = (
    RESULTS_ROOT / "final_view_consistency_pairwise.csv"
)

PAIRWISE_CSV = "final_view_consistency_pairwise.csv"
SUMMARY_CSV = "final_view_consistency_summary.csv"
COMPARISON_CSV = "final_view_consistency_method_comparison.csv"
PAIR_DIFF_CSV = "final_view_consistency_paired_differences.csv"
STATISTICS_CSV = "final_view_consistency_statistical_summary.csv"
REFERENCE_DRIFT_CSV = "baseline_reference_drift.csv"
MANIFEST_JSON = "run_manifest.json"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")
METHODS = {
    "baseline": BASELINE_ROOT,
    "final_proposed": PROPOSED_ROOT,
}

SAMPLE_COUNT = 5000
BASE_SEED = 20260804
ICP_THRESHOLD = 1e-7
ICP_MAX_ITERATIONS = 60

PAIR_KEY_FIELDS = ("method", "object", "view_a", "view_b")
METRICS = (
    "chamfer_distance",
    "hausdorff_95",
    "maximum_distance",
)


# =============================================================================
# CLI and basic I/O
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically recompute Baseline versus corrected Proposed "
            "cross-view consistency without overwriting the original results."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and print the plan without computing metrics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compute, validate, and publish corrected results.",
    )

    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file was not found: {path}")

    with path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


# =============================================================================
# Preflight validation
# =============================================================================

def expected_mesh_paths(root: Path) -> list[Path]:
    return [
        root / f"{object_name}_{view_name}" / "0" / "mesh.glb"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def validate_mesh_set(root: Path, label: str) -> list[Path]:
    paths = expected_mesh_paths(root)
    missing = [path for path in paths if not path.is_file()]

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{label} is missing {len(missing)} mesh file(s):\n{details}"
        )

    return paths


def row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return tuple(row[field] for field in PAIR_KEY_FIELDS)  # type: ignore[return-value]


def validate_reference_table() -> int:
    """
    Validate only the structure of the old reference table.

    The old numeric values are not treated as an exact regression oracle,
    because surface sampling can vary across trimesh versions and sampling
    implementations. A drift report is generated during the corrected run.
    """

    if not REFERENCE_PAIRWISE_CSV.is_file():
        return 0

    rows = read_csv(REFERENCE_PAIRWISE_CSV)
    baseline_rows = [row for row in rows if row.get("method") == "baseline"]

    if len(rows) != 60:
        raise ValueError(
            f"Reference pairwise table has {len(rows)} rows; expected 60."
        )

    if len(baseline_rows) != 30:
        raise ValueError(
            "Reference pairwise table does not contain 30 Baseline rows."
        )

    keys = [row_key(row) for row in baseline_rows]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Reference Baseline view-pair keys are not unique."
        )

    return len(baseline_rows)


def validate_destination_is_new() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Corrected output already exists and will not be overwritten: "
            f"{OUTPUT_DIR}"
        )

    if STAGING_DIR.exists():
        raise FileExistsError(
            "A staging directory already exists. Inspect or remove it first: "
            f"{STAGING_DIR}"
        )


def preflight() -> dict[str, object]:
    baseline_paths = validate_mesh_set(
        BASELINE_ROOT,
        "Baseline",
    )
    proposed_paths = validate_mesh_set(
        PROPOSED_ROOT,
        "Corrected Proposed",
    )
    reference_count = validate_reference_table()
    validate_destination_is_new()

    return {
        "baseline_paths": baseline_paths,
        "proposed_paths": proposed_paths,
        "reference_count": reference_count,
    }


# =============================================================================
# Deterministic geometry evaluation
# =============================================================================

def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        # Sort geometry names to make concatenation order reproducible.
        geometries = [
            loaded.geometry[name]
            for name in sorted(loaded.geometry)
            if isinstance(loaded.geometry[name], trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError(f"No mesh geometry found: {mesh_path}")

        return trimesh.util.concatenate(geometries)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    raise TypeError(
        f"Unsupported geometry type for {mesh_path}: {type(loaded)}"
    )


def deterministic_surface_sample(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> np.ndarray:
    """
    Sample mesh points deterministically across old and new trimesh versions.

    Newer versions accept an explicit seed argument. Older versions use the
    global NumPy RNG, so the previous RNG state is saved and restored.
    """

    signature = inspect.signature(
        trimesh.sample.sample_surface
    )

    if "seed" in signature.parameters:
        points, _ = trimesh.sample.sample_surface(
            mesh,
            count,
            seed=seed,
        )
        return np.asarray(points, dtype=np.float64)

    old_state = np.random.get_state()

    try:
        np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(
            mesh,
            count,
        )
    finally:
        np.random.set_state(old_state)

    return np.asarray(points, dtype=np.float64)


def stable_seed(
    object_name: str,
    view_name: str,
) -> int:
    """
    Use the same seed for the same object-view under both methods.

    This avoids adding method-specific Monte Carlo variation to a paired
    Baseline-versus-Proposed comparison.
    """

    object_index = OBJECTS.index(object_name)
    view_index = VIEWS.index(view_name)

    return BASE_SEED + object_index * 100 + view_index


def sample_normalized_points(
    mesh: trimesh.Trimesh,
    seed: int,
) -> np.ndarray:
    points = deterministic_surface_sample(
        mesh,
        SAMPLE_COUNT,
        seed,
    )

    points -= points.mean(axis=0)

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    diagonal = float(np.linalg.norm(maximum - minimum))

    if diagonal <= 1e-12:
        raise ValueError(
            "The sampled point cloud has zero bounding-box diagonal."
        )

    return points / diagonal


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    _, transformed, cost = trimesh.registration.icp(
        source,
        target,
        threshold=ICP_THRESHOLD,
        max_iterations=ICP_MAX_ITERATIONS,
    )

    return np.asarray(transformed), float(cost)


def calculate_distances(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    target_tree = cKDTree(target)
    source_to_target, _ = target_tree.query(source, k=1)

    source_tree = cKDTree(source)
    target_to_source, _ = source_tree.query(target, k=1)

    chamfer_distance = float(
        (
            source_to_target.mean()
            + target_to_source.mean()
        )
        / 2.0
    )

    hausdorff_95 = float(
        max(
            np.percentile(source_to_target, 95),
            np.percentile(target_to_source, 95),
        )
    )

    maximum_distance = float(
        max(
            source_to_target.max(),
            target_to_source.max(),
        )
    )

    return (
        chamfer_distance,
        hausdorff_95,
        maximum_distance,
    )


def create_point_cache() -> dict[tuple[str, str, str], np.ndarray]:
    point_cache: dict[
        tuple[str, str, str],
        np.ndarray,
    ] = {}

    for method, root in METHODS.items():
        print(f"\nLoading {method} meshes...")

        for object_name in OBJECTS:
            for view_name in VIEWS:
                mesh_path = (
                    root
                    / f"{object_name}_{view_name}"
                    / "0"
                    / "mesh.glb"
                )

                mesh = load_mesh(mesh_path)
                seed = stable_seed(object_name, view_name)

                point_cache[
                    (method, object_name, view_name)
                ] = sample_normalized_points(
                    mesh,
                    seed,
                )

                print(
                    f"OK  {method:<14} "
                    f"{object_name}_{view_name} "
                    f"seed={seed}"
                )

    return point_cache


def calculate_pairwise_rows(
    point_cache: dict[tuple[str, str, str], np.ndarray],
) -> list[dict]:
    rows: list[dict] = []

    for method in METHODS:
        print(f"\n[{method}]")

        for object_name in OBJECTS:
            for view_a, view_b in combinations(VIEWS, 2):
                points_a = point_cache[
                    (method, object_name, view_a)
                ]
                points_b = point_cache[
                    (method, object_name, view_b)
                ]

                aligned_a, icp_cost = align_point_clouds(
                    points_a,
                    points_b,
                )

                (
                    chamfer_distance,
                    hausdorff_95,
                    maximum_distance,
                ) = calculate_distances(
                    aligned_a,
                    points_b,
                )

                row = {
                    "method": method,
                    "object": object_name,
                    "view_a": view_a,
                    "view_b": view_b,
                    "sample_count": SAMPLE_COUNT,
                    "seed_a": stable_seed(object_name, view_a),
                    "seed_b": stable_seed(object_name, view_b),
                    "icp_cost": icp_cost,
                    "chamfer_distance": chamfer_distance,
                    "hausdorff_95": hausdorff_95,
                    "maximum_distance": maximum_distance,
                }

                rows.append(row)

                print(
                    f"OK  {object_name}: "
                    f"{view_a:<5} vs {view_b:<5} | "
                    f"Chamfer={chamfer_distance:.6f}"
                )

    return rows


# =============================================================================
# Summary and paired statistics
# =============================================================================

def percentage_change(
    baseline: float,
    proposed: float,
) -> float:
    if (
        not math.isfinite(baseline)
        or not math.isfinite(proposed)
        or abs(baseline) <= 1e-12
    ):
        return math.nan

    return (proposed - baseline) / baseline * 100.0


def create_summary_rows(
    pairwise_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    for method in METHODS:
        for object_name in OBJECTS:
            selected = [
                row
                for row in pairwise_rows
                if row["method"] == method
                and row["object"] == object_name
            ]

            chamfer_values = np.asarray(
                [row["chamfer_distance"] for row in selected],
                dtype=float,
            )
            hausdorff_values = np.asarray(
                [row["hausdorff_95"] for row in selected],
                dtype=float,
            )
            maximum_values = np.asarray(
                [row["maximum_distance"] for row in selected],
                dtype=float,
            )

            rows.append(
                {
                    "method": method,
                    "object": object_name,
                    "comparison_count": len(selected),
                    "mean_chamfer": float(np.mean(chamfer_values)),
                    "median_chamfer": float(np.median(chamfer_values)),
                    "stdev_chamfer": float(np.std(chamfer_values, ddof=1)),
                    "minimum_chamfer": float(np.min(chamfer_values)),
                    "maximum_chamfer": float(np.max(chamfer_values)),
                    "mean_hausdorff_95": float(np.mean(hausdorff_values)),
                    "mean_maximum_distance": float(np.mean(maximum_values)),
                }
            )

    return rows


def create_method_comparison(
    summary_rows: list[dict],
) -> list[dict]:
    indexed = {
        (row["method"], row["object"]): row
        for row in summary_rows
    }

    rows: list[dict] = []

    for object_name in OBJECTS:
        baseline = indexed[("baseline", object_name)]
        proposed = indexed[("final_proposed", object_name)]

        if proposed["mean_chamfer"] < baseline["mean_chamfer"]:
            result = "Improved"
        elif proposed["mean_chamfer"] > baseline["mean_chamfer"]:
            result = "Worsened"
        else:
            result = "Unchanged"

        rows.append(
            {
                "object": object_name,
                "baseline_mean_chamfer": baseline["mean_chamfer"],
                "final_proposed_mean_chamfer": proposed["mean_chamfer"],
                "chamfer_change_percent": percentage_change(
                    baseline["mean_chamfer"],
                    proposed["mean_chamfer"],
                ),
                "baseline_mean_hausdorff_95": baseline["mean_hausdorff_95"],
                "final_proposed_mean_hausdorff_95": proposed[
                    "mean_hausdorff_95"
                ],
                "hausdorff_change_percent": percentage_change(
                    baseline["mean_hausdorff_95"],
                    proposed["mean_hausdorff_95"],
                ),
                "baseline_mean_maximum_distance": baseline[
                    "mean_maximum_distance"
                ],
                "final_proposed_mean_maximum_distance": proposed[
                    "mean_maximum_distance"
                ],
                "maximum_distance_change_percent": percentage_change(
                    baseline["mean_maximum_distance"],
                    proposed["mean_maximum_distance"],
                ),
                "consistency_result": result,
            }
        )

    return rows


def create_matched_pairs(
    pairwise_rows: list[dict],
) -> list[dict]:
    indexed = {
        (
            row["method"],
            row["object"],
            row["view_a"],
            row["view_b"],
        ): row
        for row in pairwise_rows
    }

    rows: list[dict] = []

    for object_name in OBJECTS:
        for view_a, view_b in combinations(VIEWS, 2):
            baseline = indexed[
                ("baseline", object_name, view_a, view_b)
            ]
            proposed = indexed[
                ("final_proposed", object_name, view_a, view_b)
            ]

            row: dict[str, object] = {
                "object": object_name,
                "view_a": view_a,
                "view_b": view_b,
            }

            for metric in METRICS:
                baseline_value = float(baseline[metric])
                proposed_value = float(proposed[metric])
                difference = proposed_value - baseline_value

                if difference < -1e-12:
                    result = "Improved"
                elif difference > 1e-12:
                    result = "Worsened"
                else:
                    result = "Unchanged"

                row[f"baseline_{metric}"] = baseline_value
                row[f"final_proposed_{metric}"] = proposed_value
                row[f"{metric}_difference"] = difference
                row[f"{metric}_change_percent"] = percentage_change(
                    baseline_value,
                    proposed_value,
                )
                row[f"{metric}_result"] = result

            rows.append(row)

    return rows


def matched_rank_biserial(
    differences: np.ndarray,
) -> float:
    differences = differences[
        np.isfinite(differences)
    ]
    differences = differences[
        np.abs(differences) > 1e-12
    ]

    if differences.size == 0:
        return 0.0

    ranks = rankdata(
        np.abs(differences),
        method="average",
    )

    positive_sum = float(
        ranks[differences > 0].sum()
    )
    negative_sum = float(
        ranks[differences < 0].sum()
    )

    denominator = positive_sum + negative_sum

    if denominator == 0:
        return 0.0

    raw_effect = (
        positive_sum - negative_sum
    ) / denominator

    # Lower distance is better, so reverse the raw sign.
    return -raw_effect


def effect_label(effect_size: float) -> str:
    magnitude = abs(effect_size)

    if magnitude < 0.10:
        return "Negligible"
    if magnitude < 0.30:
        return "Small"
    if magnitude < 0.50:
        return "Moderate"

    return "Large"


def run_wilcoxon(
    baseline_values: np.ndarray,
    proposed_values: np.ndarray,
) -> tuple[float, float]:
    differences = proposed_values - baseline_values

    if np.all(np.abs(differences) <= 1e-12):
        return 0.0, 1.0

    result = wilcoxon(
        proposed_values,
        baseline_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return float(result.statistic), float(result.pvalue)


def summarize_statistics(
    paired_rows: list[dict],
) -> list[dict]:
    groups: dict[str, list[dict]] = {
        "overall": paired_rows,
    }

    grouped_by_object: dict[str, list[dict]] = defaultdict(list)

    for row in paired_rows:
        grouped_by_object[str(row["object"])].append(row)

    for object_name in OBJECTS:
        groups[object_name] = grouped_by_object[object_name]

    summary_rows: list[dict] = []

    for group_name, rows in groups.items():
        for metric in METRICS:
            baseline_values = np.asarray(
                [
                    float(row[f"baseline_{metric}"])
                    for row in rows
                ],
                dtype=float,
            )
            proposed_values = np.asarray(
                [
                    float(row[f"final_proposed_{metric}"])
                    for row in rows
                ],
                dtype=float,
            )

            differences = proposed_values - baseline_values
            statistic, p_value = run_wilcoxon(
                baseline_values,
                proposed_values,
            )
            effect_size = matched_rank_biserial(differences)

            summary_rows.append(
                {
                    "group": group_name,
                    "metric": metric,
                    "pair_count": len(rows),
                    "baseline_mean": float(np.mean(baseline_values)),
                    "final_proposed_mean": float(np.mean(proposed_values)),
                    "mean_difference": float(np.mean(differences)),
                    "mean_change_percent": percentage_change(
                        float(np.mean(baseline_values)),
                        float(np.mean(proposed_values)),
                    ),
                    "baseline_median": float(np.median(baseline_values)),
                    "final_proposed_median": float(
                        np.median(proposed_values)
                    ),
                    "improved_pairs": int(
                        np.sum(differences < -1e-12)
                    ),
                    "worsened_pairs": int(
                        np.sum(differences > 1e-12)
                    ),
                    "unchanged_pairs": int(
                        np.sum(np.abs(differences) <= 1e-12)
                    ),
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                    "rank_biserial_effect": effect_size,
                    "effect_magnitude": effect_label(effect_size),
                }
            )

    return summary_rows


# =============================================================================
# Reference drift report and manifest
# =============================================================================

def create_reference_drift(
    pairwise_rows: list[dict],
) -> list[dict]:
    """
    Compare the recomputed deterministic Baseline with the old reference.

    This report is diagnostic only and never blocks publication.
    """

    if not REFERENCE_PAIRWISE_CSV.is_file():
        return []

    reference_rows = read_csv(REFERENCE_PAIRWISE_CSV)

    reference = {
        row_key(row): row
        for row in reference_rows
        if row["method"] == "baseline"
    }

    candidate = {
        (
            str(row["method"]),
            str(row["object"]),
            str(row["view_a"]),
            str(row["view_b"]),
        ): row
        for row in pairwise_rows
        if row["method"] == "baseline"
    }

    drift_rows: list[dict] = []

    for key in sorted(reference):
        if key not in candidate:
            continue

        old_row = reference[key]
        new_row = candidate[key]

        output_row: dict[str, object] = {
            "method": key[0],
            "object": key[1],
            "view_a": key[2],
            "view_b": key[3],
        }

        for metric in (
            "icp_cost",
            "chamfer_distance",
            "hausdorff_95",
            "maximum_distance",
        ):
            old_value = float(old_row[metric])
            new_value = float(new_row[metric])

            output_row[f"reference_{metric}"] = old_value
            output_row[f"deterministic_{metric}"] = new_value
            output_row[f"{metric}_absolute_drift"] = abs(
                new_value - old_value
            )
            output_row[f"{metric}_relative_drift_percent"] = (
                percentage_change(old_value, new_value)
            )

        drift_rows.append(output_row)

    return drift_rows


def create_manifest(
    baseline_paths: list[Path],
    proposed_paths: list[Path],
) -> dict:
    mesh_records = []

    for method, paths in (
        ("baseline", baseline_paths),
        ("final_proposed", proposed_paths),
    ):
        for mesh_path in paths:
            mesh_records.append(
                {
                    "method": method,
                    "relative_path": str(
                        mesh_path.relative_to(PROJECT_ROOT)
                    ),
                    "size_bytes": mesh_path.stat().st_size,
                    "sha256": sha256_file(mesh_path),
                }
            )

    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "trimesh_version": getattr(trimesh, "__version__", "unknown"),
        "sample_count": SAMPLE_COUNT,
        "base_seed": BASE_SEED,
        "seed_strategy": (
            "Same explicit seed for the same object-view under both methods."
        ),
        "normalization": (
            "Subtract sampled-point centroid, then divide by sampled "
            "axis-aligned bounding-box diagonal."
        ),
        "icp_threshold": ICP_THRESHOLD,
        "icp_max_iterations": ICP_MAX_ITERATIONS,
        "reference_csv": (
            str(REFERENCE_PAIRWISE_CSV)
            if REFERENCE_PAIRWISE_CSV.is_file()
            else None
        ),
        "mesh_files": mesh_records,
    }


# =============================================================================
# Output validation and atomic publication
# =============================================================================

def require_unique_rows(
    rows: list[dict],
    key_fields: tuple[str, ...],
    expected_count: int,
    label: str,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"{label} has {len(rows)} rows; expected {expected_count}."
        )

    keys = [
        tuple(str(row[field]) for field in key_fields)
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            f"{label} contains duplicate keys: {key_fields}"
        )


def validate_results(
    pairwise_rows: list[dict],
    summary_rows: list[dict],
    comparison_rows: list[dict],
    paired_rows: list[dict],
    statistical_rows: list[dict],
) -> None:
    require_unique_rows(
        pairwise_rows,
        PAIR_KEY_FIELDS,
        60,
        "Pairwise consistency table",
    )
    require_unique_rows(
        summary_rows,
        ("method", "object"),
        6,
        "View-consistency summary table",
    )
    require_unique_rows(
        comparison_rows,
        ("object",),
        3,
        "Method comparison table",
    )
    require_unique_rows(
        paired_rows,
        ("object", "view_a", "view_b"),
        30,
        "Paired-difference table",
    )
    require_unique_rows(
        statistical_rows,
        ("group", "metric"),
        12,
        "Statistical summary table",
    )

    method_counts = Counter(
        str(row["method"])
        for row in pairwise_rows
    )

    expected_counts = Counter(
        {
            "baseline": 30,
            "final_proposed": 30,
        }
    )

    if method_counts != expected_counts:
        raise ValueError(
            f"Invalid method coverage: {dict(method_counts)}"
        )


def publish_staging() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(OUTPUT_DIR)


# =============================================================================
# Main workflow
# =============================================================================

def print_preflight(summary: dict[str, object]) -> None:
    baseline_paths = summary["baseline_paths"]
    proposed_paths = summary["proposed_paths"]

    if not isinstance(baseline_paths, list):
        raise TypeError("Invalid Baseline path list.")
    if not isinstance(proposed_paths, list):
        raise TypeError("Invalid Proposed path list.")

    print(f"Baseline meshes: {len(baseline_paths)}/15")
    print(f"Corrected Proposed meshes: {len(proposed_paths)}/15")
    print(
        "Reference Baseline view pairs: "
        f"{summary['reference_count']}/30"
    )
    print(f"Output: {OUTPUT_DIR}")


def run_evaluation(
    preflight_summary: dict[str, object],
) -> None:
    CORRECTED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    STAGING_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    try:
        point_cache = create_point_cache()
        pairwise_rows = calculate_pairwise_rows(point_cache)
        summary_rows = create_summary_rows(pairwise_rows)
        comparison_rows = create_method_comparison(summary_rows)
        paired_rows = create_matched_pairs(pairwise_rows)
        statistical_rows = summarize_statistics(paired_rows)
        drift_rows = create_reference_drift(pairwise_rows)

        validate_results(
            pairwise_rows,
            summary_rows,
            comparison_rows,
            paired_rows,
            statistical_rows,
        )

        write_csv(
            STAGING_DIR / PAIRWISE_CSV,
            pairwise_rows,
        )
        write_csv(
            STAGING_DIR / SUMMARY_CSV,
            summary_rows,
        )
        write_csv(
            STAGING_DIR / COMPARISON_CSV,
            comparison_rows,
        )
        write_csv(
            STAGING_DIR / PAIR_DIFF_CSV,
            paired_rows,
        )
        write_csv(
            STAGING_DIR / STATISTICS_CSV,
            statistical_rows,
        )

        if drift_rows:
            write_csv(
                STAGING_DIR / REFERENCE_DRIFT_CSV,
                drift_rows,
            )

        baseline_paths = preflight_summary["baseline_paths"]
        proposed_paths = preflight_summary["proposed_paths"]

        if not isinstance(baseline_paths, list):
            raise TypeError("Invalid Baseline path list.")
        if not isinstance(proposed_paths, list):
            raise TypeError("Invalid Proposed path list.")

        manifest = create_manifest(
            baseline_paths,
            proposed_paths,
        )

        (
            STAGING_DIR / MANIFEST_JSON
        ).write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        publish_staging()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )
        raise

    overall_statistics = {
        str(row["metric"]): row
        for row in statistical_rows
        if row["group"] == "overall"
    }

    print("\n" + "=" * 90)
    print("CORRECTED VIEW-CONSISTENCY VALIDATION")
    print("=" * 90)
    print(f"Pairwise rows: {len(pairwise_rows)}/60")
    print(f"Paired differences: {len(paired_rows)}/30")
    print(f"Statistical summary rows: {len(statistical_rows)}/12")

    if drift_rows:
        print(
            "Reference drift rows: "
            f"{len(drift_rows)}/30 "
            "(diagnostic only; not a pass/fail regression)"
        )

    print("\nOverall Baseline versus Corrected Proposed:")

    for metric in METRICS:
        row = overall_statistics[metric]

        print(
            f"  {metric}: "
            f"{float(row['baseline_mean']):.6f} -> "
            f"{float(row['final_proposed_mean']):.6f} | "
            f"improved={row['improved_pairs']}, "
            f"worsened={row['worsened_pairs']}, "
            f"unchanged={row['unchanged_pairs']}, "
            f"p={float(row['wilcoxon_p_value']):.4f}, "
            f"effect={float(row['rank_biserial_effect']):+.3f}"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print("CORRECTED VIEW-CONSISTENCY EVALUATION PASSED.")
    print(
        "Note: view-pair rows within an object share source meshes; "
        "treat p-values as exploratory."
    )


def main() -> None:
    args = parse_args()

    print("=" * 90)
    print("Deterministic Corrected Final Cross-View Consistency Evaluation")
    print("=" * 90)

    preflight_summary = preflight()
    print_preflight(preflight_summary)

    if args.check_only:
        print(
            "\nCHECK PASSED: no view-consistency metrics were computed."
        )
        print(
            "Run again with --run after reviewing this deterministic plan."
        )
        return

    run_evaluation(preflight_summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
