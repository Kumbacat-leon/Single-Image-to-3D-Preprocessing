from __future__ import annotations

import argparse
import csv
import inspect
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.stats import binomtest, rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_ROOTS = {
    "baseline": PROJECT_ROOT / "baseline_outputs",
    "ratio_70": (
        PROJECT_ROOT
        / "pipeline_v2_ratio_outputs"
        / "ratio_70"
    ),
    "ratio_80": (
        PROJECT_ROOT
        / "pipeline_v2_ratio_outputs"
        / "ratio_80"
    ),
    "ratio_90": (
        PROJECT_ROOT
        / "pipeline_v2_ratio_outputs"
        / "ratio_90"
    ),
}

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

METRICS = (
    "chamfer_distance",
    "hausdorff_95",
    "maximum_distance",
)

METHOD_COMPARISONS = tuple(
    (
        method_a,
        method_b,
        f"{method_a} -> {method_b}",
    )
    for method_a, method_b in combinations(
        METHOD_ROOTS.keys(),
        2,
    )
)

SAMPLE_COUNT = 5000
BASE_SEED = 20260804
ICP_THRESHOLD = 1e-7
ICP_MAX_ITERATIONS = 60

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = (
    CORRECTED_ROOT
    / "pipeline_v2_ratio_view_consistency"
)

STAGING_DIR = (
    CORRECTED_ROOT
    / "_pipeline_v2_ratio_view_consistency_staging"
)

PAIRWISE_CSV = "pipeline_v2_ratio_view_pairwise.csv"
METHOD_SUMMARY_CSV = "pipeline_v2_ratio_view_method_summary.csv"
COMPARISON_CHANGES_CSV = "pipeline_v2_ratio_view_comparison_changes.csv"
COMPARISON_STATISTICS_CSV = "pipeline_v2_ratio_view_comparison_statistics.csv"
RANKING_CSV = "pipeline_v2_ratio_ranking.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically evaluate cross-view consistency for "
            "Baseline and Pipeline V2 ratios 70%, 80%, and 90%."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all 60 mesh files without computing metrics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compute and publish Pipeline V2 ratio evaluation results.",
    )

    return parser.parse_args()


def expected_mesh_paths(root: Path) -> list[Path]:
    return [
        root
        / f"{object_name}_{view_name}"
        / "0"
        / "mesh.glb"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def validate_mesh_set(
    root: Path,
    method: str,
) -> list[Path]:
    paths = expected_mesh_paths(root)
    missing = [
        path
        for path in paths
        if not path.is_file()
    ]

    if missing:
        details = "\n".join(
            f"  - {path}"
            for path in missing
        )
        raise FileNotFoundError(
            f"{method} is missing {len(missing)} mesh file(s):\n"
            f"{details}"
        )

    return paths


def validate_destination_is_new() -> None:
    for path in (
        OUTPUT_DIR,
        STAGING_DIR,
    ):
        if path.exists():
            raise FileExistsError(
                "Pipeline V2 ratio evaluation output already exists "
                f"and will not be overwritten: {path}"
            )


def preflight() -> dict[str, object]:
    method_paths: dict[str, list[Path]] = {}

    for method, root in METHOD_ROOTS.items():
        method_paths[method] = validate_mesh_set(
            root,
            method,
        )

    validate_destination_is_new()

    view_pairs_per_object = len(
        tuple(
            combinations(
                VIEWS,
                2,
            )
        )
    )

    return {
        "method_paths": method_paths,
        "expected_meshes": (
            len(METHOD_ROOTS)
            * len(OBJECTS)
            * len(VIEWS)
        ),
        "expected_pairwise_rows": (
            len(METHOD_ROOTS)
            * len(OBJECTS)
            * view_pairs_per_object
        ),
        "expected_comparison_rows": (
            len(METHOD_COMPARISONS)
            * len(OBJECTS)
            * view_pairs_per_object
        ),
        "expected_statistics_rows": (
            len(METHOD_COMPARISONS)
            * (1 + len(OBJECTS))
            * len(METRICS)
        ),
    }


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(
        mesh_path,
        force="scene",
    )

    if isinstance(
        loaded,
        trimesh.Scene,
    ):
        geometries = [
            loaded.geometry[name]
            for name in sorted(
                loaded.geometry
            )
            if isinstance(
                loaded.geometry[name],
                trimesh.Trimesh,
            )
        ]

        if not geometries:
            raise ValueError(
                f"No mesh geometry found: {mesh_path}"
            )

        return trimesh.util.concatenate(
            geometries
        )

    if isinstance(
        loaded,
        trimesh.Trimesh,
    ):
        return loaded

    raise TypeError(
        f"Unsupported geometry type in {mesh_path}: "
        f"{type(loaded)}"
    )


def stable_seed(
    object_name: str,
    view_name: str,
) -> int:
    object_index = OBJECTS.index(
        object_name
    )
    view_index = VIEWS.index(
        view_name
    )

    return (
        BASE_SEED
        + object_index * 100
        + view_index
    )


def deterministic_surface_sample(
    mesh: trimesh.Trimesh,
    count: int,
    seed: int,
) -> np.ndarray:
    signature = inspect.signature(
        trimesh.sample.sample_surface
    )

    if "seed" in signature.parameters:
        points, _ = trimesh.sample.sample_surface(
            mesh,
            count,
            seed=seed,
        )
        return np.asarray(
            points,
            dtype=np.float64,
        )

    old_state = np.random.get_state()

    try:
        np.random.seed(
            seed
        )
        points, _ = trimesh.sample.sample_surface(
            mesh,
            count,
        )
    finally:
        np.random.set_state(
            old_state
        )

    return np.asarray(
        points,
        dtype=np.float64,
    )


def sample_normalized_points(
    mesh: trimesh.Trimesh,
    seed: int,
) -> np.ndarray:
    points = deterministic_surface_sample(
        mesh,
        SAMPLE_COUNT,
        seed,
    )

    points -= points.mean(
        axis=0
    )

    minimum = points.min(
        axis=0
    )
    maximum = points.max(
        axis=0
    )

    diagonal = float(
        np.linalg.norm(
            maximum - minimum
        )
    )

    if diagonal <= 1e-12:
        raise ValueError(
            "Sampled point cloud has zero bounding-box diagonal."
        )

    return points / diagonal


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    _, transformed, cost = (
        trimesh.registration.icp(
            source,
            target,
            threshold=ICP_THRESHOLD,
            max_iterations=ICP_MAX_ITERATIONS,
        )
    )

    return (
        np.asarray(
            transformed,
            dtype=np.float64,
        ),
        float(
            cost
        ),
    )


def calculate_distances(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float, float]:
    target_tree = cKDTree(
        target
    )
    source_to_target, _ = target_tree.query(
        source,
        k=1,
    )

    source_tree = cKDTree(
        source
    )
    target_to_source, _ = source_tree.query(
        target,
        k=1,
    )

    chamfer_distance = float(
        (
            source_to_target.mean()
            + target_to_source.mean()
        )
        / 2.0
    )

    hausdorff_95 = float(
        max(
            np.percentile(
                source_to_target,
                95,
            ),
            np.percentile(
                target_to_source,
                95,
            ),
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


def create_point_cache() -> dict[
    tuple[str, str, str],
    np.ndarray,
]:
    cache: dict[
        tuple[str, str, str],
        np.ndarray,
    ] = {}

    for method, root in METHOD_ROOTS.items():
        print(
            f"\nLoading {method} meshes ..."
        )

        for object_name in OBJECTS:
            for view_name in VIEWS:
                mesh_path = (
                    root
                    / f"{object_name}_{view_name}"
                    / "0"
                    / "mesh.glb"
                )

                mesh = load_mesh(
                    mesh_path
                )

                seed = stable_seed(
                    object_name,
                    view_name,
                )

                cache[
                    (
                        method,
                        object_name,
                        view_name,
                    )
                ] = sample_normalized_points(
                    mesh,
                    seed,
                )

                print(
                    f"OK  {method:<10} "
                    f"{object_name}_{view_name:<12} "
                    f"seed={seed}"
                )

    return cache


def create_pairwise_rows(
    cache: dict[
        tuple[str, str, str],
        np.ndarray,
    ],
) -> list[dict]:
    rows: list[dict] = []

    for method in METHOD_ROOTS:
        print(
            f"\n[{method}]"
        )

        for object_name in OBJECTS:
            for view_a, view_b in combinations(
                VIEWS,
                2,
            ):
                points_a = cache[
                    (
                        method,
                        object_name,
                        view_a,
                    )
                ]

                points_b = cache[
                    (
                        method,
                        object_name,
                        view_b,
                    )
                ]

                (
                    aligned_a,
                    icp_cost,
                ) = align_point_clouds(
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

                rows.append(
                    {
                        "method": method,
                        "object": object_name,
                        "view_a": view_a,
                        "view_b": view_b,
                        "sample_count": SAMPLE_COUNT,
                        "seed_a": stable_seed(
                            object_name,
                            view_a,
                        ),
                        "seed_b": stable_seed(
                            object_name,
                            view_b,
                        ),
                        "icp_cost": icp_cost,
                        "chamfer_distance": chamfer_distance,
                        "hausdorff_95": hausdorff_95,
                        "maximum_distance": maximum_distance,
                    }
                )

                print(
                    f"OK  {object_name}: "
                    f"{view_a:<5} vs {view_b:<5} | "
                    f"Chamfer={chamfer_distance:.6f}"
                )

    return rows


def create_method_summary_rows(
    pairwise_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    groups = (
        "overall",
        *OBJECTS,
    )

    for group_name in groups:
        for method in METHOD_ROOTS:
            selected = [
                row
                for row in pairwise_rows
                if row["method"] == method
                and (
                    group_name == "overall"
                    or row["object"] == group_name
                )
            ]

            output_row: dict[
                str,
                object,
            ] = {
                "group": group_name,
                "method": method,
                "comparison_count": len(
                    selected
                ),
            }

            for metric in METRICS:
                values = np.asarray(
                    [
                        float(
                            row[metric]
                        )
                        for row in selected
                    ],
                    dtype=float,
                )

                output_row[
                    f"mean_{metric}"
                ] = float(
                    np.mean(
                        values
                    )
                )

                output_row[
                    f"median_{metric}"
                ] = float(
                    np.median(
                        values
                    )
                )

                output_row[
                    f"stdev_{metric}"
                ] = float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )

            rows.append(
                output_row
            )

    return rows


def percentage_change(
    old_value: float,
    new_value: float,
) -> float:
    if abs(
        old_value
    ) <= 1e-12:
        return math.nan

    return (
        (
            new_value
            - old_value
        )
        / old_value
        * 100.0
    )


def classify_lower_is_better(
    old_value: float,
    new_value: float,
) -> str:
    if new_value < old_value:
        return "Improved"

    if new_value > old_value:
        return "Worsened"

    return "Unchanged"


def create_comparison_rows(
    pairwise_rows: list[dict],
) -> list[dict]:
    indexed = {
        (
            str(
                row["method"]
            ),
            str(
                row["object"]
            ),
            str(
                row["view_a"]
            ),
            str(
                row["view_b"]
            ),
        ): row
        for row in pairwise_rows
    }

    rows: list[dict] = []

    for (
        from_method,
        to_method,
        comparison_name,
    ) in METHOD_COMPARISONS:
        for object_name in OBJECTS:
            for view_a, view_b in combinations(
                VIEWS,
                2,
            ):
                old_row = indexed[
                    (
                        from_method,
                        object_name,
                        view_a,
                        view_b,
                    )
                ]

                new_row = indexed[
                    (
                        to_method,
                        object_name,
                        view_a,
                        view_b,
                    )
                ]

                output_row: dict[
                    str,
                    object,
                ] = {
                    "comparison": comparison_name,
                    "from_method": from_method,
                    "to_method": to_method,
                    "object": object_name,
                    "view_a": view_a,
                    "view_b": view_b,
                }

                for metric in METRICS:
                    old_value = float(
                        old_row[
                            metric
                        ]
                    )

                    new_value = float(
                        new_row[
                            metric
                        ]
                    )

                    output_row[
                        f"from_{metric}"
                    ] = old_value

                    output_row[
                        f"to_{metric}"
                    ] = new_value

                    output_row[
                        f"{metric}_difference"
                    ] = (
                        new_value
                        - old_value
                    )

                    output_row[
                        f"{metric}_change_percent"
                    ] = percentage_change(
                        old_value,
                        new_value,
                    )

                    output_row[
                        f"{metric}_result"
                    ] = classify_lower_is_better(
                        old_value,
                        new_value,
                    )

                rows.append(
                    output_row
                )

    return rows


def matched_rank_biserial(
    differences: np.ndarray,
) -> float:
    differences = differences[
        np.abs(
            differences
        ) > 1e-12
    ]

    if differences.size == 0:
        return 0.0

    ranks = rankdata(
        np.abs(
            differences
        ),
        method="average",
    )

    positive_sum = float(
        ranks[
            differences > 0
        ].sum()
    )

    negative_sum = float(
        ranks[
            differences < 0
        ].sum()
    )

    denominator = (
        positive_sum
        + negative_sum
    )

    if denominator == 0:
        return 0.0

    raw_effect = (
        positive_sum
        - negative_sum
    ) / denominator

    # Distances are lower-is-better.
    return -raw_effect


def effect_label(
    effect_size: float,
) -> str:
    magnitude = abs(
        effect_size
    )

    if magnitude < 0.10:
        return "Negligible"

    if magnitude < 0.30:
        return "Small"

    if magnitude < 0.50:
        return "Moderate"

    return "Large"


def run_wilcoxon(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> tuple[float, float]:
    differences = (
        new_values
        - old_values
    )

    if np.all(
        np.abs(
            differences
        ) <= 1e-12
    ):
        return (
            0.0,
            1.0,
        )

    result = wilcoxon(
        new_values,
        old_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(
            result.statistic
        ),
        float(
            result.pvalue
        ),
    )


def exact_sign_test(
    improved: int,
    worsened: int,
) -> float:
    non_tied = (
        improved
        + worsened
    )

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
    comparison_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    groups = (
        "overall",
        *OBJECTS,
    )

    for (
        from_method,
        to_method,
        comparison_name,
    ) in METHOD_COMPARISONS:
        for group_name in groups:
            selected = [
                row
                for row in comparison_rows
                if row["comparison"]
                == comparison_name
                and (
                    group_name
                    == "overall"
                    or row["object"]
                    == group_name
                )
            ]

            for metric in METRICS:
                old_values = np.asarray(
                    [
                        float(
                            row[
                                f"from_{metric}"
                            ]
                        )
                        for row in selected
                    ],
                    dtype=float,
                )

                new_values = np.asarray(
                    [
                        float(
                            row[
                                f"to_{metric}"
                            ]
                        )
                        for row in selected
                    ],
                    dtype=float,
                )

                differences = (
                    new_values
                    - old_values
                )

                (
                    statistic,
                    p_value,
                ) = run_wilcoxon(
                    old_values,
                    new_values,
                )

                improved = int(
                    np.sum(
                        differences
                        < -1e-12
                    )
                )

                worsened = int(
                    np.sum(
                        differences
                        > 1e-12
                    )
                )

                unchanged = int(
                    np.sum(
                        np.abs(
                            differences
                        ) <= 1e-12
                    )
                )

                old_mean = float(
                    np.mean(
                        old_values
                    )
                )

                new_mean = float(
                    np.mean(
                        new_values
                    )
                )

                effect_size = matched_rank_biserial(
                    differences
                )

                rows.append(
                    {
                        "comparison": comparison_name,
                        "from_method": from_method,
                        "to_method": to_method,
                        "group": group_name,
                        "metric": metric,
                        "pair_count": len(
                            selected
                        ),
                        "from_mean": old_mean,
                        "to_mean": new_mean,
                        "mean_difference": float(
                            np.mean(
                                differences
                            )
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
                        "exact_sign_test_p_value": exact_sign_test(
                            improved,
                            worsened,
                        ),
                        "rank_biserial_effect": effect_size,
                        "effect_magnitude": effect_label(
                            effect_size
                        ),
                    }
                )

    return rows


def create_ranking_rows(
    method_summary_rows: list[dict],
) -> list[dict]:
    overall = [
        row
        for row in method_summary_rows
        if row["group"] == "overall"
        and row["method"] != "baseline"
    ]

    sorted_rows = sorted(
        overall,
        key=lambda row: (
            float(
                row[
                    "mean_chamfer_distance"
                ]
            ),
            float(
                row[
                    "mean_hausdorff_95"
                ]
            ),
            float(
                row[
                    "mean_maximum_distance"
                ]
            ),
        ),
    )

    output_rows: list[dict] = []

    for rank, row in enumerate(
        sorted_rows,
        start=1,
    ):
        output_rows.append(
            {
                "rank": rank,
                "method": row["method"],
                "primary_metric": "mean_chamfer_distance",
                "mean_chamfer_distance": row[
                    "mean_chamfer_distance"
                ],
                "mean_hausdorff_95": row[
                    "mean_hausdorff_95"
                ],
                "mean_maximum_distance": row[
                    "mean_maximum_distance"
                ],
                "selection_rule": (
                    "Lowest mean Chamfer; Hausdorff 95 and maximum "
                    "distance used as tie-breakers."
                ),
            }
        )

    return output_rows


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


def validate_counts(
    summary: dict[str, object],
    pairwise_rows: list[dict],
    method_summary_rows: list[dict],
    comparison_rows: list[dict],
    statistics_rows: list[dict],
    ranking_rows: list[dict],
) -> None:
    expected_method_summary = (
        len(METHOD_ROOTS)
        * (
            1
            + len(OBJECTS)
        )
    )

    checks = (
        (
            len(
                pairwise_rows
            ),
            int(
                summary[
                    "expected_pairwise_rows"
                ]
            ),
            "pairwise rows",
        ),
        (
            len(
                method_summary_rows
            ),
            expected_method_summary,
            "method-summary rows",
        ),
        (
            len(
                comparison_rows
            ),
            int(
                summary[
                    "expected_comparison_rows"
                ]
            ),
            "comparison rows",
        ),
        (
            len(
                statistics_rows
            ),
            int(
                summary[
                    "expected_statistics_rows"
                ]
            ),
            "statistical rows",
        ),
        (
            len(
                ranking_rows
            ),
            len(
                METHOD_ROOTS
            )
            - 1,
            "ranking rows",
        ),
    )

    for actual, expected, label in checks:
        if actual != expected:
            raise ValueError(
                f"Created {actual} {label}; "
                f"expected {expected}."
            )


def publish() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(
        OUTPUT_DIR
    )


def run_evaluation(
    summary: dict[str, object],
) -> None:
    STAGING_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        cache = create_point_cache()

        pairwise_rows = create_pairwise_rows(
            cache
        )

        method_summary_rows = (
            create_method_summary_rows(
                pairwise_rows
            )
        )

        comparison_rows = create_comparison_rows(
            pairwise_rows
        )

        statistics_rows = create_statistics_rows(
            comparison_rows
        )

        ranking_rows = create_ranking_rows(
            method_summary_rows
        )

        validate_counts(
            summary,
            pairwise_rows,
            method_summary_rows,
            comparison_rows,
            statistics_rows,
            ranking_rows,
        )

        write_csv(
            STAGING_DIR
            / PAIRWISE_CSV,
            pairwise_rows,
        )

        write_csv(
            STAGING_DIR
            / METHOD_SUMMARY_CSV,
            method_summary_rows,
        )

        write_csv(
            STAGING_DIR
            / COMPARISON_CHANGES_CSV,
            comparison_rows,
        )

        write_csv(
            STAGING_DIR
            / COMPARISON_STATISTICS_CSV,
            statistics_rows,
        )

        write_csv(
            STAGING_DIR
            / RANKING_CSV,
            ranking_rows,
        )

        publish()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )

        raise

    overall_summary = [
        row
        for row in method_summary_rows
        if row["group"] == "overall"
    ]

    baseline_comparisons = [
        row
        for row in statistics_rows
        if row["group"] == "overall"
        and row["metric"]
        == "chamfer_distance"
        and row["from_method"]
        == "baseline"
    ]

    print(
        "\n"
        + "=" * 100
    )
    print(
        "PIPELINE V2 RATIO VIEW-CONSISTENCY RESULTS"
    )
    print(
        "=" * 100
    )

    print(
        f"Pairwise rows: "
        f"{len(pairwise_rows)}/"
        f"{summary['expected_pairwise_rows']}"
    )

    print(
        f"Comparison rows: "
        f"{len(comparison_rows)}/"
        f"{summary['expected_comparison_rows']}"
    )

    print(
        f"Statistical rows: "
        f"{len(statistics_rows)}/"
        f"{summary['expected_statistics_rows']}"
    )

    print(
        "\nOverall method means:"
    )

    for row in overall_summary:
        print(
            f"{row['method']:<10} "
            f"Chamfer="
            f"{float(row['mean_chamfer_distance']):.6f} | "
            f"H95="
            f"{float(row['mean_hausdorff_95']):.6f} | "
            f"Max="
            f"{float(row['mean_maximum_distance']):.6f}"
        )

    print(
        "\nBaseline-to-ratio Chamfer comparisons:"
    )

    for row in baseline_comparisons:
        print(
            f"{row['comparison']:<24} "
            f"{float(row['from_mean']):.6f} -> "
            f"{float(row['to_mean']):.6f} | "
            f"change="
            f"{float(row['mean_change_percent']):+.2f}% | "
            f"improved={row['improved_pairs']}, "
            f"worsened={row['worsened_pairs']} | "
            f"p="
            f"{float(row['wilcoxon_p_value']):.4f} | "
            f"effect="
            f"{float(row['rank_biserial_effect']):+.3f}"
        )

    best = ranking_rows[0]

    print(
        "\nPilot ratio ranking:"
    )

    for row in ranking_rows:
        print(
            f"{row['rank']}. "
            f"{row['method']:<8} "
            f"Chamfer="
            f"{float(row['mean_chamfer_distance']):.6f}"
        )

    print(
        f"\nBest pilot ratio by the predefined rule: "
        f"{best['method']}"
    )

    print(
        f"Saved: {OUTPUT_DIR}"
    )

    print(
        "PIPELINE V2 RATIO VIEW-CONSISTENCY EVALUATION PASSED."
    )

    print(
        "Note: this is development-set screening. Lock the selected ratio "
        "before evaluating the expanded final dataset."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    method_paths = summary[
        "method_paths"
    ]

    if not isinstance(
        method_paths,
        dict,
    ):
        raise TypeError(
            "Invalid method-path preflight summary."
        )

    print(
        "=" * 100
    )
    print(
        "Pipeline V2 Adaptive Ratio - Cross-View Consistency Evaluation"
    )
    print(
        "=" * 100
    )

    for method in METHOD_ROOTS:
        print(
            f"{method:<10} "
            f"{len(method_paths[method])}/15"
        )

    print(
        f"Expected meshes: "
        f"{summary['expected_meshes']}"
    )

    print(
        f"Expected pairwise rows: "
        f"{summary['expected_pairwise_rows']}"
    )

    print(
        f"Expected comparison rows: "
        f"{summary['expected_comparison_rows']}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no Pipeline V2 ratio "
            "view-consistency metrics were computed."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_evaluation(
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
