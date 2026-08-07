from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "ablation_mesh_metrics"
    / "ablation_mesh_metrics.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "ablation_mesh_statistics"
)

STAGING_DIR = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "_ablation_mesh_statistics_staging"
)

PAIRWISE_CSV = "ablation_stage_pairwise_changes.csv"
SUMMARY_CSV = "ablation_stage_statistical_summary.csv"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

METHODS = (
    "baseline",
    "nobg_only",
    "nobg_crop_pad",
    "final_proposed",
)

TRANSITIONS = (
    ("baseline", "nobg_only", "Background removal"),
    ("nobg_only", "nobg_crop_pad", "Crop and padding"),
    ("nobg_crop_pad", "final_proposed", "Enhancement"),
    ("baseline", "final_proposed", "Full pipeline"),
)

LOWER_IS_BETTER = (
    "connected_components",
    "degenerate_faces",
)

DESCRIPTIVE_ONLY = (
    "vertex_count",
    "face_count",
    "file_size_mb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired corrected ablation statistics for mesh topology "
            "and complexity metrics."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the ablation metric table without computing statistics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Compute and publish the ablation statistical comparison.",
    )

    return parser.parse_args()


def read_rows() -> list[dict[str, str]]:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Ablation metric CSV was not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def validate_destination_is_new() -> None:
    for path in (OUTPUT_DIR, STAGING_DIR):
        if path.exists():
            raise FileExistsError(
                "Ablation statistics output already exists and will not be "
                f"overwritten: {path}"
            )


def validate_rows(
    rows: list[dict[str, str]],
) -> dict[tuple[str, str, str], dict[str, str]]:
    expected_count = (
        len(METHODS)
        * len(OBJECTS)
        * len(VIEWS)
    )

    if len(rows) != expected_count:
        raise ValueError(
            f"Metric table has {len(rows)} rows; expected {expected_count}."
        )

    indexed: dict[
        tuple[str, str, str],
        dict[str, str],
    ] = {}

    required_fields = {
        "method",
        "object",
        "view",
        *LOWER_IS_BETTER,
        *DESCRIPTIVE_ONLY,
    }

    for row in rows:
        missing_fields = (
            required_fields
            - set(row)
        )

        if missing_fields:
            raise ValueError(
                "Metric table is missing required columns: "
                f"{sorted(missing_fields)}"
            )

        key = (
            row["method"],
            row["object"],
            row["view"],
        )

        if key in indexed:
            raise ValueError(
                f"Duplicate metric row detected: {key}"
            )

        indexed[key] = row

    for method in METHODS:
        for object_name in OBJECTS:
            for view_name in VIEWS:
                key = (
                    method,
                    object_name,
                    view_name,
                )

                if key not in indexed:
                    raise ValueError(
                        f"Missing metric row: {key}"
                    )

    return indexed


def to_float(value: str) -> float:
    value = value.strip()

    if not value or value.lower() == "nan":
        return math.nan

    return float(value)


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

    return (
        (new_value - old_value)
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


def run_wilcoxon(
    old_values: np.ndarray,
    new_values: np.ndarray,
) -> tuple[float, float]:
    differences = (
        new_values - old_values
    )

    if np.all(
        np.abs(differences) <= 1e-12
    ):
        return 0.0, 1.0

    result = wilcoxon(
        new_values,
        old_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def run_sign_test(
    improved_count: int,
    worsened_count: int,
) -> float:
    non_tied = (
        improved_count
        + worsened_count
    )

    if non_tied == 0:
        return 1.0

    return float(
        binomtest(
            improved_count,
            n=non_tied,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )


def create_pairwise_rows(
    indexed: dict[
        tuple[str, str, str],
        dict[str, str],
    ],
) -> list[dict]:
    output_rows: list[dict] = []

    all_metrics = (
        *LOWER_IS_BETTER,
        *DESCRIPTIVE_ONLY,
    )

    for (
        from_method,
        to_method,
        stage_name,
    ) in TRANSITIONS:
        for object_name in OBJECTS:
            for view_name in VIEWS:
                old_row = indexed[
                    (
                        from_method,
                        object_name,
                        view_name,
                    )
                ]

                new_row = indexed[
                    (
                        to_method,
                        object_name,
                        view_name,
                    )
                ]

                output_row: dict[
                    str,
                    object,
                ] = {
                    "stage": stage_name,
                    "from_method": from_method,
                    "to_method": to_method,
                    "object": object_name,
                    "view": view_name,
                    "pair_id": (
                        f"{object_name}_{view_name}"
                    ),
                }

                for metric in all_metrics:
                    old_value = to_float(
                        old_row[metric]
                    )
                    new_value = to_float(
                        new_row[metric]
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
                        new_value - old_value
                    )

                    output_row[
                        f"{metric}_change_percent"
                    ] = percentage_change(
                        old_value,
                        new_value,
                    )

                    if metric in LOWER_IS_BETTER:
                        output_row[
                            f"{metric}_result"
                        ] = (
                            classify_lower_is_better(
                                old_value,
                                new_value,
                            )
                        )

                output_rows.append(
                    output_row
                )

    return output_rows


def summarize_transition(
    rows: list[dict],
    stage_name: str,
    from_method: str,
    to_method: str,
    group_name: str,
    metric: str,
) -> dict:
    selected = [
        row
        for row in rows
        if row["stage"] == stage_name
        and row["from_method"] == from_method
        and row["to_method"] == to_method
        and (
            group_name == "overall"
            or row["object"] == group_name
        )
    ]

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

    statistic, wilcoxon_p = (
        run_wilcoxon(
            old_values,
            new_values,
        )
    )

    differences = (
        new_values - old_values
    )

    improved = int(
        np.sum(
            differences < -1e-12
        )
    )
    worsened = int(
        np.sum(
            differences > 1e-12
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
        np.mean(old_values)
    )
    new_mean = float(
        np.mean(new_values)
    )

    return {
        "stage": stage_name,
        "from_method": from_method,
        "to_method": to_method,
        "group": group_name,
        "metric": metric,
        "pair_count": len(selected),
        "from_mean": old_mean,
        "to_mean": new_mean,
        "mean_difference": float(
            np.mean(differences)
        ),
        "mean_change_percent": percentage_change(
            old_mean,
            new_mean,
        ),
        "improved_pairs": improved,
        "worsened_pairs": worsened,
        "unchanged_pairs": unchanged,
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_value": wilcoxon_p,
        "exact_sign_test_p_value": run_sign_test(
            improved,
            worsened,
        ),
    }


def create_summary_rows(
    pairwise_rows: list[dict],
) -> list[dict]:
    output_rows: list[dict] = []

    groups = (
        "overall",
        *OBJECTS,
    )

    for (
        from_method,
        to_method,
        stage_name,
    ) in TRANSITIONS:
        for group_name in groups:
            for metric in LOWER_IS_BETTER:
                output_rows.append(
                    summarize_transition(
                        pairwise_rows,
                        stage_name,
                        from_method,
                        to_method,
                        group_name,
                        metric,
                    )
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


def publish() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(
        OUTPUT_DIR
    )


def run_analysis(
    indexed: dict[
        tuple[str, str, str],
        dict[str, str],
    ],
) -> None:
    STAGING_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        pairwise_rows = (
            create_pairwise_rows(
                indexed
            )
        )

        summary_rows = (
            create_summary_rows(
                pairwise_rows
            )
        )

        expected_pairwise = (
            len(TRANSITIONS)
            * len(OBJECTS)
            * len(VIEWS)
        )

        expected_summary = (
            len(TRANSITIONS)
            * (
                1
                + len(OBJECTS)
            )
            * len(LOWER_IS_BETTER)
        )

        if len(pairwise_rows) != expected_pairwise:
            raise ValueError(
                f"Created {len(pairwise_rows)} pairwise rows; "
                f"expected {expected_pairwise}."
            )

        if len(summary_rows) != expected_summary:
            raise ValueError(
                f"Created {len(summary_rows)} summary rows; "
                f"expected {expected_summary}."
            )

        write_csv(
            STAGING_DIR
            / PAIRWISE_CSV,
            pairwise_rows,
        )

        write_csv(
            STAGING_DIR
            / SUMMARY_CSV,
            summary_rows,
        )

        publish()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )

        raise

    print("\n" + "=" * 96)
    print("CORRECTED ABLATION MESH STATISTICS")
    print("=" * 96)
    print(
        f"Pairwise transition rows: "
        f"{len(pairwise_rows)}/"
        f"{expected_pairwise}"
    )
    print(
        f"Statistical summary rows: "
        f"{len(summary_rows)}/"
        f"{expected_summary}"
    )

    overall_rows = [
        row
        for row in summary_rows
        if row["group"] == "overall"
    ]

    for row in overall_rows:
        print(
            f"{row['stage']:<20} "
            f"{row['metric']:<22} "
            f"{float(row['from_mean']):.3f} -> "
            f"{float(row['to_mean']):.3f} | "
            f"improved={row['improved_pairs']}, "
            f"worsened={row['worsened_pairs']}, "
            f"unchanged={row['unchanged_pairs']} | "
            f"Wilcoxon p="
            f"{float(row['wilcoxon_p_value']):.4f} | "
            f"sign p="
            f"{float(row['exact_sign_test_p_value']):.4f}"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print(
        "CORRECTED ABLATION MESH STATISTICS PASSED."
    )
    print(
        "Interpretation note: connected components and degenerate faces "
        "are topology/mesh-cleanliness indicators, not direct measures of "
        "semantic shape accuracy. Vertex and face counts are reported only "
        "as descriptive complexity measures."
    )


def main() -> None:
    args = parse_args()
    rows = read_rows()
    indexed = validate_rows(
        rows
    )
    validate_destination_is_new()

    print("=" * 96)
    print("Corrected Ablation Mesh Statistical Analysis")
    print("=" * 96)
    print(
        f"Metric rows: "
        f"{len(rows)}/60"
    )
    print(
        f"Matched method/object/view keys: "
        f"{len(indexed)}/60"
    )
    print(
        f"Planned transitions: "
        f"{len(TRANSITIONS)}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no ablation statistics were computed."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_analysis(
        indexed
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
