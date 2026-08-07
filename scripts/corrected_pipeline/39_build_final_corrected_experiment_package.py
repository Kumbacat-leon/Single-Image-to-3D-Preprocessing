from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "final_experiment_package"
STAGING_DIR = CORRECTED_ROOT / "_final_experiment_package_staging"

FILES = {
    "view_statistics": (
        CORRECTED_ROOT
        / "view_consistency"
        / "final_view_consistency_statistical_summary.csv"
    ),
    "runtime_paired": (
        CORRECTED_ROOT
        / "end_to_end_runtime"
        / "end_to_end_runtime_paired_comparison.csv"
    ),
    "ablation_view_statistics": (
        CORRECTED_ROOT
        / "ablation_view_consistency"
        / "ablation_view_consistency_stage_statistics.csv"
    ),
    "ablation_view_pairwise": (
        CORRECTED_ROOT
        / "ablation_view_consistency"
        / "ablation_view_consistency_pairwise.csv"
    ),
    "ablation_mesh_statistics": (
        CORRECTED_ROOT
        / "ablation_mesh_statistics"
        / "ablation_stage_statistical_summary.csv"
    ),
    "ablation_input_statistics": (
        CORRECTED_ROOT
        / "ablation_input_quality"
        / "ablation_input_quality_stage_statistics.csv"
    ),
    "mesh_metrics": (
        CORRECTED_ROOT
        / "ablation_mesh_metrics"
        / "ablation_mesh_metrics.csv"
    ),
    "generation_runtime_raw": (
        CORRECTED_ROOT
        / "runtime_generation"
        / "generation_runtime_raw.csv"
    ),
    "preprocessing_runtime_raw": (
        CORRECTED_ROOT
        / "preprocessing_runtime"
        / "preprocessing_runtime_raw.csv"
    ),
    "end_to_end_runtime_raw": (
        CORRECTED_ROOT
        / "end_to_end_runtime"
        / "end_to_end_runtime_raw.csv"
    ),
}

AUDIT_CSV = "final_experiment_audit.csv"
KEY_RESULTS_CSV = "final_key_results.csv"
SUMMARY_TXT = "final_experiment_summary.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and consolidate all corrected experimental results into "
            "one final experiment package without writing the thesis."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate required result files without creating the package.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Create the final corrected experiment package.",
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def to_float(value: str | float | int | None) -> float:
    if value is None:
        return math.nan

    if isinstance(value, (float, int)):
        return float(value)

    text = value.strip()

    if not text or text.lower() == "nan":
        return math.nan

    return float(text)


def preflight() -> list[dict[str, Any]]:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Final experiment package already exists and will not be "
            f"overwritten: {OUTPUT_DIR}"
        )

    if STAGING_DIR.exists():
        raise FileExistsError(
            "A staging directory already exists. Inspect or remove it: "
            f"{STAGING_DIR}"
        )

    audit_rows: list[dict[str, Any]] = []
    missing: list[Path] = []

    for label, path in FILES.items():
        exists = path.is_file()

        if not exists:
            missing.append(path)

        row_count = 0

        if exists:
            row_count = len(read_csv(path))

        audit_rows.append(
            {
                "result_name": label,
                "path": str(path),
                "exists": exists,
                "row_count": row_count,
                "status": "PASS" if exists and row_count > 0 else "FAIL",
            }
        )

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)

        raise FileNotFoundError(
            "Required corrected experiment result files are missing:\n"
            f"{details}"
        )

    return audit_rows


def find_row(
    rows: list[dict[str, str]],
    **conditions: str,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one row for {conditions}, found {len(matches)}."
        )

    return matches[0]


def significance_label(p_value: float) -> str:
    if not math.isfinite(p_value):
        return "Not tested"

    return "Statistically significant" if p_value < 0.05 else "Not significant"


def create_key_results() -> list[dict[str, Any]]:
    view_rows = read_csv(FILES["view_statistics"])
    runtime_rows = read_csv(FILES["runtime_paired"])
    ablation_view_rows = read_csv(FILES["ablation_view_statistics"])
    ablation_mesh_rows = read_csv(FILES["ablation_mesh_statistics"])
    ablation_input_rows = read_csv(FILES["ablation_input_statistics"])

    output_rows: list[dict[str, Any]] = []

    for metric in (
        "chamfer_distance",
        "hausdorff_95",
        "maximum_distance",
    ):
        row = find_row(
            view_rows,
            group="overall",
            metric=metric,
        )

        p_value = to_float(row["wilcoxon_p_value"])

        output_rows.append(
            {
                "section": "Final cross-view consistency",
                "comparison": "Baseline -> Final Proposed",
                "metric": metric,
                "from_mean": to_float(row["baseline_mean"]),
                "to_mean": to_float(row["final_proposed_mean"]),
                "change_percent": to_float(row["mean_change_percent"]),
                "improved_pairs": int(float(row["improved_pairs"])),
                "worsened_pairs": int(float(row["worsened_pairs"])),
                "unchanged_pairs": int(float(row["unchanged_pairs"])),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

    for metric in (
        "end_to_end_wall_seconds",
        "triposr_subprocess_wall_seconds",
        "model_inference_ms",
        "mesh_extraction_ms",
    ):
        row = find_row(
            runtime_rows,
            metric=metric,
        )

        p_value = to_float(row["wilcoxon_p_value"])

        output_rows.append(
            {
                "section": "Runtime",
                "comparison": "Baseline -> Final Proposed",
                "metric": metric,
                "from_mean": to_float(row["baseline_mean"]),
                "to_mean": to_float(row["corrected_proposed_mean"]),
                "change_percent": to_float(row["change_percent"]),
                "improved_pairs": int(float(row["proposed_faster_pairs"])),
                "worsened_pairs": int(float(row["baseline_faster_pairs"])),
                "unchanged_pairs": int(float(row["tied_pairs"])),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

    for stage in (
        "Background removal",
        "Crop and padding",
        "Enhancement",
        "Full pipeline",
    ):
        row = find_row(
            ablation_view_rows,
            stage=stage,
            group="overall",
            metric="chamfer_distance",
        )

        p_value = to_float(row["wilcoxon_p_value"])

        output_rows.append(
            {
                "section": "Ablation cross-view consistency",
                "comparison": stage,
                "metric": "chamfer_distance",
                "from_mean": to_float(row["from_mean"]),
                "to_mean": to_float(row["to_mean"]),
                "change_percent": to_float(row["mean_change_percent"]),
                "improved_pairs": int(float(row["improved_pairs"])),
                "worsened_pairs": int(float(row["worsened_pairs"])),
                "unchanged_pairs": int(float(row["unchanged_pairs"])),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

    for metric in (
        "connected_components",
        "degenerate_faces",
    ):
        row = find_row(
            ablation_mesh_rows,
            stage="Full pipeline",
            group="overall",
            metric=metric,
        )

        p_value = to_float(row["wilcoxon_p_value"])

        output_rows.append(
            {
                "section": "Mesh topology",
                "comparison": "Baseline -> Final Proposed",
                "metric": metric,
                "from_mean": to_float(row["from_mean"]),
                "to_mean": to_float(row["to_mean"]),
                "change_percent": to_float(row["mean_change_percent"]),
                "improved_pairs": int(float(row["improved_pairs"])),
                "worsened_pairs": int(float(row["worsened_pairs"])),
                "unchanged_pairs": int(float(row["unchanged_pairs"])),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

    for metric in (
        "center_offset_ratio",
        "margin_imbalance_ratio",
        "background_std",
        "background_edge_density",
        "foreground_background_difference",
    ):
        row = find_row(
            ablation_input_rows,
            stage="Full pipeline",
            group="overall",
            metric=metric,
        )

        p_value = to_float(row["wilcoxon_p_value"])

        output_rows.append(
            {
                "section": "Input quality",
                "comparison": "Baseline -> Final Proposed",
                "metric": metric,
                "from_mean": to_float(row["from_mean"]),
                "to_mean": to_float(row["to_mean"]),
                "change_percent": to_float(row["mean_change_percent"]),
                "improved_pairs": int(float(row["improved_pairs"])),
                "worsened_pairs": int(float(row["worsened_pairs"])),
                "unchanged_pairs": int(float(row["unchanged_pairs"])),
                "p_value": p_value,
                "significance": significance_label(p_value),
            }
        )

    return output_rows


def create_summary_text(
    audit_rows: list[dict[str, Any]],
    key_rows: list[dict[str, Any]],
) -> str:
    by_key = {
        (row["section"], row["comparison"], row["metric"]): row
        for row in key_rows
    }

    chamfer = by_key[
        (
            "Final cross-view consistency",
            "Baseline -> Final Proposed",
            "chamfer_distance",
        )
    ]

    end_to_end = by_key[
        (
            "Runtime",
            "Baseline -> Final Proposed",
            "end_to_end_wall_seconds",
        )
    ]

    background_removal = by_key[
        (
            "Ablation cross-view consistency",
            "Background removal",
            "chamfer_distance",
        )
    ]

    crop_padding = by_key[
        (
            "Ablation cross-view consistency",
            "Crop and padding",
            "chamfer_distance",
        )
    ]

    enhancement = by_key[
        (
            "Ablation cross-view consistency",
            "Enhancement",
            "chamfer_distance",
        )
    ]

    center = by_key[
        (
            "Input quality",
            "Baseline -> Final Proposed",
            "center_offset_ratio",
        )
    ]

    bg_edges = by_key[
        (
            "Input quality",
            "Baseline -> Final Proposed",
            "background_edge_density",
        )
    ]

    separation = by_key[
        (
            "Input quality",
            "Baseline -> Final Proposed",
            "foreground_background_difference",
        )
    ]

    lines = [
        "FINAL CORRECTED EXPERIMENT SUMMARY",
        "=" * 72,
        "",
        f"Audit files passed: {sum(row['status'] == 'PASS' for row in audit_rows)}/{len(audit_rows)}",
        "",
        "1. Final cross-view consistency",
        (
            "The Final Proposed method changed mean Chamfer distance from "
            f"{chamfer['from_mean']:.6f} to {chamfer['to_mean']:.6f} "
            f"({chamfer['change_percent']:+.2f}%). "
            f"The paired result was {chamfer['significance'].lower()} "
            f"(p={chamfer['p_value']:.4f})."
        ),
        "",
        "2. Stage contribution",
        (
            "Background removal produced the clearest positive Chamfer trend "
            f"({background_removal['change_percent']:+.2f}%)."
        ),
        (
            "Crop and padding showed a negative Chamfer trend "
            f"({crop_padding['change_percent']:+.2f}%)."
        ),
        (
            "Enhancement recovered part of that loss "
            f"({enhancement['change_percent']:+.2f}%)."
        ),
        "",
        "3. Input-image effects",
        (
            "Object centering improved directionally: center offset changed "
            f"from {center['from_mean']:.4f} to {center['to_mean']:.4f}."
        ),
        (
            "However, the full pipeline increased background edge density "
            f"from {bg_edges['from_mean']:.4f} to {bg_edges['to_mean']:.4f} "
            f"(p={bg_edges['p_value']:.4f})."
        ),
        (
            "Foreground-background separation changed from "
            f"{separation['from_mean']:.4f} to {separation['to_mean']:.4f} "
            f"(p={separation['p_value']:.4f})."
        ),
        "",
        "4. Efficiency",
        (
            "Mean end-to-end time changed from "
            f"{end_to_end['from_mean']:.4f}s to {end_to_end['to_mean']:.4f}s "
            f"({end_to_end['change_percent']:+.2f}%, "
            f"p={end_to_end['p_value']:.4f})."
        ),
        "",
        "5. Overall experimental conclusion",
        (
            "The automated preprocessing workflow improved input centering "
            "and produced a small average improvement in cross-view geometry "
            "consistency, but the improvement was not statistically "
            "significant. Background removal was the strongest positive "
            "stage. Crop/padding and enhancement introduced mixed effects, "
            "including increased background edge activity and reduced "
            "foreground-background separation in the implemented pipeline. "
            "The full workflow also required additional end-to-end runtime."
        ),
        "",
        "Important limitation:",
        (
            "The dataset contains 15 images from three object categories. "
            "Cross-view pairs within the same object share source meshes, so "
            "their p-values must be treated as exploratory rather than fully "
            "independent-sample evidence."
        ),
    ]

    return "\n".join(lines) + "\n"


def create_figures(key_rows: list[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    created: list[str] = []

    view_rows = [
        row
        for row in key_rows
        if row["section"] == "Final cross-view consistency"
    ]

    for row in view_rows:
        metric = str(row["metric"])

        plt.figure(figsize=(6, 4))
        plt.bar(
            ["Baseline", "Final Proposed"],
            [row["from_mean"], row["to_mean"]],
        )
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Final Cross-View Consistency: {metric}")
        plt.tight_layout()

        filename = f"figure_{metric}.png"
        plt.savefig(
            STAGING_DIR / filename,
            dpi=200,
        )
        plt.close()
        created.append(filename)

    runtime_row = next(
        row
        for row in key_rows
        if row["section"] == "Runtime"
        and row["metric"] == "end_to_end_wall_seconds"
    )

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Baseline", "Final Proposed"],
        [runtime_row["from_mean"], runtime_row["to_mean"]],
    )
    plt.ylabel("Seconds per image")
    plt.title("End-to-End Runtime")
    plt.tight_layout()
    filename = "figure_end_to_end_runtime.png"
    plt.savefig(
        STAGING_DIR / filename,
        dpi=200,
    )
    plt.close()
    created.append(filename)

    pairwise_rows = read_csv(FILES["ablation_view_pairwise"])
    method_order = [
        "baseline",
        "nobg_only",
        "nobg_crop_pad",
        "final_proposed",
    ]
    method_labels = [
        "Baseline",
        "NoBG",
        "NoBG + Crop/Pad",
        "Full",
    ]

    method_means = []

    for method in method_order:
        values = [
            to_float(row["chamfer_distance"])
            for row in pairwise_rows
            if row["method"] == method
        ]
        method_means.append(sum(values) / len(values))

    plt.figure(figsize=(7, 4))
    plt.plot(
        method_labels,
        method_means,
        marker="o",
    )
    plt.ylabel("Mean Chamfer Distance")
    plt.title("Cumulative Ablation: Cross-View Consistency")
    plt.tight_layout()
    filename = "figure_ablation_chamfer.png"
    plt.savefig(
        STAGING_DIR / filename,
        dpi=200,
    )
    plt.close()
    created.append(filename)

    input_rows = [
        row
        for row in key_rows
        if row["section"] == "Input quality"
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(
        [
            str(row["metric"]).replace("_", " ")
            for row in input_rows
        ],
        [row["change_percent"] for row in input_rows],
    )
    plt.axhline(0.0, linewidth=1)
    plt.ylabel("Change from Baseline (%)")
    plt.title("Full-Pipeline Input-Quality Changes")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    filename = "figure_input_quality_changes.png"
    plt.savefig(
        STAGING_DIR / filename,
        dpi=200,
    )
    plt.close()
    created.append(filename)

    return created


def publish() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(OUTPUT_DIR)


def run_package(audit_rows: list[dict[str, Any]]) -> None:
    STAGING_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        key_rows = create_key_results()

        write_csv(
            STAGING_DIR / AUDIT_CSV,
            audit_rows,
        )
        write_csv(
            STAGING_DIR / KEY_RESULTS_CSV,
            key_rows,
        )

        summary_text = create_summary_text(
            audit_rows,
            key_rows,
        )

        (
            STAGING_DIR / SUMMARY_TXT
        ).write_text(
            summary_text,
            encoding="utf-8",
        )

        figure_names = create_figures(key_rows)

        publish()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )
        raise

    print("\n" + "=" * 92)
    print("FINAL CORRECTED EXPERIMENT PACKAGE")
    print("=" * 92)
    print(
        f"Audit files passed: "
        f"{sum(row['status'] == 'PASS' for row in audit_rows)}/"
        f"{len(audit_rows)}"
    )
    print(f"Key result rows: {len(key_rows)}")
    print(f"Figures created: {len(figure_names)}")
    print(f"Saved: {OUTPUT_DIR}")
    print("FINAL CORRECTED EXPERIMENT PACKAGE PASSED.")
    print(
        "This package consolidates experimental evidence only; it does not "
        "write or modify the thesis."
    )


def main() -> None:
    args = parse_args()
    audit_rows = preflight()

    print("=" * 92)
    print("Final Corrected Experiment Audit and Consolidation")
    print("=" * 92)

    for row in audit_rows:
        print(
            f"{row['status']:<5} "
            f"{row['result_name']:<28} "
            f"rows={row['row_count']}"
        )

    print(f"Output: {OUTPUT_DIR}")

    if args.check_only:
        print(
            "\nCHECK PASSED: all required corrected result files are present."
        )
        print(
            "Run again with --run to create the final experiment package."
        )
        return

    run_package(audit_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
