from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRIPOSR_ROOT = PROJECT_ROOT / "TripoSR"
RUN_PY = TRIPOSR_ROOT / "run.py"
MODEL_ROOT = TRIPOSR_ROOT / "models" / "TripoSR"

METHOD_ROOTS = {
    "baseline": PROJECT_ROOT / "baseline_outputs",
    "corrected_proposed": (
        PROJECT_ROOT
        / "final_proposed_outputs_corrected_20260804_final"
    ),
}

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "runtime_generation"
STAGING_DIR = CORRECTED_ROOT / "_runtime_generation_staging"
TEMP_DIR = CORRECTED_ROOT / "_runtime_generation_temp"

RAW_CSV = "generation_runtime_raw.csv"
SUMMARY_CSV = "generation_runtime_summary.csv"
PAIRED_CSV = "generation_runtime_paired_comparison.csv"

TIMING_PATTERNS = {
    "initialization_ms": re.compile(
        r"Initializing model finished in\s+([0-9.]+)\s*ms",
        re.IGNORECASE,
    ),
    "image_processing_ms": re.compile(
        r"Processing images finished in\s+([0-9.]+)\s*ms",
        re.IGNORECASE,
    ),
    "model_inference_ms": re.compile(
        r"Running model finished in\s+([0-9.]+)\s*ms",
        re.IGNORECASE,
    ),
    "mesh_extraction_ms": re.compile(
        r"Extracting mesh finished in\s+([0-9.]+)\s*ms",
        re.IGNORECASE,
    ),
    "mesh_export_ms": re.compile(
        r"Exporting mesh finished in\s+([0-9.]+)\s*ms",
        re.IGNORECASE,
    ),
}

RUNTIME_FIELDS = (
    "wall_clock_seconds",
    "initialization_ms",
    "image_processing_ms",
    "model_inference_ms",
    "mesh_extraction_ms",
    "mesh_export_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark TripoSR generation runtime using the exact saved "
            "model-ready input.png files for Baseline and corrected Proposed."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all inputs without running the benchmark.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the benchmark.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated runs per image and method. Default: 3.",
    )

    return parser.parse_args()


def expected_input_paths(root: Path) -> list[Path]:
    return [
        root / f"{object_name}_{view_name}" / "0" / "input.png"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def validate_input_set(
    root: Path,
    label: str,
) -> list[Path]:
    paths = expected_input_paths(root)
    missing = [path for path in paths if not path.is_file()]

    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"{label} is missing {len(missing)} input.png file(s):\n"
            f"{details}"
        )

    return paths


def validate_destination_is_new() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            "Runtime output already exists and will not be overwritten: "
            f"{OUTPUT_DIR}"
        )

    for temporary_path in (STAGING_DIR, TEMP_DIR):
        if temporary_path.exists():
            raise FileExistsError(
                "A previous temporary directory exists. Inspect or remove it: "
                f"{temporary_path}"
            )


def preflight(repeats: int) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("--repeats must be at least 1.")

    if not RUN_PY.is_file():
        raise FileNotFoundError(f"TripoSR run.py was not found: {RUN_PY}")

    if not MODEL_ROOT.exists():
        raise FileNotFoundError(
            f"Local TripoSR model folder was not found: {MODEL_ROOT}"
        )

    baseline_paths = validate_input_set(
        METHOD_ROOTS["baseline"],
        "Baseline",
    )

    proposed_paths = validate_input_set(
        METHOD_ROOTS["corrected_proposed"],
        "Corrected Proposed",
    )

    validate_destination_is_new()

    return {
        "baseline_inputs": baseline_paths,
        "proposed_inputs": proposed_paths,
        "repeats": repeats,
        "expected_runs": (
            len(METHOD_ROOTS)
            * len(OBJECTS)
            * len(VIEWS)
            * repeats
        ),
    }


def extract_last_timing(
    output_text: str,
    pattern: re.Pattern[str],
) -> float:
    matches = pattern.findall(output_text)

    if not matches:
        return math.nan

    return float(matches[-1])


def run_one(
    method: str,
    object_name: str,
    view_name: str,
    repeat_number: int,
) -> dict:
    pair_id = f"{object_name}_{view_name}"

    input_path = (
        METHOD_ROOTS[method]
        / pair_id
        / "0"
        / "input.png"
    )

    run_root = (
        TEMP_DIR
        / method
        / pair_id
        / f"repeat_{repeat_number}"
    )

    # This local run.py expects the numbered output folder to exist.
    (run_root / "0").mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(RUN_PY),
        str(input_path),
        "--pretrained-model-name-or-path",
        str(MODEL_ROOT),
        "--output-dir",
        str(run_root),
        "--device",
        "cuda:0",
        "--chunk-size",
        "1024",
        "--mc-resolution",
        "96",
        "--model-save-format",
        "glb",
        "--no-remove-bg",
    ]

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    start_time = time.perf_counter()

    completed = subprocess.run(
        command,
        cwd=str(TRIPOSR_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    wall_clock_seconds = time.perf_counter() - start_time

    output_text = completed.stdout + "\n" + completed.stderr
    mesh_path = run_root / "0" / "mesh.glb"

    log_dir = STAGING_DIR / "logs"
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_dir
        / f"{method}_{pair_id}_repeat_{repeat_number}.txt"
    )

    log_path.write_text(
        output_text,
        encoding="utf-8",
    )

    success = (
        completed.returncode == 0
        and mesh_path.is_file()
    )

    row = {
        "method": method,
        "object": object_name,
        "view": view_name,
        "pair_id": pair_id,
        "repeat": repeat_number,
        "success": success,
        "return_code": completed.returncode,
        "input_path": str(input_path),
        "wall_clock_seconds": wall_clock_seconds,
        "initialization_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS["initialization_ms"],
        ),
        "image_processing_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS["image_processing_ms"],
        ),
        "model_inference_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS["model_inference_ms"],
        ),
        "mesh_extraction_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS["mesh_extraction_ms"],
        ),
        "mesh_export_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS["mesh_export_ms"],
        ),
        "mesh_size_mb": (
            mesh_path.stat().st_size / (1024 * 1024)
            if mesh_path.is_file()
            else math.nan
        ),
        "log_path": str(log_path),
    }

    shutil.rmtree(
        run_root,
        ignore_errors=True,
    )

    return row


def finite_values(
    rows: list[dict],
    field: str,
) -> list[float]:
    values: list[float] = []

    for row in rows:
        if not row["success"]:
            continue

        value = float(row[field])

        if math.isfinite(value):
            values.append(value)

    return values


def summarize_method(
    rows: list[dict],
    group: str,
    method: str,
) -> dict:
    selected = [
        row
        for row in rows
        if row["method"] == method
    ]

    summary: dict[str, object] = {
        "group": group,
        "method": method,
        "run_count": len(selected),
        "successful_runs": sum(
            1
            for row in selected
            if row["success"]
        ),
    }

    for field in RUNTIME_FIELDS:
        values = finite_values(
            selected,
            field,
        )

        summary[f"mean_{field}"] = (
            statistics.mean(values)
            if values
            else math.nan
        )

        summary[f"median_{field}"] = (
            statistics.median(values)
            if values
            else math.nan
        )

        summary[f"stdev_{field}"] = (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        )

    return summary


def create_summary_rows(
    raw_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    for method in METHOD_ROOTS:
        rows.append(
            summarize_method(
                raw_rows,
                "overall",
                method,
            )
        )

    for object_name in OBJECTS:
        object_rows = [
            row
            for row in raw_rows
            if row["object"] == object_name
        ]

        for method in METHOD_ROOTS:
            rows.append(
                summarize_method(
                    object_rows,
                    object_name,
                    method,
                )
            )

    return rows


def run_paired_wilcoxon(
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


def create_paired_rows(
    raw_rows: list[dict],
) -> list[dict]:
    indexed = {
        (
            row["method"],
            row["object"],
            row["view"],
            int(row["repeat"]),
        ): row
        for row in raw_rows
        if row["success"]
    }

    rows: list[dict] = []

    for field in RUNTIME_FIELDS:
        baseline_values = []
        proposed_values = []

        for object_name in OBJECTS:
            for view_name in VIEWS:
                repeats = sorted(
                    {
                        int(row["repeat"])
                        for row in raw_rows
                        if row["object"] == object_name
                        and row["view"] == view_name
                    }
                )

                for repeat_number in repeats:
                    baseline_key = (
                        "baseline",
                        object_name,
                        view_name,
                        repeat_number,
                    )

                    proposed_key = (
                        "corrected_proposed",
                        object_name,
                        view_name,
                        repeat_number,
                    )

                    if (
                        baseline_key not in indexed
                        or proposed_key not in indexed
                    ):
                        continue

                    baseline_value = float(
                        indexed[baseline_key][field]
                    )
                    proposed_value = float(
                        indexed[proposed_key][field]
                    )

                    if (
                        math.isfinite(baseline_value)
                        and math.isfinite(proposed_value)
                    ):
                        baseline_values.append(
                            baseline_value
                        )
                        proposed_values.append(
                            proposed_value
                        )

        baseline_array = np.asarray(
            baseline_values,
            dtype=float,
        )
        proposed_array = np.asarray(
            proposed_values,
            dtype=float,
        )

        statistic, p_value = run_paired_wilcoxon(
            baseline_array,
            proposed_array,
        )

        baseline_mean = float(
            np.mean(baseline_array)
        )
        proposed_mean = float(
            np.mean(proposed_array)
        )

        rows.append(
            {
                "metric": field,
                "paired_run_count": len(baseline_array),
                "baseline_mean": baseline_mean,
                "corrected_proposed_mean": proposed_mean,
                "mean_difference": (
                    proposed_mean - baseline_mean
                ),
                "change_percent": (
                    (proposed_mean - baseline_mean)
                    / baseline_mean
                    * 100.0
                    if abs(baseline_mean) > 1e-12
                    else math.nan
                ),
                "proposed_faster_pairs": int(
                    np.sum(proposed_array < baseline_array)
                ),
                "baseline_faster_pairs": int(
                    np.sum(proposed_array > baseline_array)
                ),
                "tied_pairs": int(
                    np.sum(
                        np.abs(
                            proposed_array - baseline_array
                        ) <= 1e-12
                    )
                ),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
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
    TEMP_DIR.mkdir(
        parents=False,
        exist_ok=False,
    )

    repeats = int(
        preflight_summary["repeats"]
    )

    raw_rows: list[dict] = []

    try:
        for method in METHOD_ROOTS:
            print(f"\n[{method}]")

            for object_name in OBJECTS:
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
                            method,
                            object_name,
                            view_name,
                            repeat_number,
                        )

                        raw_rows.append(row)

                        if row["success"]:
                            print(
                                f"OK  wall="
                                f"{row['wall_clock_seconds']:.2f}s | "
                                f"inference="
                                f"{row['model_inference_ms']:.2f}ms | "
                                f"mesh="
                                f"{row['mesh_extraction_ms']:.2f}ms"
                            )
                        else:
                            print(
                                f"ERROR return_code="
                                f"{row['return_code']} | "
                                f"log={row['log_path']}"
                            )

        summary_rows = create_summary_rows(
            raw_rows
        )

        paired_rows = create_paired_rows(
            raw_rows
        )

        write_csv(
            STAGING_DIR / RAW_CSV,
            raw_rows,
        )

        write_csv(
            STAGING_DIR / SUMMARY_CSV,
            summary_rows,
        )

        write_csv(
            STAGING_DIR / PAIRED_CSV,
            paired_rows,
        )

        successful_runs = sum(
            1
            for row in raw_rows
            if row["success"]
        )

        expected_runs = int(
            preflight_summary["expected_runs"]
        )

        if successful_runs != expected_runs:
            raise RuntimeError(
                f"Only {successful_runs}/{expected_runs} runs succeeded. "
                f"Inspect logs in {STAGING_DIR / 'logs'}"
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

    paired_index = {
        row["metric"]: row
        for row in paired_rows
    }

    print("\n" + "=" * 90)
    print("CORRECTED GENERATION-RUNTIME RESULTS")
    print("=" * 90)
    print(
        f"Successful runs: "
        f"{successful_runs}/{expected_runs}"
    )

    for field in (
        "wall_clock_seconds",
        "model_inference_ms",
        "mesh_extraction_ms",
    ):
        row = paired_index[field]

        print(
            f"{field}: "
            f"{float(row['baseline_mean']):.4f} -> "
            f"{float(row['corrected_proposed_mean']):.4f} | "
            f"change={float(row['change_percent']):+.2f}% | "
            f"proposed_faster={row['proposed_faster_pairs']}, "
            f"baseline_faster={row['baseline_faster_pairs']} | "
            f"p={float(row['wilcoxon_p_value']):.4f}"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print("CORRECTED GENERATION-RUNTIME BENCHMARK PASSED.")
    print(
        "Scope: this benchmark measures TripoSR generation from saved "
        "model-ready input.png files. It does not include the external "
        "image-preprocessing pipeline time."
    )


def main() -> None:
    args = parse_args()
    summary = preflight(args.repeats)

    print("=" * 90)
    print("Corrected TripoSR Generation-Runtime Benchmark")
    print("=" * 90)
    print(
        f"Baseline model-ready inputs: "
        f"{len(summary['baseline_inputs'])}/15"
    )
    print(
        f"Corrected Proposed inputs: "
        f"{len(summary['proposed_inputs'])}/15"
    )
    print(f"Repeats per image: {summary['repeats']}")
    print(f"Expected runs: {summary['expected_runs']}")
    print(f"Output: {OUTPUT_DIR}")

    if args.check_only:
        print(
            "\nCHECK PASSED: no TripoSR benchmark runs were executed."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_benchmark(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
