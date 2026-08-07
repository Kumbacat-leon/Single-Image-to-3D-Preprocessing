from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
from rembg import new_session
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = Path(__file__).resolve().parent

PREPROCESS_SCRIPT = (
    SCRIPTS_ROOT / "32_benchmark_corrected_preprocessing_runtime.py"
)

TRIPOSR_ROOT = PROJECT_ROOT / "TripoSR"
RUN_PY = TRIPOSR_ROOT / "run.py"
MODEL_ROOT = TRIPOSR_ROOT / "models" / "TripoSR"

ORIGINAL_ROOT = PROJECT_ROOT / "dataset_original"

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "end_to_end_runtime"
STAGING_DIR = CORRECTED_ROOT / "_end_to_end_runtime_staging"
TEMP_DIR = CORRECTED_ROOT / "_end_to_end_runtime_temp"

RAW_CSV = "end_to_end_runtime_raw.csv"
SUMMARY_CSV = "end_to_end_runtime_summary.csv"
PAIRED_CSV = "end_to_end_runtime_paired_comparison.csv"
SESSION_TXT = "preprocessing_session_initialization.txt"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")
METHODS = ("baseline", "corrected_proposed")

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

SUMMARY_FIELDS = (
    "end_to_end_wall_seconds",
    "external_preprocessing_seconds",
    "triposr_subprocess_wall_seconds",
    "initialization_ms",
    "image_processing_ms",
    "model_inference_ms",
    "mesh_extraction_ms",
    "mesh_export_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare true command-line end-to-end runtime for Baseline and "
            "corrected Proposed workflows."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the full plan without running the benchmark.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the end-to-end benchmark.",
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeated paired runs per image. Default: 3.",
    )

    return parser.parse_args()


def load_module(
    path: Path,
    module_name: str,
) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(
            f"Could not load module from: {path}"
        )

    module = importlib.util.module_from_spec(
        specification
    )
    specification.loader.exec_module(
        module
    )

    return module


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
                "A previous end-to-end runtime path exists. "
                f"Inspect or remove it before rerunning: {path}"
            )


def preflight(repeats: int) -> dict[str, object]:
    if repeats < 1:
        raise ValueError(
            "--repeats must be at least 1."
        )

    for required_path in (
        PREPROCESS_SCRIPT,
        RUN_PY,
        MODEL_ROOT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path was not found: {required_path}"
            )

    inputs = [
        find_original(
            object_name,
            view_name,
        )
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    validate_destination_is_new()

    return {
        "inputs": inputs,
        "repeats": repeats,
        "expected_runs": (
            len(METHODS)
            * len(inputs)
            * repeats
        ),
        "expected_pairs": (
            len(inputs)
            * repeats
        ),
    }


def extract_last_timing(
    output_text: str,
    pattern: re.Pattern[str],
) -> float:
    matches = pattern.findall(
        output_text
    )

    if not matches:
        return math.nan

    return float(
        matches[-1]
    )


def build_ready_input(
    preprocessing_module: ModuleType,
    session: object,
    original_path: Path,
    run_directory: Path,
) -> tuple[Path, float]:
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

    start = time.perf_counter()

    preprocessing_module.stage_background_removal(
        original_path,
        no_background_path,
        session,
    )
    preprocessing_module.stage_crop(
        no_background_path,
        cropped_path,
    )
    preprocessing_module.stage_padding(
        cropped_path,
        padded_path,
    )
    preprocessing_module.stage_enhancement(
        padded_path,
        enhanced_path,
    )
    preprocessing_module.stage_triposr_ready(
        enhanced_path,
        ready_path,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return ready_path, elapsed


def run_triposr(
    input_path: Path,
    output_root: Path,
    no_remove_bg: bool,
) -> tuple[subprocess.CompletedProcess[str], float]:
    (output_root / "0").mkdir(
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
        str(output_root),
        "--device",
        "cuda:0",
        "--chunk-size",
        "1024",
        "--mc-resolution",
        "96",
        "--model-save-format",
        "glb",
    ]

    if no_remove_bg:
        command.append(
            "--no-remove-bg"
        )

    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    start = time.perf_counter()

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

    elapsed = (
        time.perf_counter()
        - start
    )

    return completed, elapsed


def run_one(
    preprocessing_module: ModuleType,
    session: object,
    method: str,
    object_name: str,
    view_name: str,
    repeat_number: int,
    order_index: int,
) -> dict:
    pair_id = (
        f"{object_name}_{view_name}"
    )

    original_path = find_original(
        object_name,
        view_name,
    )

    run_directory = (
        TEMP_DIR
        / method
        / pair_id
        / f"repeat_{repeat_number}"
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    output_root = (
        run_directory / "triposr_output"
    )

    total_start = (
        time.perf_counter()
    )

    external_preprocessing_seconds = 0.0

    if method == "baseline":
        model_input = original_path
        no_remove_bg = False

    elif method == "corrected_proposed":
        (
            model_input,
            external_preprocessing_seconds,
        ) = build_ready_input(
            preprocessing_module,
            session,
            original_path,
            run_directory,
        )
        no_remove_bg = True

    else:
        raise ValueError(
            f"Unsupported method: {method}"
        )

    (
        completed,
        triposr_subprocess_wall_seconds,
    ) = run_triposr(
        model_input,
        output_root,
        no_remove_bg,
    )

    end_to_end_wall_seconds = (
        time.perf_counter()
        - total_start
    )

    output_text = (
        completed.stdout
        + "\n"
        + completed.stderr
    )

    mesh_path = (
        output_root
        / "0"
        / "mesh.glb"
    )

    log_directory = (
        STAGING_DIR / "logs"
    )
    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path = (
        log_directory
        / (
            f"{method}_{pair_id}"
            f"_repeat_{repeat_number}.txt"
        )
    )

    log_path.write_text(
        output_text,
        encoding="utf-8",
    )

    success = (
        completed.returncode == 0
        and mesh_path.is_file()
    )

    verification_input = ""

    if (
        method == "corrected_proposed"
        and repeat_number == 1
    ):
        verification_directory = (
            STAGING_DIR
            / "verification_inputs"
            / object_name
        )
        verification_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        verification_path = (
            verification_directory
            / (
                f"{object_name.upper()} "
                f"{view_name.upper()}"
                "_triposr_ready.png"
            )
        )

        shutil.copy2(
            model_input,
            verification_path,
        )

        verification_input = str(
            verification_path
        )

    row = {
        "method": method,
        "object": object_name,
        "view": view_name,
        "pair_id": pair_id,
        "repeat": repeat_number,
        "order_index": order_index,
        "success": success,
        "return_code": completed.returncode,
        "original_input_path": str(
            original_path
        ),
        "actual_model_input_path": str(
            model_input
        ),
        "external_preprocessing_seconds": (
            external_preprocessing_seconds
        ),
        "triposr_subprocess_wall_seconds": (
            triposr_subprocess_wall_seconds
        ),
        "end_to_end_wall_seconds": (
            end_to_end_wall_seconds
        ),
        "initialization_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS[
                "initialization_ms"
            ],
        ),
        "image_processing_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS[
                "image_processing_ms"
            ],
        ),
        "model_inference_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS[
                "model_inference_ms"
            ],
        ),
        "mesh_extraction_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS[
                "mesh_extraction_ms"
            ],
        ),
        "mesh_export_ms": extract_last_timing(
            output_text,
            TIMING_PATTERNS[
                "mesh_export_ms"
            ],
        ),
        "mesh_size_mb": (
            mesh_path.stat().st_size
            / (1024 * 1024)
            if mesh_path.is_file()
            else math.nan
        ),
        "verification_input": (
            verification_input
        ),
        "log_path": str(
            log_path
        ),
    }

    shutil.rmtree(
        run_directory,
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

        value = float(
            row[field]
        )

        if math.isfinite(value):
            values.append(
                value
            )

    return values


def summarize_method(
    rows: list[dict],
    group_name: str,
    method: str,
) -> dict:
    selected = [
        row
        for row in rows
        if row["method"] == method
    ]

    summary: dict[str, object] = {
        "group": group_name,
        "method": method,
        "run_count": len(selected),
        "successful_runs": sum(
            1
            for row in selected
            if row["success"]
        ),
    }

    for field in SUMMARY_FIELDS:
        values = finite_values(
            selected,
            field,
        )

        summary[
            f"mean_{field}"
        ] = (
            statistics.mean(values)
            if values
            else math.nan
        )

        summary[
            f"median_{field}"
        ] = (
            statistics.median(values)
            if values
            else math.nan
        )

        summary[
            f"stdev_{field}"
        ] = (
            statistics.stdev(values)
            if len(values) >= 2
            else 0.0
        )

    return summary


def create_summary_rows(
    raw_rows: list[dict],
) -> list[dict]:
    rows: list[dict] = []

    for method in METHODS:
        rows.append(
            summarize_method(
                raw_rows,
                "overall",
                method,
            )
        )

    for object_name in OBJECTS:
        selected = [
            row
            for row in raw_rows
            if row["object"] == object_name
        ]

        for method in METHODS:
            rows.append(
                summarize_method(
                    selected,
                    object_name,
                    method,
                )
            )

    return rows


def paired_wilcoxon(
    baseline_values: np.ndarray,
    proposed_values: np.ndarray,
) -> tuple[float, float]:
    differences = (
        proposed_values
        - baseline_values
    )

    if np.all(
        np.abs(differences) <= 1e-12
    ):
        return 0.0, 1.0

    result = wilcoxon(
        proposed_values,
        baseline_values,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return (
        float(result.statistic),
        float(result.pvalue),
    )


def create_paired_rows(
    raw_rows: list[dict],
) -> list[dict]:
    indexed = {
        (
            str(row["method"]),
            str(row["object"]),
            str(row["view"]),
            int(row["repeat"]),
        ): row
        for row in raw_rows
        if row["success"]
    }

    output_rows: list[dict] = []

    for field in SUMMARY_FIELDS:
        baseline_values: list[float] = []
        proposed_values: list[float] = []

        for object_name in OBJECTS:
            for view_name in VIEWS:
                repeat_numbers = sorted(
                    {
                        int(row["repeat"])
                        for row in raw_rows
                        if row["object"] == object_name
                        and row["view"] == view_name
                    }
                )

                for repeat_number in repeat_numbers:
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
                        indexed[
                            baseline_key
                        ][field]
                    )
                    proposed_value = float(
                        indexed[
                            proposed_key
                        ][field]
                    )

                    if (
                        math.isfinite(
                            baseline_value
                        )
                        and math.isfinite(
                            proposed_value
                        )
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

        statistic, p_value = (
            paired_wilcoxon(
                baseline_array,
                proposed_array,
            )
        )

        baseline_mean = float(
            np.mean(
                baseline_array
            )
        )
        proposed_mean = float(
            np.mean(
                proposed_array
            )
        )

        output_rows.append(
            {
                "metric": field,
                "paired_run_count": (
                    len(
                        baseline_array
                    )
                ),
                "baseline_mean": (
                    baseline_mean
                ),
                "corrected_proposed_mean": (
                    proposed_mean
                ),
                "mean_difference": (
                    proposed_mean
                    - baseline_mean
                ),
                "change_percent": (
                    (
                        proposed_mean
                        - baseline_mean
                    )
                    / baseline_mean
                    * 100.0
                    if abs(
                        baseline_mean
                    ) > 1e-12
                    else math.nan
                ),
                "proposed_faster_pairs": int(
                    np.sum(
                        proposed_array
                        < baseline_array
                    )
                ),
                "baseline_faster_pairs": int(
                    np.sum(
                        proposed_array
                        > baseline_array
                    )
                ),
                "tied_pairs": int(
                    np.sum(
                        np.abs(
                            proposed_array
                            - baseline_array
                        ) <= 1e-12
                    )
                ),
                "wilcoxon_statistic": (
                    statistic
                ),
                "wilcoxon_p_value": (
                    p_value
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

    preprocessing_module = load_module(
        PREPROCESS_SCRIPT,
        "corrected_preprocessing_runtime_source",
    )

    print(
        "\nInitializing external U2-Net session ..."
    )

    session_start = (
        time.perf_counter()
    )
    session = new_session(
        "u2net"
    )
    session_initialization_seconds = (
        time.perf_counter()
        - session_start
    )

    print(
        "External U2-Net session initialized in "
        f"{session_initialization_seconds:.3f}s"
    )

    repeats = int(
        preflight_summary["repeats"]
    )

    raw_rows: list[dict] = []
    order_index = 0

    try:
        for object_name in OBJECTS:
            print(
                f"\n[{object_name}]"
            )

            for view_name in VIEWS:
                for repeat_number in range(
                    1,
                    repeats + 1,
                ):
                    # Alternate method order to reduce systematic thermal
                    # drift and cache-order bias.
                    if repeat_number % 2 == 1:
                        method_order = (
                            "baseline",
                            "corrected_proposed",
                        )
                    else:
                        method_order = (
                            "corrected_proposed",
                            "baseline",
                        )

                    for method in method_order:
                        order_index += 1

                        print(
                            f"Running {method:<18} "
                            f"{object_name}_{view_name} "
                            f"repeat {repeat_number}/{repeats} ..."
                        )

                        row = run_one(
                            preprocessing_module,
                            session,
                            method,
                            object_name,
                            view_name,
                            repeat_number,
                            order_index,
                        )

                        raw_rows.append(
                            row
                        )

                        if row["success"]:
                            print(
                                "OK  end-to-end="
                                f"{row['end_to_end_wall_seconds']:.2f}s | "
                                "external="
                                f"{row['external_preprocessing_seconds']:.2f}s | "
                                "TripoSR="
                                f"{row['triposr_subprocess_wall_seconds']:.2f}s"
                            )
                        else:
                            print(
                                "ERROR return_code="
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

        (
            STAGING_DIR / SESSION_TXT
        ).write_text(
            (
                "External U2-Net session initialization seconds: "
                f"{session_initialization_seconds:.9f}\n"
                "This one-time startup cost is reported separately and is "
                "not added to every Proposed image.\n"
            ),
            encoding="utf-8",
        )

        successful_runs = sum(
            1
            for row in raw_rows
            if row["success"]
        )

        expected_runs = int(
            preflight_summary[
                "expected_runs"
            ]
        )

        if successful_runs != expected_runs:
            raise RuntimeError(
                f"Only {successful_runs}/{expected_runs} runs succeeded. "
                f"Inspect logs in: {STAGING_DIR / 'logs'}"
            )

        publish()

    except Exception:
        if TEMP_DIR.exists():
            shutil.rmtree(
                TEMP_DIR,
                ignore_errors=True,
            )

        # Preserve staging logs on failure for diagnosis.
        raise

    if TEMP_DIR.exists():
        shutil.rmtree(
            TEMP_DIR,
            ignore_errors=True,
        )

    paired_index = {
        str(row["metric"]): row
        for row in paired_rows
    }

    print(
        "\n"
        + "=" * 96
    )
    print(
        "CORRECTED END-TO-END RUNTIME RESULTS"
    )
    print(
        "=" * 96
    )
    print(
        f"Successful runs: "
        f"{successful_runs}/{expected_runs}"
    )
    print(
        f"Paired comparisons: "
        f"{preflight_summary['expected_pairs']}/"
        f"{preflight_summary['expected_pairs']}"
    )

    for field in (
        "end_to_end_wall_seconds",
        "external_preprocessing_seconds",
        "triposr_subprocess_wall_seconds",
        "image_processing_ms",
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
            f"baseline_faster={row['baseline_faster_pairs']}, "
            f"ties={row['tied_pairs']} | "
            f"p={float(row['wilcoxon_p_value']):.4f}"
        )

    print(
        "\nExternal U2-Net session initialization: "
        f"{session_initialization_seconds:.3f}s "
        "(one-time, reported separately)"
    )
    print(
        f"Saved: {OUTPUT_DIR}"
    )
    print(
        "CORRECTED END-TO-END RUNTIME BENCHMARK PASSED."
    )
    print(
        "Interpretation scope: Baseline uses original images and TripoSR's "
        "default preprocessing. Corrected Proposed uses the external "
        "five-stage preprocessing pipeline and then TripoSR with "
        "--no-remove-bg."
    )


def main() -> None:
    args = parse_args()
    summary = preflight(
        args.repeats
    )

    print(
        "=" * 96
    )
    print(
        "Corrected Fair End-to-End Runtime Benchmark"
    )
    print(
        "=" * 96
    )
    print(
        f"Original input images: "
        f"{len(summary['inputs'])}/15"
    )
    print(
        f"Repeats per image: "
        f"{summary['repeats']}"
    )
    print(
        f"Expected method runs: "
        f"{summary['expected_runs']}"
    )
    print(
        f"Expected paired comparisons: "
        f"{summary['expected_pairs']}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no end-to-end benchmark runs were executed."
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
