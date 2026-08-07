from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_ROOT = PROJECT_ROOT / "pipeline_v2_ratio_inputs"
OUTPUT_ROOT = PROJECT_ROOT / "pipeline_v2_ratio_outputs"
STAGING_OUTPUT_ROOT = PROJECT_ROOT / "_pipeline_v2_ratio_outputs_staging"

TRIPOSR_ROOT = PROJECT_ROOT / "TripoSR"
RUN_PY = TRIPOSR_ROOT / "run.py"
MODEL_ROOT = TRIPOSR_ROOT / "models" / "TripoSR"

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

RESULTS_ROOT = CORRECTED_ROOT / "pipeline_v2_ratio_generation"
STAGING_RESULTS_ROOT = (
    CORRECTED_ROOT
    / "_pipeline_v2_ratio_generation_staging"
)

RATIOS = ("ratio_70", "ratio_80", "ratio_90")
OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

MANIFEST_CSV = "pipeline_v2_ratio_generation_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 45 TripoSR meshes for Pipeline V2 adaptive-ratio "
            "screening using the saved ratio candidate input.png files."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate candidate inputs and TripoSR paths without generating.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate all 45 ratio-screening meshes.",
    )

    return parser.parse_args()


def expected_input_path(
    ratio_name: str,
    object_name: str,
    view_name: str,
) -> Path:
    return (
        INPUT_ROOT
        / ratio_name
        / f"{object_name}_{view_name}"
        / "0"
        / "input.png"
    )


def expected_mesh_path(
    root: Path,
    ratio_name: str,
    object_name: str,
    view_name: str,
) -> Path:
    return (
        root
        / ratio_name
        / f"{object_name}_{view_name}"
        / "0"
        / "mesh.glb"
    )


def validate_destination_is_new() -> None:
    existing = [
        path
        for path in (
            OUTPUT_ROOT,
            STAGING_OUTPUT_ROOT,
            RESULTS_ROOT,
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
            "Pipeline V2 ratio-generation output already exists and will "
            f"not be overwritten:\n{details}"
        )


def preflight() -> dict[str, object]:
    for required_path in (
        INPUT_ROOT,
        TRIPOSR_ROOT,
        RUN_PY,
        MODEL_ROOT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path was not found: {required_path}"
            )

    input_paths = [
        expected_input_path(
            ratio_name,
            object_name,
            view_name,
        )
        for ratio_name in RATIOS
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    missing = [
        path
        for path in input_paths
        if not path.is_file()
    ]

    if missing:
        details = "\n".join(
            f"  - {path}"
            for path in missing
        )

        raise FileNotFoundError(
            f"Missing {len(missing)} ratio candidate input(s):\n"
            f"{details}"
        )

    validate_destination_is_new()

    return {
        "input_paths": input_paths,
        "expected_meshes": len(input_paths),
    }


def run_triposr(
    input_path: Path,
    output_directory: Path,
    log_path: Path,
) -> tuple[bool, float, int]:
    # TripoSR exports the first mesh to <output-dir>/0/mesh.glb, but this
    # local run.py does not create the numbered subdirectory automatically.
    # Create the exact required folder before launching the subprocess.
    numbered_output_directory = (
        output_directory
        / "0"
    )

    if output_directory.exists():
        raise FileExistsError(
            f"TripoSR output directory already exists: {output_directory}"
        )

    numbered_output_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    command = [
        sys.executable,
        str(RUN_PY),
        str(input_path),
        "--pretrained-model-name-or-path",
        str(MODEL_ROOT),
        "--output-dir",
        str(output_directory),
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

    runtime_seconds = (
        time.perf_counter()
        - start
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_path.write_text(
        completed.stdout
        + "\n"
        + completed.stderr,
        encoding="utf-8",
    )

    mesh_path = (
        output_directory
        / "0"
        / "mesh.glb"
    )

    success = (
        completed.returncode == 0
        and mesh_path.is_file()
    )

    return (
        success,
        runtime_seconds,
        completed.returncode,
    )


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise ValueError(
            f"No records were available for: {path}"
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
        writer.writerows(rows)


def validate_completed_outputs(
    rows: list[dict],
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Recorded {len(rows)} generations; "
            f"expected {expected_count}."
        )

    successful = sum(
        1
        for row in rows
        if bool(row["success"])
    )

    if successful != expected_count:
        raise RuntimeError(
            f"Only {successful}/{expected_count} "
            "ratio-screening meshes succeeded."
        )

    missing_meshes = [
        expected_mesh_path(
            STAGING_OUTPUT_ROOT,
            ratio_name,
            object_name,
            view_name,
        )
        for ratio_name in RATIOS
        for object_name in OBJECTS
        for view_name in VIEWS
        if not expected_mesh_path(
            STAGING_OUTPUT_ROOT,
            ratio_name,
            object_name,
            view_name,
        ).is_file()
    ]

    if missing_meshes:
        details = "\n".join(
            f"  - {path}"
            for path in missing_meshes
        )

        raise RuntimeError(
            "Generated-mesh validation failed:\n"
            f"{details}"
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

    manifest_rows: list[dict] = []

    try:
        total = int(
            summary["expected_meshes"]
        )
        completed_count = 0

        for ratio_name in RATIOS:
            print(f"\n[{ratio_name}]")

            for object_name in OBJECTS:
                for view_name in VIEWS:
                    pair_id = (
                        f"{object_name}_{view_name}"
                    )

                    input_path = (
                        expected_input_path(
                            ratio_name,
                            object_name,
                            view_name,
                        )
                    )

                    output_directory = (
                        STAGING_OUTPUT_ROOT
                        / ratio_name
                        / pair_id
                    )

                    log_path = (
                        STAGING_RESULTS_ROOT
                        / "logs"
                        / ratio_name
                        / f"{pair_id}.txt"
                    )

                    print(
                        f"Running {ratio_name:<9} "
                        f"{pair_id} ..."
                    )

                    (
                        success,
                        runtime_seconds,
                        return_code,
                    ) = run_triposr(
                        input_path,
                        output_directory,
                        log_path,
                    )

                    generated_input_path = (
                        output_directory
                        / "0"
                        / "input.png"
                    )

                    generated_input_path.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    shutil.copy2(
                        input_path,
                        generated_input_path,
                    )

                    mesh_path = (
                        output_directory
                        / "0"
                        / "mesh.glb"
                    )

                    manifest_rows.append(
                        {
                            "ratio_name": ratio_name,
                            "object": object_name,
                            "view": view_name,
                            "pair_id": pair_id,
                            "success": success,
                            "return_code": return_code,
                            "runtime_seconds": runtime_seconds,
                            "source_input_path": str(
                                input_path
                            ),
                            "saved_input_path": str(
                                generated_input_path
                            ),
                            "mesh_path": str(
                                mesh_path
                            ),
                            "mesh_size_mb": (
                                mesh_path.stat().st_size
                                / (1024 * 1024)
                                if mesh_path.is_file()
                                else math.nan
                            ),
                            "log_path": str(
                                log_path
                            ),
                        }
                    )

                    if not success:
                        log_tail = ""

                        if log_path.is_file():
                            log_lines = log_path.read_text(
                                encoding="utf-8",
                                errors="replace",
                            ).splitlines()

                            log_tail = "\n".join(
                                log_lines[-30:]
                            )

                        raise RuntimeError(
                            f"TripoSR failed for "
                            f"{ratio_name} {pair_id}. "
                            f"Return code: {return_code}. "
                            f"See log: {log_path}\n"
                            f"Last log lines:\n{log_tail}"
                        )

                    completed_count += 1

                    print(
                        f"OK  {ratio_name:<9} "
                        f"{pair_id:<16} "
                        f"{runtime_seconds:.2f}s "
                        f"[{completed_count}/{total}]"
                    )

        validate_completed_outputs(
            manifest_rows,
            int(
                summary["expected_meshes"]
            ),
        )

        write_csv(
            STAGING_RESULTS_ROOT
            / MANIFEST_CSV,
            manifest_rows,
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

    runtimes = [
        float(row["runtime_seconds"])
        for row in manifest_rows
    ]

    print("\n" + "=" * 96)
    print("PIPELINE V2 RATIO MODEL-GENERATION RESULTS")
    print("=" * 96)
    print(
        f"Generated meshes: "
        f"{len(manifest_rows)}/"
        f"{summary['expected_meshes']}"
    )
    print(
        f"Mean generation runtime: "
        f"{sum(runtimes) / len(runtimes):.3f}s"
    )
    print(
        f"Minimum generation runtime: "
        f"{min(runtimes):.3f}s"
    )
    print(
        f"Maximum generation runtime: "
        f"{max(runtimes):.3f}s"
    )
    print(
        f"Outputs: {OUTPUT_ROOT}"
    )
    print(
        f"Records: {RESULTS_ROOT}"
    )
    print(
        "PIPELINE V2 RATIO MODEL GENERATION PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print("=" * 96)
    print("Pipeline V2 Adaptive Ratio Screening - Model Generation")
    print("=" * 96)
    print(
        f"Candidate inputs: "
        f"{len(summary['input_paths'])}/45"
    )
    print(
        f"Ratios: "
        f"{', '.join(RATIOS)}"
    )
    print(
        f"Planned meshes: "
        f"{summary['expected_meshes']}"
    )
    print(
        "TripoSR settings: "
        "CUDA, chunk-size=1024, mc-resolution=96, no-remove-bg"
    )
    print(
        f"Output: {OUTPUT_ROOT}"
    )
    print(
        f"Results: {RESULTS_ROOT}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no Pipeline V2 ratio meshes were generated."
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
