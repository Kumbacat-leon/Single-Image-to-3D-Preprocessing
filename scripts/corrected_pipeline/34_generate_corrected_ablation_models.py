from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
from rembg import new_session


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_ROOT = Path(__file__).resolve().parent

PREPROCESS_SCRIPT = (
    SCRIPTS_ROOT / "32_benchmark_corrected_preprocessing_runtime.py"
)

TRIPOSR_ROOT = PROJECT_ROOT / "TripoSR"
RUN_PY = TRIPOSR_ROOT / "run.py"
MODEL_ROOT = TRIPOSR_ROOT / "models" / "TripoSR"

ORIGINAL_ROOT = PROJECT_ROOT / "dataset_original"
FULL_CORRECTED_ROOT = (
    PROJECT_ROOT
    / "final_proposed_outputs_corrected_20260804_final"
)

ABLATION_ROOTS = {
    "nobg_only": (
        PROJECT_ROOT
        / "ablation_nobg_outputs_corrected_20260804_final"
    ),
    "nobg_crop_pad": (
        PROJECT_ROOT
        / "ablation_nobg_crop_pad_outputs_corrected_20260804_final"
    ),
}

CORRECTED_RESULTS_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

RESULTS_DIR = CORRECTED_RESULTS_ROOT / "ablation_generation"
STAGING_RESULTS_DIR = (
    CORRECTED_RESULTS_ROOT
    / "_ablation_generation_staging"
)
TEMP_DIR = (
    CORRECTED_RESULTS_ROOT
    / "_ablation_generation_temp"
)

MANIFEST_CSV = "ablation_generation_manifest.csv"
FULL_CHECK_CSV = "full_input_reproduction_check.csv"
SESSION_TXT = "u2net_session_initialization.txt"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate corrected cumulative ablation meshes for NoBG-only "
            "and NoBG+Crop/Pad using the same TripoSR-ready adapter and "
            "TripoSR settings as the corrected full method."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the plan without generating inputs or meshes.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Generate 30 corrected ablation meshes.",
    )

    return parser.parse_args()


def load_module(path: Path, module_name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load module from: {path}")

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def find_original(object_name: str, view_name: str) -> Path:
    folder = ORIGINAL_ROOT / object_name

    if not folder.is_dir():
        raise FileNotFoundError(f"Object folder was not found: {folder}")

    expected = normalize_name(f"{object_name} {view_name}")
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed
        and normalize_name(path.stem) == expected
    ]

    if not matches:
        available = ", ".join(
            sorted(path.name for path in folder.iterdir() if path.is_file())
        )
        raise FileNotFoundError(
            f"Original image was not found for {object_name}_{view_name}. "
            f"Available files: {available}"
        )

    return matches[0]


def expected_full_input(object_name: str, view_name: str) -> Path:
    return (
        FULL_CORRECTED_ROOT
        / f"{object_name}_{view_name}"
        / "0"
        / "input.png"
    )


def validate_destination_is_new() -> None:
    existing_paths = [
        path
        for path in (
            *ABLATION_ROOTS.values(),
            RESULTS_DIR,
            STAGING_RESULTS_DIR,
            TEMP_DIR,
        )
        if path.exists()
    ]

    if existing_paths:
        details = "\n".join(f"  - {path}" for path in existing_paths)
        raise FileExistsError(
            "Corrected ablation output already exists and will not "
            f"be overwritten:\n{details}"
        )


def preflight() -> dict[str, object]:
    for required_path in (
        PREPROCESS_SCRIPT,
        RUN_PY,
        MODEL_ROOT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path was not found: {required_path}"
            )

    originals = [
        find_original(object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    full_inputs = [
        expected_full_input(object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    missing_full = [path for path in full_inputs if not path.is_file()]

    if missing_full:
        details = "\n".join(f"  - {path}" for path in missing_full)
        raise FileNotFoundError(
            "Corrected Full input.png files are missing:\n"
            f"{details}"
        )

    validate_destination_is_new()

    return {
        "originals": originals,
        "full_inputs": full_inputs,
        "expected_ablation_meshes": (
            len(ABLATION_ROOTS) * len(OBJECTS) * len(VIEWS)
        ),
        "expected_full_checks": len(OBJECTS) * len(VIEWS),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def compare_images(
    reproduced_path: Path,
    reference_path: Path,
) -> dict[str, object]:
    reproduced = cv2.imread(str(reproduced_path), cv2.IMREAD_COLOR)
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)

    if reproduced is None:
        raise ValueError(f"Could not read reproduced image: {reproduced_path}")

    if reference is None:
        raise ValueError(f"Could not read reference image: {reference_path}")

    same_shape = reproduced.shape == reference.shape

    if same_shape:
        difference = np.abs(
            reproduced.astype(np.int16)
            - reference.astype(np.int16)
        )
        mean_absolute_difference = float(np.mean(difference))
        maximum_absolute_difference = int(np.max(difference))
        pixel_exact_match = bool(np.array_equal(reproduced, reference))
    else:
        mean_absolute_difference = math.nan
        maximum_absolute_difference = -1
        pixel_exact_match = False

    return {
        "same_shape": same_shape,
        "pixel_exact_match": pixel_exact_match,
        "sha256_match": (
            sha256_file(reproduced_path)
            == sha256_file(reference_path)
        ),
        "mean_absolute_pixel_difference": mean_absolute_difference,
        "maximum_absolute_pixel_difference": maximum_absolute_difference,
    }


def run_triposr(
    input_path: Path,
    output_root: Path,
) -> tuple[bool, float, Path]:
    (output_root / "0").mkdir(parents=True, exist_ok=True)

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

    elapsed_seconds = time.perf_counter() - start

    log_path = (
        STAGING_RESULTS_DIR
        / "logs"
        / f"{output_root.parent.name}_{output_root.name}.txt"
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)

    log_path.write_text(
        completed.stdout + "\n" + completed.stderr,
        encoding="utf-8",
    )

    mesh_path = output_root / "0" / "mesh.glb"
    success = completed.returncode == 0 and mesh_path.is_file()

    return success, elapsed_seconds, log_path


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


def run_generation(summary: dict[str, object]) -> None:
    CORRECTED_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_RESULTS_DIR.mkdir(parents=False, exist_ok=False)
    TEMP_DIR.mkdir(parents=False, exist_ok=False)

    for output_root in ABLATION_ROOTS.values():
        output_root.mkdir(parents=False, exist_ok=False)

    preprocessing = load_module(
        PREPROCESS_SCRIPT,
        "corrected_preprocessing_source",
    )

    print("\nInitializing U2-Net session ...")
    session_start = time.perf_counter()
    session = new_session("u2net")
    session_seconds = time.perf_counter() - session_start
    print(f"U2-Net session initialized in {session_seconds:.3f}s")

    manifest_rows: list[dict] = []
    full_check_rows: list[dict] = []

    try:
        for object_name in OBJECTS:
            print(f"\n[{object_name}]")

            for view_name in VIEWS:
                pair_id = f"{object_name}_{view_name}"
                original_path = find_original(object_name, view_name)

                working_directory = TEMP_DIR / pair_id
                working_directory.mkdir(parents=True, exist_ok=False)

                no_background_path = working_directory / "01_nobg.png"
                cropped_path = working_directory / "02_crop.png"
                padded_path = working_directory / "03_pad.png"
                enhanced_path = working_directory / "04_enhanced.png"
                nobg_ready_path = working_directory / "nobg_only_ready.png"
                crop_pad_ready_path = (
                    working_directory / "nobg_crop_pad_ready.png"
                )
                full_ready_path = (
                    working_directory / "full_reproduced_ready.png"
                )

                preprocessing.stage_background_removal(
                    original_path,
                    no_background_path,
                    session,
                )

                preprocessing.stage_triposr_ready(
                    no_background_path,
                    nobg_ready_path,
                )

                preprocessing.stage_crop(
                    no_background_path,
                    cropped_path,
                )

                preprocessing.stage_padding(
                    cropped_path,
                    padded_path,
                )

                preprocessing.stage_triposr_ready(
                    padded_path,
                    crop_pad_ready_path,
                )

                preprocessing.stage_enhancement(
                    padded_path,
                    enhanced_path,
                )

                preprocessing.stage_triposr_ready(
                    enhanced_path,
                    full_ready_path,
                )

                full_reference_path = expected_full_input(
                    object_name,
                    view_name,
                )

                full_comparison = compare_images(
                    full_ready_path,
                    full_reference_path,
                )

                full_check_rows.append(
                    {
                        "object": object_name,
                        "view": view_name,
                        "pair_id": pair_id,
                        "reproduced_input": str(full_ready_path),
                        "reference_input": str(full_reference_path),
                        **full_comparison,
                    }
                )

                ready_inputs = {
                    "nobg_only": nobg_ready_path,
                    "nobg_crop_pad": crop_pad_ready_path,
                }

                for method, ready_path in ready_inputs.items():
                    output_root = ABLATION_ROOTS[method] / pair_id

                    print(f"Running {method:<15} {pair_id} ...")

                    success, runtime_seconds, log_path = run_triposr(
                        ready_path,
                        output_root,
                    )

                    saved_input = output_root / "0" / "input.png"
                    shutil.copy2(ready_path, saved_input)

                    mesh_path = output_root / "0" / "mesh.glb"

                    manifest_rows.append(
                        {
                            "method": method,
                            "object": object_name,
                            "view": view_name,
                            "pair_id": pair_id,
                            "success": success,
                            "runtime_seconds": runtime_seconds,
                            "input_path": str(saved_input),
                            "mesh_path": str(mesh_path),
                            "mesh_size_mb": (
                                mesh_path.stat().st_size / (1024 * 1024)
                                if mesh_path.is_file()
                                else math.nan
                            ),
                            "log_path": str(log_path),
                        }
                    )

                    if success:
                        print(
                            f"OK  {method:<15} {pair_id} | "
                            f"{runtime_seconds:.2f}s"
                        )
                    else:
                        raise RuntimeError(
                            f"TripoSR failed for {method} {pair_id}. "
                            f"See: {log_path}"
                        )

                shutil.rmtree(
                    working_directory,
                    ignore_errors=True,
                )

        write_csv(
            STAGING_RESULTS_DIR / MANIFEST_CSV,
            manifest_rows,
        )

        write_csv(
            STAGING_RESULTS_DIR / FULL_CHECK_CSV,
            full_check_rows,
        )

        (
            STAGING_RESULTS_DIR / SESSION_TXT
        ).write_text(
            (
                "U2-Net session initialization seconds: "
                f"{session_seconds:.9f}\n"
            ),
            encoding="utf-8",
        )

        expected_meshes = int(summary["expected_ablation_meshes"])
        completed_meshes = sum(
            1 for row in manifest_rows if row["success"]
        )

        if completed_meshes != expected_meshes:
            raise RuntimeError(
                f"Only {completed_meshes}/{expected_meshes} "
                "ablation meshes were generated."
            )

        if len(full_check_rows) != int(summary["expected_full_checks"]):
            raise RuntimeError(
                "Full-input reproduction checks are incomplete."
            )

        STAGING_RESULTS_DIR.rename(RESULTS_DIR)

    except Exception:
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        raise

    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    exact_matches = sum(
        1 for row in full_check_rows if row["pixel_exact_match"]
    )

    finite_differences = [
        float(row["mean_absolute_pixel_difference"])
        for row in full_check_rows
        if math.isfinite(
            float(row["mean_absolute_pixel_difference"])
        )
    ]

    mean_pixel_difference = float(
        np.mean(finite_differences)
    )

    print("\n" + "=" * 94)
    print("CORRECTED ABLATION GENERATION RESULTS")
    print("=" * 94)
    print(
        f"Generated ablation meshes: "
        f"{len(manifest_rows)}/"
        f"{summary['expected_ablation_meshes']}"
    )
    print(
        f"Full-input exact reproductions: "
        f"{exact_matches}/"
        f"{summary['expected_full_checks']}"
    )
    print(
        "Mean full-input absolute pixel difference: "
        f"{mean_pixel_difference:.6f}"
    )
    print(
        f"NoBG outputs: {ABLATION_ROOTS['nobg_only']}"
    )
    print(
        f"NoBG+Crop/Pad outputs: "
        f"{ABLATION_ROOTS['nobg_crop_pad']}"
    )
    print(f"Saved records: {RESULTS_DIR}")
    print("CORRECTED ABLATION GENERATION PASSED.")
    print(
        "Ablation interpretation: the shared TripoSR-ready adapter is "
        "applied to every non-baseline variant. Comparisons isolate the "
        "incremental effect of Crop/Pad and then Enhancement within the "
        "implemented pipeline."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print("=" * 94)
    print("Corrected Cumulative Ablation Generation")
    print("=" * 94)
    print(f"Original images: {len(summary['originals'])}/15")
    print(
        f"Corrected Full reference inputs: "
        f"{len(summary['full_inputs'])}/15"
    )
    print(
        f"Planned ablation meshes: "
        f"{summary['expected_ablation_meshes']}"
    )
    print("Variants:")
    print("  1. NoBG only + shared TripoSR-ready adapter")
    print("  2. NoBG + Crop/Pad + shared TripoSR-ready adapter")
    print("  3. Full corrected method: existing reference outputs")
    print(f"Results: {RESULTS_DIR}")

    if args.check_only:
        print(
            "\nCHECK PASSED: no ablation inputs or meshes were generated."
        )
        print("Run again with --run after reviewing this plan.")
        return

    run_generation(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
