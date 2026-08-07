from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SELECTED_METHOD = "clahe_mild"

PILOT_INPUT_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_enhancement_inputs"
    / SELECTED_METHOD
)

PILOT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "pipeline_v2_enhancement_outputs"
    / SELECTED_METHOD
)

RANKING_CSV = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "pipeline_v2_enhancement_view_consistency"
    / "pipeline_v2_enhancement_ranking.csv"
)

METHOD_SUMMARY_CSV = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
    / "pipeline_v2_enhancement_view_consistency"
    / "pipeline_v2_enhancement_view_method_summary.csv"
)

LOCK_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "pipeline_v2_locked_20260806"
)

STAGING_LOCK_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "_pipeline_v2_locked_20260806_staging"
)

DATASET_ROOT = PROJECT_ROOT / "dataset_expanded"

OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

LOCK_CONFIG_JSON = "pipeline_v2_locked_config.json"
LOCK_README = "README_PIPELINE_V2_LOCK.txt"
SELECTED_EVIDENCE_CSV = "selected_pilot_evidence.csv"

MANIFEST_PATH = (
    DATASET_ROOT
    / "metadata"
    / "expanded_dataset_manifest.csv"
)

DATASET_README = (
    DATASET_ROOT
    / "README_EXPANDED_DATASET.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lock the selected Pipeline V2 configuration and initialize "
            "the independent expanded-dataset workspace."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the selected V2 pilot evidence without writing files.",
    )

    mode.add_argument(
        "--run",
        action="store_true",
        help="Create the locked V2 record and expanded-dataset workspace.",
    )

    return parser.parse_args()


def expected_input_paths() -> list[Path]:
    return [
        PILOT_INPUT_ROOT
        / f"{object_name}_{view_name}"
        / "0"
        / "input.png"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def expected_mesh_paths() -> list[Path]:
    return [
        PILOT_OUTPUT_ROOT
        / f"{object_name}_{view_name}"
        / "0"
        / "mesh.glb"
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def validate_selected_ranking() -> dict[str, str]:
    if not RANKING_CSV.is_file():
        raise FileNotFoundError(
            f"Ranking file was not found: {RANKING_CSV}"
        )

    ranking_rows = read_csv(RANKING_CSV)

    if not ranking_rows:
        raise ValueError(
            f"Ranking file is empty: {RANKING_CSV}"
        )

    top_row = min(
        ranking_rows,
        key=lambda row: int(row["rank"]),
    )

    if top_row["method"] != SELECTED_METHOD:
        raise ValueError(
            "The locked method does not match the pilot ranking. "
            f"Expected {SELECTED_METHOD}, but rank 1 is "
            f"{top_row['method']}."
        )

    return top_row


def validate_selected_summary() -> dict[str, str]:
    if not METHOD_SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            f"Method summary was not found: {METHOD_SUMMARY_CSV}"
        )

    rows = read_csv(METHOD_SUMMARY_CSV)

    matches = [
        row
        for row in rows
        if row.get("group") == "overall"
        and row.get("method") == SELECTED_METHOD
    ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one overall selected-method summary row, "
            f"found {len(matches)}."
        )

    return matches[0]


def validate_destinations_are_new() -> None:
    existing = [
        path
        for path in (
            LOCK_ROOT,
            STAGING_LOCK_ROOT,
            DATASET_ROOT,
        )
        if path.exists()
    ]

    if existing:
        details = "\n".join(
            f"  - {path}"
            for path in existing
        )

        raise FileExistsError(
            "A lock or expanded-dataset workspace already exists and "
            f"will not be overwritten:\n{details}"
        )


def preflight() -> dict[str, object]:
    input_paths = expected_input_paths()
    mesh_paths = expected_mesh_paths()

    missing_inputs = [
        path
        for path in input_paths
        if not path.is_file()
    ]

    missing_meshes = [
        path
        for path in mesh_paths
        if not path.is_file()
    ]

    if missing_inputs:
        details = "\n".join(
            f"  - {path}"
            for path in missing_inputs
        )

        raise FileNotFoundError(
            f"Missing selected pilot input(s):\n{details}"
        )

    if missing_meshes:
        details = "\n".join(
            f"  - {path}"
            for path in missing_meshes
        )

        raise FileNotFoundError(
            f"Missing selected pilot mesh(es):\n{details}"
        )

    top_row = validate_selected_ranking()
    summary_row = validate_selected_summary()

    validate_destinations_are_new()

    return {
        "input_paths": input_paths,
        "mesh_paths": mesh_paths,
        "top_row": top_row,
        "summary_row": summary_row,
    }


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def create_lock_files(
    summary: dict[str, object],
) -> None:
    STAGING_LOCK_ROOT.mkdir(
        parents=True,
        exist_ok=False,
    )

    top_row = summary["top_row"]
    summary_row = summary["summary_row"]

    if not isinstance(top_row, dict):
        raise TypeError("Invalid ranking row.")

    if not isinstance(summary_row, dict):
        raise TypeError("Invalid summary row.")

    lock_config = {
        "locked_at": datetime.now().astimezone().isoformat(),
        "status": "LOCKED_FOR_EXPANDED_DATASET_EVALUATION",
        "selected_method": SELECTED_METHOD,
        "selection_basis": (
            "Lowest mean Chamfer distance on the pilot development set; "
            "Hausdorff 95 and maximum distance used as tie-breakers."
        ),
        "pilot_dataset_role": "development_parameter_screening",
        "expanded_dataset_role": "independent_final_evaluation",
        "pipeline": [
            "U2-Net background removal",
            "Alpha-mask cleanup",
            "Foreground bounding-box crop",
            "Adaptive scaling: longest object side = 80% of 512 pixels",
            "Centering on a 512x512 uniform grey canvas",
            "Foreground-only mild CLAHE",
            "TripoSR with internal background removal disabled",
        ],
        "parameters": {
            "canvas_width": 512,
            "canvas_height": 512,
            "background_rgb": [128, 128, 128],
            "maximum_object_side_ratio": 0.80,
            "alpha_threshold": 16,
            "clahe_clip_limit": 1.5,
            "clahe_tile_grid": [8, 8],
            "sharpening": "disabled",
            "triposr_chunk_size": 1024,
            "triposr_mc_resolution": 96,
            "triposr_device": "cuda:0",
            "triposr_remove_background": False,
        },
        "pilot_selected_metrics": {
            "mean_chamfer_distance": float(
                summary_row["mean_chamfer_distance"]
            ),
            "mean_hausdorff_95": float(
                summary_row["mean_hausdorff_95"]
            ),
            "mean_maximum_distance": float(
                summary_row["mean_maximum_distance"]
            ),
        },
        "important_rule": (
            "Do not retune these parameters using the expanded final dataset."
        ),
    }

    (
        STAGING_LOCK_ROOT
        / LOCK_CONFIG_JSON
    ).write_text(
        json.dumps(
            lock_config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    evidence_rows = [
        {
            "selected_method": SELECTED_METHOD,
            "rank": top_row["rank"],
            "mean_chamfer_distance": top_row[
                "mean_chamfer_distance"
            ],
            "mean_hausdorff_95": top_row[
                "mean_hausdorff_95"
            ],
            "mean_maximum_distance": top_row[
                "mean_maximum_distance"
            ],
            "pilot_input_root": str(PILOT_INPUT_ROOT),
            "pilot_output_root": str(PILOT_OUTPUT_ROOT),
            "ranking_source": str(RANKING_CSV),
            "method_summary_source": str(METHOD_SUMMARY_CSV),
        }
    ]

    write_csv(
        STAGING_LOCK_ROOT / SELECTED_EVIDENCE_CSV,
        list(evidence_rows[0].keys()),
        evidence_rows,
    )

    readme_text = """PIPELINE V2 LOCK RECORD
=======================

Selected pilot configuration:
  ratio_80 + foreground-only mild CLAHE

Locked parameters:
  Canvas: 512 x 512
  Background: RGB (128, 128, 128)
  Maximum object side: 80% of canvas
  CLAHE clip limit: 1.5
  CLAHE tile grid: 8 x 8
  Sharpening: disabled
  TripoSR chunk size: 1024
  TripoSR marching-cubes resolution: 96
  TripoSR internal background removal: disabled

Experimental rule:
  The pilot dataset was used for parameter development and screening.
  The expanded dataset must be evaluated with these parameters unchanged.
  Do not tune the pipeline after observing expanded-dataset results.

Selected pilot files:
  pipeline_v2_enhancement_inputs\\clahe_mild
  pipeline_v2_enhancement_outputs\\clahe_mild
"""

    (
        STAGING_LOCK_ROOT
        / LOCK_README
    ).write_text(
        readme_text,
        encoding="utf-8",
    )


def create_dataset_workspace() -> None:
    directories = (
        DATASET_ROOT / "raw",
        DATASET_ROOT / "processed_baseline",
        DATASET_ROOT / "processed_pipeline_v2",
        DATASET_ROOT / "metadata",
        DATASET_ROOT / "licenses",
        DATASET_ROOT / "quality_control",
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=False,
        )

    manifest_fields = [
        "sample_id",
        "object_id",
        "category",
        "view",
        "source_type",
        "source_dataset",
        "source_file",
        "license_or_permission",
        "included",
        "exclusion_reason",
        "notes",
    ]

    write_csv(
        MANIFEST_PATH,
        manifest_fields,
        [],
    )

    dataset_readme = """EXPANDED DATASET WORKSPACE
==========================

Purpose
-------
This folder is the independent final-evaluation dataset. The Pipeline V2
parameters are already locked and must not be changed after inspecting these
images or their 3D results.

Recommended target
------------------
Minimum: 10 objects x 5 canonical views = 50 images
Preferred: 15-20 objects x 5 canonical views = 75-100 images

Required views per object
-------------------------
front
back
left
right
top

Recommended design
------------------
1. Use objects that are not the same physical mouse, bottle, and shoe used in
   the pilot dataset.
2. Include varied shapes, materials, textures, lighting, and backgrounds.
3. Keep the same five view labels for every object.
4. Record the source and license/permission for every image.
5. Mark exclusions before model generation and preserve the reason.
6. Do not alter the locked Pipeline V2 parameters.

Raw-image naming
----------------
Preferred format:
  <object_id>_<view>.<extension>

Examples:
  mug01_front.jpg
  mug01_back.jpg
  mug01_left.jpg
  mug01_right.jpg
  mug01_top.jpg

Manifest
--------
Complete:
  metadata\\expanded_dataset_manifest.csv

Do not place generated meshes inside raw/.
"""

    DATASET_README.write_text(
        dataset_readme,
        encoding="utf-8",
    )

    (
        DATASET_ROOT
        / "raw"
        / "PLACE_RAW_IMAGES_HERE.txt"
    ).write_text(
        (
            "Place expanded-dataset source images here. "
            "Use one object_id with five canonical views.\n"
        ),
        encoding="utf-8",
    )


def publish_lock() -> None:
    if LOCK_ROOT.exists():
        raise FileExistsError(
            f"Lock output appeared during execution: {LOCK_ROOT}"
        )

    STAGING_LOCK_ROOT.rename(
        LOCK_ROOT
    )


def run_initialization(
    summary: dict[str, object],
) -> None:
    try:
        create_lock_files(summary)
        create_dataset_workspace()
        publish_lock()

    except Exception:
        if STAGING_LOCK_ROOT.exists():
            shutil.rmtree(
                STAGING_LOCK_ROOT,
                ignore_errors=True,
            )

        if DATASET_ROOT.exists():
            shutil.rmtree(
                DATASET_ROOT,
                ignore_errors=True,
            )

        raise

    print("\n" + "=" * 96)
    print("PIPELINE V2 LOCK AND EXPANDED-DATASET INITIALIZATION")
    print("=" * 96)
    print("Selected configuration: clahe_mild")
    print("Selected pilot inputs: 15/15")
    print("Selected pilot meshes: 15/15")
    print(f"Lock record: {LOCK_ROOT}")
    print(f"Expanded dataset workspace: {DATASET_ROOT}")
    print(
        "PIPELINE V2 LOCK AND EXPANDED-DATASET INITIALIZATION PASSED."
    )


def main() -> None:
    args = parse_args()
    summary = preflight()

    print("=" * 96)
    print("Pipeline V2 Lock and Expanded-Dataset Initialization")
    print("=" * 96)
    print(
        f"Selected pilot inputs: "
        f"{len(summary['input_paths'])}/15"
    )
    print(
        f"Selected pilot meshes: "
        f"{len(summary['mesh_paths'])}/15"
    )
    print("Pilot ranking winner: clahe_mild")
    print("Locked ratio: 80%")
    print("Locked CLAHE clip limit: 1.5")
    print("Sharpening: disabled")
    print(f"Lock output: {LOCK_ROOT}")
    print(f"Expanded dataset root: {DATASET_ROOT}")

    if args.check_only:
        print(
            "\nCHECK PASSED: no lock record or expanded-dataset "
            "workspace was created."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_initialization(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"\nERROR: {error}",
            file=sys.stderr,
        )
        raise
