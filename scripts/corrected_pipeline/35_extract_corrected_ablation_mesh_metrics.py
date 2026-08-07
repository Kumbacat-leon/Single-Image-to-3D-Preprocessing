from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import trimesh


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

CORRECTED_ROOT = (
    PROJECT_ROOT
    / "comparison_results"
    / "corrected_20260804_final"
)

OUTPUT_DIR = CORRECTED_ROOT / "ablation_mesh_metrics"
STAGING_DIR = CORRECTED_ROOT / "_ablation_mesh_metrics_staging"

METRICS_CSV = "ablation_mesh_metrics.csv"
METHOD_SUMMARY_CSV = "ablation_mesh_method_summary.csv"
OBJECT_SUMMARY_CSV = "ablation_mesh_object_summary.csv"
MANIFEST_JSON = "ablation_mesh_metrics_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract corrected mesh metrics for Baseline, NoBG-only, "
            "NoBG+Crop/Pad, and Final Proposed outputs."
        )
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all 60 mesh paths without extracting metrics.",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="Extract and publish corrected ablation mesh metrics.",
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
                "Corrected ablation mesh-metric output already exists "
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

    return {
        "method_paths": method_paths,
        "expected_meshes": (
            len(METHOD_ROOTS)
            * len(OBJECTS)
            * len(VIEWS)
        ),
    }


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(
        mesh_path,
        force="scene",
    )

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            loaded.geometry[name]
            for name in sorted(loaded.geometry)
            if isinstance(
                loaded.geometry[name],
                trimesh.Trimesh,
            )
        ]

        if not geometries:
            raise ValueError(
                f"No mesh geometry found in: {mesh_path}"
            )

        return trimesh.util.concatenate(
            geometries
        )

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    raise TypeError(
        f"Unsupported geometry type in {mesh_path}: "
        f"{type(loaded)}"
    )


def count_degenerate_faces(
    mesh: trimesh.Trimesh,
) -> int:
    areas = np.asarray(
        mesh.area_faces,
        dtype=float,
    )

    return int(
        np.sum(areas <= 1e-12)
    )


def extract_metrics(
    mesh: trimesh.Trimesh,
    mesh_path: Path,
    method: str,
    object_name: str,
    view_name: str,
) -> dict:
    extents = np.asarray(
        mesh.extents,
        dtype=float,
    )

    connected_components = mesh.split(
        only_watertight=False,
    )

    volume = (
        float(mesh.volume)
        if mesh.is_watertight
        else math.nan
    )

    return {
        "method": method,
        "object": object_name,
        "view": view_name,
        "pair_id": f"{object_name}_{view_name}",
        "mesh_path": str(mesh_path),
        "file_size_mb": (
            mesh_path.stat().st_size
            / (1024 * 1024)
        ),
        "vertex_count": int(
            len(mesh.vertices)
        ),
        "face_count": int(
            len(mesh.faces)
        ),
        "connected_components": int(
            len(connected_components)
        ),
        "degenerate_faces": count_degenerate_faces(
            mesh
        ),
        "surface_area": float(
            mesh.area
        ),
        "volume": volume,
        "bounding_box_volume": float(
            np.prod(extents)
        ),
        "extent_x": float(
            extents[0]
        ),
        "extent_y": float(
            extents[1]
        ),
        "extent_z": float(
            extents[2]
        ),
        "watertight": bool(
            mesh.is_watertight
        ),
        "winding_consistent": bool(
            mesh.is_winding_consistent
        ),
        "euler_number": int(
            mesh.euler_number
        ),
    }


def mean_numeric(
    rows: list[dict],
    field: str,
) -> float:
    values = np.asarray(
        [
            float(row[field])
            for row in rows
            if math.isfinite(
                float(row[field])
            )
        ],
        dtype=float,
    )

    if values.size == 0:
        return math.nan

    return float(
        np.mean(values)
    )


def create_summary_row(
    rows: list[dict],
    group: str,
    method: str,
) -> dict:
    selected = [
        row
        for row in rows
        if row["method"] == method
    ]

    return {
        "group": group,
        "method": method,
        "mesh_count": len(selected),
        "mean_file_size_mb": mean_numeric(
            selected,
            "file_size_mb",
        ),
        "mean_vertex_count": mean_numeric(
            selected,
            "vertex_count",
        ),
        "mean_face_count": mean_numeric(
            selected,
            "face_count",
        ),
        "mean_connected_components": mean_numeric(
            selected,
            "connected_components",
        ),
        "mean_degenerate_faces": mean_numeric(
            selected,
            "degenerate_faces",
        ),
        "mean_surface_area": mean_numeric(
            selected,
            "surface_area",
        ),
        "mean_bounding_box_volume": mean_numeric(
            selected,
            "bounding_box_volume",
        ),
        "watertight_meshes": sum(
            1
            for row in selected
            if bool(row["watertight"])
        ),
        "winding_consistent_meshes": sum(
            1
            for row in selected
            if bool(
                row["winding_consistent"]
            )
        ),
    }


def create_method_summary_rows(
    rows: list[dict],
) -> list[dict]:
    return [
        create_summary_row(
            rows,
            "overall",
            method,
        )
        for method in METHOD_ROOTS
    ]


def create_object_summary_rows(
    rows: list[dict],
) -> list[dict]:
    output_rows: list[dict] = []

    for object_name in OBJECTS:
        object_rows = [
            row
            for row in rows
            if row["object"] == object_name
        ]

        for method in METHOD_ROOTS:
            output_rows.append(
                create_summary_row(
                    object_rows,
                    object_name,
                    method,
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


def validate_rows(
    rows: list[dict],
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"Extracted {len(rows)} metric rows; "
            f"expected {expected_count}."
        )

    keys = [
        (
            str(row["method"]),
            str(row["object"]),
            str(row["view"]),
        )
        for row in rows
    ]

    if len(keys) != len(set(keys)):
        raise ValueError(
            "Duplicate method/object/view rows were detected."
        )


def publish() -> None:
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output appeared during execution: {OUTPUT_DIR}"
        )

    STAGING_DIR.rename(
        OUTPUT_DIR
    )


def run_extraction(
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

    rows: list[dict] = []

    try:
        for method, root in METHOD_ROOTS.items():
            print(f"\n[{method}]")

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

                    metrics = extract_metrics(
                        mesh,
                        mesh_path,
                        method,
                        object_name,
                        view_name,
                    )

                    rows.append(
                        metrics
                    )

                    print(
                        f"OK  {object_name}_{view_name:<12} "
                        f"vertices={metrics['vertex_count']:<7} "
                        f"faces={metrics['face_count']:<7} "
                        f"components={metrics['connected_components']} "
                        f"degenerate={metrics['degenerate_faces']}"
                    )

        expected_count = int(
            preflight_summary[
                "expected_meshes"
            ]
        )

        validate_rows(
            rows,
            expected_count,
        )

        method_summary_rows = (
            create_method_summary_rows(
                rows
            )
        )

        object_summary_rows = (
            create_object_summary_rows(
                rows
            )
        )

        write_csv(
            STAGING_DIR / METRICS_CSV,
            rows,
        )

        write_csv(
            STAGING_DIR / METHOD_SUMMARY_CSV,
            method_summary_rows,
        )

        write_csv(
            STAGING_DIR / OBJECT_SUMMARY_CSV,
            object_summary_rows,
        )

        manifest = {
            "created_at": (
                datetime.now()
                .astimezone()
                .isoformat()
            ),
            "project_root": str(
                PROJECT_ROOT
            ),
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "trimesh_version": getattr(
                trimesh,
                "__version__",
                "unknown",
            ),
            "methods": {
                method: str(root)
                for method, root in METHOD_ROOTS.items()
            },
            "objects": list(
                OBJECTS
            ),
            "views": list(
                VIEWS
            ),
            "mesh_count": len(
                rows
            ),
        }

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

        publish()

    except Exception:
        if STAGING_DIR.exists():
            shutil.rmtree(
                STAGING_DIR,
                ignore_errors=True,
            )
        raise

    print("\n" + "=" * 92)
    print("CORRECTED ABLATION MESH-METRIC RESULTS")
    print("=" * 92)
    print(
        f"Metric rows: "
        f"{len(rows)}/"
        f"{preflight_summary['expected_meshes']}"
    )

    for summary_row in method_summary_rows:
        print(
            f"{summary_row['method']:<16} "
            f"components="
            f"{float(summary_row['mean_connected_components']):.3f} | "
            f"degenerate="
            f"{float(summary_row['mean_degenerate_faces']):.3f} | "
            f"vertices="
            f"{float(summary_row['mean_vertex_count']):.1f}"
        )

    print(f"\nSaved: {OUTPUT_DIR}")
    print(
        "CORRECTED ABLATION MESH-METRIC EXTRACTION PASSED."
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
            "Invalid method path summary."
        )

    print("=" * 92)
    print("Corrected Ablation Mesh-Metric Extraction")
    print("=" * 92)

    for method in METHOD_ROOTS:
        print(
            f"{method:<16} "
            f"{len(method_paths[method])}/15"
        )

    print(
        f"Expected mesh metrics: "
        f"{summary['expected_meshes']}"
    )
    print(
        f"Output: {OUTPUT_DIR}"
    )

    if args.check_only:
        print(
            "\nCHECK PASSED: no mesh metrics were extracted."
        )
        print(
            "Run again with --run after reviewing this plan."
        )
        return

    run_extraction(
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
