from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.stats import wilcoxon


OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")

BASELINE_FOLDER = "baseline_outputs"
CORRECTED_PROPOSED_FOLDER = (
    "final_proposed_outputs_corrected_20260804_final"
)
REFERENCE_RESULTS_FOLDER = "comparison_results"
CORRECTED_RESULTS_FOLDER = Path(
    "comparison_results"
) / "corrected_20260804_final"

METRICS_FILENAME = "final_mesh_metrics.csv"
PAIRWISE_FILENAME = "final_mesh_pairwise_comparison.csv"
SUMMARY_FILENAME = "final_mesh_comparison_summary.csv"

INTEGER_METRICS = (
    "vertex_count",
    "face_count",
    "connected_components",
    "degenerate_faces",
    "euler_number",
)
FLOAT_METRICS = (
    "file_size_mb",
    "surface_area",
    "volume",
    "bounding_box_volume",
    "extent_x",
    "extent_y",
    "extent_z",
)
BOOLEAN_METRICS = (
    "watertight",
    "winding_consistent",
)


def sha256_file(path: Path) -> str:
    """Return the uppercase SHA-256 digest of one file."""
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def find_project_root(explicit_root: str | None) -> Path:
    """Locate the project from an argument or common script locations."""
    if explicit_root:
        candidates = [Path(explicit_root).expanduser()]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir.parent,
            script_dir,
            Path.cwd(),
            Path.cwd().parent,
        ]

    checked: list[Path] = []

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved in checked:
            continue

        checked.append(resolved)

        if (
            (resolved / BASELINE_FOLDER).is_dir()
            and (resolved / CORRECTED_PROPOSED_FOLDER).is_dir()
            and (
                resolved
                / REFERENCE_RESULTS_FOLDER
                / METRICS_FILENAME
            ).is_file()
        ):
            return resolved

    searched = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not locate the project root. Searched:\n" + searched
    )


def expected_case_keys() -> set[tuple[str, str]]:
    """Return the fixed 3-object by 5-view case key set."""
    return {
        (object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    }


def mesh_path_for(
    root: Path,
    object_name: str,
    view_name: str,
) -> Path:
    """Return the expected GLB path for one object-view case."""
    return root / f"{object_name}_{view_name}" / "0" / "mesh.glb"


def validate_mesh_inventory(root: Path, label: str) -> list[Path]:
    """Require exactly one non-empty expected mesh for every case."""
    expected_paths = [
        mesh_path_for(root, object_name, view_name)
        for object_name in OBJECTS
        for view_name in VIEWS
    ]

    missing = [path for path in expected_paths if not path.is_file()]
    empty = [
        path
        for path in expected_paths
        if path.is_file() and path.stat().st_size <= 0
    ]

    discovered = {
        path.resolve()
        for path in root.rglob("mesh.glb")
        if path.is_file()
    }
    expected = {path.resolve() for path in expected_paths}
    unexpected = sorted(discovered - expected)

    if missing or empty or unexpected:
        details = [
            f"{label} inventory validation failed:",
            f"  missing={len(missing)}",
            f"  empty={len(empty)}",
            f"  unexpected={len(unexpected)}",
        ]
        details.extend(f"  MISSING: {path}" for path in missing)
        details.extend(f"  EMPTY: {path}" for path in empty)
        details.extend(f"  UNEXPECTED: {path}" for path in unexpected)
        raise RuntimeError("\n".join(details))

    return expected_paths


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a non-empty UTF-8 CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"CSV file was not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    return rows


def validate_corrected_manifest(
    proposed_root: Path,
) -> int:
    """Verify all corrected meshes against their generation manifest."""
    manifest_path = proposed_root / "generation_manifest.csv"
    rows = read_csv_rows(manifest_path)
    required_fields = {"sample_id", "mesh_bytes", "mesh_sha256"}
    missing_fields = required_fields - set(rows[0])

    if missing_fields:
        raise ValueError(
            "Corrected generation manifest is missing fields: "
            f"{sorted(missing_fields)}"
        )

    indexed: dict[str, dict[str, str]] = {}

    for row in rows:
        sample_id = row["sample_id"]

        if sample_id in indexed:
            raise ValueError(
                f"Duplicate sample_id in generation manifest: {sample_id}"
            )

        indexed[sample_id] = row

    expected_ids = {
        f"{object_name}_{view_name}"
        for object_name in OBJECTS
        for view_name in VIEWS
    }

    if set(indexed) != expected_ids:
        missing = sorted(expected_ids - set(indexed))
        unexpected = sorted(set(indexed) - expected_ids)
        raise ValueError(
            "Corrected generation manifest case mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for sample_id in sorted(expected_ids):
        row = indexed[sample_id]
        mesh_path = proposed_root / sample_id / "0" / "mesh.glb"

        if not mesh_path.is_file():
            raise FileNotFoundError(f"Corrected mesh was not found: {mesh_path}")

        actual_bytes = mesh_path.stat().st_size
        expected_bytes = int(row["mesh_bytes"])

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"Mesh size does not match the manifest for {sample_id}: "
                f"actual={actual_bytes}, expected={expected_bytes}"
            )

        actual_hash = sha256_file(mesh_path)
        expected_hash = row["mesh_sha256"].strip().upper()

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Mesh SHA-256 does not match the manifest for {sample_id}."
            )

    return len(rows)


def load_reference_baselines(
    reference_csv: Path,
) -> dict[tuple[str, str], dict[str, str]]:
    """Load the unchanged Baseline rows from the original metric table."""
    rows = read_csv_rows(reference_csv)
    required_fields = {
        "method",
        "object",
        "view",
        *INTEGER_METRICS,
        *FLOAT_METRICS,
        *BOOLEAN_METRICS,
    }
    missing_fields = required_fields - set(rows[0])

    if missing_fields:
        raise ValueError(
            "Reference metric CSV is missing fields: "
            f"{sorted(missing_fields)}"
        )

    baseline_rows = [row for row in rows if row["method"] == "baseline"]
    indexed: dict[tuple[str, str], dict[str, str]] = {}

    for row in baseline_rows:
        key = (row["object"], row["view"])

        if key in indexed:
            raise ValueError(f"Duplicate Baseline row in reference CSV: {key}")

        indexed[key] = row

    expected = expected_case_keys()

    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        unexpected = sorted(set(indexed) - expected)
        raise ValueError(
            "Reference Baseline rows do not match the 15-case plan: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return indexed


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load and concatenate mesh geometry exactly as in the original study."""
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]

        if not geometries:
            raise ValueError(f"No mesh geometry found in: {mesh_path}")

        return trimesh.util.concatenate(geometries)

    if isinstance(loaded, trimesh.Trimesh):
        return loaded

    raise TypeError(f"Unsupported geometry type: {type(loaded)}")


def count_degenerate_faces(mesh: trimesh.Trimesh) -> int:
    """Count faces whose area is at or below the original threshold."""
    areas = np.asarray(mesh.area_faces, dtype=float)
    return int(np.sum(areas <= 1e-12))


def extract_metrics(
    mesh: trimesh.Trimesh,
    mesh_path: Path,
    method: str,
    object_name: str,
    view_name: str,
) -> dict[str, object]:
    """Extract the original study's geometric and topology metrics."""
    extents = np.asarray(mesh.extents, dtype=float)
    components = mesh.split(only_watertight=False)

    return {
        "method": method,
        "object": object_name,
        "view": view_name,
        "mesh_path": str(mesh_path),
        "file_size_mb": mesh_path.stat().st_size / (1024 * 1024),
        "vertex_count": len(mesh.vertices),
        "face_count": len(mesh.faces),
        "connected_components": len(components),
        "degenerate_faces": count_degenerate_faces(mesh),
        "surface_area": float(mesh.area),
        "volume": float(mesh.volume) if mesh.is_watertight else math.nan,
        "bounding_box_volume": float(np.prod(extents)),
        "extent_x": extents[0],
        "extent_y": extents[1],
        "extent_z": extents[2],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "euler_number": int(mesh.euler_number),
    }


def extract_all_metrics(
    baseline_root: Path,
    proposed_root: Path,
) -> list[dict[str, object]]:
    """Extract all 30 rows in the original deterministic case order."""
    rows: list[dict[str, object]] = []
    method_roots = (
        ("baseline", baseline_root),
        ("final_proposed", proposed_root),
    )
    total = len(method_roots) * len(OBJECTS) * len(VIEWS)
    index = 0

    for method, root in method_roots:
        for object_name in OBJECTS:
            for view_name in VIEWS:
                index += 1
                mesh_path = mesh_path_for(root, object_name, view_name)
                mesh = load_mesh(mesh_path)
                metrics = extract_metrics(
                    mesh,
                    mesh_path,
                    method,
                    object_name,
                    view_name,
                )
                rows.append(metrics)
                print(
                    f"[{index:02d}/{total}] {method:<14} "
                    f"{object_name}_{view_name:<12} "
                    f"vertices={metrics['vertex_count']:<7} "
                    f"faces={metrics['face_count']:<7} "
                    f"components={metrics['connected_components']}"
                )

    return rows


def to_bool(value: object) -> bool:
    """Convert a CSV-compatible Boolean value to bool."""
    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Unsupported Boolean value: {value!r}")


def to_float(value: object) -> float:
    """Convert a CSV-compatible value to float, preserving NaN."""
    if value is None:
        return math.nan

    normalized = str(value).strip()

    if not normalized or normalized.lower() == "nan":
        return math.nan

    return float(normalized)


def to_int(value: object) -> int:
    """Convert a CSV-compatible numeric value to int."""
    return int(float(str(value)))


def floats_match(left: object, right: object) -> bool:
    """Compare finite values tightly while treating two NaNs as equal."""
    left_value = to_float(left)
    right_value = to_float(right)

    if math.isnan(left_value) and math.isnan(right_value):
        return True

    return math.isclose(
        left_value,
        right_value,
        rel_tol=1e-10,
        abs_tol=1e-12,
    )


def validate_baseline_regression(
    computed_rows: list[dict[str, object]],
    reference_rows: dict[tuple[str, str], dict[str, str]],
) -> int:
    """Prove that the metric implementation reproduces old Baseline rows."""
    baseline_rows = [
        row for row in computed_rows if row["method"] == "baseline"
    ]
    mismatches: list[str] = []

    for row in baseline_rows:
        key = (str(row["object"]), str(row["view"]))
        reference = reference_rows[key]

        for metric in INTEGER_METRICS:
            if to_int(row[metric]) != to_int(reference[metric]):
                mismatches.append(
                    f"{key[0]}_{key[1]} {metric}: "
                    f"computed={row[metric]!r}, reference={reference[metric]!r}"
                )

        for metric in FLOAT_METRICS:
            if not floats_match(row[metric], reference[metric]):
                mismatches.append(
                    f"{key[0]}_{key[1]} {metric}: "
                    f"computed={row[metric]!r}, reference={reference[metric]!r}"
                )

        for metric in BOOLEAN_METRICS:
            if to_bool(row[metric]) != to_bool(reference[metric]):
                mismatches.append(
                    f"{key[0]}_{key[1]} {metric}: "
                    f"computed={row[metric]!r}, reference={reference[metric]!r}"
                )

    if mismatches:
        preview = "\n".join(f"  - {item}" for item in mismatches[:20])
        suffix = "" if len(mismatches) <= 20 else "\n  - ..."
        raise RuntimeError(
            "Baseline metric regression failed. The corrected evaluation "
            "was not written.\n"
            f"{preview}{suffix}"
        )

    if len(baseline_rows) != 15:
        raise RuntimeError(
            f"Baseline regression checked {len(baseline_rows)} rows; expected 15."
        )

    return len(baseline_rows)


def percentage_change(
    baseline_value: float,
    proposed_value: float,
) -> float:
    """Calculate percentage change relative to the Baseline value."""
    if (
        not math.isfinite(baseline_value)
        or not math.isfinite(proposed_value)
        or abs(baseline_value) <= 1e-12
    ):
        return math.nan

    return (proposed_value - baseline_value) / baseline_value * 100.0


def format_optional(value: float) -> str:
    """Format a float while preserving NaN as an empty CSV field."""
    if not math.isfinite(value):
        return ""

    return f"{value:.8f}"


def lower_is_better(
    baseline_value: int,
    proposed_value: int,
) -> str:
    """Classify a metric for which a lower value is preferred."""
    if proposed_value < baseline_value:
        return "Improved"

    if proposed_value > baseline_value:
        return "Worsened"

    return "Unchanged"


def boolean_quality_result(
    baseline_value: bool,
    proposed_value: bool,
) -> str:
    """Classify a Boolean quality property where True is preferred."""
    if proposed_value and not baseline_value:
        return "Improved"

    if baseline_value and not proposed_value:
        return "Worsened"

    return "Unchanged"


def create_pairwise_row(
    baseline: dict[str, object],
    proposed: dict[str, object],
    object_name: str,
    view_name: str,
) -> dict[str, object]:
    """Create one Baseline-versus-corrected-Proposed comparison row."""
    baseline_vertices = to_int(baseline["vertex_count"])
    proposed_vertices = to_int(proposed["vertex_count"])
    baseline_faces = to_int(baseline["face_count"])
    proposed_faces = to_int(proposed["face_count"])
    baseline_components = to_int(baseline["connected_components"])
    proposed_components = to_int(proposed["connected_components"])
    baseline_degenerate = to_int(baseline["degenerate_faces"])
    proposed_degenerate = to_int(proposed["degenerate_faces"])
    baseline_file_size = to_float(baseline["file_size_mb"])
    proposed_file_size = to_float(proposed["file_size_mb"])
    baseline_area = to_float(baseline["surface_area"])
    proposed_area = to_float(proposed["surface_area"])
    baseline_bbox = to_float(baseline["bounding_box_volume"])
    proposed_bbox = to_float(proposed["bounding_box_volume"])
    baseline_watertight = to_bool(baseline["watertight"])
    proposed_watertight = to_bool(proposed["watertight"])
    baseline_winding = to_bool(baseline["winding_consistent"])
    proposed_winding = to_bool(proposed["winding_consistent"])

    return {
        "object": object_name,
        "view": view_name,
        "baseline_mesh_path": baseline["mesh_path"],
        "final_proposed_mesh_path": proposed["mesh_path"],
        "baseline_file_size_mb": baseline_file_size,
        "final_proposed_file_size_mb": proposed_file_size,
        "file_size_change_percent": format_optional(
            percentage_change(baseline_file_size, proposed_file_size)
        ),
        "baseline_vertex_count": baseline_vertices,
        "final_proposed_vertex_count": proposed_vertices,
        "vertex_change": proposed_vertices - baseline_vertices,
        "vertex_change_percent": format_optional(
            percentage_change(float(baseline_vertices), float(proposed_vertices))
        ),
        "baseline_face_count": baseline_faces,
        "final_proposed_face_count": proposed_faces,
        "face_change": proposed_faces - baseline_faces,
        "face_change_percent": format_optional(
            percentage_change(float(baseline_faces), float(proposed_faces))
        ),
        "baseline_connected_components": baseline_components,
        "final_proposed_connected_components": proposed_components,
        "component_change": proposed_components - baseline_components,
        "component_result": lower_is_better(
            baseline_components,
            proposed_components,
        ),
        "baseline_degenerate_faces": baseline_degenerate,
        "final_proposed_degenerate_faces": proposed_degenerate,
        "degenerate_face_change": proposed_degenerate - baseline_degenerate,
        "degenerate_face_result": lower_is_better(
            baseline_degenerate,
            proposed_degenerate,
        ),
        "baseline_surface_area": baseline_area,
        "final_proposed_surface_area": proposed_area,
        "surface_area_change_percent": format_optional(
            percentage_change(baseline_area, proposed_area)
        ),
        "baseline_bounding_box_volume": baseline_bbox,
        "final_proposed_bounding_box_volume": proposed_bbox,
        "bounding_box_volume_change_percent": format_optional(
            percentage_change(baseline_bbox, proposed_bbox)
        ),
        "baseline_watertight": baseline_watertight,
        "final_proposed_watertight": proposed_watertight,
        "watertight_result": boolean_quality_result(
            baseline_watertight,
            proposed_watertight,
        ),
        "baseline_winding_consistent": baseline_winding,
        "final_proposed_winding_consistent": proposed_winding,
        "winding_result": boolean_quality_result(
            baseline_winding,
            proposed_winding,
        ),
        "baseline_euler_number": to_int(baseline["euler_number"]),
        "final_proposed_euler_number": to_int(proposed["euler_number"]),
    }


def create_pairwise_rows(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Create and strictly validate all 15 paired comparison rows."""
    indexed: dict[tuple[str, str, str], dict[str, object]] = {}

    for row in metric_rows:
        key = (
            str(row["method"]),
            str(row["object"]),
            str(row["view"]),
        )

        if key in indexed:
            raise ValueError(f"Duplicate metric row: {key}")

        indexed[key] = row

    expected = {
        (method, object_name, view_name)
        for method in ("baseline", "final_proposed")
        for object_name in OBJECTS
        for view_name in VIEWS
    }

    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        unexpected = sorted(set(indexed) - expected)
        raise ValueError(
            "Metric row pairing failed: "
            f"missing={missing}, unexpected={unexpected}"
        )

    return [
        create_pairwise_row(
            indexed[("baseline", object_name, view_name)],
            indexed[("final_proposed", object_name, view_name)],
            object_name,
            view_name,
        )
        for object_name in OBJECTS
        for view_name in VIEWS
    ]


def run_paired_wilcoxon(
    baseline_values: list[float],
    proposed_values: list[float],
) -> tuple[float, float]:
    """Run the original two-sided paired Wilcoxon test."""
    baseline_array = np.asarray(baseline_values, dtype=float)
    proposed_array = np.asarray(proposed_values, dtype=float)
    valid = np.isfinite(baseline_array) & np.isfinite(proposed_array)
    baseline_array = baseline_array[valid]
    proposed_array = proposed_array[valid]

    if baseline_array.size == 0:
        return math.nan, math.nan

    differences = proposed_array - baseline_array

    if np.all(np.abs(differences) <= 1e-12):
        return 0.0, 1.0

    result = wilcoxon(
        proposed_array,
        baseline_array,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )
    return float(result.statistic), float(result.pvalue)


def create_summary_rows(
    pairwise_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Create the original overall and object-level topology summaries."""
    groups = {
        "overall": pairwise_rows,
        **{
            object_name: [
                row
                for row in pairwise_rows
                if row["object"] == object_name
            ]
            for object_name in OBJECTS
        },
    }
    summary_rows: list[dict[str, object]] = []

    for group_name, rows in groups.items():
        component_counts = Counter(str(row["component_result"]) for row in rows)
        degenerate_counts = Counter(
            str(row["degenerate_face_result"]) for row in rows
        )
        watertight_counts = Counter(
            str(row["watertight_result"]) for row in rows
        )
        winding_counts = Counter(str(row["winding_result"]) for row in rows)

        baseline_components = [
            float(row["baseline_connected_components"]) for row in rows
        ]
        proposed_components = [
            float(row["final_proposed_connected_components"]) for row in rows
        ]
        baseline_degenerate = [
            float(row["baseline_degenerate_faces"]) for row in rows
        ]
        proposed_degenerate = [
            float(row["final_proposed_degenerate_faces"]) for row in rows
        ]

        component_stat, component_p = run_paired_wilcoxon(
            baseline_components,
            proposed_components,
        )
        degenerate_stat, degenerate_p = run_paired_wilcoxon(
            baseline_degenerate,
            proposed_degenerate,
        )

        summary_rows.append(
            {
                "group": group_name,
                "pair_count": len(rows),
                "baseline_mean_components": float(np.mean(baseline_components)),
                "final_proposed_mean_components": float(
                    np.mean(proposed_components)
                ),
                "component_improved_pairs": component_counts["Improved"],
                "component_worsened_pairs": component_counts["Worsened"],
                "component_unchanged_pairs": component_counts["Unchanged"],
                "component_wilcoxon_statistic": component_stat,
                "component_wilcoxon_p_value": component_p,
                "baseline_mean_degenerate_faces": float(
                    np.mean(baseline_degenerate)
                ),
                "final_proposed_mean_degenerate_faces": float(
                    np.mean(proposed_degenerate)
                ),
                "degenerate_improved_pairs": degenerate_counts["Improved"],
                "degenerate_worsened_pairs": degenerate_counts["Worsened"],
                "degenerate_unchanged_pairs": degenerate_counts["Unchanged"],
                "degenerate_wilcoxon_statistic": degenerate_stat,
                "degenerate_wilcoxon_p_value": degenerate_p,
                "watertight_improved_pairs": watertight_counts["Improved"],
                "watertight_worsened_pairs": watertight_counts["Worsened"],
                "watertight_unchanged_pairs": watertight_counts["Unchanged"],
                "winding_improved_pairs": winding_counts["Improved"],
                "winding_worsened_pairs": winding_counts["Worsened"],
                "winding_unchanged_pairs": winding_counts["Unchanged"],
            }
        )

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a non-empty CSV with the original UTF-8 BOM convention."""
    if not rows:
        raise ValueError(f"No rows were available for: {path}")

    with path.open("x", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs_atomically(
    results_dir: Path,
    metric_rows: list[dict[str, object]],
    pairwise_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    """Publish all three new CSVs without touching prior result tables."""
    if results_dir.exists():
        raise FileExistsError(
            f"Corrected result directory already exists: {results_dir}\n"
            "No files were overwritten."
        )

    results_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = results_dir.with_name(
        f".{results_dir.name}.staging-{os.getpid()}"
    )

    if staging_dir.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_dir}")

    staging_dir.mkdir(parents=False)

    try:
        write_csv(staging_dir / METRICS_FILENAME, metric_rows)
        write_csv(staging_dir / PAIRWISE_FILENAME, pairwise_rows)
        write_csv(staging_dir / SUMMARY_FILENAME, summary_rows)
        staging_dir.rename(results_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    """Parse arguments; read-only validation is the default action."""
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Baseline-versus-corrected-Proposed mesh metrics "
            "without running TripoSR."
        )
    )
    parser.add_argument(
        "--project-root",
        help="Project root containing the mesh output directories.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--run",
        action="store_true",
        help="Extract metrics and write a new corrected result directory.",
    )
    action.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all inputs without loading meshes or writing results.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate inputs, then optionally compute all corrected metric tables."""
    args = parse_args()
    project_root = find_project_root(args.project_root)
    baseline_root = project_root / BASELINE_FOLDER
    proposed_root = project_root / CORRECTED_PROPOSED_FOLDER
    reference_csv = (
        project_root
        / REFERENCE_RESULTS_FOLDER
        / METRICS_FILENAME
    )
    results_dir = project_root / CORRECTED_RESULTS_FOLDER

    baseline_paths = validate_mesh_inventory(baseline_root, "Baseline")
    proposed_paths = validate_mesh_inventory(
        proposed_root,
        "Corrected Proposed",
    )
    manifest_count = validate_corrected_manifest(proposed_root)
    reference_rows = load_reference_baselines(reference_csv)

    print("=" * 80)
    print("Corrected Final Mesh Metric Preflight")
    print("=" * 80)
    print(f"Project root: {project_root}")
    print(f"Baseline root: {baseline_root}")
    print(f"Corrected Proposed root: {proposed_root}")
    print(f"Reference metrics: {reference_csv}")
    print(f"New results: {results_dir}")
    print(f"Baseline meshes: {len(baseline_paths)}/15")
    print(f"Corrected Proposed meshes: {len(proposed_paths)}/15")
    print(f"Manifest meshes verified: {manifest_count}/15")
    print(f"Reference Baseline rows: {len(reference_rows)}/15")

    if not args.run:
        if results_dir.exists():
            raise FileExistsError(
                f"Corrected result directory already exists: {results_dir}"
            )

        print("\nCHECK PASSED: no mesh was modified and no CSV was written.")
        print("Run again with --run to compute the corrected metric tables.")
        return

    if results_dir.exists():
        raise FileExistsError(
            f"Corrected result directory already exists: {results_dir}\n"
            "No files were overwritten."
        )

    print("\nExtracting metrics from existing GLB files only...")
    metric_rows = extract_all_metrics(baseline_root, proposed_root)
    regression_count = validate_baseline_regression(
        metric_rows,
        reference_rows,
    )
    pairwise_rows = create_pairwise_rows(metric_rows)
    summary_rows = create_summary_rows(pairwise_rows)

    if len(metric_rows) != 30 or len(pairwise_rows) != 15:
        raise RuntimeError(
            "Corrected metric output count validation failed: "
            f"metrics={len(metric_rows)}, pairs={len(pairwise_rows)}"
        )

    write_outputs_atomically(
        results_dir,
        metric_rows,
        pairwise_rows,
        summary_rows,
    )

    overall = next(row for row in summary_rows if row["group"] == "overall")
    print("\n" + "=" * 80)
    print(f"Metric rows: {len(metric_rows)}/30")
    print(f"Paired comparisons: {len(pairwise_rows)}/15")
    print(f"Baseline regression matches: {regression_count}/15")
    print("Connected components:")
    print(
        f"  Improved={overall['component_improved_pairs']}, "
        f"Worsened={overall['component_worsened_pairs']}, "
        f"Unchanged={overall['component_unchanged_pairs']}, "
        f"p={overall['component_wilcoxon_p_value']:.4f}"
    )
    print("Degenerate faces:")
    print(
        f"  Improved={overall['degenerate_improved_pairs']}, "
        f"Worsened={overall['degenerate_worsened_pairs']}, "
        f"Unchanged={overall['degenerate_unchanged_pairs']}, "
        f"p={overall['degenerate_wilcoxon_p_value']:.4f}"
    )
    print(f"Saved: {results_dir / METRICS_FILENAME}")
    print(f"Saved: {results_dir / PAIRWISE_FILENAME}")
    print(f"Saved: {results_dir / SUMMARY_FILENAME}")
    print("CORRECTED MESH METRIC EVALUATION PASSED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
