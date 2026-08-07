from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


OBJECTS = ("mouse", "bottle", "shoe")
VIEWS = ("front", "back", "left", "right", "top")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

DEFAULT_OUTPUT_NAME = (
    "final_proposed_outputs_corrected_20260804_final"
)
REQUIRED_OUTPUT_PREFIX = "final_proposed_outputs_corrected_"


@dataclass(frozen=True)
class Case:
    sample_id: str
    object_name: str
    view_name: str
    input_path: Path


def normalize_name(value: str) -> str:
    """Normalize a filename stem for strict, case-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def sha256_file(path: Path) -> str:
    """Return the uppercase SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest().upper()


def find_project_root(explicit_root: str | None) -> Path:
    """Locate the project from either an argument or common script locations."""
    if explicit_root:
        candidates = [Path(explicit_root).expanduser()]
    else:
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir,
            script_dir.parent,
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
            (resolved / "TripoSR" / "run.py").is_file()
            and (resolved / "dataset_triposr_ready").is_dir()
        ):
            return resolved

    searched = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "Could not locate the project root. Searched:\n" + searched
    )


def find_input_image(folder: Path, expected_stem: str) -> Path:
    """Find exactly one prepared image with the expected normalized stem."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder was not found: {folder}")

    expected = normalize_name(expected_stem)
    matches = [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and normalize_name(path.stem) == expected
    ]

    if len(matches) != 1:
        available = ", ".join(
            sorted(
                path.name
                for path in folder.iterdir()
                if path.is_file()
            )
        )
        raise RuntimeError(
            f"Expected one image matching '{expected_stem}' in {folder}; "
            f"found {len(matches)}. Available files: {available}"
        )

    return matches[0].resolve()


def build_plan(project_root: Path) -> list[Case]:
    """Build and validate the fixed 3-object by 5-view experiment plan."""
    input_root = project_root / "dataset_triposr_ready"
    all_images = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(all_images) != 15:
        raise RuntimeError(
            f"dataset_triposr_ready contains {len(all_images)} images; "
            "expected exactly 15."
        )

    plan: list[Case] = []

    for object_name in OBJECTS:
        for view_name in VIEWS:
            input_path = find_input_image(
                input_root / object_name,
                f"{object_name} {view_name} triposr ready",
            )
            plan.append(
                Case(
                    sample_id=f"{object_name}_{view_name}",
                    object_name=object_name,
                    view_name=view_name,
                    input_path=input_path,
                )
            )

    input_paths = [case.input_path for case in plan]

    if len(set(input_paths)) != 15:
        raise RuntimeError("The experiment plan contains duplicate input files.")

    return plan


def parse_local_dino_path(config_path: Path, triposr_root: Path) -> Path:
    """Resolve and validate the local DINO directory used by TripoSR."""
    if not config_path.is_file():
        raise FileNotFoundError(f"TripoSR config was not found: {config_path}")

    config_text = config_path.read_text(encoding="utf-8-sig")
    matches = re.findall(
        r"^\s*pretrained_model_name_or_path\s*:\s*"
        r"['\"]?([^'\"\r\n#]+?)['\"]?\s*(?:#.*)?$",
        config_text,
        flags=re.MULTILINE,
    )

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one pretrained_model_name_or_path field in "
            f"{config_path}; found {len(matches)}."
        )

    configured_value = matches[0].strip()
    configured_path = Path(configured_value).expanduser()

    if not configured_path.is_absolute():
        configured_path = triposr_root / configured_path

    dino_root = configured_path.resolve()
    dino_config = dino_root / "config.json"

    if not dino_root.is_dir() or not dino_config.is_file():
        raise FileNotFoundError(
            "The configured local DINO directory is incomplete.\n"
            f"Configured value: {configured_value}\n"
            f"Resolved path: {dino_root}\n"
            f"Expected file: {dino_config}"
        )

    with dino_config.open("r", encoding="utf-8-sig") as config_file:
        dino_data = json.load(config_file)

    expected_values = {
        "model_type": "vit",
        "hidden_size": 768,
        "image_size": 224,
        "patch_size": 16,
    }
    mismatches = {
        key: (dino_data.get(key), expected)
        for key, expected in expected_values.items()
        if dino_data.get(key) != expected
    }

    if mismatches:
        details = ", ".join(
            f"{key}={actual!r} (expected {expected!r})"
            for key, (actual, expected) in mismatches.items()
        )
        raise RuntimeError(f"Unexpected DINO configuration: {details}")

    return dino_root


def resolve_output_root(project_root: Path, value: str) -> Path:
    """Resolve an output root and reject protected or ambiguous targets."""
    candidate = Path(value).expanduser()

    if not candidate.is_absolute():
        candidate = project_root / candidate

    output_root = candidate.resolve()
    protected_roots = {
        project_root.resolve(),
        (project_root / "TripoSR").resolve(),
        (project_root / "baseline_outputs").resolve(),
        (project_root / "final_proposed_outputs").resolve(),
        (project_root / "ablation_nobg_outputs").resolve(),
        (project_root / "ablation_nobg_crop_pad_outputs").resolve(),
    }

    if output_root in protected_roots:
        raise RuntimeError(f"Protected output directory selected: {output_root}")

    if output_root.parent != project_root.resolve():
        raise RuntimeError(
            "The corrected output directory must be a direct child of the "
            f"project root: {project_root}"
        )

    if not output_root.name.startswith(REQUIRED_OUTPUT_PREFIX):
        raise RuntimeError(
            "The output directory name must start with "
            f"'{REQUIRED_OUTPUT_PREFIX}'."
        )

    if output_root.exists() and output_root.is_symlink():
        raise RuntimeError(f"Symbolic-link output roots are not allowed: {output_root}")

    return output_root


def relative_to_project(path: Path, project_root: Path) -> str:
    """Return a stable project-relative path for manifests."""
    return str(path.resolve().relative_to(project_root.resolve()))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    """Atomically write the current generation manifest."""
    fieldnames = [
        "sample_id",
        "object",
        "view",
        "status",
        "input_image",
        "input_sha256",
        "mesh_file",
        "mesh_bytes",
        "mesh_sha256",
        "old_mesh_file",
        "old_mesh_sha256",
        "hash_matches_old",
        "generation_seconds",
        "log_file",
    ]
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary_path, path)


def validate_partial_case(case_root: Path, input_path: Path) -> None:
    """Allow safe resume only when a partial case contains the expected input."""
    files = sorted(
        path.relative_to(case_root)
        for path in case_root.rglob("*")
        if path.is_file()
    )
    allowed = [Path("0") / "input.png"]

    if not files:
        return

    if files != allowed:
        formatted = ", ".join(str(path) for path in files)
        raise RuntimeError(
            f"Unexpected partial output in {case_root}: {formatted}. "
            "Nothing was deleted or overwritten."
        )

    saved_input = case_root / "0" / "input.png"

    if sha256_file(saved_input) != sha256_file(input_path):
        raise RuntimeError(
            f"The saved input does not match the planned input: {saved_input}"
        )


def build_result_row(
    case: Case,
    project_root: Path,
    mesh_path: Path,
    log_path: Path,
    status: str,
    generation_seconds: float | str,
) -> dict[str, object]:
    """Build one auditable manifest row, including comparison to old Final."""
    old_mesh = (
        project_root
        / "final_proposed_outputs"
        / case.sample_id
        / "0"
        / "mesh.glb"
    )
    old_exists = old_mesh.is_file()
    mesh_hash = sha256_file(mesh_path)
    old_hash = sha256_file(old_mesh) if old_exists else ""

    return {
        "sample_id": case.sample_id,
        "object": case.object_name,
        "view": case.view_name,
        "status": status,
        "input_image": relative_to_project(case.input_path, project_root),
        "input_sha256": sha256_file(case.input_path),
        "mesh_file": relative_to_project(mesh_path, project_root),
        "mesh_bytes": mesh_path.stat().st_size,
        "mesh_sha256": mesh_hash,
        "old_mesh_file": (
            relative_to_project(old_mesh, project_root)
            if old_exists
            else ""
        ),
        "old_mesh_sha256": old_hash,
        "hash_matches_old": (
            str(mesh_hash == old_hash)
            if old_exists
            else ""
        ),
        "generation_seconds": generation_seconds,
        "log_file": relative_to_project(log_path, project_root),
    }


def run_case(
    case: Case,
    index: int,
    total: int,
    project_root: Path,
    triposr_root: Path,
    model_path: Path,
    output_root: Path,
    log_root: Path,
    resume: bool,
) -> dict[str, object]:
    """Generate or safely resume one Final Proposed mesh."""
    case_root = output_root / case.sample_id
    numbered_root = case_root / "0"
    mesh_path = numbered_root / "mesh.glb"
    saved_input = numbered_root / "input.png"
    log_path = log_root / f"{case.sample_id}.log"

    if mesh_path.exists():
        if not resume:
            raise RuntimeError(
                f"Existing mesh found without --resume: {mesh_path}"
            )

        if not mesh_path.is_file() or mesh_path.stat().st_size <= 0:
            raise RuntimeError(f"Invalid existing mesh: {mesh_path}")

        if not saved_input.is_file():
            raise RuntimeError(
                f"Existing mesh has no recorded input image: {saved_input}"
            )

        if sha256_file(saved_input) != sha256_file(case.input_path):
            raise RuntimeError(
                f"Existing output uses a different input image: {case_root}"
            )

        print(f"[{index:02d}/{total}] SKIP existing: {case.sample_id}")
        return build_result_row(
            case=case,
            project_root=project_root,
            mesh_path=mesh_path,
            log_path=log_path,
            status="existing",
            generation_seconds="",
        )

    if case_root.exists():
        if not resume:
            raise RuntimeError(
                f"Existing case directory found without --resume: {case_root}"
            )
        validate_partial_case(case_root, case.input_path)

    numbered_root.mkdir(parents=True, exist_ok=True)

    if saved_input.exists():
        if sha256_file(saved_input) != sha256_file(case.input_path):
            raise RuntimeError(
                f"Refusing to overwrite a different saved input: {saved_input}"
            )
    else:
        shutil.copy2(case.input_path, saved_input)

    command = [
        sys.executable,
        str(triposr_root / "run.py"),
        str(case.input_path),
        "--pretrained-model-name-or-path",
        str(model_path),
        "--output-dir",
        str(case_root),
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

    print(f"[{index:02d}/{total}] Generating: {case.sample_id}")
    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=str(triposr_root),
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - start_time
    output_text = completed.stdout + "\n" + completed.stderr
    log_path.write_text(output_text, encoding="utf-8")

    if completed.returncode != 0 or not mesh_path.is_file():
        raise RuntimeError(
            f"Generation failed for {case.sample_id}; return code "
            f"{completed.returncode}. See: {log_path}"
        )

    if mesh_path.stat().st_size <= 0:
        raise RuntimeError(f"Generated mesh is empty: {mesh_path}")

    row = build_result_row(
        case=case,
        project_root=project_root,
        mesh_path=mesh_path,
        log_path=log_path,
        status="generated",
        generation_seconds=round(elapsed, 3),
    )
    print(
        f"             OK: {row['mesh_bytes']} bytes, "
        f"matches old={row['hash_matches_old']}"
    )
    return row


def run_generation(
    plan: list[Case],
    project_root: Path,
    triposr_root: Path,
    model_path: Path,
    output_root: Path,
    resume: bool,
) -> None:
    """Run the protected 15-case generation and maintain its manifest."""
    if output_root.exists() and not resume:
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Use a new directory, or pass --resume for outputs created by "
            "this script."
        )

    output_root.mkdir(parents=False, exist_ok=resume)
    log_root = output_root / "_logs"
    log_root.mkdir(parents=False, exist_ok=True)
    manifest_path = output_root / "generation_manifest.csv"
    rows: list[dict[str, object]] = []

    for index, case in enumerate(plan, start=1):
        try:
            row = run_case(
                case=case,
                index=index,
                total=len(plan),
                project_root=project_root,
                triposr_root=triposr_root,
                model_path=model_path,
                output_root=output_root,
                log_root=log_root,
                resume=resume,
            )
        except Exception:
            if rows:
                write_manifest(manifest_path, rows)
            raise

        rows.append(row)
        write_manifest(manifest_path, rows)

    matching_old = sum(
        row["hash_matches_old"] == "True"
        for row in rows
    )
    old_available = sum(
        bool(row["old_mesh_sha256"])
        for row in rows
    )

    print("\nGeneration completed successfully.")
    print(f"Generated or verified meshes: {len(rows)}/{len(plan)}")
    print(f"Old Final meshes available: {old_available}/{len(plan)}")
    print(f"SHA-256 matches with old Final: {matching_old}/{old_available}")
    print(f"Output: {output_root}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments; validation is the default action."""
    parser = argparse.ArgumentParser(
        description=(
            "Safely generate the 15 corrected Final Proposed TripoSR outputs."
        )
    )
    parser.add_argument(
        "--project-root",
        help="Project root containing TripoSR and dataset_triposr_ready.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_NAME,
        help=(
            "New project-relative corrected output directory. "
            f"Default: {DEFAULT_OUTPUT_NAME}"
        ),
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--run",
        action="store_true",
        help="Run all 15 generations after validation.",
    )
    action.add_argument(
        "--check-only",
        action="store_true",
        help="Validate only. This is also the default action.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only verified outputs previously created by this script.",
    )
    return parser.parse_args()


def main() -> None:
    """Validate the environment, then optionally run the 15-case batch."""
    args = parse_args()

    if args.resume and not args.run:
        raise RuntimeError("--resume requires --run.")

    project_root = find_project_root(args.project_root)
    triposr_root = project_root / "TripoSR"
    model_path = triposr_root / "models" / "TripoSR"
    config_path = model_path / "config.yaml"
    output_root = resolve_output_root(project_root, args.output_root)

    if not model_path.is_dir():
        raise FileNotFoundError(f"Local TripoSR model was not found: {model_path}")

    dino_root = parse_local_dino_path(config_path, triposr_root)
    plan = build_plan(project_root)

    print("=" * 72)
    print("Corrected Final Proposed preflight")
    print("=" * 72)
    print(f"Project root: {project_root}")
    print(f"Python: {Path(sys.executable).resolve()}")
    print(f"TripoSR model: {model_path}")
    print(f"Local DINO: {dino_root}")
    print(f"Output root: {output_root}")
    print(f"Planned cases: {len(plan)}")

    for case in plan:
        print(f"  - {case.sample_id}: {case.input_path.name}")

    if not args.run:
        print("\nCHECK PASSED: no model was generated.")
        print("Run again with --run after reviewing this plan.")
        return

    run_generation(
        plan=plan,
        project_root=project_root,
        triposr_root=triposr_root,
        model_path=model_path,
        output_root=output_root,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
