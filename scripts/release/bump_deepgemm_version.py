#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

from utils import compare_versions, get_repo_root, normalize_version, validate_version

FILES_TO_UPDATE = [
    Path("python/pyproject.toml"),
    Path("docker/Dockerfile"),
]

VERSION_PATTERN = r"\d+\.\d+\.\d+(?:rc\d+|\.post\d+)?"


def read_current_deepgemm_version(repo_root: Path) -> str:
    """Read the current sgl-deep-gemm version from python/pyproject.toml."""
    pyproject = repo_root / "python" / "pyproject.toml"
    content = pyproject.read_text()
    match = re.search(rf"sgl-deep-gemm==({VERSION_PATTERN})", content)
    if not match:
        raise ValueError(f"Could not find sgl-deep-gemm version in {pyproject}")
    return match.group(1)


def replace_deepgemm_version(
    file_path: Path, old_version: str, new_version: str
) -> bool:
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping")
        return False

    content = file_path.read_text()
    new_content = content

    name = file_path.name
    if name == "pyproject.toml":
        new_content = new_content.replace(
            f"sgl-deep-gemm=={old_version}", f"sgl-deep-gemm=={new_version}"
        )
    elif name == "Dockerfile":
        new_content = re.sub(
            rf"(ARG SGL_DEEP_GEMM_VERSION=){re.escape(old_version)}",
            rf"\g<1>{new_version}",
            new_content,
        )

    if content == new_content:
        print(f"No changes needed in {file_path}")
        return False

    file_path.write_text(new_content)
    print(f"Updated {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bump sgl-deep-gemm version across all relevant files"
    )
    parser.add_argument(
        "new_version",
        help="New version (e.g., 0.1.2, 0.1.2rc0, or 0.1.2.post1)",
    )
    args = parser.parse_args()

    new_version = normalize_version(args.new_version)

    if not validate_version(new_version):
        print(f"Error: Invalid version format: {new_version}")
        print("Expected format: X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN")
        print("Examples: 0.1.2, 0.1.2rc0, 0.1.2.post1")
        sys.exit(1)

    repo_root = get_repo_root()
    old_version = read_current_deepgemm_version(repo_root)
    print(f"Current sgl-deep-gemm version: {old_version}")
    print(f"New sgl-deep-gemm version: {new_version}")
    print()

    comparison = compare_versions(new_version, old_version)
    if comparison == 0:
        print("Error: New version is the same as current version")
        sys.exit(1)
    elif comparison < 0:
        print(
            f"Error: New version ({new_version}) is older than current version ({old_version})"
        )
        print("Version must be greater than the current version")
        sys.exit(1)

    updated_count = 0
    for file_rel in FILES_TO_UPDATE:
        file_abs = repo_root / file_rel
        if replace_deepgemm_version(file_abs, old_version, new_version):
            updated_count += 1

    print()
    print(f"Successfully updated {updated_count} file(s)")
    print(f"sgl-deep-gemm version bumped from {old_version} to {new_version}")

    print("\nValidating version updates...")
    failed_files = []
    for file_rel in FILES_TO_UPDATE:
        file_abs = repo_root / file_rel
        if not file_abs.exists():
            print(f"Warning: File {file_rel} does not exist, skipping validation.")
            continue

        content = file_abs.read_text()
        if new_version not in content:
            failed_files.append(file_rel)
            print(f"{file_rel} does not contain version {new_version}")
        else:
            print(f"{file_rel} validated")

    if failed_files:
        print(f"\nError: {len(failed_files)} file(s) were not updated correctly:")
        for file_rel in failed_files:
            print(f"  - {file_rel}")
        sys.exit(1)

    print("\nAll files validated successfully!")


if __name__ == "__main__":
    main()
