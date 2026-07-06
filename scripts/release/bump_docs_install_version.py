#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

from utils import (
    compare_versions,
    get_repo_root,
    normalize_version,
    validate_version,
)

# Docs pages that pin a release branch in their "install from source" snippet,
# e.g. `git clone -b v0.5.12 https://github.com/sgl-project/sglang.git`.
FILES_TO_UPDATE = [
    Path("docs_new/docs/get-started/install.mdx"),
    Path("docs_new/docs/hardware-platforms/amd_gpu.mdx"),
]

# Matches `git clone -b v<version> https://github.com/sgl-project/sglang.git`,
# capturing the version (without the leading `v`) in group 2.
CLONE_RE = re.compile(
    r"(git clone -b )v([0-9][0-9A-Za-z.\-]*)"
    r"( https://github\.com/sgl-project/sglang\.git)"
)


def read_current_version(file_path: Path) -> str:
    """Read the pinned source-install version from a docs page."""
    match = CLONE_RE.search(file_path.read_text())
    if not match:
        raise ValueError(
            f"Could not find a 'git clone -b v<version> ...sglang.git' line in {file_path}"
        )
    return match.group(2)


def replace_clone_version(file_path: Path, new_version: str) -> bool:
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping")
        return False

    content = file_path.read_text()
    new_content = CLONE_RE.sub(rf"\g<1>v{new_version}\g<3>", content)

    if content == new_content:
        print(f"No changes needed in {file_path}")
        return False

    file_path.write_text(new_content)
    print(f"✓ Updated {file_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Bump the 'install from source' release-branch version in the docs"
    )
    parser.add_argument(
        "new_version",
        help="New version (e.g., 0.5.13, 0.5.13rc0, or 0.5.13.post1)",
    )
    args = parser.parse_args()

    new_version = normalize_version(args.new_version)

    if not validate_version(new_version):
        print(f"Error: Invalid version format: {new_version}")
        print("Expected format: X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN")
        print("Examples: 0.5.13, 0.5.13rc0, 0.5.13.post1")
        sys.exit(1)

    repo_root = get_repo_root()

    # Determine the current version from the primary install page for logging.
    primary = repo_root / FILES_TO_UPDATE[0]
    old_version = read_current_version(primary)
    print(f"Current docs install version: {old_version}")
    print(f"New docs install version: {new_version}")
    print()

    comparison = compare_versions(new_version, old_version)
    if comparison == 0:
        print("Docs are already at this version; nothing to do.")
        return
    elif comparison < 0:
        print(
            f"Warning: new version ({new_version}) is older than the docs version "
            f"({old_version}); proceeding anyway."
        )

    updated_count = 0
    for file_rel in FILES_TO_UPDATE:
        file_abs = repo_root / file_rel
        if replace_clone_version(file_abs, new_version):
            updated_count += 1

    print()
    print(f"Successfully updated {updated_count} file(s)")
    print(f"Docs install version bumped from {old_version} to {new_version}")

    print("\nValidating version updates...")
    failed_files = []
    for file_rel in FILES_TO_UPDATE:
        file_abs = repo_root / file_rel
        if not file_abs.exists():
            print(f"Warning: File {file_rel} does not exist, skipping validation.")
            continue

        found = read_current_version(file_abs)
        if found != new_version:
            failed_files.append(file_rel)
            print(f"✗ {file_rel} still pins v{found}")
        else:
            print(f"✓ {file_rel} validated")

    if failed_files:
        print(f"\nError: {len(failed_files)} file(s) were not updated correctly:")
        for file_rel in failed_files:
            print(f"  - {file_rel}")
        sys.exit(1)

    print("\nAll files validated successfully!")


if __name__ == "__main__":
    main()
