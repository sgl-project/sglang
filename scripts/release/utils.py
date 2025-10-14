import re
import sys
from pathlib import Path
from typing import List, Tuple


def normalize_version(version: str) -> str:
    """Remove 'v' prefix from version string if present."""
    return version.lstrip("v")


def validate_version(version: str, version_type: str = "sglang") -> bool:
    if version_type == "sglang":
        pattern = r"^\d+\.\d+\.\d+(rc\d+)?$"
    else:
        pattern = r"^\d+\.\d+\.\d+$"

    if not re.match(pattern, version):
        return False
    return True


def get_repo_root() -> Path:
    return Path(__file__).parent.parent.parent


def read_current_version(version_file: Path) -> str:
    content = version_file.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError(f"Could not find version in {version_file}")
    return match.group(1)


def replace_in_file(file_path: Path, old_version: str, new_version: str) -> bool:
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist, skipping")
        return False

    content = file_path.read_text()
    new_content = content.replace(old_version, new_version)

    if content == new_content:
        print(f"No changes needed in {file_path}")
        return False

    file_path.write_text(new_content)
    print(f"✓ Updated {file_path}")
    return True


def bump_version(
    new_version: str,
    version_file: Path,
    files_to_update: List[Path],
    version_type: str = "sglang",
) -> None:
    # Normalize version (remove 'v' prefix if present)
    new_version = normalize_version(new_version)

    if not validate_version(new_version, version_type):
        print(f"Error: Invalid version format: {new_version}")
        if version_type == "sglang":
            print("Expected format: X.Y.Z or X.Y.ZrcN (e.g., 0.5.3 or 0.5.3rc0)")
        else:
            print("Expected format: X.Y.Z (e.g., 0.3.12)")
        sys.exit(1)

    repo_root = get_repo_root()
    version_file_abs = repo_root / version_file

    if not version_file_abs.exists():
        print(f"Error: Version file {version_file_abs} does not exist")
        sys.exit(1)

    old_version = read_current_version(version_file_abs)
    print(f"Current version: {old_version}")
    print(f"New version: {new_version}")
    print()

    if old_version == new_version:
        print("Warning: New version is the same as current version")
        sys.exit(0)

    updated_count = 0
    for file_rel in files_to_update:
        file_abs = repo_root / file_rel
        if replace_in_file(file_abs, old_version, new_version):
            updated_count += 1

    print()
    print(f"Successfully updated {updated_count} file(s)")
    print(f"Version bumped from {old_version} to {new_version}")
