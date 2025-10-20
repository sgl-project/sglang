import re
import sys
from pathlib import Path
from typing import List, Tuple


def normalize_version(version: str) -> str:
    """Remove 'v' prefix from version string if present."""
    return version.lstrip("v")


def validate_version(version: str) -> bool:
    """Validate version format: X.Y.Z, X.Y.Zrc0, or X.Y.Z.post1"""
    pattern = r"^\d+\.\d+\.\d+(rc\d+|\.post\d+)?$"
    return bool(re.match(pattern, version))


def parse_version(version: str) -> Tuple[int, int, int, int, int]:
    """
    Parse version string into comparable components.

    Returns: (major, minor, patch, pre_release, post_release)
    - pre_release: -1000 + rc_number for rcN, 0 for stable (rc0 < rc1 < stable)
    - post_release: N for .postN, 0 otherwise

    The pre_release field uses negative numbers to ensure RC versions come before
    stable versions when tuples are compared. Python compares tuples element by
    element, so (0, 5, 3, -1000, 0) < (0, 5, 3, 0, 0) ensures rc0 < stable.

    Examples:
    - "0.5.3rc0" → (0, 5, 3, -1000, 0)  # rc0 comes before stable
    - "0.5.3rc1" → (0, 5, 3, -999, 0)   # rc1 comes after rc0
    - "0.5.3"    → (0, 5, 3, 0, 0)      # stable version
    - "0.5.3.post1" → (0, 5, 3, 0, 1)   # post comes after stable
    """
    # Match version components
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)(?:rc(\d+)|\.post(\d+))?$", version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major, minor, patch, rc, post = match.groups()
    major, minor, patch = int(major), int(minor), int(patch)

    if rc is not None:
        # RC version: pre_release = -1000 + rc_number (ensures rc0 < rc1 < ... < stable)
        return (major, minor, patch, -1000 + int(rc), 0)
    elif post is not None:
        # Post version: post_release = N
        return (major, minor, patch, 0, int(post))
    else:
        # Stable version
        return (major, minor, patch, 0, 0)


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings following PEP 440 ordering.

    Returns:
    - -1 if v1 < v2
    -  0 if v1 == v2
    -  1 if v1 > v2

    Version ordering: X.Y.ZrcN < X.Y.Z < X.Y.Z.postN < X.Y.(Z+1)
    """
    parsed_v1 = parse_version(v1)
    parsed_v2 = parse_version(v2)

    if parsed_v1 < parsed_v2:
        return -1
    elif parsed_v1 > parsed_v2:
        return 1
    else:
        return 0


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

    # For TOML files, use regex to match version field regardless of current value
    if file_path.suffix == ".toml":
        # Match: version = "X.Y.Z..." (with optional quotes and whitespace)
        # Captures quotes (or lack thereof) to preserve original quoting style
        pattern = r'(version\s*=\s*)(["\']?)([^"\'\n]+)(["\']?)'
        new_content = re.sub(pattern, rf"\g<1>\g<2>{new_version}\g<4>", content)
    else:
        # For non-TOML files, use simple string replacement
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
) -> None:
    # Normalize version (remove 'v' prefix if present)
    new_version = normalize_version(new_version)

    if not validate_version(new_version):
        print(f"Error: Invalid version format: {new_version}")
        print("Expected format: X.Y.Z, X.Y.ZrcN, or X.Y.Z.postN")
        print("Examples: 0.5.4, 0.5.3rc0, 0.5.3.post1")
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

    # Compare versions
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
    for file_rel in files_to_update:
        file_abs = repo_root / file_rel
        if replace_in_file(file_abs, old_version, new_version):
            updated_count += 1

    print()
    print(f"Successfully updated {updated_count} file(s)")
    print(f"Version bumped from {old_version} to {new_version}")

    # Validate that all files now contain the new version
    print("\nValidating version updates...")
    failed_files = []
    for file_rel in files_to_update:
        file_abs = repo_root / file_rel
        if not file_abs.exists():
            print(f"Warning: File {file_rel} does not exist, skipping validation.")
            continue

        content = file_abs.read_text()

        # For TOML files, use regex to specifically check the version field
        if file_abs.suffix == ".toml":
            # Match version field with optional quotes
            pattern = r'version\s*=\s*["\']?' + re.escape(new_version) + r'["\']?'
            if not re.search(pattern, content):
                failed_files.append(file_rel)
                print(f"✗ {file_rel} does not contain version {new_version}")
            else:
                print(f"✓ {file_rel} validated")
        else:
            # For non-TOML files, use simple string search
            if new_version not in content:
                failed_files.append(file_rel)
                print(f"✗ {file_rel} does not contain version {new_version}")
            else:
                print(f"✓ {file_rel} validated")

    if failed_files:
        print(f"\nError: {len(failed_files)} file(s) were not updated correctly:")
        for file_rel in failed_files:
            print(f"  - {file_rel}")
        sys.exit(1)

    print("\nAll files validated successfully!")
