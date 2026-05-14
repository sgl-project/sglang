#!/usr/bin/env python3
"""
Check if sgl-kernel version from sgl-kernel/pyproject.toml matches the versions
used in SGLang files (python/pyproject.toml, engine.py, and Dockerfile).
Sets GitHub Actions output variables to indicate if sync is needed.
"""

import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions


def get_kernel_version_from_source() -> str:
    """Extract version from sgl-kernel/pyproject.toml (line 11)"""
    pyproject_path = Path("sgl-kernel/pyproject.toml")

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    version = data.get("project", {}).get("version")
    if not version:
        print("Error: Could not find version in sgl-kernel/pyproject.toml")
        sys.exit(1)

    return version


def get_kernel_version_from_python_pyproject() -> str:
    """Extract sgl-kernel version from python/pyproject.toml"""
    pyproject_path = Path("python/pyproject.toml")

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    content = pyproject_path.read_text()

    # Match "sgl-kernel==x.x.x"
    match = re.search(r'"sgl-kernel==([^"]+)"', content)
    if not match:
        print("Error: Could not find sgl-kernel version in python/pyproject.toml")
        sys.exit(1)

    return match.group(1)


def get_kernel_version_from_engine() -> str:
    """Extract sgl-kernel version from python/sglang/srt/entrypoints/engine.py"""
    engine_path = Path("python/sglang/srt/entrypoints/engine.py")

    if not engine_path.exists():
        print(f"Error: {engine_path} not found")
        sys.exit(1)

    content = engine_path.read_text()

    # Find the assert_pkg_version call for sgl-kernel
    # Look for the pattern: assert_pkg_version("sgl-kernel", "version", ...)
    match = re.search(
        r'assert_pkg_version\s*\(\s*"sgl-kernel"\s*,\s*"([^"]+)"', content
    )
    if not match:
        print("Error: Could not find sgl-kernel version in engine.py")
        sys.exit(1)

    return match.group(1)


def get_kernel_version_from_dockerfile() -> str:
    """Extract SGL_KERNEL_VERSION from docker/Dockerfile"""
    dockerfile_path = Path("docker/Dockerfile")

    if not dockerfile_path.exists():
        print(f"Error: {dockerfile_path} not found")
        sys.exit(1)

    content = dockerfile_path.read_text()

    # Match ARG SGL_KERNEL_VERSION=x.x.x
    match = re.search(r"^ARG\s+SGL_KERNEL_VERSION=(.+)$", content, re.MULTILINE)
    if not match:
        print("Error: Could not find SGL_KERNEL_VERSION in Dockerfile")
        sys.exit(1)

    return match.group(1).strip()


def main():
    kernel_version = get_kernel_version_from_source()
    pyproject_version = get_kernel_version_from_python_pyproject()
    engine_version = get_kernel_version_from_engine()
    dockerfile_version = get_kernel_version_from_dockerfile()

    print(f"Kernel version in sgl-kernel/pyproject.toml: {kernel_version}")
    print(f"Kernel version in python/pyproject.toml: {pyproject_version}")
    print(f"Kernel version in engine.py: {engine_version}")
    print(f"Kernel version in Dockerfile: {dockerfile_version}")

    # Check if any version differs from the source
    needs_sync = (
        kernel_version != pyproject_version
        or kernel_version != engine_version
        or kernel_version != dockerfile_version
    )

    # Set GitHub Actions output
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"needs_sync={'true' if needs_sync else 'false'}\n")
            f.write(f"kernel_version={kernel_version}\n")

    if needs_sync:
        print(f"\n✓ Sync needed to version: {kernel_version}")
        mismatches = []
        if kernel_version != pyproject_version:
            mismatches.append(
                f"  - python/pyproject.toml: {pyproject_version} → {kernel_version}"
            )
        if kernel_version != engine_version:
            mismatches.append(f"  - engine.py: {engine_version} → {kernel_version}")
        if kernel_version != dockerfile_version:
            mismatches.append(
                f"  - Dockerfile: {dockerfile_version} → {kernel_version}"
            )

        print("Changes needed:")
        for mismatch in mismatches:
            print(mismatch)

        sys.exit(0)
    else:
        print("\n✓ All versions are in sync, no action needed")
        sys.exit(0)


if __name__ == "__main__":
    main()
