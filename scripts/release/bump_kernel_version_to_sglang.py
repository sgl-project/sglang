#!/usr/bin/env python3
"""
Bump sgl-kernel version in SGLang files to match the version in sgl-kernel/pyproject.toml.
Updates:
  - python/pyproject.toml
  - python/sglang/srt/entrypoints/engine.py
  - docker/Dockerfile
"""

import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions


def get_kernel_version_from_source() -> str:
    """Extract version from sgl-kernel/pyproject.toml"""
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


def update_python_pyproject(new_version: str) -> bool:
    """Update sgl-kernel version in python/pyproject.toml"""
    pyproject_path = Path("python/pyproject.toml")

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found")
        sys.exit(1)

    content = pyproject_path.read_text()

    # Replace "sgl-kernel==x.x.x" with new version
    new_content = re.sub(
        r'"sgl-kernel==[^"]+"',
        f'"sgl-kernel=={new_version}"',
        content,
    )

    if content == new_content:
        print("No changes needed in python/pyproject.toml")
        return False

    pyproject_path.write_text(new_content)
    print(f"✓ Updated python/pyproject.toml to version {new_version}")
    return True


def update_engine_py(new_version: str) -> bool:
    """Update sgl-kernel version in python/sglang/srt/entrypoints/engine.py"""
    engine_path = Path("python/sglang/srt/entrypoints/engine.py")

    if not engine_path.exists():
        print(f"Error: {engine_path} not found")
        sys.exit(1)

    content = engine_path.read_text()

    # Replace version in assert_pkg_version("sgl-kernel", "version", ...)
    new_content = re.sub(
        r'(assert_pkg_version\s*\(\s*"sgl-kernel"\s*,\s*)"[^"]+"',
        rf'\1"{new_version}"',
        content,
    )

    if content == new_content:
        print("No changes needed in engine.py")
        return False

    engine_path.write_text(new_content)
    print(f"✓ Updated engine.py to version {new_version}")
    return True


def update_dockerfile(new_version: str) -> bool:
    """Update SGL_KERNEL_VERSION in docker/Dockerfile"""
    dockerfile_path = Path("docker/Dockerfile")

    if not dockerfile_path.exists():
        print(f"Error: {dockerfile_path} not found")
        sys.exit(1)

    content = dockerfile_path.read_text()

    # Replace ARG SGL_KERNEL_VERSION=x.x.x with new version
    new_content = re.sub(
        r"^(ARG\s+SGL_KERNEL_VERSION=)(.+)$",
        rf"\g<1>{new_version}",
        content,
        flags=re.MULTILINE,
    )

    if content == new_content:
        print("No changes needed in Dockerfile")
        return False

    dockerfile_path.write_text(new_content)
    print(f"✓ Updated Dockerfile to version {new_version}")
    return True


def main():
    kernel_version = get_kernel_version_from_source()
    print(f"Bumping sgl-kernel version to: {kernel_version}\n")

    updated_files = []

    if update_python_pyproject(kernel_version):
        updated_files.append("python/pyproject.toml")

    if update_engine_py(kernel_version):
        updated_files.append("python/sglang/srt/entrypoints/engine.py")

    if update_dockerfile(kernel_version):
        updated_files.append("docker/Dockerfile")

    print()
    if updated_files:
        print(f"✓ Successfully updated {len(updated_files)} file(s):")
        for file in updated_files:
            print(f"  - {file}")
    else:
        print("✓ All files already have the correct version")


if __name__ == "__main__":
    main()
