#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils import bump_version


def main():
    parser = argparse.ArgumentParser(
        description="Bump sgl-kernel version across all relevant files"
    )
    parser.add_argument(
        "new_version",
        help="New version (e.g., 0.3.12, 0.3.11rc0, or 0.3.11.post1)",
    )
    args = parser.parse_args()

    version_file = Path("sgl-kernel/python/sgl_kernel/version.py")

    files_to_update = [
        Path("sgl-kernel/pyproject.toml"),
        Path("sgl-kernel/pyproject_cpu.toml"),
        Path("sgl-kernel/pyproject_rocm.toml"),
        Path("sgl-kernel/python/sgl_kernel/version.py"),
    ]

    bump_version(args.new_version, version_file, files_to_update)


if __name__ == "__main__":
    main()
