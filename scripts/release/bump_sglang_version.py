#!/usr/bin/env python3

import argparse
from pathlib import Path

from utils import bump_version


def main():
    parser = argparse.ArgumentParser(
        description="Bump SGLang version across all relevant files"
    )
    parser.add_argument(
        "new_version",
        help="New version (e.g., 0.5.4, 0.5.3rc0, or 0.5.3.post1)",
    )
    args = parser.parse_args()

    version_file = Path("python/sglang/version.py")

    files_to_update = [
        Path("benchmark/deepseek_v3/README.md"),
        Path("docker/Dockerfile"),
        Path("docker/rocm.Dockerfile"),
        Path("docs/get_started/install.md"),
        Path("docs/platforms/amd_gpu.md"),
        Path("docs/platforms/ascend_npu.md"),
        Path("python/pyproject.toml"),
        Path("python/pyproject_other.toml"),
        Path("python/pyproject_cpu.toml"),
        Path("python/pyproject_xpu.toml"),
        Path("python/sglang/version.py"),
    ]

    bump_version(args.new_version, version_file, files_to_update)


if __name__ == "__main__":
    main()
