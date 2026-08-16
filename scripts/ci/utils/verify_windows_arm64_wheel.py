#!/usr/bin/env python3
"""Verify the native Windows ARM64 SGLang wheel produced by this interpreter."""

import argparse
import sys
import zipfile
from email.parser import Parser
from pathlib import Path


def verify_wheel(wheel_dir: Path) -> Path:
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel in {wheel_dir}, found {len(wheels)}")

    wheel_path = wheels[0]
    python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    expected_suffix = f"-{python_tag}-{python_tag}-win_arm64.whl"
    if not wheel_path.name.endswith(expected_suffix):
        raise RuntimeError(f"unexpected Windows ARM64 wheel tag: {wheel_path.name}")

    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA file, found {metadata_names}")
        metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode("utf-8"))

    multimodal = [
        name
        for name in names
        if name.startswith("sglang/srt/multimodal/_core") and name.endswith(".pyd")
    ]
    unsupported = [
        name
        for name in names
        if name.startswith(
            (
                "sglang/srt/grpc/_core",
                "sglang/srt/server/_core",
            )
        )
        and name.endswith(".pyd")
    ]
    if len(multimodal) != 1:
        raise RuntimeError(f"expected one multimodal extension, found {multimodal}")
    if unsupported:
        raise RuntimeError(
            f"found unsupported Windows extension modules: {unsupported}"
        )

    version = metadata["Version"]
    if not version or version.startswith("0.0.0"):
        raise RuntimeError(f"setuptools-scm produced an invalid version: {version}")

    print(f"verified {wheel_path} ({version})")
    return wheel_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel_dir)


if __name__ == "__main__":
    main()
