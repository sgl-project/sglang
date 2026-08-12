#!/usr/bin/env python3
"""Verify the reduced native Windows ARM64 sglang-kernel wheel."""

import argparse
import zipfile
from email.parser import Parser
from pathlib import Path


def verify_wheel(wheel_dir: Path) -> Path:
    wheels = list(wheel_dir.glob("sglang_kernel-*-win_arm64.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one kernel wheel, found {len(wheels)}")

    wheel_path = wheels[0]
    with zipfile.ZipFile(wheel_path) as wheel:
        names = wheel.namelist()
        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"expected one METADATA file, found {metadata_names}")
        metadata = Parser().parsestr(wheel.read(metadata_names[0]).decode("utf-8"))

    native_modules = [
        name
        for name in names
        if name.startswith("sgl_kernel/sm100/common_ops") and name.endswith(".pyd")
    ]
    if len(native_modules) != 1:
        raise RuntimeError(
            f"expected one SM100+ common_ops module, found {native_modules}"
        )
    if any(name.startswith("sgl_kernel/sm90/common_ops") for name in names):
        raise RuntimeError("Windows ARM64 wheel unexpectedly contains the SM90 module")
    if metadata["Name"] != "sglang-kernel":
        raise RuntimeError(f"unexpected distribution name: {metadata['Name']}")

    print(f"verified {wheel_path} ({metadata['Version']})")
    return wheel_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()
    verify_wheel(args.wheel_dir)


if __name__ == "__main__":
    main()
