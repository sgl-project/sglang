#!/usr/bin/env python3
"""Repair an SGLang wheel and smoke-test its production Rust TreeCore."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

_LIBTORCH_EXCLUDES = (
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_python.so",
)
_TREE_CORE_DIR = PurePosixPath("sglang/srt/mem_cache/rust_tree_core")
_BINDING_CLASSES = (
    "RustUnifiedTreeCoreBinding",
    "RustBigramUnifiedTreeCoreBinding",
    "TreeCoreInitParamsBinding",
)


def _single_wheel(directory: Path) -> Path:
    wheels = sorted(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel in {directory}, found {wheels}")
    return wheels[0]


def _metadata(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"expected one METADATA file in {wheel}, found {metadata_files}"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    return str(metadata["Name"]), str(metadata["Version"])


def _smoke_test_tree_core(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="sglang-wheel-smoke-") as temp_dir:
        root = Path(temp_dir)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            inspection_modules = [
                name
                for name in names
                if PurePosixPath(name).parent == _TREE_CORE_DIR
                and PurePosixPath(name).name.startswith("mem_cache_inspection")
                and name.endswith(".so")
            ]
            if inspection_modules:
                raise RuntimeError(
                    f"production wheel contains inspection modules: {inspection_modules}"
                )
            production_modules = [
                name
                for name in names
                if PurePosixPath(name).parent == _TREE_CORE_DIR
                and PurePosixPath(name).name.startswith("mem_cache.")
                and name.endswith(".so")
            ]
            if len(production_modules) != 1:
                raise RuntimeError(
                    "expected one production Rust TreeCore module, found "
                    f"{production_modules}"
                )

        install_dir = root / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-compile",
                "--no-deps",
                "--no-index",
                "--target",
                os.fspath(install_dir),
                os.fspath(wheel),
            ],
            check=True,
        )

        smoke_script = textwrap.dedent(f"""
            import sys
            import types
            from pathlib import Path

            site_packages = Path({os.fspath(install_dir)!r}).resolve()
            sys.path.insert(0, str(site_packages))
            package = types.ModuleType("sglang")
            package.__package__ = "sglang"
            package.__path__ = [str(site_packages / "sglang")]
            sys.modules["sglang"] = package

            from sglang.srt.mem_cache.rust_tree_core.extension import bindings

            module_path = Path(bindings.__file__).resolve()
            if site_packages not in module_path.parents:
                raise RuntimeError(
                    f"loaded TreeCore outside installed wheel: {{module_path}}"
                )
            if bindings.__name__ != "sglang.srt.mem_cache.rust_tree_core.mem_cache":
                raise RuntimeError(
                    f"loaded unexpected TreeCore module: {{bindings.__name__}}"
                )
            for class_name in {_BINDING_CLASSES!r}:
                binding = getattr(bindings, class_name, None)
                if binding is None:
                    raise RuntimeError(
                        f"production TreeCore is missing {{class_name}}"
                    )
                inspection_methods = [
                    name for name in dir(binding) if name.startswith("inspect_")
                ]
                if inspection_methods:
                    raise RuntimeError(
                        f"production {{class_name}} exposes inspection methods: "
                        f"{{inspection_methods}}"
                    )

            from array import array

            hashes = bindings.get_hash_str(array("q", [1, 2]), None, 1)
            if len(hashes) != 2 or any(len(value) != 64 for value in hashes):
                raise RuntimeError(f"unexpected TreeCore hash result: {{hashes}}")
            """)
        environment = os.environ.copy()
        environment["SGLANG_RUST_BUILD_MODE"] = "never"
        environment.pop("PYTHONPATH", None)
        subprocess.run(
            [sys.executable, "-I", "-c", smoke_script],
            cwd=root,
            env=environment,
            check=True,
        )


def _write_github_outputs(path: Path, *, wheel: Path, version: str) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(f"wheel_filename={wheel.name}\n")
        output.write(f"wheel_version={version}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    wheel_dir = args.wheel_dir.resolve()
    source_wheel = _single_wheel(wheel_dir)
    with tempfile.TemporaryDirectory(
        prefix="sglang-wheel-repair-", dir=wheel_dir.parent
    ) as repair_dir:
        command = [
            sys.executable,
            "-m",
            "auditwheel",
            "repair",
            os.fspath(source_wheel),
            "--wheel-dir",
            repair_dir,
        ]
        for library in _LIBTORCH_EXCLUDES:
            command.extend(("--exclude", library))
        subprocess.run(command, check=True)

        repaired_wheel = _single_wheel(Path(repair_dir))
        name, version = _metadata(repaired_wheel)
        if name.casefold() != "sglang":
            raise RuntimeError(f"expected sglang wheel, found {name!r}")
        _smoke_test_tree_core(repaired_wheel)

        destination = wheel_dir / repaired_wheel.name
        source_wheel.unlink()
        shutil.move(repaired_wheel, destination)

    if args.github_output is not None:
        _write_github_outputs(
            args.github_output.resolve(), wheel=destination, version=version
        )
    print(f"Prepared {destination.name} (sglang {version})")


if __name__ == "__main__":
    main()
