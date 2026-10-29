#!/usr/bin/env python3
"""Move runtime-built Rust extensions between AMD CI containers.

The AMD counterpart of _pr-test-rust-ext-build.yml. CUDA ships prebuilt modules
into the package and trusts them with SGLANG_RUST_BUILD_MODE=never; AMD cannot
build them off the test image, because sglang-radix-tree links the image's own
ROCm libtorch. So stage-a builds them in its container, and later stages copy
the loader's cache into theirs:

  export DEST  build _multimodal, mem_cache and mem_cache_inspection through
               load_rust_extension, then copy its artifacts/ tree to DEST
  import SRC   copy SRC's artifacts/ tree into this container's cache

Nothing here decides whether a module is reusable. The cache is keyed by the
loader's own fingerprint (Rust sources, cargo/rustc, Python ABI, torch version
and ABI), and the build mode stays auto, so a module whose fingerprint differs
from what this container computes is never looked up: it rebuilds as it would
have without the import.

Run inside the container through amd_ci_exec.sh, which gives the tests'
environment; amd_ci_rust_ext_cache.sh is the host-side entry point.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path


def _artifacts_root() -> Path:
    # The loader's own resolution, so this tracks SGLANG_CACHE_DIR with it.
    from sglang.srt.rust_extensions.loader import _cache_root

    return _cache_root(None) / "artifacts"


def _tree_core_supported() -> bool:
    import torch

    from sglang.srt.rust_extensions.torch_build import (
        _MAX_SUPPORTED_TORCH,
        _MIN_SUPPORTED_TORCH,
    )

    # The registry falls back to the Python TreeCore outside this range, so the
    # tests would never build it there.
    match = re.match(r"^(\d+)\.(\d+)", str(torch.__version__))
    return match is not None and (
        _MIN_SUPPORTED_TORCH
        <= (int(match.group(1)), int(match.group(2)))
        <= _MAX_SUPPORTED_TORCH
    )


def _load_tree_core(inspection: bool = False):
    # Importing this module builds mem_cache, so it must stay inside the try.
    from sglang.srt.mem_cache.rust_tree_core.extension import load_tree_core_extension

    return load_tree_core_extension(inspection=inspection)


def export(dest: Path) -> int:
    from sglang.srt.rust_extensions import load_rust_extension

    builds = {
        "_multimodal": lambda: load_rust_extension(
            "sglang.srt.rust_extensions._multimodal"
        ),
    }
    if _tree_core_supported():
        builds["mem_cache"] = _load_tree_core
        builds["mem_cache_inspection"] = lambda: _load_tree_core(inspection=True)
    else:
        print("torch is outside the Rust TreeCore range; not building mem_cache")

    for name, build in builds.items():
        try:
            build()
            print(f"built {name}")
        except Exception as exc:  # A later stage just builds it itself.
            print(f"::warning::could not build {name} for later stages: {exc}")

    source = _artifacts_root()
    if not source.is_dir():
        print("::warning::no Rust extension was built; nothing to export")
        return 1
    shutil.copytree(source, dest / "artifacts", dirs_exist_ok=True)
    for path in sorted((dest / "artifacts").rglob("*.so")):
        print(f"exported {path.relative_to(dest)}")
    return 0


def import_(source: Path) -> int:
    artifacts = source / "artifacts"
    if not artifacts.is_dir():
        print(f"::warning::{artifacts} does not exist; nothing to import")
        return 1
    destination = _artifacts_root()
    # Only files absent here: a container never holds a different module under
    # the same fingerprint, and leaving any existing one alone keeps this a no-op
    # for whatever is already in the cache.
    for path in sorted(artifacts.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(artifacts)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        print(f"imported {path.relative_to(source)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("export", "import"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.action == "export":
        return export(args.directory)
    return import_(args.directory)


if __name__ == "__main__":
    sys.exit(main())
