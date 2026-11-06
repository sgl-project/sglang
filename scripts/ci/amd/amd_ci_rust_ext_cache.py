#!/usr/bin/env python3
"""Cache runtime-built Rust extensions on the persistent /sgl-data volume.

The AMD counterpart of _pr-test-rust-ext-build.yml. CUDA ships prebuilt modules
into the package and trusts them with SGLANG_RUST_BUILD_MODE=never; AMD cannot
build them off the test image, because sglang-radix-tree links the image's own
ROCm libtorch. So the first job to need a module builds it in its container and
publishes it, and later jobs on the same image copy it into their own cache:

  sync --root ROOT PACKAGE=ENTRY...
      For each Cargo package, copy ROOT/ENTRY (if any) into the loader's
      artifacts/PACKAGE/, then load the package's modules through
      load_rust_extension. Each load finds its module in the cache or builds it.
      If ROOT/ENTRY did not supply every module that loaded, publish the loaded
      ones as ROOT/ENTRY, replacing an entry that fell short.

ENTRY is computed by amd_ci_install_dependency.sh from the source trees, the
image and the installed torch. Nothing here decides whether a module is
reusable: the cache is keyed by the loader's own fingerprint (Rust sources,
cargo/rustc, Python ABI, torch version and ABI), and the build mode stays auto,
so a module whose fingerprint differs from what this container computes is
never looked up: it rebuilds as it would have without the cache. Loading every
module here is what tells a hit from an entry the key matched but the loader
cannot use (a different rustc, say, or files lost to an interrupted prune).
Such an entry is replaced, not refreshed, so it never keeps a key from caching.

Publishing mirrors the AOT wheel cache: copy into a temporary directory under
ROOT and rename it onto ENTRY, so readers only see complete entries and the
loser of a race drops its copy. Entries unused for 7 days are pruned, renamed
aside first so an interrupted prune never leaves a partial entry under its
name. Any /sgl-data error is a warning; the modules then build at runtime as
before.

Run inside the container through amd_ci_exec.sh, which gives the tests'
environment.
"""

import argparse
import errno
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

PRUNE_AFTER_SECONDS = 7 * 24 * 3600


def _artifacts_root() -> Path:
    # The loader's own resolution, so this tracks SGLANG_CACHE_DIR with it.
    from sglang.srt.rust_extensions.loader import _cache_root

    return _cache_root(None) / "artifacts"


def _tree_core_supported() -> bool:
    # Without importing torch, or sglang, which imports it: that takes seconds on
    # ROCm, wasted when the registry falls back to the Python TreeCore and the
    # tests never build it. torch_build.py itself only imports the stdlib.
    import importlib.util
    from importlib.metadata import PackageNotFoundError, version

    spec = importlib.util.find_spec("sglang")
    path = Path(spec.submodule_search_locations[0]) / "srt/rust_extensions"
    spec = importlib.util.spec_from_file_location(
        "_torch_build", path / "torch_build.py"
    )
    torch_build = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = torch_build  # @dataclass looks itself up there.
    spec.loader.exec_module(torch_build)
    try:
        torch_version = version("torch")
    except PackageNotFoundError:
        torch_version = "none"
    match = re.match(r"^(\d+)\.(\d+)", torch_version)
    supported = match is not None and (
        torch_build._MIN_SUPPORTED_TORCH
        <= (int(match.group(1)), int(match.group(2)))
        <= torch_build._MAX_SUPPORTED_TORCH
    )
    if not supported:
        print(f"[RUST] torch {torch_version} is outside the Rust TreeCore range")
    return supported


def _load_tree_core(inspection: bool = False):
    # Importing this module builds mem_cache, so it must stay inside the try.
    from sglang.srt.mem_cache.rust_tree_core.extension import load_tree_core_extension

    return load_tree_core_extension(inspection=inspection)


def _load_multimodal():
    from sglang.srt.rust_extensions import load_rust_extension

    return load_rust_extension("sglang.srt.rust_extensions._multimodal")


def _loaders(package: str) -> dict:
    """The modules the tests load from PACKAGE, by name."""
    if package == "sglang-mm":
        return {"_multimodal": _load_multimodal}
    if package == "sglang-radix-tree":
        if not _tree_core_supported():
            return {}
        return {
            "mem_cache": _load_tree_core,
            "mem_cache_inspection": lambda: _load_tree_core(inspection=True),
        }
    print(f"::warning::unknown Rust package {package}; not caching it")
    return {}


def _load(package: str, loaders: dict) -> dict:
    """Load each module, building it if the cache lacks it.

    Returns the files they were loaded from, relative to artifacts/PACKAGE.
    """
    artifacts = (_artifacts_root() / package).resolve()
    loaded = {}
    for name, load in loaders.items():
        try:
            path = Path(load().__file__).resolve()
        except Exception as exc:  # The tests just build it themselves.
            print(f"::warning::could not build {name}: {exc}")
            continue
        if artifacts not in path.parents:
            print(f"[RUST] {name} was loaded from {path}; not caching it")
            continue
        loaded[name] = path.relative_to(artifacts)
    return loaded


def _copy_missing(source: Path, destination: Path) -> list:
    # Only files absent here: a container never holds a different module under
    # the same fingerprint, and leaving any existing one alone keeps this a no-op
    # for whatever is already in the cache. Each file is staged and renamed, so
    # a copy cut short (e.g. by a concurrent prune) never leaves a partial .so
    # at a path the loader would load. Unlike copytree, the first error is
    # raised as is.
    copied = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.{os.getpid()}.import")
        try:
            shutil.copy2(path, staging)
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
        copied.append(path)
    return copied


def _reuse(cached: Path, package: str) -> bool:
    """Copy CACHED into the loader's cache; False if any file could not be."""
    try:
        copied = _copy_missing(cached, _artifacts_root() / package)
    except OSError as exc:
        print(f"::warning::could not reuse {cached}: {exc}")
        return False
    for path in copied:
        print(f"[RUST] Imported {path.relative_to(cached.parent)}")
    return True


def _remove(path: Path, reason: str) -> None:
    # Rename first: a name is either a complete entry or gone, even if the job
    # is killed during rmtree. A leftover .tmp.* is pruned 7 days later.
    aside = path.with_name(f".tmp.{reason}.{os.getpid()}.{path.name}")
    os.rename(path, aside)
    shutil.rmtree(aside)


def _discard(cached: Path, inode: int) -> None:
    # Only the entry this job read: another job may have replaced it already.
    try:
        if cached.stat().st_ino == inode:
            _remove(cached, "stale")
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            print(f"::warning::could not remove {cached}: {exc}")


def _publish(root: Path, package: str, entry: str, files: list) -> bool:
    source = _artifacts_root() / package
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=root, prefix=".tmp."))
    except OSError as exc:
        print(f"::warning::could not store {entry} in {root}: {exc}")
        return False
    try:
        tmp.chmod(0o755)
        for relative in files:
            (tmp / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, tmp / relative)
        # Like `mv -T`: fails if another job already published this entry.
        os.rename(tmp, root / entry)
        print(f"[RUST] Stored {entry}")
        return True
    except OSError as exc:
        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
            print(f"[RUST] Another job already stored {entry}")
        else:
            print(f"::warning::could not store {entry} in {root}: {exc}")
        shutil.rmtree(tmp, ignore_errors=True)
        return False


def _prune(root: Path) -> None:
    # The AOT wheel cache's rule (`find -mindepth 1 -maxdepth 1 -mtime +7`);
    # it also clears .tmp.* directories left by killed jobs.
    cutoff = time.time() - PRUNE_AFTER_SECONDS
    try:
        children = list(root.iterdir())
    except OSError:
        return
    for child in children:
        try:
            if child.lstat().st_mtime >= cutoff:
                continue
            if child.is_dir() and not child.is_symlink():
                _remove(child, "prune")
            else:
                child.unlink()
            print(f"[RUST] Pruned {child.name}")
        except OSError:
            pass


def _sync(root: Path, package: str, entry: str) -> bool:
    """Returns whether ROOT changed."""
    loaders = _loaders(package)
    if not loaders:
        print(f"[RUST] Not caching {package}")
        return False
    cached = root / entry
    try:
        inode = cached.stat().st_ino if cached.is_dir() else None
    except OSError:
        inode = None
    if inode is None:
        print(f"[RUST] Building {package} (cache key: {entry})")
        reused = False
    else:
        reused = _reuse(cached, package)
    loaded = _load(package, loaders)
    if not loaded:
        return False
    if reused and all((cached / path).is_file() for path in loaded.values()):
        # Refreshed only after the loads, so an unusable entry is never kept.
        try:
            os.utime(cached)
        except OSError as exc:
            print(f"::warning::could not refresh {cached}: {exc}")
        print(f"[RUST] Reusing cached {entry}")
        return False
    if inode is not None:
        print(
            f"::warning::cached {entry} did not supply every module this "
            "container loads; replacing it"
        )
        _discard(cached, inode)
    if len(loaded) < len(loaders):
        print(f"[RUST] Caching only {', '.join(loaded)} from {package}")
    return _publish(root, package, entry, sorted(loaded.values()))


def sync(root: Path, entries: dict) -> None:
    changed = False
    for package, entry in entries.items():
        changed |= _sync(root, package, entry)
    if changed:
        _prune(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("sync",))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("entries", nargs="+", metavar="PACKAGE=ENTRY")
    args = parser.parse_args()
    entries = dict(item.split("=", 1) for item in args.entries)
    sys.stdout.reconfigure(line_buffering=True)  # In order with cargo's output.
    try:
        sync(args.root, entries)
    except Exception as exc:  # Never fail the job: the tests build at runtime.
        print(f"::warning::Rust extension cache failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
