from __future__ import annotations

import fcntl
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

logger = logging.getLogger(__name__)


def _get_build_directory(name: str) -> Path:
    try:
        from torch.utils.cpp_extension import _get_build_directory

        return Path(_get_build_directory(name, False))
    except (ImportError, AttributeError):
        from torch.utils.cpp_extension import get_default_build_root

        root = os.environ.get("TORCH_EXTENSIONS_DIR") or get_default_build_root()
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            cu_str = (
                "cpu"
                if torch.version.cuda is None
                else f"cu{torch.version.cuda.replace('.', '')}"
            )
            py_str = (
                f"py{sys.version_info.major}{sys.version_info.minor}"
                f"{getattr(sys, 'abiflags', '')}"
            )
            root = os.path.join(root, f"{py_str}_{cu_str}")
        return Path(root) / name


def _is_recoverable_load_error(
    exc: BaseException, name: str, build_directory: Path
) -> bool:
    message = str(exc).lower()
    current = exc.__cause__ or exc.__context__
    while current is not None:
        message += f"\n{current}".lower()
        current = current.__cause__ or current.__context__

    if any(
        marker in message
        for marker in (
            "error building extension",
            "error compiling objects for extension",
            "ninja",
            "nvcc",
            "gcc",
            "g++",
            "fatal error:",
            "compilation terminated",
        )
    ):
        return False

    if not any(
        marker in message
        for marker in (str(build_directory / f"{name}.so").lower(), f"{name}.so")
    ):
        return False

    return any(
        marker in message
        for marker in (
            "undefined symbol",
            "cannot open shared object file",
            "no such file or directory",
            "file too short",
            "invalid elf header",
            "wrong elf class",
            "elf load command",
            "dlopen",
            "version `glibcxx",
        )
    )


@contextmanager
def _extension_build_lock(build_directory: Path) -> Iterator[None]:
    """Serialize builds and discard PyTorch lock files left by dead processes."""
    build_directory.parent.mkdir(parents=True, exist_ok=True)
    lock_path = build_directory.parent / f".{build_directory.name}.sglang.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            torch_lock_path = build_directory / "lock"
            if torch_lock_path.exists():
                logger.warning(
                    "Removing stale PyTorch extension lock for %s at %s",
                    build_directory.name,
                    torch_lock_path,
                )
                torch_lock_path.unlink(missing_ok=True)
            build_directory.mkdir(parents=True, exist_ok=True)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_extension_with_recovery(
    name: str,
    sources: Sequence[str],
    extra_cflags: Sequence[str] | None = None,
    extra_cuda_cflags: Sequence[str] | None = None,
    verbose: bool = False,
) -> Any:
    from torch.utils.cpp_extension import load

    build_directory = _get_build_directory(name)
    load_kwargs = {
        "name": name,
        "sources": list(sources),
        "extra_cflags": None if extra_cflags is None else list(extra_cflags),
        "extra_cuda_cflags": (
            None if extra_cuda_cflags is None else list(extra_cuda_cflags)
        ),
        "build_directory": str(build_directory),
        "verbose": verbose,
    }

    with _extension_build_lock(build_directory):
        try:
            return load(**load_kwargs)
        except Exception as exc:
            if not _is_recoverable_load_error(exc, name, build_directory):
                raise

            logger.warning(
                "Detected a stale or broken JIT extension for %s at %s; clearing "
                "its cache and retrying once.",
                name,
                build_directory,
            )
            sys.modules.pop(name, None)
            if build_directory.exists():
                shutil.rmtree(build_directory)
            build_directory.mkdir(parents=True)
            return load(**load_kwargs)


__all__ = ["load_extension_with_recovery"]
