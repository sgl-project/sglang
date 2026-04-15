# Adapted from https://github.com/vllm-project/vllm-metal/blob/a06cd65a35b5c61c9a7f9d5f5ae00b30d9603379/vllm_metal/metal/build.py
# SPDX-License-Identifier: Apache-2.0
"""JIT build script for the native paged-attention Metal extension.

Compiles ``paged_ops.cpp`` + nanobind into a shared library that dispatches
Metal shaders through MLX's own command encoder.
"""

from __future__ import annotations

import logging
import subprocess
import sysconfig
from pathlib import Path

PARTITION_SIZE = 512

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_SRC = _THIS_DIR / "paged_ops.cpp"
_BUILD = _THIS_DIR / "build.py"
_EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
_CACHE_DIR = Path.home() / ".cache" / "vllm-metal"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_OUT = _CACHE_DIR / f"_paged_ops{_EXT_SUFFIX}"


def _find_package_path(name: str) -> Path:
    """Resolve a Python package's root directory."""
    import importlib

    mod = importlib.import_module(name)
    paths = getattr(mod, "__path__", None)
    if paths:
        return Path(list(paths)[0])
    f = getattr(mod, "__file__", None)
    if f:
        return Path(f).parent
    raise RuntimeError(f"Cannot locate package '{name}'")


def needs_rebuild() -> bool:
    """Return True if the .so is missing or older than the source."""
    if not _OUT.exists():
        return True
    latest_input_mtime = max(
        _SRC.stat().st_mtime,
        _BUILD.stat().st_mtime,
    )
    return _OUT.stat().st_mtime < latest_input_mtime


def build() -> Path:
    """JIT-build the native extension, returning the path to the .so."""
    if not needs_rebuild():
        return _OUT

    logger.info("Building native paged-attention extension ...")

    py_include = sysconfig.get_paths()["include"]
    nb_path = _find_package_path("nanobind")
    mlx_path = _find_package_path("mlx")
    mlx_include = mlx_path / "include"
    mlx_lib = mlx_path / "lib"
    metal_cpp = mlx_include / "metal_cpp"

    # Verify critical paths exist
    for p, label in [
        (py_include, "Python include"),
        (nb_path, "nanobind"),
        (mlx_include, "MLX include"),
        (mlx_lib / "libmlx.dylib", "MLX lib"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    nb_src = nb_path / "src" / "nb_combined.cpp"
    if not nb_src.exists():
        raise FileNotFoundError(f"nanobind source not found: {nb_src}")

    cmd = [
        "clang++",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-O2",
        "-fvisibility=default",
        f"-I{py_include}",
        f"-I{nb_path / 'include'}",
        f"-I{nb_path / 'src'}",
        f"-I{nb_path / 'ext' / 'robin_map' / 'include'}",
        f"-I{mlx_include}",
        f"-I{metal_cpp}",
        f"-L{mlx_lib}",
        "-lmlx",
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        f"-Wl,-rpath,{mlx_lib}",
        "-D_METAL_",
        "-DACCELERATE_NEW_LAPACK",
        f"-DVLLM_METAL_PARTITION_SIZE={PARTITION_SIZE}",
        "-undefined",
        "dynamic_lookup",
        str(nb_src),
        str(_SRC),
        "-o",
        str(_OUT),
    ]

    logger.info("  %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to build paged_ops extension:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    logger.info("Built %s", _OUT)
    return _OUT