# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Lazy native extension loading for OmniDreams single-view acceleration."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator

# Vendored layout: this loader and the native source tree are siblings under
# ``multimodal_gen/native/`` (FlashDreams kept them one level apart).
_ROOT = Path(__file__).resolve().parent / "omnidreams_singleview"
_NATIVE_BUILD_PATH = _ROOT / "tools" / "native_build.py"
_SOURCE_DIR = _ROOT / "src"
_PYTHON_DIR = _ROOT / "python"
_EXTENSION_SOURCE = _SOURCE_DIR / "omnidreams_singleview_ext.cpp"
_NATIVE_PRIMITIVES_SOURCE = _SOURCE_DIR / "native_primitives.cpp"
_NATIVE_PRIMITIVES_CUDA_SOURCE = _SOURCE_DIR / "native_primitives_cuda.cu"
_NATIVE_COMMON_HEADER_DIR = _SOURCE_DIR / "native_common"
_DIT_STREAMING_DIR = _SOURCE_DIR / "dit_streaming"
_DIT_STREAMING_KERNEL_DIR = _DIT_STREAMING_DIR / "kernels"
_DIT_STREAMING_PYEXT_DIR = _DIT_STREAMING_DIR / "pyext"
_DIT_STREAMING_COMMON_DIR = _DIT_STREAMING_DIR / "common"
_VAE_STREAMING_DIR = _SOURCE_DIR / "vae_streaming"
_PYTORCH_MAX_JOBS_ENV = "MAX_JOBS"
_DEFAULT_MAX_JOBS_CAP = 8
_NATIVE_CUDA_ARCH_LIST_ENV = "OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST"
_PYTORCH_CUDA_ARCH_LIST_ENV = "TORCH_CUDA_ARCH_LIST"
_DEFAULT_CUDA_ARCH_LIST = "12.0a"

_native_build_module: ModuleType | None = None
_extension: ModuleType | None = None
_extension_load_error: Exception | None = None
_state_lock = threading.RLock()


def _native_build() -> ModuleType:
    global _native_build_module
    with _state_lock:
        if _native_build_module is not None:
            return _native_build_module

        spec = importlib.util.spec_from_file_location(
            "omnidreams_singleview_native_build",
            _NATIVE_BUILD_PATH,
        )
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Cannot import native build helpers from {_NATIVE_BUILD_PATH}"
            )

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _native_build_module = module
        return module


def validate_thirdparty() -> dict[str, Any]:
    """Validate native source checkouts and return their pinned provenance.

    If a prebuilt .so already exists, skip stamp validation entirely —
    the tree_sha256 in stamps drifts after rsync / git operations, and
    the only thing that matters is that the 3rdparty headers are present."""

    # Fast path: prebuilt .so exists → skip stamp verification.
    _build_dir = _ROOT / "build" / "torch_extensions"
    if any(glob.glob(str(_build_dir / "omnidreams_singleview_native_*/*.so"))):
        return _thirdparty_info_no_validation()

    return {
        name: info.as_dict()
        for name, info in _native_build().validate_thirdparty().items()
    }


def sync_thirdparty(*, force: bool = False) -> dict[str, Any]:
    """Synchronize native source checkouts and return their pinned provenance."""

    return {
        name: info.as_dict()
        for name, info in _native_build().sync_thirdparty(force=force).items()
    }


def load_python_module(name: str) -> ModuleType:
    """Load a helper module shipped with the single-view native sources."""

    if not name.isidentifier():
        raise ValueError(
            f"Native helper module name must be an identifier, got {name!r}"
        )
    path = _PYTHON_DIR / f"{name}.py"
    if not path.is_file():
        raise ImportError(f"Unknown OmniDreams single-view native helper {name!r}")
    module_name = f"omnidreams_singleview_native_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot import OmniDreams single-view native helper from {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    python_dir = str(_PYTHON_DIR)
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    spec.loader.exec_module(module)
    return module


def _first_library_dir(library_name: str, candidates: tuple[Path, ...]) -> Path | None:
    for directory in candidates:
        if (directory / library_name).is_file():
            return directory
    return None


def build_info(
    build_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return native source provenance without compiling the extension."""

    return _native_build().native_provenance(build_root=build_root)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension_sources() -> list[Path]:
    return [
        _EXTENSION_SOURCE,
        _NATIVE_PRIMITIVES_SOURCE,
        _NATIVE_PRIMITIVES_CUDA_SOURCE,
        _DIT_STREAMING_DIR / "streaming_dit_bindings.cpp",
        _VAE_STREAMING_DIR / "vae_streaming_bindings.cpp",
        _VAE_STREAMING_DIR / "lightvae_ops.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_ops.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_direct_stages.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_warp_mma_stages.cu",
        _VAE_STREAMING_DIR / "lightvae_fp8_attention.cu",
        _DIT_STREAMING_PYEXT_DIR / "streaming_dit_bridge.cu",
        _DIT_STREAMING_PYEXT_DIR / "sage3_blackwell_api_shim.cu",
        _DIT_STREAMING_PYEXT_DIR / "sage3_fp4_quant_shim.cu",
        _DIT_STREAMING_KERNEL_DIR / "attention.cu",
        _DIT_STREAMING_KERNEL_DIR / "block_quant.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_adaln_lora.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_block.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_flash.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_flash_tc.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_tc_probe.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_fp8_two_gemm.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_gemm_bf16.cu",
        _DIT_STREAMING_KERNEL_DIR / "cosmos_modulate.cu",
        _DIT_STREAMING_KERNEL_DIR / "ops.cu",
        _DIT_STREAMING_KERNEL_DIR / "sage3_attention.cu",
        _DIT_STREAMING_KERNEL_DIR / "transformer_block.cu",
    ]


def _extension_fingerprint_sources() -> list[Path]:
    return [
        *_extension_sources(),
        *sorted(_NATIVE_COMMON_HEADER_DIR.glob("*.h")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.h")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.cuh")),
        *sorted(_DIT_STREAMING_DIR.rglob("*.hpp")),
        *sorted(_VAE_STREAMING_DIR.rglob("*.h")),
        *sorted(_VAE_STREAMING_DIR.rglob("*.hpp")),
        *sorted(_PYTHON_DIR.glob("*.py")),
    ]


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for source in _extension_fingerprint_sources():
        digest.update(source.relative_to(_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(source).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _extension_name(thirdparty_info: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(_source_fingerprint().encode("ascii"))
    digest.update(json.dumps(thirdparty_info, sort_keys=True).encode("utf-8"))
    return f"omnidreams_singleview_native_{digest.hexdigest()[:12]}"


def _validate_max_jobs(value: int | str) -> str:
    text = str(value).strip()
    try:
        jobs = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Native max jobs must be a positive integer, got {value!r}"
        ) from exc
    if jobs < 1:
        raise ValueError(f"Native max jobs must be a positive integer, got {value!r}")
    return str(jobs)


def _resolved_max_jobs(max_jobs: int | str | None) -> str | None:
    if max_jobs is not None:
        return _validate_max_jobs(max_jobs)
    if os.environ.get(_PYTORCH_MAX_JOBS_ENV):
        return None
    return str(min(os.cpu_count() or 1, _DEFAULT_MAX_JOBS_CAP))


def _normalize_blackwell_arch(arch_list: str) -> str:
    """Force the arch-specific ``a`` target on consumer Blackwell tokens.

    The FP8 CUTLASS GEMMs use arch-conditional MMA atoms that only resolve to
    real instructions under ``sm_120a`` (consumer Blackwell: RTX 50xx / RTX PRO
    6000).  A bare ``12.0``/``12.1`` token compiles them to a device-side trap
    ("Arch conditional MMA instruction used without targeting appropriate
    compute capability"), aborting at the first FP8 GEMM.  Normalize those
    tokens (``12.0`` -> ``12.0a``, ``8.9;12.0`` -> ``8.9;12.0a``); leave every
    other token untouched.
    """
    tokens: list[str] = []
    for raw in arch_list.replace(",", ";").split(";"):
        tok = raw.strip()
        if not tok:
            continue
        base, sep, ptx = tok.partition("+")
        if base in ("12.0", "12.1"):
            base += "a"
        tokens.append(base + sep + ptx)
    return ";".join(tokens)


def _resolved_cuda_arch_list() -> str | None:
    env = os.environ.get(_PYTORCH_CUDA_ARCH_LIST_ENV)
    if env:
        normalized = _normalize_blackwell_arch(env)
        # Only override the caller env when normalization changed something
        # (i.e. it lacked the required ``a`` suffix on a Blackwell token).
        return normalized if normalized != env else None
    raw = os.environ.get(_NATIVE_CUDA_ARCH_LIST_ENV, _DEFAULT_CUDA_ARCH_LIST)
    return _normalize_blackwell_arch(raw)


def _effective_cuda_arch_list() -> str:
    return _normalize_blackwell_arch(
        os.environ.get(
            _PYTORCH_CUDA_ARCH_LIST_ENV,
            os.environ.get(_NATIVE_CUDA_ARCH_LIST_ENV, _DEFAULT_CUDA_ARCH_LIST),
        )
    )


def _python_package_dir(package: str) -> Path | None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        return None
    locations = list(spec.submodule_search_locations)
    if not locations:
        return None
    return Path(locations[0])


@contextlib.contextmanager
def _scoped_torch_max_jobs(max_jobs: int | str | None) -> Iterator[None]:
    resolved = _resolved_max_jobs(max_jobs)
    if resolved is None:
        yield
        return

    previous = os.environ.get(_PYTORCH_MAX_JOBS_ENV)
    os.environ[_PYTORCH_MAX_JOBS_ENV] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PYTORCH_MAX_JOBS_ENV, None)
        else:
            os.environ[_PYTORCH_MAX_JOBS_ENV] = previous


@contextlib.contextmanager
def _scoped_cuda_arch_list() -> Iterator[None]:
    resolved = _resolved_cuda_arch_list()
    if resolved is None:
        yield
        return

    previous = os.environ.get(_PYTORCH_CUDA_ARCH_LIST_ENV)
    os.environ[_PYTORCH_CUDA_ARCH_LIST_ENV] = resolved
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PYTORCH_CUDA_ARCH_LIST_ENV, None)
        else:
            os.environ[_PYTORCH_CUDA_ARCH_LIST_ENV] = previous


def load_extension(
    build_root: Path | str | None = None,
    *,
    max_jobs: int | str | None = None,
    verbose: bool = False,
) -> ModuleType | None:
    """Compile and load the CUDA native extension on demand.

    PyTorch's extension builder uses ``MAX_JOBS`` for Ninja fanout. If the caller
    has not already set it, this loader sets a modest default cap to avoid
    runaway memory use in local clean builds.

    Returns ``None`` if the extension cannot be built on the current host. The
    full exception is retained and exposed through ``extension_load_error()``.
    """

    global _extension, _extension_load_error
    with _state_lock:
        if _extension is not None:
            return _extension
        _extension_load_error = None

        # Fast path: prebuilt .so already exists
        prebuilt = _load_prebuilt_extension()
        if prebuilt is not None:
            _extension = prebuilt
            return _extension

        try:
            from torch.utils.cpp_extension import load as load_torch_extension

            thirdparty_info = validate_thirdparty()
            extension_name = _extension_name(thirdparty_info)
            cutlass_dir = Path(thirdparty_info["cutlass"]["path"])
            cutlass_include = cutlass_dir / "include"
            cudnn_frontend_include = (
                Path(thirdparty_info["cudnn-frontend"]["path"]) / "include"
            )
            cudnn_package_dir = _python_package_dir("nvidia.cudnn")
            pip_cuda_include = (
                cudnn_package_dir.parent / "cu13" / "include"
                if cudnn_package_dir is not None
                else None
            )
            cudnn_include = (
                cudnn_package_dir / "include"
                if cudnn_package_dir is not None
                and (cudnn_package_dir / "include" / "cudnn.h").is_file()
                else None
            )
            cudnn_lib = (
                cudnn_package_dir / "lib"
                if cudnn_package_dir is not None
                and (cudnn_package_dir / "lib" / "libcudnn.so.9").is_file()
                else None
            )
            cuda_driver_lib = _first_library_dir(
                "libcuda.so",
                (
                    Path("/usr/lib/wsl/lib"),
                    Path("/usr/lib/x86_64-linux-gnu"),
                    Path("/usr/local/cuda/lib64"),
                ),
            )
            # sgl-kernel fp8_scaled_mm is an exported symbol in the installed
            # wheel's sm100/common_ops.abi3.so (default visibility). Link it via
            # -L/-l: + rpath so the shim (sgl_gemm_shim.cuh) calls it directly.
            sgl_kernel_so = None
            try:
                import sgl_kernel as _sgl_kernel  # noqa: F401
                _so = (Path(_sgl_kernel.__file__).resolve().parent
                       / "sm100" / "common_ops.abi3.so")
                if _so.is_file():
                    sgl_kernel_so = (
                        "-L" + str(_so.parent),
                        "-Wl,-rpath," + str(_so.parent),
                        "-l:common_ops.abi3.so",
                    )
            except Exception:
                sgl_kernel_so = None
            # sgl-kernel's sage3_ops.so (separate SM120a-gated library,
            # cmake/sage3.cmake) exports the upstream SageAttention symbols
            # (mha_fwd, scaled_fp4_quant / _permute / _trans). Link it so the
            # sgl_sage3_shim.cuh calls resolve cross-.so (no more #include of
            # upstream SageAttention .cu).
            sage3_ops_so = None
            try:
                import sgl_kernel as _sgl_kernel  # noqa: F401
                _sa = (Path(_sgl_kernel.__file__).resolve().parent
                       / "sage3_ops.so")
                if _sa.is_file():
                    sage3_ops_so = (
                        "-L" + str(_sa.parent),
                        "-Wl,-rpath," + str(_sa.parent),
                        "-l:sage3_ops.so",
                    )
            except Exception:
                sage3_ops_so = None
            extension_build_dir = _native_build().torch_extension_build_dir(
                extension_name,
                build_root=build_root,
            )
            extension_build_dir.mkdir(parents=True, exist_ok=True)

            with _scoped_torch_max_jobs(max_jobs), _scoped_cuda_arch_list():
                _extension = load_torch_extension(
                    name=extension_name,
                    sources=[str(source) for source in _extension_sources()],
                    build_directory=str(extension_build_dir),
                    extra_include_paths=[
                        str(_SOURCE_DIR),
                        str(_DIT_STREAMING_DIR),
                        str(_DIT_STREAMING_KERNEL_DIR),
                        str(_DIT_STREAMING_PYEXT_DIR),
                        str(_DIT_STREAMING_COMMON_DIR),
                        str(_VAE_STREAMING_DIR),
                        str(cutlass_include),
                        str(cutlass_dir / "tools" / "util" / "include"),
                        str(cutlass_dir / "examples" / "common"),
                        str(cutlass_dir / "examples" / "41_fused_multi_head_attention"),
                        str(cudnn_frontend_include),
                        *([] if cudnn_include is None else [str(cudnn_include)]),
                        *([] if pip_cuda_include is None or not pip_cuda_include.is_dir() else [str(pip_cuda_include)]),
                    ],
                    extra_cflags=[
                        "-O3",
                        "-std=c++20",
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
                        "-DOMNIDREAMS_SINGLEVIEW_USE_CUTLASS",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=1",
                        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SHA="
                        f'\\"{thirdparty_info["cutlass"]["commit"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUTLASS_SOURCE_SHA="
                        f'\\"{thirdparty_info["cutlass"]["source_sha256"]}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SOURCE_SHA="
                        f'\\"{_file_sha256(_EXTENSION_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_SOURCE_FINGERPRINT_SHA="
                        f'\\"{_source_fingerprint()}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_NATIVE_PRIMITIVES_SOURCE_SHA="
                        f'\\"{_file_sha256(_NATIVE_PRIMITIVES_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUDA_SOURCE_SHA="
                        f'\\"{_file_sha256(_NATIVE_PRIMITIVES_CUDA_SOURCE)}\\"',
                        "-DOMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST="
                        f'\\"{_effective_cuda_arch_list()}\\"',
                    ],
                    extra_cuda_cflags=[
                        "-O3",
                        "-std=c++20",
                        "--expt-relaxed-constexpr",
                        "--expt-extended-lambda",
                        "-lineinfo",
                        "--use_fast_math",
                        "-U__CUDA_NO_HALF_OPERATORS__",
                        "-U__CUDA_NO_HALF_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                        "-DQBLKSIZE=128",
                        "-DKBLKSIZE=128",
                        "-DCTA256",
                        "-DDQINRMEM",
                        "-DEXECMODE=0",
                        "-DNDEBUG",
                        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                        "-DOMNIDREAMS_SINGLEVIEW_WITH_CUDA",
                        "-DOMNIDREAMS_SINGLEVIEW_USE_CUTLASS",
                        "-DOMNIDREAMS_SINGLEVIEW_HAS_SAGE3=1",
                    ],
                    extra_ldflags=[
                        *([] if sgl_kernel_so is None else list(sgl_kernel_so)),
                        *([] if sage3_ops_so is None else list(sage3_ops_so)),
                        "-lcublas",
                        "-lcublasLt",
                        *([] if cudnn_lib is None else [f"-L{cudnn_lib}"]),
                        "-lcudnn" if cudnn_lib is None else "-l:libcudnn.so.9",
                        *([] if cuda_driver_lib is None else [f"-L{cuda_driver_lib}"]),
                        "-lcuda",
                        "-lnvrtc",
                    ],
                    with_cuda=True,
                    verbose=verbose,
                )
        except Exception as exc:  # pragma: no cover - environment-specific build path
            _extension_load_error = exc
            # Fallback: try loading a previously compiled .so
            _extension = _load_prebuilt_extension()
            if _extension is not None:
                _extension_load_error = None
                return _extension
            return None
        return _extension


def _thirdparty_info_no_validation() -> dict[str, Any]:
    """Return 3rdparty paths without stamp validation (prebuilt .so exists)."""
    manifest_path = _ROOT / "thirdparty_sources.json"
    manifest = json.loads(manifest_path.read_text())
    info: dict[str, Any] = {}
    for src in manifest["sources"]:
        d = _ROOT / "3rdparty" / src["directory"]
        info[src["name"]] = {
            "name": src["name"],
            "path": str(d),
            "commit": src["commit"],
            "source_sha256": "",
            "tree_sha256": "",
            "stamp_path": "",
        }
    return info


def _load_prebuilt_extension() -> ModuleType | None:
    """Try to load a previously compiled .so from the build directory."""
    # Add pip CUDA + torch lib paths so the prebuilt .so can resolve
    # libcudnn.so.9 and libc10.so at load time.
    _extra_lib = []
    _cudnn_pkg = _python_package_dir("nvidia.cudnn")
    if _cudnn_pkg is not None:
        _extra_lib.append(str(_cudnn_pkg / "lib"))
    # Torch libs (libc10, libtorch, etc.)
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib):
        _extra_lib.append(_torch_lib)
    if _extra_lib:
        _existing = os.environ.get("LD_LIBRARY_PATH", "")
        _parts = [p for p in _existing.split(":") if p]
        for lib in _extra_lib:
            if lib not in _parts:
                os.environ["LD_LIBRARY_PATH"] = lib + ":" + _existing

    build_dir = _ROOT / "build" / "torch_extensions"
    for pattern in ("omnidreams_singleview_native_*/omnidreams*.so",
                     "*/omnidreams*.so"):
        candidates = sorted(glob.glob(str(build_dir / pattern)))
        for so_path in candidates:
            try:
                # PyTorch C++ extension exports PyInit_<dirname>(), so match
                # the directory name, not a hardcoded string.
                _mod_name = os.path.basename(os.path.dirname(so_path))
                spec = importlib.util.spec_from_file_location(_mod_name, so_path)
                if spec is None or spec.loader is None:
                    continue
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "optimized_dit_forward"):
                    return mod
            except Exception:
                continue
    return None


def extension_load_error() -> Exception | None:
    """Return the last native extension load error, if any."""

    with _state_lock:
        return _extension_load_error
