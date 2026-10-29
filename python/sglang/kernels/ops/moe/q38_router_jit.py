# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""NVRTC builds, cached on disk by the digest of source, headers, options and toolchain,
and CUDA driver launches with programmatic dependent launch (PDL)."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
from typing import NamedTuple, Optional

import torch


def driver():
    from cuda.bindings import driver

    return driver


def check(result: tuple, what: str) -> tuple:
    if int(result[0]) != 0:
        raise RuntimeError(f"{what} failed: {result[0]!r}")
    return result[1:]


def include_dirs() -> list[str]:
    candidates: list[Path] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(variable)
        if root:
            candidates.append(Path(root) / "include")
    candidates.append(Path("/usr/local/cuda/include"))
    try:
        import nvidia

        for base in nvidia.__path__:
            candidates.extend(sorted(Path(base).glob("*/include")))
    except ImportError:
        pass
    found: list[str] = []
    for directory in candidates:
        if (directory / "cuda.h").is_file() and str(directory) not in found:
            found.append(str(directory))
            if (directory / "cccl" / "cuda" / "std").is_dir():
                found.append(str(directory / "cccl"))
    return found


def nvrtc_version() -> tuple[int, int]:
    from cuda.bindings import nvrtc

    major, minor = check(nvrtc.nvrtcVersion(), "nvrtcVersion")
    return int(major), int(minor)


def cache_dir() -> Path:
    """Where compiled cubins persist: SGLang's cache directory, as SGLang's JIT kernels."""
    root = os.environ.get("SGLANG_CACHE_DIR") or "~/.cache/sglang"
    return Path(os.path.expanduser(root)) / "sgl_nvidia_mega"


def compile_source(
    source: str,
    name: str,
    arch: str,
    options: tuple[str, ...],
    headers: Optional[dict[str, str]] = None,
) -> bytes:
    """Compile ``source`` to a cubin for ``arch`` with NVRTC, cached on disk by the
    hash of source, headers, options and NVRTC version. Refuses to run under CUDA-graph
    capture: every kernel compiles at build."""
    from cuda.bindings import nvrtc

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(f"compiling {name} inside CUDA-graph capture")

    headers = dict(headers or {})
    options = [f"--gpu-architecture={arch}", *options]
    options += [f"-I{d}" for d in include_dirs()]
    if nvrtc_version() >= (12, 9):
        options.append("--no-cache")
    header_text = [f"{k}\0{v}" for k, v in sorted(headers.items())]
    key = hashlib.sha256(
        "\0".join([source, *header_text, *options, repr(nvrtc_version())]).encode()
    ).hexdigest()
    cached = cache_dir() / f"{name}-{key[:24]}.cubin"
    if cached.is_file():
        return cached.read_bytes()
    (program,) = check(
        nvrtc.nvrtcCreateProgram(
            source.encode(),
            f"{name}.cu".encode(),
            len(headers),
            [text.encode() for text in headers.values()],
            [header.encode() for header in headers],
        ),
        "nvrtcCreateProgram",
    )
    try:
        encoded = [option.encode() for option in options]
        (status,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
        if int(status) != 0:
            (size,) = check(nvrtc.nvrtcGetProgramLogSize(program), "log size")
            log = b" " * size
            check(nvrtc.nvrtcGetProgramLog(program, log), "nvrtcGetProgramLog")
            raise RuntimeError(
                f"NVRTC failed to compile {name}: {log.decode(errors='replace')}"
            )
        (size,) = check(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
        cubin = b" " * size
        check(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
    finally:
        nvrtc.nvrtcDestroyProgram(program)
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(cubin)
    temporary.replace(cached)
    return cubin


class Kernel(NamedTuple):
    function: object
    block: tuple[int, int, int]
    shared_memory_bytes: int
    cubin: bytes


def load(cubin: bytes, symbol: str, block, shared: int) -> Kernel:
    cu = driver()
    (module,) = check(cu.cuModuleLoadData(cubin), "cuModuleLoadData")
    (function,) = check(cu.cuModuleGetFunction(module, symbol.encode()), symbol)
    if shared > 48 * 1024:
        check(
            cu.cuFuncSetAttribute(
                function,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared,
            ),
            "cuFuncSetAttribute",
        )
    return Kernel(function, tuple(block), shared, cubin)


def pointer(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


def null_or_pointer(tensor: Optional[torch.Tensor]) -> ctypes.c_void_p:
    return ctypes.c_void_p(0) if tensor is None else pointer(tensor)


def launch(
    kernel: Kernel,
    grid: tuple[int, int, int],
    arguments: list,
    stream: int,
    cluster: Optional[tuple[int, int, int]] = None,
) -> None:
    """Launch with programmatic stream serialization (PDL), in thread-block clusters
    of ``cluster`` blocks when given; graph-capturable."""
    cu = driver()
    addresses = [
        argument.getPtr()
        if isinstance(argument, cu.CUtensorMap)
        else ctypes.addressof(argument)
        for argument in arguments
    ]
    pointer_array = (ctypes.c_void_p * len(addresses))(*addresses)
    config = cu.CUlaunchConfig()
    config.gridDimX, config.gridDimY, config.gridDimZ = grid
    config.blockDimX, config.blockDimY, config.blockDimZ = kernel.block
    config.sharedMemBytes = kernel.shared_memory_bytes
    config.hStream = cu.CUstream(stream)
    attribute = cu.CUlaunchAttribute()
    attribute.id = (
        cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION
    )
    attribute.value.programmaticStreamSerializationAllowed = 1
    attributes = [attribute]
    if cluster is not None:
        dims = cu.CUlaunchAttribute()
        dims.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        dims.value.clusterDim.x, dims.value.clusterDim.y, dims.value.clusterDim.z = (
            cluster
        )
        attributes.append(dims)
    config.attrs = attributes
    config.numAttrs = len(attributes)
    check(
        cu.cuLaunchKernelEx(
            config, kernel.function, ctypes.addressof(pointer_array), 0
        ),
        "cuLaunchKernelEx",
    )
