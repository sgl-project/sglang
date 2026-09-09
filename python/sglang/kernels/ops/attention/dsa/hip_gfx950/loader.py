"""JIT build + load of the four gfx950 DSA-indexer kernels, at import time of
the first gated call -- never in a timed region or a graph capture.  Four
extensions because an extension holds one PYBIND11_MODULE."""

from __future__ import annotations

import logging
import os
import pathlib
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# The accepted configuration of each kernel.  Changing one of these means
# editing the kernel it belongs to; the sources reject any other value rather
# than mis-launching.
LOGITS_HIST_BITS = 12  # must match logits_hist_m<8, 12> and topk_transform.cu
LOGITS_BLOCKS_PER_ROW = 48
TOPK_G = 64


def _aiter_include_paths() -> List[str]:
    """qk_rope_hadamard_quant.cu includes aiter's hip_reduce.h and opus.hpp so the
    numeric helpers are the same code aiter's own fused indexer uses.
    """
    import aiter

    root = os.path.abspath(os.path.join(os.path.dirname(aiter.__file__), ".."))
    inc = os.path.join(root, "csrc", "include")
    if not os.path.isfile(os.path.join(inc, "hip_reduce.h")):
        raise RuntimeError(
            f"aiter C++ headers not found under {inc}; the gfx950 fused DSA "
            "indexer needs aiter's csrc/include on the include path"
        )
    return [inc]


def _stage(build_dir: str, sources: List[str]) -> List[str]:
    """Compile from a copy in the build directory: torch's ROCm hipify rewrites
    sources IN PLACE, which litters an installed package and fails outright on
    a read-only install.  Verified on torch 2.9.1+rocm7.2.0."""
    staged = []
    for s in sources:
        src, dst = os.path.join(_CSRC, s), os.path.join(build_dir, s)
        want = pathlib.Path(src).read_bytes()
        if not os.path.exists(dst) or pathlib.Path(dst).read_bytes() != want:
            # Every TP rank stages into the same directory, so the write must be atomic:
            # a partially written source is a build error in another rank.
            tmp = f"{dst}.{os.getpid()}.tmp"
            pathlib.Path(tmp).write_bytes(want)
            os.replace(tmp, dst)
        staged.append(dst)
    return staged


def _load(
    name: str,
    sources: List[str],
    extra_flags: Optional[List[str]] = None,
    std: str = "c++17",
    include_aiter: bool = False,
):
    """Build one extension for gfx950 alone, into torch's own extension cache."""
    from torch.utils.cpp_extension import _get_build_directory, load

    build_dir = _get_build_directory(name, verbose=False)

    # torch derives its --offload-arch list from this; without it every build
    # also emits gfx942.  Restored so we do not edit the process environment.
    prev = os.environ.get("PYTORCH_ROCM_ARCH")
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    try:
        return load(
            name=name,
            sources=_stage(build_dir, sources),
            build_directory=build_dir,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", f"-std={std}"]
            + (extra_flags or []),
            extra_include_paths=_aiter_include_paths() if include_aiter else [],
            verbose=False,
        )
    finally:
        if prev is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = prev


@lru_cache(maxsize=1)
def modules():
    """(gemv, qk, logits, topk) extension modules, or raise."""
    gemv = _load("sglang_dsa_gfx950_gemv", ["dual_gemv_bf16.cu"])
    # qk_rope_hadamard_quant.cu needs C++20 (aiter's opus.hpp) and aiter's headers.
    qk = _load(
        "sglang_dsa_gfx950_qk",
        ["qk_rope_hadamard_quant.cu"],
        extra_flags=["-Wno-unused-result"],
        std="c++20",
        include_aiter=True,
    )
    logits = _load(
        "sglang_dsa_gfx950_logits",
        ["paged_mqa_logits.cu"],
        extra_flags=[
            "-fno-honor-nans",
            "-mllvm",
            "-amdgpu-mfma-vgpr-form",
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-amdgpu-function-calls=false",
            "-Wno-unused-result",
        ],
    )
    topk = _load(
        "sglang_dsa_gfx950_topk",
        ["topk_transform.cu"],
        extra_flags=[f"-DDSA_TOPK_HIST_BITS={LOGITS_HIST_BITS}"],
    )
    return gemv, qk, logits, topk


@lru_cache(maxsize=1)
def modules_or_none():
    """Build once; on any failure log and return None so callers fall back."""
    try:
        return modules()
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        logger.warning(
            "gfx950 fused DSA indexer unavailable (build failed: %s); "
            "falling back to the standard ROCm indexer path",
            e,
        )
        return None
