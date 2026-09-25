"""JIT build + load of the four gfx950 DSA-indexer kernels, on the first gated
call -- never in a timed region or a graph capture.

Four modules rather than one because the flags differ per kernel: the logits
kernel wants its own -mllvm set, the qk kernel needs aiter's headers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# The accepted configuration of each kernel.  Changing one of these means
# editing the kernel it belongs to; the sources reject any other value rather
# than mis-launching.
LOGITS_HIST_BITS = 12  # must match logits_hist_m<8, 12> and topk_transform.cuh
LOGITS_BLOCKS_PER_ROW = 48
TOPK_G = 64


@cache_once
def modules() -> Tuple[Module, Module, Module, Module]:
    """(gemv, qk, logits, topk) modules, or raise. qk goes early: it is the one
    that can fail for a reason outside this tree (a missing or stale aiter)."""
    gemv = load_jit(
        "dsa_gfx950_dual_gemv",
        cuda_files=["dsa_gfx950/dual_gemv_bf16.cuh"],
        cuda_wrappers=[("dual_gemv_bf16", "DualGemvBf16Kernel::run")],
        extra_cuda_cflags=["-O3"],
    )
    # Includes aiter's hip_reduce.h and opus.hpp, so the numeric helpers are the
    # same code aiter's own fused indexer uses. opus.hpp needs the C++20 the JIT
    # defaults already pass.
    qk = load_jit(
        "dsa_gfx950_qk",
        cuda_files=["dsa_gfx950/qk_rope_hadamard_quant.cuh"],
        cuda_wrappers=[
            (
                "indexer_qk_rope_hadamard_quant_and_cache",
                "QkRopeHadamardQuantKernel::run",
            )
        ],
        extra_cuda_cflags=["-O3", "-Wno-unused-result"],
        extra_dependencies=["aiter"],
    )
    logits = load_jit(
        "dsa_gfx950_logits",
        cuda_files=["dsa_gfx950/paged_mqa_logits.cuh"],
        cuda_wrappers=[("logits_hist", "PagedMqaLogitsKernel::run")],
        extra_cuda_cflags=[
            "-O3",
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
    # The width is a module argument as well as a -D. The -D already keys the
    # cache; the argument keeps two widths in separate modules, so their exports
    # cannot collide in one process.
    topk = load_jit(
        "dsa_gfx950_topk",
        f"hist_bits{LOGITS_HIST_BITS}",
        cuda_files=["dsa_gfx950/topk_transform.cuh"],
        cuda_wrappers=[
            ("topk_transform", "TopKTransformKernel::run"),
            ("hist_stride", "TopKHistStride::run"),
        ],
        extra_cuda_cflags=["-O3", f"-DDSA_TOPK_HIST_BITS={LOGITS_HIST_BITS}"],
    )
    return gemv, qk, logits, topk


@cache_once
def _build() -> Tuple[Optional[Tuple[Module, ...]], Optional[Exception]]:
    """(modules, error) for the one build attempt.  Never raises."""
    try:
        return modules(), None
    except Exception as e:  # noqa: BLE001 - a build failure must never be fatal
        return None, e


@cache_once
def modules_or_none() -> Optional[Tuple[Module, ...]]:
    """Build once; on failure log and return None. Whether the caller falls back
    or refuses is the gate's decision, so this does not claim either."""
    mods, error = _build()
    if error is not None:
        logger.warning("gfx950 fused DSA indexer unavailable, build failed: %s", error)
    return mods


def build_error() -> Optional[Exception]:
    """The exception from the one build attempt, or None. Triggers the build if
    nothing has yet, so call it after modules_or_none() rather than before."""
    return _build()[1]
