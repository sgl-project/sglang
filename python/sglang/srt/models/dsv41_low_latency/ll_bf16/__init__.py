# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Low-M BF16 x BF16 -> FP32 GEMM in CuTe DSL: ``out[M, N] = x[M, K] @ w[N, K].T``.

Two kernels: a one-CTA-per-column dot product (M and K compiled in) and a
clustered split-K MMA kernel (shape-dynamic, compiled per (split_k, stages)).
Kernels (_dotprod.py, _splitk.py) and the dispatch are taken unchanged from vLLM
(vllm-project/vllm@75e3b17c, vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py),
with the tuned table of the DSV4.1 compressor projections.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any, Literal, NamedTuple

import torch

logger = logging.getLogger(__name__)

_DEFAULT_DOTPROD_BS = 128
_DEFAULT_DOTPROD_MAX_M = 4
_DEFAULT_SPLITK_CONFIG = (6, 4)
# Narrow outputs give split-K too few CTAs (one per 16 columns and split).
_DOTPROD_MAX_N = 128

# (K, N) -> {M: ("dotprod", bs) | ("splitk", (split_k, num_stages))}, GB300, cold weights,
# CUDA graph, PDL; listed where it beat the default dispatch by >= 0.2 us.
_TUNED: dict[tuple[int, int], dict[int, tuple[str, Any]]] = {
    # DSV4.1 compressor ratio 2 (kv | score), N = 2 * head_dim.
    (5120, 1024): {m: ("splitk", (4, 4)) for m in (5, 6, 7, 8)},
    # DSV4.1 compressor ratio 1 (kv only).
    (5120, 512): {5: ("dotprod", 128), 6: ("dotprod", 64), 8: ("splitk", (4, 6))},
}


class CompileKey(NamedTuple):
    backend: Literal["dotprod", "splitk"]
    M: int = 0
    K: int = 0
    bs: int = 0
    split_k: int = 0
    num_stages: int = 0


@lru_cache(maxsize=1)
def is_available() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    if torch.cuda.get_device_capability()[0] < 9:
        return False
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        from quack.compile_utils import make_fake_tensor  # noqa: F401
    except ImportError:
        return False
    return True


@lru_cache(maxsize=1)
def _use_pdl() -> bool:
    from sglang.kernels.jit.utils import is_arch_support_pdl

    return is_arch_support_pdl()


def dispatch(*, M: int, K: int, N: int) -> CompileKey:
    tuned = _TUNED.get((K, N), {}).get(M)
    if tuned is not None:
        backend, config = tuned
        if backend == "dotprod":
            return CompileKey(backend="dotprod", M=M, K=K, bs=config)
        split_k, num_stages = config
        return CompileKey(backend="splitk", split_k=split_k, num_stages=num_stages)
    if M <= _DEFAULT_DOTPROD_MAX_M or K < 2048 or N < _DOTPROD_MAX_N:
        return CompileKey(backend="dotprod", M=M, K=K, bs=_DEFAULT_DOTPROD_BS)
    split_k, num_stages = _DEFAULT_SPLITK_CONFIG
    return CompileKey(backend="splitk", split_k=split_k, num_stages=num_stages)


def _fake_gemm_tensors(*, M, K, N, divisibility: int):
    from cutlass import BFloat16, Float32
    from quack.compile_utils import make_fake_tensor

    x = make_fake_tensor(BFloat16, (M, K), divisibility=divisibility)
    w = make_fake_tensor(BFloat16, (N, K), divisibility=divisibility)
    out = make_fake_tensor(Float32, (M, N), divisibility=1)
    return x, w, out


def _stream():
    from cuda.bindings.driver import CUstream

    return CUstream(torch.cuda.current_stream().cuda_stream)


_compiled: dict[CompileKey, Any] = {}


def compile_kernel(key: CompileKey) -> Any:
    """Compile (once) the kernel for ``key``; must not run under CUDA graph capture."""
    kernel = _compiled.get(key)
    if kernel is not None:
        return kernel
    import cutlass.cute as cute

    if key.backend == "splitk":
        from ._splitk import LLBf16SplitK

        x, w, out = _fake_gemm_tensors(
            M=cute.sym_int(), K=cute.sym_int(), N=cute.sym_int(), divisibility=8
        )
        gemm = LLBf16SplitK(
            tile_n=16,
            tile_k=256,
            num_stages=key.num_stages,
            num_dma_warps=4,
            split_k=key.split_k,
            use_pdl=_use_pdl(),
        )
        kernel = cute.compile(gemm, x, w, out, _stream(), options="--enable-tvm-ffi")
    else:
        from ._dotprod import LLBf16Dotprod

        x, w, out = _fake_gemm_tensors(
            M=key.M, K=key.K, N=cute.sym_int(), divisibility=math.gcd(8, key.K)
        )
        gemm = LLBf16Dotprod(k=key.K, bs=key.bs, use_pdl=_use_pdl())
        kernel = cute.compile(
            gemm,
            x,
            w,
            out,
            key.M,
            key.K,
            1,  # runtime N placeholder for the fake-tensor compile
            _stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
    _compiled[key] = kernel
    return kernel


def ll_bf16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 ``x @ weight.T`` for contiguous BF16 ``x`` [M, K] and ``weight`` [N, K], K % 8 == 0."""
    # DLPack refuses tensors that require grad, such as an nn.Linear weight; detach is a view.
    x, weight = x.detach(), weight.detach()
    M, K = x.shape
    N = weight.shape[0]
    key = dispatch(M=M, K=K, N=N)
    kernel = compile_kernel(key)
    out = torch.empty(M, N, dtype=torch.float32, device=x.device)
    if key.backend == "splitk":
        kernel(x, weight, out, _stream(), 1.0)
    else:
        kernel(x, weight, out, N, _stream())
    return out
