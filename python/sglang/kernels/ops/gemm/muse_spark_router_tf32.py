"""MuseSpark decode router logits through FlashInfer's TF32 cluster split-K GEMM (``flashinfer.mm_tf32``).

With TF32 matmuls allowed, cuBLAS runs the multi-row FP32 router logits as a TF32 split-K GEMM plus a
separate reduction kernel on the router side stream. ``mm_tf32`` splits K over a thread-block cluster and
reduces the partial sums in distributed shared memory in the same launch, with the same TF32-rounded
inputs and FP32 accumulation; only the (fixed) summation order differs from cuBLAS.
"""

from functools import cache
from typing import Optional

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import is_arch_support_pdl

# Up to 512 tokens (DFlash decode c <= 64): from 1024 cuBLAS split-K was faster in SM107 serving.
MAX_TOKENS = 512

# Tactics (mma_m, mma_n, split_k, ab_stages, cta_k) by max tokens, tuned at each captured decode size (TP4 and
# TP8) beside the main stream's gathered pre-expert projection, then routing on this stream: the default grid is
# sized for the kernel alone, so it waits for the projection's CTAs or slows them. None keeps the default.
_SM107_TACTICS = (
    (8, None),
    (16, (64, 16, 16, 8, 64)),
    (32, (64, 32, 8, 12, 64)),
    (64, (64, 64, 8, 19, 32)),
    (96, (128, 32, 8, 15, 32)),
    (128, (64, 64, 8, 19, 32)),
    (144, (128, 64, 8, 12, 32)),
    (160, (64, 64, 4, 19, 32)),
    (208, (64, 64, 8, 19, 32)),
    (416, (128, 64, 8, 6, 64)),
    (512, (128, 64, 4, 6, 64)),
)
_SM103_TACTICS = (
    (8, (64, 8, 8, 6, 128)),
    (16, (64, 16, 8, 10, 64)),
    (32, (64, 32, 8, 8, 64)),
    (64, (64, 64, 8, 13, 32)),
    (96, (64, 32, 4, 9, 64)),
    (112, (64, 64, 4, 13, 32)),
    (144, (64, 64, 8, 13, 32)),
    (192, (64, 64, 8, 6, 64)),
    (224, (128, 64, 4, 8, 32)),
    (352, (128, 64, 8, 8, 32)),
    (384, (128, 64, 4, 4, 64)),
    (448, (128, 64, 8, 8, 32)),
    (512, (64, 64, 4, 13, 32)),
)


@cache
def _mm_tf32():
    """``flashinfer.mm_tf32`` when this FlashInfer has it and CuTe DSL is installed, else None."""
    try:
        from flashinfer import mm_tf32
        from flashinfer.cute_dsl import is_cute_dsl_available
    except ImportError:
        return None
    return mm_tf32 if is_cute_dsl_available() else None


def router_tf32_eligible(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Multi-row FP32 decode logits on SM100-family GPUs, only with TF32 matmuls allowed:
    under ``float32_matmul_precision("highest")`` the logits stay FP32."""
    return (
        x.is_cuda
        and x.dtype == torch.float32
        and w.dtype == torch.float32
        and x.dim() == 2
        and w.dim() == 2
        and 2 <= x.shape[0] <= MAX_TOKENS
        and torch.get_float32_matmul_precision() != "highest"
        and x.shape[1] == w.shape[1]
        and x.shape[1] % 64 == 0
        and w.shape[0] % 2 == 0
        and x.device == w.device
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 16 == 0
        and w.data_ptr() % 16 == 0
        and current_arch().is_sm(10)
        and _mm_tf32() is not None
    )


def router_tf32(
    x: torch.Tensor, w: torch.Tensor, use_pdl: Optional[bool] = None
) -> torch.Tensor:
    """``x @ w.T`` as a contiguous FP32 ``[T, E]`` tensor (TF32 MMA, FP32 accumulate)."""
    if use_pdl is None:
        use_pdl = is_arch_support_pdl()
    tactic = None
    if current_arch().is_sm(10, 7):
        tactic = next((t for m, t in _SM107_TACTICS if x.shape[0] <= m), None)
    elif current_arch().is_sm(10, 3):
        tactic = next((t for m, t in _SM103_TACTICS if x.shape[0] <= m), None)
    return _mm_tf32()(x, w.t(), pdl=bool(use_pdl), backend="cute-dsl", tactic=tactic)
