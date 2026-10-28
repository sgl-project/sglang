"""gfx95 bpreshuffle block-scale GEMM of aiter_w8a8_block_fp8_linear as aiter's asm split-K
kernel, launched into the output its fused producer already zero-filled instead of behind
the memset aiter's launcher puts before it."""

from functools import lru_cache, partial
from typing import Callable, Optional, Tuple

import torch
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config, get_zero_bias_buf

from sglang.kernels.ops.gemm.aiter_asm_blockscale import (
    asm_split_k_supported,
    gemm_a8w8_blockscale_bpreshuffle_asm_prezeroed,
)


@lru_cache(maxsize=4096)
def asm_split_k_kernel(m: int, n: int, k: int) -> Optional[Tuple[str, int]]:
    """(kernel name, split-K) when aiter_w8a8_block_fp8_linear runs the [m, k] x [n, k]
    bpreshuffle GEMM as aiter's asm split-K kernel, which accumulates into a zeroed output
    and which we can launch; None otherwise."""
    # fp8_utils imports this module while it initializes
    from sglang.srt.layers.quantization.fp8_utils import (
        use_aiter_triton_gemm_w8a8_tuned_gfx950,
    )

    if use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
        return None
    config = get_CKGEMM_config(
        m, n, k, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE
    )
    if config is None or config["libtype"] != "asm" or int(config["splitK"]) <= 1:
        return None
    name, split_k = config["kernelName"], int(config["splitK"])
    if not asm_split_k_supported(name, n=n, k=k, split_k=split_k):
        return None
    return name, split_k


def zeroed_gemm(input: torch.Tensor, gemm: Callable) -> Callable:
    """``gemm``, or a drop-in for it launching aiter's asm split-K kernel into the output
    the producer of ``input`` zero-filled."""
    # set once by gemma_norm_fp8; dynamo cannot trace the tensor-attribute handoff
    if torch.compiler.is_compiling():
        return gemm
    zeroed_out = input.__dict__.pop("zeroed_gemm_out", None)
    if zeroed_out is None:
        return gemm
    return partial(_gemm_into_zeroed, gemm=gemm, out=zeroed_out)


def _gemm_into_zeroed(xq, wq, x_scale, w_scale, dtype, *, gemm, out):
    m, (n, k) = xq.shape[0], wq.shape
    kernel = asm_split_k_kernel(m, n, k)
    # the asm kernel accumulates bf16 into the zeroed output
    if (
        kernel is None
        or out.shape != (m, n)
        or not dtype == out.dtype == torch.bfloat16
    ):
        return gemm(xq, wq, x_scale, w_scale, dtype=dtype)
    return gemm_a8w8_blockscale_bpreshuffle_asm_prezeroed(
        xq=xq,
        wq=wq,
        x_scale=x_scale,
        w_scale=w_scale,
        out=out,
        kernel_name=kernel[0],
        split_k=kernel[1],
        bias=get_zero_bias_buf(wq),
    )
