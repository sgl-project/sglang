from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    import torch

_is_npu = is_npu()

_K3_N_GEMM_DISPATCH_MAP = {
    (144, 7168): 16,
    (896, 7168): 8,
}
_K3_K_GEMM_DISPATCH_MAP = {
    (1536, 128): 12,
}


def situ_and_mul(
    input: torch.Tensor,
    out: Optional[torch.Tensor],
    beta: float,
    linear_beta: Optional[float],
) -> torch.Tensor:
    from .activation import situ_and_mul as impl

    return impl(input, out, beta, linear_beta)


def situ_and_mul_masked_post_quant(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    beta: float,
    linear_beta: float,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swizzle: bool = False,
) -> None:
    from .moe import situ_and_mul_masked_post_quant as impl

    return impl(
        input,
        output,
        output_scale,
        quant_group_size,
        masked_m,
        beta,
        linear_beta,
        scale_ue8m0,
        topk,
        transposed,
        swizzle,
    )


def kimi_k3_tiny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    import torch

    from ..gemm.tiny_gemm import tiny_k_gemm_bf16, tiny_n_gemm_bf16

    m, k = x.shape
    n, _ = w.shape
    if not _is_npu:
        if max_num_tokens := _K3_N_GEMM_DISPATCH_MAP.get((n, k)):
            if 0 < m <= max_num_tokens:
                return tiny_n_gemm_bf16(x, w)
        if max_num_tokens := _K3_K_GEMM_DISPATCH_MAP.get((n, k)):
            if 0 < m <= max_num_tokens:
                return tiny_k_gemm_bf16(x, w)
    return torch.nn.functional.linear(x, w)


__all__ = [
    "situ_and_mul",
    "situ_and_mul_masked_post_quant",
    "kimi_k3_tiny_gemm",
]
