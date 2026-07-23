import torch

from .activation import situ_and_mul
from .moe import moe_finalize, situ_and_mul_masked_post_quant

_K3_N_GEMM_DISPATCH_MAP = {
    (144, 7168): 16,
    (896, 7168): 8,
}
_K3_K_GEMM_DISPATCH_MAP = {
    (1536, 128): 12,
}


def kimi_k3_tiny_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
) -> torch.Tensor:
    from ..tiny_gemm import tiny_k_gemm_bf16, tiny_n_gemm_bf16

    m, k = x.shape
    n, _ = w.shape
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
    "moe_finalize",
    "kimi_k3_tiny_gemm",
]
