"""ROCm-only DeepSeek-V4 main-path K norm + RoPE + FlashMLA store that also ropes the
query heads in the same launch.

The HIP counterpart of ``fused_k_norm_rope_flashmla`` in elementwise.py, which
dispatches here when it is handed ``q``.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import make_name


@cache_once
def _jit_main_k_norm_rope_flashmla_q_module(
    dtype: torch.dtype,
    head_dim: int,
    rope_dim: int,
    page_size: int,
):
    """The K kernel of ``_jit_main_k_norm_rope_flashmla_module`` plus the in-place query rope."""
    args = make_cpp_args(dtype, head_dim, rope_dim, page_size, is_arch_support_pdl())
    return load_jit(
        make_name("main_k_norm_rope_q_flashmla_hip"),
        *args,
        cuda_files=["deepseek_v4/main_norm_rope_hip.cuh"],
        cuda_wrappers=[
            ("forward_with_q", f"FusedKNormRopeQFlashMLAKernel<{args}>::forward"),
        ],
    )


def fused_k_norm_rope_flashmla_with_q(
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    freqs_real: torch.Tensor,
    positions: torch.Tensor,
    out_loc: torch.Tensor,
    kvcache: torch.Tensor,
    eps: float,
    page_size: int,
    q: torch.Tensor,
) -> None:
    """``fused_k_norm_rope_flashmla``'s K store, and the trailing ``rope_dim`` of every head of
    ``q`` ([B, H, head_dim], the same tokens) rotated in place by the same launch, bitwise what
    ``fused_rope_inplace`` produces. ``freqs_real`` is the real view of ``freqs_cis``."""
    head_dim = kv.shape[-1]
    rope_dim = freqs_real.shape[-1]
    module = _jit_main_k_norm_rope_flashmla_q_module(
        kv.dtype, head_dim, rope_dim, page_size
    )
    module.forward_with_q(
        kv, kv_weight, freqs_real, positions, out_loc, kvcache, eps, q
    )
