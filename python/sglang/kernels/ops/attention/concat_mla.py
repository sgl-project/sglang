from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_concat_mla_k_module() -> Module:
    return load_jit(
        "concat_mla_k",
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_k", "ConcatMlaKKernel::run")],
    )


@cache_once
def _jit_concat_mla_absorb_q_module(pad_head_groups=False) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), pad_head_groups)
    return load_jit(
        "concat_mla_absorb_q",
        *args,
        cuda_files=["elementwise/concat_mla.cuh"],
        cuda_wrappers=[("concat_mla_absorb_q", f"ConcatMlaAbsorbQKernel<{args}>::run")],
    )


def concat_mla_k(k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor) -> None:
    """
    Concatenate k_nope and k_rope into k for MLA (Multi-head Latent Attention).

    This kernel efficiently broadcasts k_rope across all heads while copying
    k_nope values directly.

    Args:
        k: Output tensor of shape [num_tokens, num_heads=128, k_head_dim=192], dtype=bfloat16
        k_nope: Input tensor of shape [num_tokens, num_heads=128, nope_head_dim=128], dtype=bfloat16
        k_rope: Input tensor of shape [num_tokens, 1, rope_head_dim=64], dtype=bfloat16
    """
    module = _jit_concat_mla_k_module()
    module.concat_mla_k(k, k_nope, k_rope)


def concat_mla_absorb_q(
    a: torch.Tensor, b: torch.Tensor, *, pad_head_groups: bool = False
) -> torch.Tensor:
    """
    Concatenate tensors a and b for MLA absorbed Q computation.

    Args:
        a: Input tensor of shape [dim_0, dim_1, a_last_dim], dtype=bfloat16
        b: Input tensor of shape [dim_0, dim_1, b_last_dim], dtype=bfloat16
        pad_head_groups: Place each 16-head group in a separate 32-head block of
            a 128-head tensor. Supports 16, 32, 48, or 64 live heads.

    Returns:
        BF16 tensor with concatenated channels. The head dimension is dim_1,
        or 128 when pad_head_groups is enabled.
    """
    # Isolate each live group in its own warp-wide rescaling domain.
    heads = 128 if pad_head_groups else a.shape[1]
    out = torch.empty(
        (a.shape[0], heads, a.shape[-1] + b.shape[-1]),
        dtype=a.dtype,
        device=a.device,
    )
    module = _jit_concat_mla_absorb_q_module(pad_head_groups)
    module.concat_mla_absorb_q(a, b, out)
    return out
