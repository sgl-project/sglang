from typing import TYPE_CHECKING, Optional

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
def _jit_gated_residual_combine_norm_module(
    hc_count: int, hidden_size: int, group_size: int, dtype: torch.dtype
) -> Module:
    """Compile and cache the JIT gated residual combine+norm module."""
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if hidden_size <= 0 or hidden_size % 8 != 0:
        raise RuntimeError(
            f"Unsupported hidden_size {hidden_size}. Must be a multiple of 8."
        )
    if group_size <= 0 or group_size % 512 != 0:
        raise RuntimeError(
            f"Unsupported group_size {group_size}. Must be a multiple of 512."
        )
    if hidden_size % group_size != 0:
        raise RuntimeError(
            f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
        )
    args = make_cpp_args(hc_count, hidden_size, group_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "gated_residual_combine_norm",
        *args,
        cuda_files=["elementwise/gated_residual_combine_norm.cuh"],
        cuda_wrappers=[
            ("gated_residual_combine_norm", f"GatedResidualCombineNormKernel<{args}>::run")
        ],
    )


def gated_residual_combine_norm(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    inject_logits: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    normed_out: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused gated residual combine + grouped Gemma RMSNorm.

    First computes the gated residual combine:
        combined[m, c*H + i] = residual[m, c*H + i] + a[m, c] * block_output[m, i]
        where a[m, c] = 2 * sigmoid(inject_logits[m, c] / hc_count)

    Then applies grouped RMSNorm on combined:
        normed[m, g*G + j] = combined[m, g*G + j] * rsqrt(mean(combined[m, g*G:(g+1)*G]^2) + eps) * (1 + w[g*G + j])

    Supported dtypes: torch.bfloat16, torch.float16.

    Parameters
    ----------
    block_output   : CUDA tensor [..., hidden_size]
    residual       : CUDA tensor [..., hc_count * hidden_size]
    inject_logits  : CUDA tensor [..., hc_count]
    weight         : CUDA tensor [hidden_size]
    group_size     : elements per variance group (multiple of 512)
    eps            : RMSNorm epsilon
    out            : optional pre-allocated output for combined (same shape as residual)
    normed_out     : optional pre-allocated output for normed (same shape as residual)

    Returns
    -------
    Tuple of (combined, normed), both same shape/dtype as residual.
    """
    hc_count = inject_logits.size(-1)
    hidden_size = block_output.size(-1)

    # Flatten to 2D for kernel
    block_output_2d = block_output.reshape(-1, hidden_size)
    residual_2d = residual.reshape(-1, hc_count * hidden_size)
    inject_logits_2d = inject_logits.reshape(-1, hc_count)

    if out is None:
        out = torch.empty_like(residual_2d)
    else:
        out = out.reshape(-1, hc_count * hidden_size)

    if normed_out is None:
        normed_out = torch.empty_like(residual_2d)
    else:
        normed_out = normed_out.reshape(-1, hc_count * hidden_size)

    module = _jit_gated_residual_combine_norm_module(
        hc_count, hidden_size, group_size, residual.dtype
    )
    module.gated_residual_combine_norm(
        block_output_2d,
        residual_2d,
        inject_logits_2d,
        weight,
        out,
        normed_out,
        eps,
    )
    return out.reshape(residual.shape), normed_out.reshape(residual.shape)
