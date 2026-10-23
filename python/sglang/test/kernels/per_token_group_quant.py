"""Triton-reference vs production per-token-group 8-bit quantization entry points.

Shared by the per-token-group quant test and benchmark: ``triton_*`` is the
Triton reference (``_per_token_group_quant_8bit_raw`` or the fused SiLU-and-mul
post-quant kernel), ``sglang_*`` the production int8 / fp8 path.
"""

from typing import Optional, Tuple

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    _per_token_group_quant_8bit_raw,
    sglang_per_token_group_quant_fp8,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    sglang_per_token_group_quant_int8,
)


def _triton_per_token_group_quant_8bit_fuse_silu_and_mul(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
    masked_m: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    from deep_gemm import transform_sf_into_required_layout

    from sglang.kernels.ops.moe.ep_moe_kernels import silu_and_mul_masked_post_quant_fwd

    assert column_major_scales
    assert scale_tma_aligned
    assert scale_ue8m0

    needs_unsqueeze = x.dim() == 2
    if needs_unsqueeze:
        num_tokens, _ = x.shape
        x = x.unsqueeze(0)
        assert masked_m is None
        masked_m = torch.tensor([num_tokens], device=x.device, dtype=torch.int32)

    # Use `zeros` for easier testing
    output = torch.zeros(
        (*x.shape[:-1], x.shape[-1] // 2),
        device=x.device,
        dtype=dst_dtype,
    )
    # Use `zeros` for easier testing
    output_scale_for_kernel = torch.zeros(
        (*x.shape[:-1], x.shape[-1] // 2 // group_size),
        device=x.device,
        dtype=torch.float32,
    )
    silu_and_mul_masked_post_quant_fwd(
        input=x,
        output=output,
        output_scale=output_scale_for_kernel,
        quant_group_size=group_size,
        masked_m=masked_m,
        scale_ue8m0=scale_ue8m0,
    )

    output_scale = transform_sf_into_required_layout(
        output_scale_for_kernel,
        num_groups=output.shape[0],
        mn=output.shape[-2],
        k=output.shape[-1],
        recipe=(1, group_size, group_size),
        is_sfa=True,
    )

    if needs_unsqueeze:
        output = output.squeeze(0)
        output_scale = output_scale.squeeze(0)

    return output, output_scale


def triton_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fuse_silu_and_mul:
        return _triton_per_token_group_quant_8bit_fuse_silu_and_mul(
            x=x,
            group_size=group_size,
            dst_dtype=dst_dtype,
            column_major_scales=column_major_scales,
            scale_tma_aligned=scale_tma_aligned,
            scale_ue8m0=scale_ue8m0,
            masked_m=masked_m,
        )
    return _per_token_group_quant_8bit_raw(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        dtype=dst_dtype,
    )


def sglang_per_token_group_quant_8bit(
    x: torch.Tensor,
    group_size: int,
    dst_dtype: torch.dtype,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if dst_dtype == torch.int8:
        assert not column_major_scales
        assert not scale_tma_aligned
        assert not fuse_silu_and_mul
        assert masked_m is None
        return sglang_per_token_group_quant_int8(
            x=x,
            group_size=group_size,
            eps=eps,
            dtype=dst_dtype,
        )

    return sglang_per_token_group_quant_fp8(
        x=x,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )
