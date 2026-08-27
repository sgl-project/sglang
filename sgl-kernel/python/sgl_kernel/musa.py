from typing import Optional

import torch


def musa_batched_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    rot_dim: int,
    cos_sin_cache_offsets: torch.Tensor,
) -> None:
    return torch.ops.sgl_kernel.musa_batched_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
        rot_dim,
        cos_sin_cache_offsets,
    )


def musa_rotary_embedding_contiguous(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    return torch.ops.sgl_kernel.musa_rotary_embedding_contiguous(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox,
    )


def musa_fused_moe_gemv(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale,
    B_scale,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    mul_routed_weight: bool,
    topk: int,
    use_int4_w4a16: bool,
    use_swigelu: bool,
) -> None:
    return torch.ops.sgl_kernel.musa_fused_moe_gemv(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        topk_ids,
        mul_routed_weight,
        topk,
        use_int4_w4a16,
        use_swigelu,
    )


def musa_fused_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    x_scales: Optional[torch.Tensor] = None,
    qweight_scales: Optional[torch.Tensor] = None,
    use_swigelu: bool = False,
    use_rms_norm: bool = False,
    gamma: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
):
    use_int4_w4a16 = False
    out_shape = x.shape[:-1] + (
        qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
    )
    assert not (
        use_swigelu and use_rms_norm
    ), "gemv only fused one activation (swigelu or rms_norm)!"

    if use_rms_norm:
        if gamma is None:
            assert False, "rms_norm gamma is None!"

    # fp8 grouped matmul
    if qweight.dtype == torch.float8_e4m3fn:
        assert qweight_scales is not None, "FP8 grouped matmul weight scales is None!"
        output = torch.empty(out_shape, device=x.device, dtype=torch.bfloat16)
        torch.ops.sgl_kernel.musa_fused_gemv(
            x,
            qweight,
            output,
            x_scales,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # w4a16 gemv
    elif qweight_scales is not None:
        assert (
            x.dtype == torch.bfloat16 or x.dtype == torch.float16
        ), "W4A16 gemv only support bfloat16 or float16!"
        use_int4_w4a16 = True
        out_shape = x.shape[:-1] + (
            qweight.shape[0] if not use_swigelu else qweight.shape[0] // 2,
        )
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops.sgl_kernel.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            qweight_scales,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output
    # general gemv
    else:
        output = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        torch.ops.sgl_kernel.musa_fused_gemv(
            x,
            qweight,
            output,
            None,
            None,
            use_int4_w4a16,
            use_swigelu,
            use_rms_norm,
            gamma,
            eps,
        )
        return output


def musa_fused_mul_add(
    self: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: Optional[float],
    accurate: bool = True,
):
    # if accurate == False, then we call inplace op: bias += (self * scale)
    if not accurate:
        bias.add_(self, alpha=scale)
        return bias

    # otherwise, we call custom outplace op, act: output = self * scale + bias
    output = torch.empty_like(self)
    torch.ops.sgl_kernel.musa_fused_mul_add(output, self, bias, scale)

    return output
