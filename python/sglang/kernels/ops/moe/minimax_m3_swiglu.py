# SPDX-License-Identifier: Apache-2.0
"""Fused SwiGLU-OAI (split layout) Triton kernels for MiniMax-M3.

SwiGLU-OAI on a ``[*, 2I]`` split-layout tensor (gate = first half, up = second
half): ``gate * sigmoid(alpha * gate) * (up + beta)`` with optional clamp,
computed in fp32. The standalone activation is registered as an opaque custom
op so outer ``torch.compile`` regions cannot inline it into separate pointwise
ATen kernels.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


@triton.jit
def _swiglu_oai_kernel(
    g_ptr,
    out_ptr,
    n_inter,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter
    gate = tl.load(g_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0).to(
        tl.float32
    )
    up = tl.load(
        g_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    tl.store(
        out_ptr + row * stride_om + cols * stride_on,
        out.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


@register_custom_op(
    op_name="swiglu_oai_split_inplace",
    mutates_args=["out"],
)
def _swiglu_oai_split_inplace(
    gate_up: torch.Tensor,
    out: torch.Tensor,
    alpha: float,
    beta: float,
    limit: float,
    has_limit: bool,
) -> None:
    """Launch SwiGLU-OAI behind an opaque ``torch.library`` custom op."""
    if out.numel() == 0:
        return

    two_i = gate_up.shape[-1]
    n_inter = two_i // 2
    x2 = gate_up.view(-1, two_i)
    out2 = out.view(-1, n_inter)
    m = x2.shape[0]
    # Adaptive tile (tuned on gfx950). A 512-wide tile only helps
    # once the (TP-sharded) per-rank slice is large enough to be bandwidth-bound
    # (~1.25-1.35x faster than 256 at TP=1 prefill for the dense MLP I=12288).
    # For small sharded slices (high TP) / decode the kernel is launch-bound, so
    # fall back to 256. num_warps is pinned to 4 (8 underfills this tile).
    block_i = 512 if n_inter >= 2048 else 256
    grid = (m, triton.cdiv(n_inter, block_i))
    _swiglu_oai_kernel[grid](
        x2,
        out2,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        out2.stride(0),
        out2.stride(1),
        alpha,
        beta,
        limit,
        HAS_LIMIT=has_limit,
        BLOCK_I=block_i,
        num_warps=4,
    )


def swiglu_oai_split(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: Optional[float],
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """SwiGLU-OAI on a split-layout ``[*, 2I]`` tensor -> ``[*, I]``.

    The activation uses fp32 intermediates and one Triton kernel launch. Input
    must be contiguous so this function never needs a hidden materialization
    kernel before the activation.
    """
    if gate_up.ndim == 0 or gate_up.shape[-1] % 2 != 0:
        raise ValueError(
            "SwiGLU-OAI expects a tensor whose last dimension is even, "
            f"got shape={tuple(gate_up.shape)}."
        )
    if not gate_up.is_contiguous():
        raise ValueError("SwiGLU-OAI expects a contiguous input tensor.")

    orig_shape = gate_up.shape
    n_inter = orig_shape[-1] // 2
    out = torch.empty(
        (*orig_shape[:-1], n_inter),
        dtype=out_dtype if out_dtype is not None else gate_up.dtype,
        device=gate_up.device,
    )
    has_limit = limit is not None
    _swiglu_oai_split_inplace(
        gate_up,
        out,
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        has_limit,
    )
    return out


@triton.jit
def _swiglu_oai_mxfp8_quant_kernel(
    g_ptr,
    q_ptr,
    scale_ptr,
    n_inter,
    stride_gm,
    stride_gn,
    stride_qm,
    stride_qn,
    stride_sm,
    stride_sn,
    alpha,
    beta,
    limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    row = tl.program_id(0)
    pid_i = tl.program_id(1)
    cols = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = cols < n_inter

    gate = tl.load(g_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0)
    up = tl.load(
        g_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    )
    gate = gate.to(tl.float32)
    up = up.to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)

    # Keep the activation in fp32 all the way to the E8M0 scale selection (no
    # bf16 round-trip to HBM). Matches the vLLM/ame fused swiglu+quant kernel:
    # marginally more accurate than the unfused bf16 two-kernel chain.
    activated = gate * tl.sigmoid(alpha * gate) * (up + beta)

    groups: tl.constexpr = BLOCK_I // 32
    activated_2d = tl.reshape(activated, (groups, 32))
    valid_groups = pid_i * groups + tl.arange(0, groups) < (n_inter // 32)

    amax = tl.maximum(tl.max(tl.abs(activated_2d), axis=1), 1e-30)
    # Round the E8M0 exponent up (ceil(log2(amax / e4m3_max))) so the block amax
    # stays inside the e4m3 range and the full dynamic range is used.
    scale_biased = tl.ceil(tl.log2(amax / 448.0)) + 127.0
    scale_biased = tl.minimum(tl.maximum(scale_biased, 0.0), 254.0)
    descale = tl.reshape(tl.exp2(scale_biased - 127.0), (groups, 1))

    q_2d = tl.clamp(activated_2d / descale, -448.0, 448.0)
    q = tl.reshape(q_2d, (BLOCK_I,)).to(q_ptr.dtype.element_ty)

    tl.store(q_ptr + row * stride_qm + cols * stride_qn, q, mask=mask)
    tl.store(
        scale_ptr
        + row * stride_sm
        + (pid_i * groups + tl.arange(0, groups)) * stride_sn,
        scale_biased.to(tl.uint8),
        mask=valid_groups,
    )


def swiglu_oai_mxfp8_quant(
    gate_up: torch.Tensor,
    alpha: float,
    beta: float,
    limit: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """SwiGLU-OAI on split layout, then MiniMax MXFP8 quant, in one launch.

    The activation stays in fp32 through the E8M0 scale selection (no bf16
    round-trip), matching the vLLM/ame fused swiglu+quant kernel.
    """
    orig_shape = gate_up.shape
    two_i = orig_shape[-1]
    n_inter = two_i // 2
    assert n_inter % 32 == 0, "MiniMax MXFP8 quant requires I divisible by 32."

    x2 = gate_up.reshape(-1, two_i)
    m = x2.shape[0]
    q = torch.empty((m, n_inter), dtype=torch.float8_e4m3fn, device=gate_up.device)
    scales = torch.empty((m, n_inter // 32), dtype=torch.uint8, device=gate_up.device)
    block_i = 512 if n_inter >= 2048 else 256
    grid = (m, triton.cdiv(n_inter, block_i))
    _swiglu_oai_mxfp8_quant_kernel[grid](
        x2,
        q,
        scales,
        n_inter,
        x2.stride(0),
        x2.stride(1),
        q.stride(0),
        q.stride(1),
        scales.stride(0),
        scales.stride(1),
        float(alpha),
        float(beta),
        0.0 if limit is None else float(limit),
        HAS_LIMIT=limit is not None,
        BLOCK_I=block_i,
        num_warps=4,
    )
    return q.reshape(*orig_shape[:-1], n_inter), scales.reshape(
        *orig_shape[:-1], n_inter // 32
    )
