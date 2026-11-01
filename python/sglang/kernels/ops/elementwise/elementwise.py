import math

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _fused_sigmoid_mul_kernel(
    output_ptr,
    attn_output_ptr,
    gate_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    q_ptr=None,
    s_ptr=None,
    QUANT: tl.constexpr = False,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)

    result = attn * tl.sigmoid(g)
    if QUANT:  # the block is one row: per-token FP8 as _per_token_group_quant_8bit
        y = tl.where(mask, result.to(output_ptr.dtype.element_ty).to(tl.float32), 0.0)
        y_s = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + attn_off, y_q, mask=mask)
        tl.store(s_ptr + pid_row, y_s)
    else:
        tl.store(output_ptr + attn_off, result, mask=mask)


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
    quant: bool = False,
) -> torch.Tensor:
    """
    Fused sigmoid-mul for attention output gating.

    Equivalent to: attn_output * sigmoid(gate)

    The production Qwen3.5 path passes a 3D strided gate. A single hidden-block
    Triton kernel handles both that path and flat contiguous inputs.

    When inplace=True, writes result back to attn_output and returns it.

    Supports strided gate: if gate is 3D (num_tokens, num_heads, head_dim)
    and attn_output is 2D (num_tokens, hidden_dim), the kernel reads gate
    via explicit strides without requiring a contiguous copy.
    """
    if gate.ndim == 3 and attn_output.ndim == 2:
        # Strided gate path: gate is 3D (num_tokens, num_heads, head_dim)
        num_tokens, num_heads, head_dim = gate.shape
        hidden_dim = num_heads * head_dim
        assert attn_output.shape == (num_tokens, hidden_dim)
        gate_stride_row = gate.stride(0)
        gate_stride_head = gate.stride(1)
    else:
        # Flat path: both tensors have the same shape
        assert attn_output.shape == gate.shape, (
            "attn_output and gate must have the same shape"
        )
        hidden_dim = attn_output.shape[-1]
        num_tokens = attn_output.numel() // hidden_dim
        head_dim = hidden_dim
        gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if num_tokens < 1024 else 2048
    kwargs = {}
    if quant:  # one block per row; returns per-token FP8 (q, scale [T, 1])
        block_h = triton.next_power_of_2(hidden_dim)
        q = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
        s = torch.empty((num_tokens, 1), dtype=torch.float32, device=q.device)
        kwargs = {"q_ptr": q, "s_ptr": s, "QUANT": True}
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
        **kwargs,
    )
    return (q, s) if quant else out


@triton.jit
def _fused_sigmoid_mul_mxfp4_quant_kernel(
    attn_output_ptr,
    gate_ptr,
    q_ptr,
    scale_ptr,
    gate_stride_row,
    gate_stride_head,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """sigmoid(gate) * attn_output as fused_sigmoid_mul writes it, then MXFP4 of
    that the way AITER per_1x32_f4_quant(shuffle=True) quantizes it."""
    row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM
    attn = tl.load(attn_output_ptr + row * hidden_dim + offsets).to(tl.float32)
    g = tl.load(gate_ptr + row * gate_stride_row + head * gate_stride_head + d).to(
        tl.float32
    )
    y = (attn * tl.sigmoid(g)).to(attn_output_ptr.dtype.element_ty).to(tl.float32)

    # E8M0 scale ceil_pow2(amax / 6) (RoundUp) per 32 values with amax floored
    # at 1e-10, E2M1 round to nearest even, as in the Gemma norm + MXFP4 quant
    # kernel (sglang.kernels.ops.layernorm.minimax_m3_rmsnorm).
    NB: tl.constexpr = BLOCK_H // 32
    yb = tl.reshape(y, (NB, 32))
    amax = tl.maximum(tl.max(tl.abs(yb), axis=1, keep_dims=True), 1e-10)
    u = (amax * (1.0 / 6.0)).to(tl.int32, bitcast=True)
    e = (u >> 23) & 0xFF
    e = tl.where((e < 0xFF) & ((u & 0x7FFFFF) != 0), e + 1, e)
    e = tl.minimum(e, 254)
    qx = yb * tl.exp2((127 - e).to(tl.float32))
    # gfx950 conversion, two values per instruction; four bytes (one thread's
    # eight values) per 32-bit store.
    lo, hi = tl.split(tl.reshape(qx, (NB, 16, 2)))
    byts = (
        tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
            "=v,v,v,v",
            args=[lo, hi, 1.0],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        & 0xFF
    )
    shifts = tl.arange(0, 4) * 8
    words = tl.sum(tl.reshape(byts, (NB, 4, 4)) << shifts[None, None, :], axis=2)
    wcols = pid_block * (BLOCK_H // 8) + tl.arange(0, NB * 4)
    tl.store(q_ptr + row * (hidden_dim // 8) + wcols, tl.reshape(words, (NB * 4,)))
    # E8M0 scales in the tile order of the AITER ASM FP4 GEMM.
    c = pid_block * NB + tl.arange(0, NB)
    sidx = (
        (row % 32) // 16
        + ((c % 8) // 4) * 2
        + (row % 16) * 4
        + (c % 4) * 64
        + (c // 8) * 256
        + (row // 32) * 32 * (hidden_dim // 32)
    )
    tl.store(scale_ptr + sidx, tl.reshape(e, (NB,)).to(tl.uint8))


def fused_sigmoid_mul_mxfp4_quant(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """fused_sigmoid_mul(attn_output, gate) quantized to MXFP4 for an AITER ASM
    FP4 GEMM (gfx950), in one pass.

    attn_output: (T, H * D) bf16 or fp16, contiguous; gate: (T, H, D), may be
    strided over T and H. H * D must be a multiple of 256. Returns ``(x_fp4,
    x_scales)``: uint8 ``(T, H * D // 2)`` (two E2M1 codes per byte) and uint8
    E8M0 ``(round_up(T, 256), H * D // 32)`` in the ASM GEMM tile order (rows
    >= T are not written), the bytes that per_1x32_f4_quant(shuffle=True)
    writes for fused_sigmoid_mul's output.
    """
    num_tokens, num_heads, head_dim = gate.shape
    hidden_dim = num_heads * head_dim
    assert attn_output.shape == (num_tokens, hidden_dim), (
        attn_output.shape,
        gate.shape,
    )
    assert attn_output.is_contiguous() and gate.stride(2) == 1
    assert hidden_dim % 256 == 0, hidden_dim
    x_fp4 = torch.empty(
        (num_tokens, hidden_dim // 2), dtype=torch.uint8, device=attn_output.device
    )
    x_scales = torch.empty(
        ((num_tokens + 255) // 256 * 256, hidden_dim // 32),
        dtype=torch.uint8,
        device=attn_output.device,
    )
    if num_tokens == 0:
        return x_fp4, x_scales
    block_h = math.gcd(hidden_dim, 2048)
    _fused_sigmoid_mul_mxfp4_quant_kernel[(num_tokens, hidden_dim // block_h)](
        attn_output,
        gate,
        x_fp4.view(torch.int32),
        x_scales,
        gate.stride(0),
        gate.stride(1),
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        num_warps=4,
    )
    return x_fp4, x_scales


@triton.jit
def _sigmoid_gate_mul_broadcast_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    g = tl.load(gate_ptr + row).to(tl.float32)
    g = tl.sigmoid(g)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_dim
    x = tl.load(x_ptr + row * hidden_dim + offs, mask=mask).to(tl.float32)
    out = x * g
    tl.store(
        out_ptr + row * hidden_dim + offs,
        out.to(x_ptr.dtype.element_ty),
        mask=mask,
    )


def sigmoid_gate_mul_broadcast(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Compute ``x * sigmoid(gate)`` where gate is (N, 1) and x is (N, D)."""
    bs, hidden_dim = x.shape
    out = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    max_warps = 16 if _is_hip else 32
    num_warps = max(
        min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
    )
    _sigmoid_gate_mul_broadcast_kernel[(bs,)](
        out,
        gate,
        x,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def _fused_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim], optionally also the addend
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DO_ADD: tl.constexpr = True,
    USE_PDL: tl.constexpr = False,
    GATE_ONLY: tl.constexpr = False,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    row_offset = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    w = tl.load(gate_weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    h = tl.load(hidden_states_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if not GATE_ONLY:
        s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        if DO_ADD:
            f = tl.load(output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
                tl.float32
            )

        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()

    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    if GATE_ONLY:
        # Preserve FP32 for the later shared-add FMA, without an intermediate
        # BF16 gated shared output. Keep the standalone gate's PDL ordering.
        if USE_PDL:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(output_ptr + pid, gate_val)
    else:
        result = gate_val * s
        if DO_ADD:
            result += f
        tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def _launch_fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    output: torch.Tensor,
    *,
    do_add: bool,
) -> None:
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert gate_weight.is_contiguous(), "gate_weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert output.is_contiguous(), "output must be contiguous"

    num_tokens, hidden_dim = hidden_states.shape
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert output.shape == (num_tokens, hidden_dim)

    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    if num_tokens >= 1024 or is_batch_invariant_mode_enabled():
        config["num_warps"] = min(config["num_warps"], 8)

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        hidden_dim=hidden_dim,
        DO_ADD=do_add,
        USE_PDL=use_pdl,
        GATE_ONLY=False,
        **config,
        **pdl_kwargs,
    )


def fused_gate_sigmoid_mul(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
) -> torch.Tensor:
    """Materialize the gated shared-expert contribution without an add/copy."""
    output = torch.empty_like(shared_output)
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        output,
        do_add=False,
    )
    return output


def fused_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    final_hidden_states: torch.Tensor,
) -> None:
    """Add the gated shared-expert contribution to routed-expert output."""
    _launch_fused_gate_sigmoid_mul(
        hidden_states,
        gate_weight,
        shared_output,
        final_hidden_states,
        do_add=True,
    )
