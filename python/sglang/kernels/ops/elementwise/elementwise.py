import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

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
    ROUND_SIGMOID: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    offsets = pid_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offsets < hidden_dim
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off, mask=mask, other=0.0).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off, mask=mask, other=0.0)
    sig = tl.sigmoid(g.to(tl.float32))
    if ROUND_SIGMOID:
        sig = sig.to(gate_ptr.dtype.element_ty).to(tl.float32)
    result = attn * sig
    if QUANT:  # the block is one row: per-token FP8 as _per_token_group_quant_8bit
        y = tl.where(mask, result.to(output_ptr.dtype.element_ty).to(tl.float32), 0.0)
        y_s = tl.maximum(tl.max(tl.abs(y)), 1e-10) / 448.0
        y_q = tl.clamp(y * (1.0 / y_s), -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + attn_off, y_q, mask=mask)
        tl.store(s_ptr + pid_row, y_s)
    else:
        tl.store(output_ptr + attn_off, result, mask=mask)


@triton.jit
def _fused_sigmoid_mul_block_fp8_kernel(
    attn_output_ptr,
    gate_ptr,
    q_ptr,
    s_ptr,
    gate_stride_row,
    gate_stride_head,
    s_stride_tok,
    s_stride_grp,
    hidden_dim: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GROUPS_PER_BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr = False,
    ROUND_SIGMOID: tl.constexpr = False,
    USE_PDL: tl.constexpr = False,
):
    """Fuse sigmoid(gate) * attn_output with group-128 blockwise FP8 quantization."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    pid_row = tl.program_id(0).to(tl.int64)
    pid_block = tl.program_id(1)

    grp_ids = pid_block * GROUPS_PER_BLOCK + tl.arange(0, GROUPS_PER_BLOCK)
    cols = tl.arange(0, 128)
    offsets = grp_ids[:, None] * 128 + cols[None, :]
    head = offsets // HEAD_DIM
    d = offsets - head * HEAD_DIM

    attn_off = pid_row * hidden_dim + offsets
    attn = tl.load(attn_output_ptr + attn_off).to(tl.float32)

    gate_off = pid_row * gate_stride_row + head * gate_stride_head + d
    g = tl.load(gate_ptr + gate_off)
    sig = tl.sigmoid(g.to(tl.float32))
    if ROUND_SIGMOID:
        sig = sig.to(gate_ptr.dtype.element_ty).to(tl.float32)
    y = (attn * sig).to(attn_output_ptr.dtype.element_ty).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(y), axis=1), 1e-10)
    raw_s = amax * (1.0 / 448.0)
    if SCALE_UE8M0:
        bits = raw_s.to(tl.uint32, bitcast=True)
        exp = (bits >> 23) & 0xFF
        mant = bits & 0x7FFFFF
        ue8m0_exp = tl.where(mant != 0, exp + 1, exp)
        inv_s = ((254 - ue8m0_exp) << 23).to(tl.float32, bitcast=True)
        y_q = tl.clamp(y * inv_s[:, None], -448.0, 448.0).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + attn_off, y_q)
        shifts = (tl.arange(0, 4) * 8).to(tl.uint32)
        packed = tl.sum((ue8m0_exp.to(tl.uint32) & 0xFF) << shifts, axis=0).to(tl.int32)
        tl.store(
            s_ptr + pid_row * s_stride_tok + pid_block.to(tl.int64) * s_stride_grp,
            packed,
        )
    else:
        quant_scale = 448.0 / amax
        y_q = tl.clamp(y * quant_scale[:, None], -448.0, 448.0).to(
            q_ptr.dtype.element_ty
        )
        tl.store(q_ptr + attn_off, y_q)
        tl.store(
            s_ptr + pid_row * s_stride_tok + grp_ids.to(tl.int64) * s_stride_grp,
            raw_s,
        )


def fused_sigmoid_mul(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    inplace: bool = False,
    quant: bool = False,
    round_sigmoid: bool = False,
    block_fp8_quant: bool = False,
    scale_ue8m0: bool = False,
    scale_transposed: bool = False,
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

    ``round_sigmoid`` rounds the sigmoid to the gate dtype before the multiply,
    matching the two-op torch spelling bit for bit.
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
        # A 2D gate may be a column slice of a wider GEMM output.
        if gate.ndim == 2:
            assert gate.stride(1) == 1, "a 2D gate must have inner stride 1"
            gate_stride_row = gate.stride(0)
        else:
            gate_stride_row = hidden_dim
        gate_stride_head = hidden_dim

    if block_fp8_quant:
        assert not quant, "quant and block_fp8_quant are mutually exclusive"
        assert hidden_dim % 128 == 0, "block_fp8_quant requires hidden_dim % 128 == 0"
        assert attn_output.is_contiguous(), (
            "block_fp8_quant requires contiguous attn_output"
        )
        num_groups = hidden_dim // 128
        assert not (scale_ue8m0 and (not scale_transposed or num_groups % 4 != 0))
        q = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
        from sglang.kernels.ops.quantization.fp8_kernel import (
            create_per_token_group_quant_fp8_output_scale,
        )

        s = create_per_token_group_quant_fp8_output_scale(
            x_shape=(num_tokens, hidden_dim),
            device=q.device,
            group_size=128,
            column_major_scales=scale_transposed,
            scale_tma_aligned=scale_transposed,
            scale_ue8m0=scale_ue8m0,
        )
        if num_tokens == 0:
            return (q, s)
        if scale_ue8m0:
            groups_per_block = 4
            num_warps = 2
        elif num_groups % 8 == 0 and not guard_or_true(num_tokens <= 8):
            groups_per_block = 8
            num_warps = 4
        elif num_groups % 4 == 0 and not guard_or_true(num_tokens <= 8):
            groups_per_block = 4
            num_warps = 2
        else:
            groups_per_block = 1
            num_warps = 1
        grid = (num_tokens, num_groups // groups_per_block)
        use_pdl = is_arch_support_pdl()
        pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
        _fused_sigmoid_mul_block_fp8_kernel[grid](
            attn_output,
            gate,
            q,
            s,
            gate_stride_row,
            gate_stride_head,
            s.stride(0),
            s.stride(1),
            hidden_dim,
            HEAD_DIM=head_dim,
            GROUPS_PER_BLOCK=groups_per_block,
            SCALE_UE8M0=scale_ue8m0,
            ROUND_SIGMOID=round_sigmoid,
            USE_PDL=use_pdl,
            num_warps=num_warps,
            **pdl_kwargs,
        )
        return (q, s)

    out = attn_output if inplace else torch.empty_like(attn_output)
    block_h = 1024 if guard_or_true(num_tokens < 1024) else 2048
    kwargs = {}
    if quant:  # one block per row; returns per-token FP8 (q, scale [T, 1])
        block_h = triton.next_power_of_2(hidden_dim)
        q = torch.empty_like(attn_output, dtype=torch.float8_e4m3fn)
        s = torch.empty((num_tokens, 1), dtype=torch.float32, device=q.device)
        kwargs = {"q_ptr": q, "s_ptr": s, "QUANT": True}
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}
    _fused_sigmoid_mul_kernel[grid](
        out,
        attn_output,
        gate,
        gate_stride_row,
        gate_stride_head,
        hidden_dim,
        HEAD_DIM=head_dim,
        BLOCK_H=block_h,
        ROUND_SIGMOID=round_sigmoid,
        USE_PDL=use_pdl,
        num_warps=4,
        **kwargs,
        **pdl_kwargs,
    )
    return (q, s) if quant else out


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

    if guard_or_false(num_tokens >= 1024) or is_batch_invariant_mode_enabled():
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
