import torch
import triton
import triton.language as tl
from torch.fx.experimental.symbolic_shapes import guard_or_false, guard_or_true

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.utils import is_hip

from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL

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
    PDL_SMALL: tl.constexpr = False,
):
    """Fuse sigmoid(gate) * attn_output into a single kernel."""
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
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
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


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
    block_h = 1024 if guard_or_true(num_tokens < 1024) else 2048
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
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
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


@triton.jit
def _fused_finalize_gate_sigmoid_mul_add_kernel(
    hidden_states_ptr,  # [num_tokens, hidden_dim]
    gate_weight_ptr,  # [hidden_dim]
    shared_output_ptr,  # [num_tokens, hidden_dim]
    output_ptr,  # [num_tokens, hidden_dim]
    gemm2_out_ptr,  # [num_permuted_rows, gemm2_stride] BF16
    permuted_idx_ptr,  # [num_tokens * TOP_K] int32, -1 = not local
    expert_weights_ptr,  # [num_tokens, TOP_K]
    gemm2_stride,
    hidden_dim: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    # One row per program, laid out exactly as _fused_gate_sigmoid_mul_add_kernel
    # so the gate reduction is the same; the routed term replaces its load of
    # the TRT-LLM finalize output with that kernel's arithmetic: an FP32 FMA
    # chain (flush-to-zero) over the top-k slots in slot order, rounded once
    # to BF16.
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
    s = tl.load(shared_output_ptr + row_offset + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = tl.load(permuted_idx_ptr + pid * TOP_K + k)
        scale = tl.load(expert_weights_ptr + pid * TOP_K + k).to(tl.float32)
        valid = p >= 0
        x = tl.load(
            gemm2_out_ptr + p.to(tl.int64) * gemm2_stride + offsets,
            mask=mask & valid,
            other=0.0,
        ).to(tl.float32)
        # finalizeKernel is built with -use_fast_math: fma.rn.ftz.f32.
        fma = tl.inline_asm_elementwise(
            "fma.rn.ftz.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            [scale + tl.zeros_like(x), x, acc],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        acc = tl.where(valid, fma, acc)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    f = acc.to(tl.bfloat16).to(tl.float32)
    gate_val = tl.sigmoid(tl.sum(h * w, axis=0))
    result = gate_val * s
    result += f
    tl.store(output_ptr + row_offset + offsets, result, mask=mask)


def fused_moe_finalize_gate_sigmoid_mul_add(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    shared_output: torch.Tensor,
    gemm2_out: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    expert_weights: torch.Tensor,
) -> torch.Tensor:
    """TRT-LLM MoE finalize plus the gated shared-expert add in one launch.

    Bit-identical to the TRT-LLM finalizeKernel (the small-grid variant, used
    while 10 * num_tokens < 1184) followed by fused_gate_sigmoid_mul_add.
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert gate_weight.is_contiguous(), "gate_weight must be contiguous"
    assert shared_output.is_contiguous(), "shared_output must be contiguous"
    assert gemm2_out.dtype == torch.bfloat16 and gemm2_out.stride(1) == 1
    assert expanded_idx_to_permuted_idx.dtype == torch.int32
    assert expanded_idx_to_permuted_idx.is_contiguous()
    assert expert_weights.is_contiguous()

    num_tokens, hidden_dim = hidden_states.shape
    top_k = expert_weights.shape[1]
    assert gate_weight.shape == (hidden_dim,)
    assert shared_output.shape == (num_tokens, hidden_dim)
    assert expert_weights.shape[0] == num_tokens
    assert expanded_idx_to_permuted_idx.numel() == num_tokens * top_k
    assert gemm2_out.shape[1] >= hidden_dim

    output = torch.empty_like(shared_output)
    # Same launch configuration as _launch_fused_gate_sigmoid_mul.
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

    _fused_finalize_gate_sigmoid_mul_add_kernel[(num_tokens,)](
        hidden_states,
        gate_weight,
        shared_output,
        output,
        gemm2_out,
        expanded_idx_to_permuted_idx,
        expert_weights,
        gemm2_out.stride(0),
        hidden_dim=hidden_dim,
        TOP_K=top_k,
        USE_PDL=use_pdl,
        **config,
        **pdl_kwargs,
    )
    return output
