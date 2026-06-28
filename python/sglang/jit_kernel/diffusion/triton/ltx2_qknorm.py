# Adapted from NVlabs/Sana sol-engine LTX2 qknorm + split-RoPE fusion.

import torch
import triton
import triton.language as tl


@triton.jit
def _ltx2_qknorm_split_rope_pair_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    q_cos_ptr,
    q_sin_ptr,
    k_cos_ptr,
    k_sin_ptr,
    q_rows: tl.constexpr,
    k_rows: tl.constexpr,
    q_seq: tl.constexpr,
    k_seq: tl.constexpr,
    hidden: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    q_cos_stride_b: tl.constexpr,
    q_cos_stride_h: tl.constexpr,
    q_cos_stride_t: tl.constexpr,
    q_sin_stride_b: tl.constexpr,
    q_sin_stride_h: tl.constexpr,
    q_sin_stride_t: tl.constexpr,
    k_cos_stride_b: tl.constexpr,
    k_cos_stride_h: tl.constexpr,
    k_cos_stride_t: tl.constexpr,
    k_sin_stride_b: tl.constexpr,
    k_sin_stride_h: tl.constexpr,
    k_sin_stride_t: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HALF)
    half_hidden: tl.constexpr = hidden // 2
    mask = offsets < half_hidden
    head = offsets // half_dim
    dim = offsets - head * half_dim
    first_col = head * head_dim + dim
    second_col = first_col + half_dim

    q_active = row < q_rows
    k_active = row < k_rows

    q_first = tl.load(
        q_ptr + row * hidden + first_col, mask=q_active & mask, other=0.0
    ).to(tl.float32)
    q_second = tl.load(
        q_ptr + row * hidden + second_col, mask=q_active & mask, other=0.0
    ).to(tl.float32)
    k_first = tl.load(
        k_ptr + row * hidden + first_col, mask=k_active & mask, other=0.0
    ).to(tl.float32)
    k_second = tl.load(
        k_ptr + row * hidden + second_col, mask=k_active & mask, other=0.0
    ).to(tl.float32)

    q_w_first = tl.load(q_weight_ptr + first_col, mask=mask, other=0.0).to(tl.float32)
    q_w_second = tl.load(q_weight_ptr + second_col, mask=mask, other=0.0).to(tl.float32)
    k_w_first = tl.load(k_weight_ptr + first_col, mask=mask, other=0.0).to(tl.float32)
    k_w_second = tl.load(k_weight_ptr + second_col, mask=mask, other=0.0).to(tl.float32)

    q_var = tl.sum(q_first * q_first + q_second * q_second, axis=0) / hidden
    k_var = tl.sum(k_first * k_first + k_second * k_second, axis=0) / hidden
    q_rstd = tl.rsqrt(q_var + eps)
    k_rstd = tl.rsqrt(k_var + eps)

    q_first = (q_first * q_rstd * q_w_first).to(tl.bfloat16)
    q_second = (q_second * q_rstd * q_w_second).to(tl.bfloat16)
    k_first = (k_first * k_rstd * k_w_first).to(tl.bfloat16)
    k_second = (k_second * k_rstd * k_w_second).to(tl.bfloat16)

    q_batch = row // q_seq
    q_token = row - q_batch * q_seq
    k_batch = row // k_seq
    k_token = row - k_batch * k_seq

    q_pos = (
        q_batch * q_cos_stride_b
        + head * q_cos_stride_h
        + q_token * q_cos_stride_t
        + dim
    )
    q_sin_pos = (
        q_batch * q_sin_stride_b
        + head * q_sin_stride_h
        + q_token * q_sin_stride_t
        + dim
    )
    k_pos = (
        k_batch * k_cos_stride_b
        + head * k_cos_stride_h
        + k_token * k_cos_stride_t
        + dim
    )
    k_sin_pos = (
        k_batch * k_sin_stride_b
        + head * k_sin_stride_h
        + k_token * k_sin_stride_t
        + dim
    )

    q_cos = tl.load(q_cos_ptr + q_pos, mask=q_active & mask, other=0.0)
    q_sin = tl.load(q_sin_ptr + q_sin_pos, mask=q_active & mask, other=0.0)
    k_cos = tl.load(k_cos_ptr + k_pos, mask=k_active & mask, other=0.0)
    k_sin = tl.load(k_sin_ptr + k_sin_pos, mask=k_active & mask, other=0.0)

    q_out_first = (q_first * q_cos).to(tl.bfloat16).to(tl.float32) + (
        -q_second.to(tl.float32) * q_sin.to(tl.float32)
    )
    q_out_second = (q_second * q_cos).to(tl.bfloat16).to(tl.float32) + (
        q_first.to(tl.float32) * q_sin.to(tl.float32)
    )
    k_out_first = (k_first * k_cos).to(tl.bfloat16).to(tl.float32) + (
        -k_second.to(tl.float32) * k_sin.to(tl.float32)
    )
    k_out_second = (k_second * k_cos).to(tl.bfloat16).to(tl.float32) + (
        k_first.to(tl.float32) * k_sin.to(tl.float32)
    )

    tl.store(q_out_ptr + row * hidden + first_col, q_out_first, mask=q_active & mask)
    tl.store(q_out_ptr + row * hidden + second_col, q_out_second, mask=q_active & mask)
    tl.store(k_out_ptr + row * hidden + first_col, k_out_first, mask=k_active & mask)
    tl.store(k_out_ptr + row * hidden + second_col, k_out_second, mask=k_active & mask)


@triton.jit
def _ltx2_split_rope_pair_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_cos_ptr,
    q_sin_ptr,
    k_cos_ptr,
    k_sin_ptr,
    q_rows: tl.constexpr,
    k_rows: tl.constexpr,
    q_seq: tl.constexpr,
    k_seq: tl.constexpr,
    hidden: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    q_cos_stride_b: tl.constexpr,
    q_cos_stride_h: tl.constexpr,
    q_cos_stride_t: tl.constexpr,
    q_sin_stride_b: tl.constexpr,
    q_sin_stride_h: tl.constexpr,
    q_sin_stride_t: tl.constexpr,
    k_cos_stride_b: tl.constexpr,
    k_cos_stride_h: tl.constexpr,
    k_cos_stride_t: tl.constexpr,
    k_sin_stride_b: tl.constexpr,
    k_sin_stride_h: tl.constexpr,
    k_sin_stride_t: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_HALF)
    half_hidden: tl.constexpr = hidden // 2
    mask = offsets < half_hidden
    head = offsets // half_dim
    dim = offsets - head * half_dim
    first_col = head * head_dim + dim
    second_col = first_col + half_dim

    q_active = row < q_rows
    k_active = row < k_rows

    q_first = tl.load(q_ptr + row * hidden + first_col, mask=q_active & mask, other=0.0)
    q_second = tl.load(
        q_ptr + row * hidden + second_col, mask=q_active & mask, other=0.0
    )
    k_first = tl.load(k_ptr + row * hidden + first_col, mask=k_active & mask, other=0.0)
    k_second = tl.load(
        k_ptr + row * hidden + second_col, mask=k_active & mask, other=0.0
    )

    q_batch = row // q_seq
    q_token = row - q_batch * q_seq
    k_batch = row // k_seq
    k_token = row - k_batch * k_seq

    q_pos = (
        q_batch * q_cos_stride_b
        + head * q_cos_stride_h
        + q_token * q_cos_stride_t
        + dim
    )
    q_sin_pos = (
        q_batch * q_sin_stride_b
        + head * q_sin_stride_h
        + q_token * q_sin_stride_t
        + dim
    )
    k_pos = (
        k_batch * k_cos_stride_b
        + head * k_cos_stride_h
        + k_token * k_cos_stride_t
        + dim
    )
    k_sin_pos = (
        k_batch * k_sin_stride_b
        + head * k_sin_stride_h
        + k_token * k_sin_stride_t
        + dim
    )

    q_cos = tl.load(q_cos_ptr + q_pos, mask=q_active & mask, other=0.0)
    q_sin = tl.load(q_sin_ptr + q_sin_pos, mask=q_active & mask, other=0.0)
    k_cos = tl.load(k_cos_ptr + k_pos, mask=k_active & mask, other=0.0)
    k_sin = tl.load(k_sin_ptr + k_sin_pos, mask=k_active & mask, other=0.0)

    q_out_first = (q_first * q_cos).to(tl.bfloat16).to(tl.float32) + (
        -q_second.to(tl.float32) * q_sin.to(tl.float32)
    )
    q_out_second = (q_second * q_cos).to(tl.bfloat16).to(tl.float32) + (
        q_first.to(tl.float32) * q_sin.to(tl.float32)
    )
    k_out_first = (k_first * k_cos).to(tl.bfloat16).to(tl.float32) + (
        -k_second.to(tl.float32) * k_sin.to(tl.float32)
    )
    k_out_second = (k_second * k_cos).to(tl.bfloat16).to(tl.float32) + (
        k_first.to(tl.float32) * k_sin.to(tl.float32)
    )

    tl.store(q_out_ptr + row * hidden + first_col, q_out_first, mask=q_active & mask)
    tl.store(q_out_ptr + row * hidden + second_col, q_out_second, mask=q_active & mask)
    tl.store(k_out_ptr + row * hidden + first_col, k_out_first, mask=k_active & mask)
    tl.store(k_out_ptr + row * hidden + second_col, k_out_second, mask=k_active & mask)


def _num_warps_for_hidden(hidden: int) -> int:
    return 4 if hidden >= 4096 else 8


def ltx2_qknorm_split_rope_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [batch, seq, hidden]")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q/k batch mismatch: {q.shape[0]} vs {k.shape[0]}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k hidden mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("fused qknorm+rope currently supports bfloat16 only")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    if (
        q_cos.ndim != 4
        or q_sin.shape != q_cos.shape
        or k_cos.ndim != 4
        or k_sin.shape != k_cos.shape
        or q_cos.stride(-1) != 1
        or q_sin.stride(-1) != 1
        or k_cos.stride(-1) != 1
        or k_sin.stride(-1) != 1
    ):
        raise ValueError(
            "cos/sin tensors must have shape [batch, heads, seq, half_dim] "
            "and be last-dim contiguous"
        )

    batch, q_seq, hidden = q.shape
    _, k_seq, _ = k.shape
    cos_batch, num_heads, cos_q_seq, half_dim = q_cos.shape
    k_cos_batch, k_num_heads, cos_k_seq, k_half_dim = k_cos.shape
    head_dim = half_dim * 2
    if (
        cos_batch != batch
        or k_cos_batch != batch
        or cos_q_seq != q_seq
        or cos_k_seq != k_seq
        or k_num_heads != num_heads
        or k_half_dim != half_dim
        or hidden != num_heads * head_dim
    ):
        raise ValueError(
            "LTX2 fused qknorm+rope shape mismatch: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, "
            f"q_cos={tuple(q_cos.shape)}, k_cos={tuple(k_cos.shape)}"
        )
    if q_weight.numel() != hidden or k_weight.numel() != hidden:
        raise ValueError("q/k RMSNorm weights must match hidden size")

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = int(batch * q_seq)
    k_rows = int(batch * k_seq)
    block_half = triton.next_power_of_2(num_heads * half_dim)
    _ltx2_qknorm_split_rope_pair_kernel[(max(q_rows, k_rows),)](
        q.view(-1, hidden),
        k.view(-1, hidden),
        q_out.view(-1, hidden),
        k_out.view(-1, hidden),
        q_weight,
        k_weight,
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        q_rows,
        k_rows,
        q_seq,
        k_seq,
        hidden,
        num_heads,
        head_dim,
        half_dim,
        q_cos.stride(0),
        q_cos.stride(1),
        q_cos.stride(2),
        q_sin.stride(0),
        q_sin.stride(1),
        q_sin.stride(2),
        k_cos.stride(0),
        k_cos.stride(1),
        k_cos.stride(2),
        k_sin.stride(0),
        k_sin.stride(1),
        k_sin.stride(2),
        eps,
        BLOCK_HALF=block_half,
        num_warps=_num_warps_for_hidden(hidden),
    )
    return q_out, k_out


def ltx2_split_rope_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or k.ndim != 3:
        raise ValueError("q and k must have shape [batch, seq, hidden]")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"q/k batch mismatch: {q.shape[0]} vs {k.shape[0]}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k hidden mismatch: {q.shape[-1]} vs {k.shape[-1]}")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError("fused split-RoPE currently supports bfloat16 only")
    if not q.is_contiguous() or not k.is_contiguous():
        raise ValueError("q and k must be contiguous")
    if (
        q_cos.ndim != 4
        or q_sin.shape != q_cos.shape
        or k_cos.ndim != 4
        or k_sin.shape != k_cos.shape
    ):
        raise ValueError(
            "cos/sin tensors must have shape [batch, heads, seq, half_dim]"
        )

    batch, q_seq, hidden = q.shape
    _, k_seq, _ = k.shape
    cos_batch, num_heads, cos_q_seq, half_dim = q_cos.shape
    k_cos_batch, k_num_heads, cos_k_seq, k_half_dim = k_cos.shape
    head_dim = half_dim * 2
    if (
        cos_batch != batch
        or k_cos_batch != batch
        or cos_q_seq != q_seq
        or cos_k_seq != k_seq
        or k_num_heads != num_heads
        or k_half_dim != half_dim
        or hidden != num_heads * head_dim
    ):
        raise ValueError(
            "LTX2 fused split-RoPE shape mismatch: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, "
            f"q_cos={tuple(q_cos.shape)}, k_cos={tuple(k_cos.shape)}"
        )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = int(batch * q_seq)
    k_rows = int(batch * k_seq)
    block_half = triton.next_power_of_2(num_heads * half_dim)
    _ltx2_split_rope_pair_kernel[(max(q_rows, k_rows),)](
        q.view(-1, hidden),
        k.view(-1, hidden),
        q_out.view(-1, hidden),
        k_out.view(-1, hidden),
        q_cos,
        q_sin,
        k_cos,
        k_sin,
        q_rows,
        k_rows,
        q_seq,
        k_seq,
        hidden,
        num_heads,
        head_dim,
        half_dim,
        q_cos.stride(0),
        q_cos.stride(1),
        q_cos.stride(2),
        q_sin.stride(0),
        q_sin.stride(1),
        q_sin.stride(2),
        k_cos.stride(0),
        k_cos.stride(1),
        k_cos.stride(2),
        k_sin.stride(0),
        k_sin.stride(1),
        k_sin.stride(2),
        BLOCK_HALF=block_half,
        num_warps=_num_warps_for_hidden(hidden),
    )
    return q_out, k_out
