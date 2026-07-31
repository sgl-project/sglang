"""Fused staging kernels for the NVIDIA KDA prefill backend."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_inputs_kernel(
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    q_out,
    k_out,
    v_out,
    g_out,
    beta_out,
    seq_start,
    bucket,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    g_stride_t,
    g_stride_h,
    g_stride_d,
    beta_stride_t,
    beta_stride_h,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    head = tl.program_id(2)
    token_offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, D)

    source_start = tl.load(cu_seqlens + seq_start + row)
    source_end = tl.load(cu_seqlens + seq_start + row + 1)
    valid_token = token_offsets < (source_end - source_start)
    bucket_token = token_offsets < bucket
    source_tokens = source_start + token_offsets

    q_offsets = (
        source_tokens[:, None] * q_stride_t
        + head * q_stride_h
        + dim_offsets[None, :] * q_stride_d
    )
    k_offsets = (
        source_tokens[:, None] * k_stride_t
        + head * k_stride_h
        + dim_offsets[None, :] * k_stride_d
    )
    mask = valid_token[:, None]
    q_values = tl.load(q + q_offsets, mask=mask, other=0.0).to(tl.float32)
    k_values = tl.load(k + k_offsets, mask=mask, other=0.0).to(tl.float32)
    q_values /= tl.sqrt(tl.sum(q_values * q_values, axis=1)[:, None] + 1e-6)
    k_values /= tl.sqrt(tl.sum(k_values * k_values, axis=1)[:, None] + 1e-6)

    output_offsets = (
        (row * bucket + token_offsets[:, None]) * H * D
        + head * D
        + dim_offsets[None, :]
    )
    output_mask = bucket_token[:, None]
    tl.store(q_out + output_offsets, q_values, mask=output_mask)
    tl.store(k_out + output_offsets, k_values, mask=output_mask)

    v_offsets = (
        source_tokens[:, None] * v_stride_t
        + head * v_stride_h
        + dim_offsets[None, :] * v_stride_d
    )
    g_offsets = (
        source_tokens[:, None] * g_stride_t
        + head * g_stride_h
        + dim_offsets[None, :] * g_stride_d
    )
    v_values = tl.load(v + v_offsets, mask=mask, other=0.0)
    g_values = tl.load(g + g_offsets, mask=mask, other=-1000.0)
    tl.store(v_out + output_offsets, v_values, mask=output_mask)
    tl.store(g_out + output_offsets, g_values, mask=output_mask)

    beta_offsets = source_tokens * beta_stride_t + head * beta_stride_h
    beta_values = tl.load(beta + beta_offsets, mask=valid_token, other=0.0)
    tl.store(
        beta_out + (row * bucket + token_offsets) * H + head,
        beta_values,
        mask=bucket_token,
    )


def pack_nvidia_kda_inputs(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_start: int,
    group_size: int,
    staging: dict[str, torch.Tensor],
) -> None:
    """Normalize and repack one packed-varlen group into its padded bucket."""
    q_view = q[0]
    k_view = k[0]
    v_view = v[0]
    g_view = g[0].view(q.shape[1], q.shape[2], q.shape[3])
    beta_view = beta[0]
    bucket = staging["q"].shape[1]
    heads = q.shape[2]
    dim = q.shape[3]
    assert dim == 128 and v.shape[-1] == dim
    assert staging["q"].shape[0] == group_size

    block_t = 8
    _pack_inputs_kernel[(group_size, triton.cdiv(bucket, block_t), heads)](
        q_view,
        k_view,
        v_view,
        g_view,
        beta_view,
        cu_seqlens,
        staging["q"],
        staging["k"],
        staging["v"],
        staging["g"],
        staging["beta"],
        seq_start,
        bucket,
        *q_view.stride(),
        *k_view.stride(),
        *v_view.stride(),
        *g_view.stride(),
        *beta_view.stride(),
        H=heads,
        D=dim,
        BLOCK_T=block_t,
        num_warps=8,
        num_stages=2,
    )


@triton.jit
def _unpack_output_kernel(
    source,
    output,
    cu_seqlens,
    seq_start,
    bucket,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    head = tl.program_id(2)
    token_offsets = tile * BLOCK_T + tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, D)
    destination_start = tl.load(cu_seqlens + seq_start + row)
    destination_end = tl.load(cu_seqlens + seq_start + row + 1)
    valid = token_offsets < (destination_end - destination_start)
    source_offsets = (
        (row * bucket + token_offsets[:, None]) * H * D
        + head * D
        + dim_offsets[None, :]
    )
    destination_offsets = (
        (destination_start + token_offsets)[:, None] * H * D
        + head * D
        + dim_offsets[None, :]
    )
    values = tl.load(source + source_offsets, mask=valid[:, None])
    tl.store(output + destination_offsets, values, mask=valid[:, None])


def unpack_nvidia_kda_output(
    source: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    seq_start: int,
) -> None:
    group_size, bucket, heads, dim = source.shape
    block_t = 8
    _unpack_output_kernel[(group_size, triton.cdiv(bucket, block_t), heads)](
        source,
        output[0],
        cu_seqlens,
        seq_start,
        bucket,
        H=heads,
        D=dim,
        BLOCK_T=block_t,
        num_warps=8,
        num_stages=2,
    )


@triton.jit
def _gather_state_kernel(
    pool,
    slots,
    output,
    pool_stride_slot,
    pool_stride_head,
    pool_stride_v,
    pool_stride_k,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < D * D
    k_offsets = offsets // D
    v_offsets = offsets % D
    slot = tl.load(slots + row).to(tl.int64)
    source_offsets = (
        slot * pool_stride_slot
        + head * pool_stride_head
        + v_offsets * pool_stride_v
        + k_offsets * pool_stride_k
    )
    destination_offsets = ((row * H + head) * D + k_offsets) * D + v_offsets
    values = tl.load(pool + source_offsets, mask=valid)
    tl.store(output + destination_offsets, values, mask=valid)


def gather_nvidia_kda_state(
    pool: torch.Tensor,
    slots: torch.Tensor,
    output: torch.Tensor,
) -> None:
    rows, heads, dim_k, dim_v = output.shape
    assert dim_k == dim_v == 128 and slots.numel() == rows
    block = 256
    _gather_state_kernel[(rows, heads, triton.cdiv(dim_k * dim_v, block))](
        pool,
        slots,
        output,
        *pool.stride(),
        H=heads,
        D=dim_k,
        BLOCK=block,
        num_warps=8,
    )


@triton.jit
def _scatter_state_kernel(
    source,
    slots,
    pool,
    pool_stride_slot,
    pool_stride_head,
    pool_stride_v,
    pool_stride_k,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < D * D
    k_offsets = offsets // D
    v_offsets = offsets % D
    slot = tl.load(slots + row).to(tl.int64)
    source_offsets = ((row * H + head) * D + k_offsets) * D + v_offsets
    destination_offsets = (
        slot * pool_stride_slot
        + head * pool_stride_head
        + v_offsets * pool_stride_v
        + k_offsets * pool_stride_k
    )
    values = tl.load(source + source_offsets, mask=valid)
    tl.store(pool + destination_offsets, values, mask=valid)


def scatter_nvidia_kda_state(
    source: torch.Tensor,
    slots: torch.Tensor,
    pool: torch.Tensor,
) -> None:
    rows, heads, dim_k, dim_v = source.shape
    assert dim_k == dim_v == 128 and slots.numel() == rows
    block = 256
    _scatter_state_kernel[(rows, heads, triton.cdiv(dim_k * dim_v, block))](
        source,
        slots,
        pool,
        *pool.stride(),
        H=heads,
        D=dim_k,
        BLOCK=block,
        num_warps=8,
    )
