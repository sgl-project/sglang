# Fused compressor decode kernels for the OLD path on ROCm.
# c4: 8 blocks register softmax + overlap shift.
# c128: online softmax over 128 blocks.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _compress_c4_decode_old_kernel(
    pool_ptr,  # [max_num_reqs, 8, 4*head_dim] fp32, in/out
    kv_score_input_ptr,  # [bs, 4*head_dim]              fp32
    ape_ptr,  # [4, 2*head_dim]               fp32 (hotfixed)
    out_ptr,  # [bs, head_dim]                fp32, output
    seq_lens_ptr,  # [bs]   int (32/64)
    req_pool_indices_ptr,  # [bs]   int (32/64)
    stride_pool_req,
    stride_pool_slot,
    stride_input_row,
    stride_ape_row,
    stride_out_row,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # 2D grid (request, head_dim-stripe): per-element softmax along head_dim is
    # independent, and overlap-shift stores from different pid_d touch disjoint
    # bytes within the same pool slot, so striping head_dim is race-free.
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_bs).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + pid_bs).to(tl.int64)

    write_pos = (seq_len - 1) % 4 + 4
    should_shift = (seq_len % 4) == 0
    is_first_compress = seq_len == 4

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    in_base = pid_bs.to(tl.int64) * stride_input_row
    pool_req_base = req * stride_pool_req

    # Write current token's 4 sections into pool slot `write_pos`.
    write_slot_base = pool_req_base + write_pos.to(tl.int64) * stride_pool_slot
    for sec in tl.static_range(4):
        x = tl.load(
            kv_score_input_ptr + in_base + sec * HEAD_DIM + d_offs,
            mask=d_mask,
            other=0.0,
        )
        tl.store(
            pool_ptr + write_slot_base + sec * HEAD_DIM + d_offs,
            x,
            mask=d_mask,
        )

    NEG_BIG = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for slot in tl.static_range(8):
        slot_base = pool_req_base + slot * stride_pool_slot

        if slot < 4:
            kv_off = 0 * HEAD_DIM
            score_off = 2 * HEAD_DIM
            ape_off = 0
        else:
            kv_off = 1 * HEAD_DIM
            score_off = 3 * HEAD_DIM
            ape_off = HEAD_DIM

        kv_b = tl.load(
            pool_ptr + slot_base + kv_off + d_offs,
            mask=d_mask,
            other=0.0,
        )
        score_b = tl.load(
            pool_ptr + slot_base + score_off + d_offs,
            mask=d_mask,
            other=0.0,
        )
        ratio_idx = slot % 4
        ape_b = tl.load(
            ape_ptr + ratio_idx * stride_ape_row + ape_off + d_offs,
            mask=d_mask,
            other=0.0,
        )

        # Overlap shift slots 4..7 -> 0..3. Load the OTHER halves only when
        # shift is actually needed to avoid wasted reads.
        if slot >= 4 and should_shift:
            target_base = pool_req_base + (slot - 4) * stride_pool_slot
            kv_other = tl.load(
                pool_ptr + slot_base + 0 * HEAD_DIM + d_offs,
                mask=d_mask,
                other=0.0,
            )
            score_other = tl.load(
                pool_ptr + slot_base + 2 * HEAD_DIM + d_offs,
                mask=d_mask,
                other=0.0,
            )
            tl.store(
                pool_ptr + target_base + 0 * HEAD_DIM + d_offs,
                kv_other,
                mask=d_mask,
            )
            tl.store(
                pool_ptr + target_base + 1 * HEAD_DIM + d_offs,
                kv_b,
                mask=d_mask,
            )
            tl.store(
                pool_ptr + target_base + 2 * HEAD_DIM + d_offs,
                score_other,
                mask=d_mask,
            )
            tl.store(
                pool_ptr + target_base + 3 * HEAD_DIM + d_offs,
                score_b,
                mask=d_mask,
            )

        # First compress (seq_len==4): overlap slots 0..3 are uninitialized,
        # mask them out of the softmax.
        if is_first_compress and slot < 4:
            kv_b = tl.zeros((BLOCK_D,), tl.float32)
            score_b = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
            ape_b = tl.zeros((BLOCK_D,), tl.float32)

        s = score_b + ape_b
        new_max = tl.maximum(running_max, s)
        factor = tl.exp(running_max - new_max)
        running_sum = running_sum * factor
        weighted = weighted * factor
        e = tl.exp(s - new_max)
        running_sum = running_sum + e
        weighted = weighted + kv_b * e
        running_max = new_max

    result = weighted / running_sum
    tl.store(
        out_ptr + pid_bs.to(tl.int64) * stride_out_row + d_offs,
        result,
        mask=d_mask,
    )


_C4_BLOCK_D = 32


def fused_compress_c4_decode_old_triton(
    pool_kv: torch.Tensor,  # [max_num_reqs, 8, 2*head_dim] view of underlying [..., 4*head_dim]
    kv_score_input_kv: torch.Tensor,  # [bs, 2*head_dim]              view of underlying [..., 4*head_dim]
    ape: torch.Tensor,  # [4, 2*head_dim]               fp32
    seq_lens: torch.Tensor,  # [bs] int
    req_pool_indices: torch.Tensor,  # [bs] int
    head_dim: int,
    out: Optional[torch.Tensor] = None,  # [bs, head_dim] fp32
) -> torch.Tensor:
    """Fused c4 (ratio=4, overlap=True) compress for the OLD path."""
    bs = kv_score_input_kv.size(0)
    # `kv` tensors are 2*head_dim views of a 4*head_dim contiguous parent.
    assert pool_kv.dim() == 3 and pool_kv.dtype == torch.float32
    assert pool_kv.shape[1] == 8 and pool_kv.shape[2] == 2 * head_dim
    assert pool_kv.stride(2) == 1 and pool_kv.stride(1) == 4 * head_dim, (
        f"pool_kv must be a view of [N, 8, 4*head_dim] underlying contiguous tensor; "
        f"got strides {pool_kv.stride()}"
    )
    assert kv_score_input_kv.shape == (bs, 2 * head_dim)
    assert kv_score_input_kv.dtype == torch.float32
    assert (
        kv_score_input_kv.stride(1) == 1 and kv_score_input_kv.stride(0) == 4 * head_dim
    ), (
        f"kv_score_input_kv must be a view of [bs, 4*head_dim] underlying contiguous tensor; "
        f"got strides {kv_score_input_kv.stride()}"
    )
    assert (
        ape.shape == (4, 2 * head_dim)
        and ape.dtype == torch.float32
        and ape.is_contiguous()
    )
    assert seq_lens.shape == (bs,) and req_pool_indices.shape == (bs,)

    if out is None:
        out = kv_score_input_kv.new_empty((bs, head_dim))
    else:
        assert out.shape == (bs, head_dim)
        assert out.dtype == torch.float32 and out.is_contiguous()

    if bs == 0:
        return out

    block_d = min(_C4_BLOCK_D, triton.next_power_of_2(head_dim))
    num_d_chunks = triton.cdiv(head_dim, block_d)

    grid = (bs, num_d_chunks)
    _compress_c4_decode_old_kernel[grid](
        pool_kv,
        kv_score_input_kv,
        ape,
        out,
        seq_lens,
        req_pool_indices,
        pool_kv.stride(0),
        pool_kv.stride(1),
        kv_score_input_kv.stride(0),
        ape.stride(0),
        out.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )
    return out


# c128 (ratio=128, overlap=False)
# BLOCK_D=32, BLOCK_S=64 was empirically best on MI355X (gfx950) across
# bs=1..256 and head_dim=128/256/512.
_C128_BLOCK_D = 32
_C128_BLOCK_S = 64


def fused_compress_c128_decode_old_triton(
    pool_kv: torch.Tensor,  # [max_num_reqs, 128, head_dim] view of underlying [..., 2*head_dim]
    kv_score_input_kv: torch.Tensor,  # [bs, head_dim]                view of underlying [..., 2*head_dim]
    ape: torch.Tensor,  # [128, head_dim] fp32
    seq_lens: torch.Tensor,  # [bs] int
    req_pool_indices: torch.Tensor,  # [bs] int
    head_dim: int,
    out: Optional[torch.Tensor] = None,  # [bs, head_dim] fp32
) -> torch.Tensor:
    """Fused c128 (ratio=128, overlap=False) compress for the OLD path."""
    NUM_SLOTS = 128
    bs = kv_score_input_kv.size(0)
    assert pool_kv.dim() == 3 and pool_kv.dtype == torch.float32
    assert pool_kv.shape[1] == NUM_SLOTS and pool_kv.shape[2] == head_dim
    assert pool_kv.stride(2) == 1 and pool_kv.stride(1) == 2 * head_dim, (
        f"pool_kv must be a view of [N, 128, 2*head_dim] underlying contiguous tensor; "
        f"got strides {pool_kv.stride()}"
    )
    assert kv_score_input_kv.shape == (bs, head_dim)
    assert kv_score_input_kv.dtype == torch.float32
    assert (
        kv_score_input_kv.stride(1) == 1 and kv_score_input_kv.stride(0) == 2 * head_dim
    ), (
        f"kv_score_input_kv must be a view of [bs, 2*head_dim] underlying contiguous tensor; "
        f"got strides {kv_score_input_kv.stride()}"
    )
    assert (
        ape.shape == (NUM_SLOTS, head_dim)
        and ape.dtype == torch.float32
        and ape.is_contiguous()
    )
    assert seq_lens.shape == (bs,) and req_pool_indices.shape == (bs,)

    if out is None:
        out = kv_score_input_kv.new_empty((bs, head_dim))
    else:
        assert out.shape == (bs, head_dim)
        assert out.dtype == torch.float32 and out.is_contiguous()

    if bs == 0:
        return out

    block_d = min(_C128_BLOCK_D, triton.next_power_of_2(head_dim))
    num_d_chunks = triton.cdiv(head_dim, block_d)

    grid = (bs, num_d_chunks)
    _compress_c128_decode_chunked_kernel[grid](
        pool_kv,
        kv_score_input_kv,
        ape,
        out,
        seq_lens,
        req_pool_indices,
        pool_kv.stride(0),
        pool_kv.stride(1),
        kv_score_input_kv.stride(0),
        ape.stride(0),
        out.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        BLOCK_S=_C128_BLOCK_S,
        NUM_SLOTS=NUM_SLOTS,
    )
    return out


# Tile slot dimension into BLOCK_S chunks: parallel single-pass reduction inside
# each chunk + online softmax combine across chunks. Replaces a 128-iter serial
# dependency with NUM_SLOTS/BLOCK_S iterations.
@triton.jit
def _compress_c128_decode_chunked_kernel(
    pool_kv_ptr,
    kv_score_input_kv_ptr,
    ape_ptr,
    out_ptr,
    seq_lens_ptr,
    req_pool_indices_ptr,
    stride_pool_req,
    stride_pool_slot,
    stride_input_row,
    stride_ape_row,
    stride_out_row,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    # 2D grid (request, head_dim-stripe); per-element softmax is independent.
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_bs).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + pid_bs).to(tl.int64)
    write_pos = (seq_len - 1) % NUM_SLOTS

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    in_base = pid_bs.to(tl.int64) * stride_input_row
    pool_req_base = req * stride_pool_req

    # Read current token's (kv, score) but defer the pool[write_pos] store
    # until after the softmax pass; otherwise tile loads that span write_pos
    # would race with the store. Patched into the tile via tl.where below.
    kv_input = tl.load(
        kv_score_input_kv_ptr + in_base + 0 * HEAD_DIM + d_offs,
        mask=d_mask,
        other=0.0,
    )
    score_input = tl.load(
        kv_score_input_kv_ptr + in_base + 1 * HEAD_DIM + d_offs,
        mask=d_mask,
        other=0.0,
    )

    NEG_BIG = -1.0e9
    NUM_CHUNKS: tl.constexpr = NUM_SLOTS // BLOCK_S
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    for chunk_idx in tl.static_range(NUM_CHUNKS):
        slot_offs = chunk_idx * BLOCK_S + tl.arange(0, BLOCK_S)
        slot_base = pool_req_base + slot_offs[:, None].to(tl.int64) * stride_pool_slot

        kv_tile = tl.load(
            pool_kv_ptr + slot_base + 0 * HEAD_DIM + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )
        score_tile = tl.load(
            pool_kv_ptr + slot_base + 1 * HEAD_DIM + d_offs[None, :],
            mask=d_mask[None, :],
            other=NEG_BIG,
        )
        ape_tile = tl.load(
            ape_ptr + slot_offs[:, None] * stride_ape_row + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )

        # Patch current token at write_pos into the tile.
        is_writepos = slot_offs[:, None] == write_pos
        kv_tile = tl.where(is_writepos, kv_input[None, :], kv_tile)
        score_tile = tl.where(is_writepos, score_input[None, :], score_tile)

        s = score_tile + ape_tile
        local_max = tl.max(s, axis=0)
        new_max = tl.maximum(running_max, local_max)

        # exp_s uses new_max as reference so chunk_{sum,weighted} are addable
        # after rescaling the running state.
        exp_s = tl.exp(s - new_max[None, :])
        chunk_sum = tl.sum(exp_s, axis=0)
        chunk_weighted = tl.sum(kv_tile * exp_s, axis=0)

        factor = tl.exp(running_max - new_max)
        running_sum = running_sum * factor + chunk_sum
        weighted = weighted * factor + chunk_weighted
        running_max = new_max

    result = weighted / running_sum
    tl.store(
        out_ptr + pid_bs.to(tl.int64) * stride_out_row + d_offs,
        result,
        mask=d_mask,
    )

    # Deferred store of current token (see note at top of kernel).
    write_slot_base = pool_req_base + write_pos.to(tl.int64) * stride_pool_slot
    tl.store(
        pool_kv_ptr + write_slot_base + 0 * HEAD_DIM + d_offs,
        kv_input,
        mask=d_mask,
    )
    tl.store(
        pool_kv_ptr + write_slot_base + 1 * HEAD_DIM + d_offs,
        score_input,
        mask=d_mask,
    )
