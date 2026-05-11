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
    HEAD_DIM_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + pid).to(tl.int64)

    write_pos = (seq_len - 1) % 4 + 4
    should_shift = (seq_len % 4) == 0
    is_first_compress = seq_len == 4

    elem_offs = tl.arange(0, HEAD_DIM_BLOCK)
    elem_mask = elem_offs < HEAD_DIM

    in_base = pid.to(tl.int64) * stride_input_row
    pool_req_base = req * stride_pool_req

    # === 1. Write current token's 4 sections into pool slot `write_pos`. ===
    write_slot_base = pool_req_base + write_pos.to(tl.int64) * stride_pool_slot
    for sec in tl.static_range(4):
        x = tl.load(
            kv_score_input_ptr + in_base + sec * HEAD_DIM + elem_offs,
            mask=elem_mask,
            other=0.0,
        )
        tl.store(
            pool_ptr + write_slot_base + sec * HEAD_DIM + elem_offs,
            x,
            mask=elem_mask,
        )

    # === 2. Online safe softmax + weighted sum across 8 slots, also doing
    #        overlap shift for slots 4..7 -> 0..3 when should_shift.
    NEG_BIG = -1.0e9
    running_max = tl.full((HEAD_DIM_BLOCK,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((HEAD_DIM_BLOCK,), tl.float32)
    weighted = tl.zeros((HEAD_DIM_BLOCK,), tl.float32)

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
            pool_ptr + slot_base + kv_off + elem_offs,
            mask=elem_mask,
            other=0.0,
        )
        score_b = tl.load(
            pool_ptr + slot_base + score_off + elem_offs,
            mask=elem_mask,
            other=0.0,
        )
        ratio_idx = slot % 4
        ape_b = tl.load(
            ape_ptr + ratio_idx * stride_ape_row + ape_off + elem_offs,
            mask=elem_mask,
            other=0.0,
        )

        # Overlap shift: copy slots 4..7 into 0..3 entirely. We already loaded
        # half of kv and half of score for compression; load the OTHER halves
        # only when shift is actually needed to avoid wasted reads.
        if slot >= 4 and should_shift:
            target_base = pool_req_base + (slot - 4) * stride_pool_slot
            kv_other = tl.load(
                pool_ptr + slot_base + 0 * HEAD_DIM + elem_offs,
                mask=elem_mask,
                other=0.0,
            )
            score_other = tl.load(
                pool_ptr + slot_base + 2 * HEAD_DIM + elem_offs,
                mask=elem_mask,
                other=0.0,
            )
            tl.store(
                pool_ptr + target_base + 0 * HEAD_DIM + elem_offs,
                kv_other,
                mask=elem_mask,
            )
            tl.store(
                pool_ptr + target_base + 1 * HEAD_DIM + elem_offs,
                kv_b,
                mask=elem_mask,
            )
            tl.store(
                pool_ptr + target_base + 2 * HEAD_DIM + elem_offs,
                score_other,
                mask=elem_mask,
            )
            tl.store(
                pool_ptr + target_base + 3 * HEAD_DIM + elem_offs,
                score_b,
                mask=elem_mask,
            )

        # Edge case: very first compress (seq_len==4). Overlap slots 0..3 hold
        # uninitialized data; they should contribute nothing to the softmax.
        if is_first_compress and slot < 4:
            kv_b = tl.zeros((HEAD_DIM_BLOCK,), tl.float32)
            score_b = tl.full((HEAD_DIM_BLOCK,), NEG_BIG, tl.float32)
            ape_b = tl.zeros((HEAD_DIM_BLOCK,), tl.float32)

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
        out_ptr + pid.to(tl.int64) * stride_out_row + elem_offs,
        result,
        mask=elem_mask,
    )


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
    # The "kv" tensors are 2*head_dim views of a 4*head_dim contiguous parent
    # so per-row stride must equal 4*head_dim.
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

    HEAD_DIM_BLOCK = triton.next_power_of_2(head_dim)
    grid = (bs,)
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
        HEAD_DIM_BLOCK=HEAD_DIM_BLOCK,
    )
    return out


# ---------------------------------------------------------------------------
# c128 (ratio=128, overlap=False)
# ---------------------------------------------------------------------------


@triton.jit
def _compress_c128_decode_old_kernel(
    pool_kv_ptr,  # [N, 128, 2*head_dim] view: pool stride along slot = 2*head_dim
    kv_score_input_kv_ptr,  # [bs, 2*head_dim]    view: row stride = 2*head_dim
    ape_ptr,  # [128, head_dim]
    out_ptr,  # [bs, head_dim]
    seq_lens_ptr,
    req_pool_indices_ptr,
    stride_pool_req,
    stride_pool_slot,
    stride_input_row,
    stride_ape_row,
    stride_out_row,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,  # tile size along head_dim; CTAs split HEAD_DIM
    NUM_SLOTS: tl.constexpr,  # 128
    NUM_STAGES: tl.constexpr,  # software pipelining stages for the slot loop
):
    # Each CTA owns (request pid_bs, head_dim-stripe pid_d). Splitting along
    # HEAD_DIM is correctness-free since the online softmax state
    # (running_max/sum/weighted) is per-element of head_dim, so stripes are
    # independent. This raises CTA count from `bs` to `bs * ceil(HEAD_DIM/BLOCK_D)`,
    # which is critical when bs is small at decode time.
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_bs).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + pid_bs).to(tl.int64)
    write_pos = (seq_len - 1) % NUM_SLOTS

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    in_base = pid_bs.to(tl.int64) * stride_input_row
    pool_req_base = req * stride_pool_req

    # === 1. Write current token (kv, score) into pool slot `write_pos`. ===
    # Each CTA writes only its own head_dim stripe; stripes are disjoint at the
    # byte level, so no race across pid_d.
    write_slot_base = pool_req_base + write_pos.to(tl.int64) * stride_pool_slot
    for sec in tl.static_range(2):  # 0=kv, 1=score
        x = tl.load(
            kv_score_input_kv_ptr + in_base + sec * HEAD_DIM + d_offs,
            mask=d_mask,
            other=0.0,
        )
        tl.store(
            pool_kv_ptr + write_slot_base + sec * HEAD_DIM + d_offs,
            x,
            mask=d_mask,
        )

    # === 2. Online safe softmax + weighted sum across NUM_SLOTS slots. ===
    # Slots that haven't been filled yet still have score=-inf from clear(),
    # so softmax naturally masks them out -- no special-case needed.
    NEG_BIG = -1.0e9
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    # num_stages enables software pipelining: the compiler issues the next
    # iteration's loads while the current iteration's compute is in flight.
    # Hides HBM latency for the 128-iter sequential dependency chain.
    for slot in tl.range(0, NUM_SLOTS, num_stages=NUM_STAGES):
        slot_base = pool_req_base + slot * stride_pool_slot
        kv_b = tl.load(
            pool_kv_ptr + slot_base + 0 * HEAD_DIM + d_offs,
            mask=d_mask,
            other=0.0,
        )
        score_b = tl.load(
            pool_kv_ptr + slot_base + 1 * HEAD_DIM + d_offs,
            mask=d_mask,
            other=NEG_BIG,
        )
        ape_b = tl.load(
            ape_ptr + slot * stride_ape_row + d_offs,
            mask=d_mask,
            other=0.0,
        )

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


# BLOCK_D=32, BLOCK_S=64 was the empirically best config on MI355X (gfx950)
# across bs=1..256 and head_dim=128/256/512. BLOCK_S=64 leaves only 2 outer
# chunks for the 128 slots, killing the serial dependency that bottlenecked
# the original kernel; BLOCK_D=32 sizes each CTA to half a 64-wide wavefront,
# which maximizes CTA count (and therefore HBM pipelining) at low decode bs.
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

    # Clamp BLOCK_D to head_dim's next-power-of-2 so we never launch more
    # head_dim-stripes than there is data to cover.
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


# ---------------------------------------------------------------------------
# c128 chunked variant: parallel-within-chunk + online combine across chunks
# ---------------------------------------------------------------------------
#
# The naive c128 kernel above has a 128-iteration serial dependency on
# (running_max, running_sum, weighted) -- each iter must wait for HBM round-trip
# of the previous one. On MI355X this dominates wall time (~40 us is mostly
# load latency * 128).
#
# This variant tiles the slot dimension into BLOCK_S-sized chunks. Within a
# chunk, we issue all BLOCK_S * 3 loads in parallel and do a single-pass
# (BLOCK_S-way) reduction (no sequential dep inside the chunk). Across chunks
# we still do online softmax combine, but only NUM_SLOTS/BLOCK_S = 4 sequential
# iters (BLOCK_S=32) instead of 128.
#
# Per-CTA register footprint is the bottleneck:
#   3 tiles of (BLOCK_S, BLOCK_D) floats in registers (kv, score+ape, exp_s).
#   With BLOCK_S=32, BLOCK_D=64 -> 32 * 64 * 3 = 6 KB per tile-set per CTA.
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
    # 2D grid: (request, head_dim-stripe). Same correctness argument as the
    # other kernel -- per-element of head_dim the softmax is independent.
    pid_bs = tl.program_id(0)
    pid_d = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + pid_bs).to(tl.int32)
    req = tl.load(req_pool_indices_ptr + pid_bs).to(tl.int64)
    write_pos = (seq_len - 1) % NUM_SLOTS

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    in_base = pid_bs.to(tl.int64) * stride_input_row
    pool_req_base = req * stride_pool_req

    # === 1. Read current token's (kv, score) from input. ===
    # NOTE: we deliberately do *not* store these to pool[write_pos] up front.
    # In the chunked kernel below, tile loads pull multiple slots at once, and
    # if write_pos falls inside a tile, the read can race with the store. We
    # sidestep this by keeping the current token in registers and patching it
    # into the tile via tl.where; the actual store to pool is deferred to the
    # end of the kernel.
    kv_input = tl.load(
        kv_score_input_kv_ptr + in_base + 0 * HEAD_DIM + d_offs,
        mask=d_mask,
        other=0.0,
    )  # (BLOCK_D,)
    score_input = tl.load(
        kv_score_input_kv_ptr + in_base + 1 * HEAD_DIM + d_offs,
        mask=d_mask,
        other=0.0,
    )  # (BLOCK_D,)

    # === 2. Chunked softmax over NUM_SLOTS / BLOCK_S chunks. ===
    NEG_BIG = -1.0e9
    NUM_CHUNKS: tl.constexpr = NUM_SLOTS // BLOCK_S
    running_max = tl.full((BLOCK_D,), NEG_BIG, tl.float32)
    running_sum = tl.zeros((BLOCK_D,), tl.float32)
    weighted = tl.zeros((BLOCK_D,), tl.float32)

    # Static-unroll the chunks (no need for software pipelining at this depth;
    # the bulk of the work is the parallel intra-chunk reductions).
    for chunk_idx in tl.static_range(NUM_CHUNKS):
        slot_offs = chunk_idx * BLOCK_S + tl.arange(0, BLOCK_S)  # (BLOCK_S,)
        slot_base = pool_req_base + slot_offs[:, None].to(tl.int64) * stride_pool_slot

        # Tile load: all BLOCK_S slots' kv / score / ape in one go.
        kv_tile = tl.load(
            pool_kv_ptr + slot_base + 0 * HEAD_DIM + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )  # (BLOCK_S, BLOCK_D)
        score_tile = tl.load(
            pool_kv_ptr + slot_base + 1 * HEAD_DIM + d_offs[None, :],
            mask=d_mask[None, :],
            other=NEG_BIG,
        )  # (BLOCK_S, BLOCK_D)
        ape_tile = tl.load(
            ape_ptr + slot_offs[:, None] * stride_ape_row + d_offs[None, :],
            mask=d_mask[None, :],
            other=0.0,
        )  # (BLOCK_S, BLOCK_D)

        # Patch in the current token at write_pos. The mask is (BLOCK_S, 1)
        # which broadcasts; kv_input / score_input are (BLOCK_D,) and broadcast
        # along the slot axis.
        is_writepos = slot_offs[:, None] == write_pos  # (BLOCK_S, 1)
        kv_tile = tl.where(is_writepos, kv_input[None, :], kv_tile)
        score_tile = tl.where(is_writepos, score_input[None, :], score_tile)

        s = score_tile + ape_tile  # (BLOCK_S, BLOCK_D)

        # Single-pass softmax within the chunk, then online combine into running
        # state. The choice of `new_max = max(running_max, max(s))` followed by
        # `exp(s - new_max)` (instead of the textbook `exp(s - max(s))` then
        # rescale) lets us avoid a redundant exp pass on the chunk data.
        local_max = tl.max(s, axis=0)  # (BLOCK_D,)
        new_max = tl.maximum(running_max, local_max)

        # exp_s already uses new_max as the reference, so chunk_{sum,weighted}
        # are directly addable after rescaling the running state.
        exp_s = tl.exp(s - new_max[None, :])  # (BLOCK_S, BLOCK_D)
        chunk_sum = tl.sum(exp_s, axis=0)  # (BLOCK_D,)
        chunk_weighted = tl.sum(kv_tile * exp_s, axis=0)  # (BLOCK_D,)

        # Rescale running state to new_max basis; for chunk 0, running_max is
        # -inf so factor is 0 and running state stays clean.
        factor = tl.exp(running_max - new_max)  # (BLOCK_D,)
        running_sum = running_sum * factor + chunk_sum
        weighted = weighted * factor + chunk_weighted
        running_max = new_max

    result = weighted / running_sum
    tl.store(
        out_ptr + pid_bs.to(tl.int64) * stride_out_row + d_offs,
        result,
        mask=d_mask,
    )

    # === 3. Now (and only now) store the current token to pool[write_pos]. ===
    # Done after the softmax so the read pass never raced against the write.
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
