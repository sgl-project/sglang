"""Triton fused kernel for the OLD-path c4 (ratio=4, overlap=True) compressor decode.

Replaces the dozens of torch ops in
``deepseek_v4.py:Compressor.compress_decode_old`` with a single launch when
running on ROCm. Uses the existing OLD state-pool layout, so no env-var or
pool-shape change is needed.

Pool layout (c4):
    state [max_num_reqs, ratio*coff, 2*head_dim*coff]
        = [max_num_reqs, 8, 4*head_dim]
Each slot holds 4 head_dim-wide sections in the order
    [kv_overlap | kv_normal | score_overlap | score_normal]

Compression for a token at seq_len % 4 == 0 reads 8 (kv, score) blocks:
    slot i in [0, 4): overlap half  -> kv at [0,head_dim), score at [2,3)*head_dim
    slot i in [4, 8): normal  half  -> kv at [head_dim,2*head_dim),
                                       score at [3,4)*head_dim
ape ``[ratio=4, 2*head_dim]`` is (already-hotfixed) and split the same way:
    overlap half  : ape[ratio_idx, :head_dim]
    normal  half  : ape[ratio_idx, head_dim:]
where ratio_idx = i % 4.
"""

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
    """Fused c4 (ratio=4, overlap=True) compress for the OLD path.

    Accepts the ``KVAndScoreOld`` ``kv`` views directly. Their ``data_ptr`` is
    the start of the underlying contiguous ``[..., 4*head_dim]`` allocation
    (with score the second 2*head_dim section). Stride info from the kv view
    is used to locate slots; the kernel addresses the whole 4*head_dim slot
    region (kv + score) of the underlying tensor.

    Replaces this slice of ``compress_decode_old``::

        pool[req, write_pos] = kv_and_scores
        kv_to_compress = pool[req]
        if should_shift: pool[req, :4] = kv_to_compress[:, 4:]
        kv_to_compress.score += ape
        kv_to_compress = overlap_transform_decode(kv_to_compress)
        kv_compressed = (kv * softmax(score, dim=1)).sum(dim=1)

    Norm + RoPE on top of the returned tensor are done by the caller (already
    fused in fused_norm_rope_inplace_triton).
    """
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
