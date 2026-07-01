import torch
import triton
import triton.language as tl

from sglang.srt.speculative.triton_ops.dflash import _pick_num_warps


@triton.jit
def _dspark_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    confidence_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    confidence_row_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    confidence_threshold,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    confidence_row_ptr = confidence_ptr + row * confidence_row_stride

    match_prefix_len = tl.full((), 0, tl.int32)
    match_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (match_live != 0) & (match_i32 != 0)
        match_prefix_len += keep.to(tl.int32)
        match_live = tl.where(in_range, match_live & match_i32, match_live)

    confidence_prefix_len = tl.full((), 0, tl.int32)
    confidence_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE):
        in_range = col < block_size
        confidence = tl.load(
            confidence_row_ptr + col, mask=in_range, other=-3.4028234663852886e38
        )
        confident_i32 = (tl.sigmoid(confidence) >= confidence_threshold).to(tl.int32)
        keep = in_range & (confidence_live != 0) & (confident_i32 != 0)
        confidence_prefix_len += keep.to(tl.int32)
        confidence_live = tl.where(
            in_range, confidence_live & confident_i32, confidence_live
        )

    accept_len = tl.minimum(match_prefix_len, confidence_prefix_len)
    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)
    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dspark_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    confidence: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
    confidence_threshold: float,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DSPARK Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DSPARK Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(confidence):
        raise ValueError("DSPARK Triton accept_bonus requires contiguous confidence.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError("DSPARK Triton accept_bonus requires contiguous out_tokens_out.")
    if not commit_lens_out.is_contiguous():
        raise ValueError("DSPARK Triton accept_bonus requires contiguous commit_lens_out.")
    if not bonus_ids_out.is_contiguous():
        raise ValueError("DSPARK Triton accept_bonus requires contiguous bonus_ids_out.")
    if prefix_lens.ndim != 1:
        raise ValueError("DSPARK Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DSPARK Triton accept_bonus requires contiguous new_seq_lens_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dspark_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        confidence,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        confidence.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        float(confidence_threshold),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
