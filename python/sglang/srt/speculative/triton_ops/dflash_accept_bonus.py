import torch
import triton
import triton.language as tl


@triton.jit
def _dflash_accept_bonus_contig_kernel(
    candidates_ptr,
    target_top1_ptr,
    accept_lens_out_ptr,
    commit_lens_out_ptr,
    bonus_ids_out_ptr,
    out_tokens_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    block_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size
    draft_mask = cols < (block_size - 1)

    candidate_row_ptr = candidates_ptr + row * candidates_row_stride
    target_row_ptr = target_top1_ptr + row * target_row_stride
    candidate_tail = tl.load(candidate_row_ptr + cols + 1, mask=draft_mask, other=0)

    accept_len = tl.full((), 0, tl.int32)
    prefix_live = tl.full((), 1, tl.int32)
    for col in range(BLOCK_SIZE - 1):
        in_range = col < (block_size - 1)
        candidate_id = tl.load(candidate_row_ptr + (col + 1), mask=in_range, other=0)
        target_id = tl.load(target_row_ptr + col, mask=in_range, other=0)
        match_i32 = (candidate_id == target_id).to(tl.int32)
        keep = in_range & (prefix_live != 0) & (match_i32 != 0)
        accept_len += keep.to(tl.int32)
        prefix_live = tl.where(in_range, prefix_live & match_i32, prefix_live)

    commit_len = accept_len + 1
    bonus_id = tl.load(target_row_ptr + accept_len.to(tl.int64))

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)

    out_val = tl.where(draft_mask, candidate_tail, 0)
    out_val = tl.where(cols == accept_len, bonus_id, out_val)
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols, out_val, mask=row_mask
    )


def _pick_num_warps(block_size: int) -> int:
    if block_size <= 16:
        return 1
    if block_size <= 32:
        return 2
    if block_size <= 64:
        return 4
    return 8


def _is_row_major_contiguous_2d(x: torch.Tensor) -> bool:
    return x.ndim == 2 and x.is_contiguous()


def _compute_dflash_accept_bonus_triton_unchecked(
    candidates: torch.Tensor,
    target_top1: torch.Tensor,
    accept_lens_out: torch.Tensor,
    commit_lens_out: torch.Tensor,
    bonus_ids_out: torch.Tensor,
    out_tokens_out: torch.Tensor,
) -> None:
    batch_size, block_size = candidates.shape
    if batch_size == 0:
        return

    if not _is_row_major_contiguous_2d(candidates):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous candidates.")
    if not _is_row_major_contiguous_2d(target_top1):
        raise ValueError("DFLASH Triton accept_bonus requires contiguous target_top1.")
    if not _is_row_major_contiguous_2d(out_tokens_out):
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous out_tokens_out."
        )
    if not accept_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous accept_lens_out."
        )
    if not commit_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous commit_lens_out."
        )
    if not bonus_ids_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous bonus_ids_out."
        )

    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _dflash_accept_bonus_contig_kernel[(batch_size,)](
        candidates,
        target_top1,
        accept_lens_out,
        commit_lens_out,
        bonus_ids_out,
        out_tokens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )
