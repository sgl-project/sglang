from typing import Optional, Tuple

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
    prefix_lens_ptr,
    new_seq_lens_out_ptr,
    candidates_row_stride,
    target_row_stride,
    accept_stride,
    commit_stride,
    bonus_stride,
    out_tokens_row_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
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
    new_seq_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride) + commit_len

    tl.store(accept_lens_out_ptr + row * accept_stride, accept_len)
    tl.store(commit_lens_out_ptr + row * commit_stride, commit_len)
    tl.store(bonus_ids_out_ptr + row * bonus_stride, bonus_id)
    tl.store(new_seq_lens_out_ptr + row * new_seq_lens_stride, new_seq_len)

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
    prefix_lens: torch.Tensor,
    new_seq_lens_out: torch.Tensor,
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
    if prefix_lens.ndim != 1:
        raise ValueError("DFLASH Triton accept_bonus requires 1D prefix_lens.")
    if not new_seq_lens_out.is_contiguous():
        raise ValueError(
            "DFLASH Triton accept_bonus requires contiguous new_seq_lens_out."
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
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        target_top1.stride(0),
        accept_lens_out.stride(0),
        commit_lens_out.stride(0),
        bonus_ids_out.stride(0),
        out_tokens_out.stride(0),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit(do_not_specialize=["forced_num_accept_tokens", "fixed_token"])
def _dflash_simulated_accept_kernel(
    candidates_ptr,
    target_bonus_tokens_ptr,
    num_correct_drafts_ptr,
    num_accept_tokens_ptr,
    bonus_tokens_ptr,
    out_tokens_ptr,
    prefix_lens_ptr,
    new_seq_lens_ptr,
    candidates_row_stride,
    candidates_col_stride,
    target_bonus_tokens_stride,
    num_correct_drafts_stride,
    num_accept_tokens_stride,
    bonus_tokens_stride,
    out_tokens_row_stride,
    out_tokens_col_stride,
    prefix_lens_stride,
    new_seq_lens_stride,
    block_size,
    forced_num_accept_tokens,
    fixed_token,
    REAL_DRAFT_TOKENS: tl.constexpr,
    WRITE_SEQ_LENS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    num_correct_drafts = forced_num_accept_tokens - 1

    tl.store(
        num_correct_drafts_ptr + row * num_correct_drafts_stride, num_correct_drafts
    )
    tl.store(
        num_accept_tokens_ptr + row * num_accept_tokens_stride, forced_num_accept_tokens
    )
    if WRITE_SEQ_LENS:
        prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
        tl.store(
            new_seq_lens_ptr + row * new_seq_lens_stride,
            prefix_len + forced_num_accept_tokens,
        )

    out_dtype = out_tokens_ptr.dtype.element_ty
    if REAL_DRAFT_TOKENS:
        # Two casts, as in the eager path: target -> bonus dtype -> out dtype.
        bonus_token = tl.load(
            target_bonus_tokens_ptr + row * target_bonus_tokens_stride
        ).to(bonus_tokens_ptr.dtype.element_ty)
        tl.store(bonus_tokens_ptr + row * bonus_tokens_stride, bonus_token)
        drafts = tl.load(
            candidates_ptr
            + row * candidates_row_stride
            + (cols + 1) * candidates_col_stride,
            mask=cols < num_correct_drafts,
            other=0,
        )
        out = tl.where(
            cols == num_correct_drafts, bonus_token.to(out_dtype), drafts.to(out_dtype)
        )
    else:
        tl.store(bonus_tokens_ptr + row * bonus_tokens_stride, fixed_token)
        out = tl.full((BLOCK,), 0, out_dtype) + fixed_token
    tl.store(
        out_tokens_ptr + row * out_tokens_row_stride + cols * out_tokens_col_stride,
        out.to(out_dtype),
        mask=cols < block_size,
    )


def dflash_simulated_accept(
    *,
    candidates: torch.Tensor,
    target_bonus_tokens: Optional[torch.Tensor],
    forced_num_accept_tokens: int,
    num_correct_drafts_out: torch.Tensor,
    num_accept_tokens_out: torch.Tensor,
    bonus_tokens_out: torch.Tensor,
    out_tokens: torch.Tensor,
    fixed_token: int,
    prefix_lens: Optional[torch.Tensor] = None,
    new_seq_lens_out: Optional[torch.Tensor] = None,
) -> None:
    """Writes a forced DFlash acceptance of ``n = forced_num_accept_tokens``
    (bonus included) for every request in one launch.

    ``target_bonus_tokens`` is the target token at the forced row, ``[bs]``;
    None selects fixed mode, which writes ``fixed_token`` to the bonus and to
    every ``out_tokens`` column. Otherwise ``out_tokens`` holds the drafts
    ``candidates[:, 1:n]``, the bonus at column ``n - 1`` and zeros after it.
    ``new_seq_lens_out = prefix_lens + n`` is written when it is given.
    """
    bs, block_size = candidates.shape
    assert out_tokens.shape == (bs, block_size)
    assert 1 <= forced_num_accept_tokens <= block_size
    for t in (num_correct_drafts_out, num_accept_tokens_out, bonus_tokens_out):
        assert t.shape == (bs,)
    real_draft_tokens = target_bonus_tokens is not None
    if real_draft_tokens:
        assert target_bonus_tokens.shape == (bs,)
    write_seq_lens = new_seq_lens_out is not None
    if write_seq_lens:
        assert prefix_lens is not None
        assert prefix_lens.shape == (bs,) and new_seq_lens_out.shape == (bs,)
    for t in (
        candidates,
        target_bonus_tokens,
        num_correct_drafts_out,
        num_accept_tokens_out,
        bonus_tokens_out,
        out_tokens,
        prefix_lens,
        new_seq_lens_out,
    ):
        if t is not None:
            assert t.device == candidates.device
            assert t.dtype in (torch.int32, torch.int64)
    if bs == 0:
        return

    # A disabled branch's pointers alias an output; the kernel never reads them.
    if not real_draft_tokens:
        target_bonus_tokens = bonus_tokens_out
    if not write_seq_lens:
        prefix_lens = new_seq_lens_out = num_accept_tokens_out
    block = triton.next_power_of_2(block_size)
    _dflash_simulated_accept_kernel[(bs,)](
        candidates,
        target_bonus_tokens,
        num_correct_drafts_out,
        num_accept_tokens_out,
        bonus_tokens_out,
        out_tokens,
        prefix_lens,
        new_seq_lens_out,
        candidates.stride(0),
        candidates.stride(1),
        target_bonus_tokens.stride(0),
        num_correct_drafts_out.stride(0),
        num_accept_tokens_out.stride(0),
        bonus_tokens_out.stride(0),
        out_tokens.stride(0),
        out_tokens.stride(1),
        prefix_lens.stride(0),
        new_seq_lens_out.stride(0),
        block_size,
        forced_num_accept_tokens,
        fixed_token,
        REAL_DRAFT_TOKENS=real_draft_tokens,
        WRITE_SEQ_LENS=write_seq_lens,
        BLOCK=block,
        num_warps=_pick_num_warps(block),
    )


@triton.jit
def _prepare_dflash_draft_block_contig_kernel(
    bonus_tokens_ptr,
    prefix_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    block_ids_out_ptr,
    positions_out_ptr,
    cache_loc_out_ptr,
    bonus_tokens_stride,
    prefix_lens_stride,
    req_pool_indices_stride,
    req_to_token_row_stride,
    block_ids_row_stride,
    positions_row_stride,
    cache_loc_row_stride,
    req_to_token_width,
    block_size,
    mask_token_id,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    row_mask = cols < block_size

    prefix_len = tl.load(prefix_lens_ptr + row * prefix_lens_stride)
    req_idx = tl.load(req_pool_indices_ptr + row * req_pool_indices_stride)
    bonus_token = tl.load(bonus_tokens_ptr + row * bonus_tokens_stride)

    logical_pos = prefix_len.to(tl.int64) + cols
    valid = row_mask & (logical_pos < req_to_token_width)
    req_row_ptr = req_to_token_ptr + req_idx * req_to_token_row_stride
    slot_ids = tl.load(req_row_ptr + logical_pos, mask=valid, other=0)

    block_ids = tl.full((BLOCK_SIZE,), mask_token_id, tl.int64)
    block_ids = tl.where(cols == 0, bonus_token.to(tl.int64), block_ids)
    tl.store(
        block_ids_out_ptr + row * block_ids_row_stride + cols, block_ids, mask=row_mask
    )
    tl.store(
        positions_out_ptr + row * positions_row_stride + cols,
        logical_pos,
        mask=row_mask,
    )
    tl.store(
        cache_loc_out_ptr + row * cache_loc_row_stride + cols,
        slot_ids.to(tl.int64),
        mask=row_mask,
    )


def _prepare_dflash_draft_block_unchecked(
    bonus_tokens: torch.Tensor,
    prefix_lens: torch.Tensor,
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    block_ids_out: torch.Tensor,
    positions_out: torch.Tensor,
    cache_loc_out: torch.Tensor,
    mask_token_id: int,
) -> None:
    batch_size = int(bonus_tokens.numel())
    if batch_size == 0:
        return

    if req_to_token.ndim != 2 or req_to_token.stride(1) != 1:
        raise ValueError("DFLASH Triton prepare_block requires row-major req_to_token.")
    if not _is_row_major_contiguous_2d(block_ids_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous block_ids_out."
        )
    if not _is_row_major_contiguous_2d(positions_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous positions_out."
        )
    if not _is_row_major_contiguous_2d(cache_loc_out):
        raise ValueError(
            "DFLASH Triton prepare_block requires contiguous cache_loc_out."
        )

    block_size = int(block_ids_out.shape[1])
    block = triton.next_power_of_2(block_size)
    num_warps = _pick_num_warps(block)
    _prepare_dflash_draft_block_contig_kernel[(batch_size,)](
        bonus_tokens,
        prefix_lens,
        req_pool_indices,
        req_to_token,
        block_ids_out,
        positions_out,
        cache_loc_out,
        bonus_tokens.stride(0),
        prefix_lens.stride(0),
        req_pool_indices.stride(0),
        req_to_token.stride(0),
        block_ids_out.stride(0),
        positions_out.stride(0),
        cache_loc_out.stride(0),
        int(req_to_token.shape[1]),
        block_size,
        int(mask_token_id),
        BLOCK_SIZE=block,
        num_warps=num_warps,
    )


@triton.jit
def _selector_walk_kernel(
    scores_ptr,
    candidate_ptr,
    uniforms_ptr,
    temperatures_ptr,
    greedy_ptr,
    tokens_ptr,
    q_ptr,
    slots: tl.constexpr,
    top_k: tl.constexpr,
):
    """One program per request: a slot's K scores stay in registers and the walk is a
    loop, so the slot-to-slot dependency costs nothing instead of one kernel each."""
    row = tl.program_id(0)
    offsets = tl.arange(0, top_k)
    temperature = tl.load(temperatures_ptr + row)
    greedy = tl.load(greedy_ptr + row) != 0
    previous = 0
    for slot in range(slots):
        base = (row * slots + slot) * top_k
        scores = tl.load(scores_ptr + (base + previous) * top_k + offsets).to(
            tl.float32
        )
        if greedy:
            # Same pick as torch.argmax: the first NaN if any, else the first max.
            is_nan = scores != scores
            best = tl.max(tl.where(is_nan, -float("inf"), scores), axis=0)
            has_nan = tl.max(is_nan.to(tl.int32), axis=0) > 0
            hit = tl.where(has_nan, is_nan, scores == best)
            index = tl.min(tl.where(hit, offsets, top_k), axis=0)
            probabilities = tl.where(offsets == index, 1.0, 0.0)
        else:
            scaled = scores / temperature
            exponentials = tl.exp(scaled - tl.max(scaled, axis=0))
            probabilities = exponentials / tl.sum(exponentials, axis=0)
            uniform = tl.load(uniforms_ptr + row * slots + slot)
            index = tl.sum(
                tl.where(uniform >= tl.cumsum(probabilities, axis=0), 1, 0), axis=0
            )
            index = tl.minimum(index, top_k - 1)
        tl.store(q_ptr + base + offsets, probabilities)
        tl.store(tokens_ptr + row * slots + slot, tl.load(candidate_ptr + base + index))
        previous = index


def selector_walk_triton(
    *,
    candidate_ids,
    scores,
    uniforms,
    temperatures,
    greedy_mask,
):
    batch, slots, top_k = candidate_ids.shape
    tokens = torch.empty((batch, slots), dtype=torch.int64, device=scores.device)
    q_rows = torch.empty(
        (batch, slots, top_k), dtype=torch.float32, device=scores.device
    )
    _selector_walk_kernel[(batch,)](
        scores.contiguous(),
        candidate_ids.contiguous(),
        uniforms.contiguous(),
        temperatures.contiguous(),
        greedy_mask.contiguous(),
        tokens,
        q_rows,
        slots=slots,
        top_k=top_k,
        num_warps=1,
    )
    return tokens, q_rows


@triton.jit
def _fused_correct_drafts_and_bonus_kernel(
    candidates_ptr,
    target_predict_ptr,
    num_correct_drafts_ptr,
    bonus_tokens_ptr,
    block_size,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    in_row = offs < block_size - 1
    drafts = tl.load(candidates_ptr + b * block_size + 1 + offs, mask=in_row, other=-1)
    targets = tl.load(target_predict_ptr + b * block_size + offs, mask=in_row, other=-2)
    eq = (drafts == targets) & in_row
    # Leading-match count = index of the first mismatch lane; lanes past the
    # row and all-match rows both resolve to block_size - 1 via the min.
    num_correct = tl.min(tl.where(eq, BLOCK, offs), 0)
    bonus_token = tl.load(target_predict_ptr + b * block_size + num_correct)
    tl.store(num_correct_drafts_ptr + b, num_correct.to(tl.int32))
    tl.store(bonus_tokens_ptr + b, bonus_token.to(tl.int64))


def compute_dflash_correct_drafts_and_bonus(
    *,
    candidates: torch.Tensor,
    target_predict: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute DFlash accept lengths and bonus tokens (greedy verify rule).

    Args:
        candidates: Token ids proposed by the DFlash draft, including the current token.
            Shape: [bs, block_size]. candidates[:, 0] is the current token.
        target_predict: Token ids predicted by the target model for each position in the block.
            Shape: [bs, block_size]. target_predict[:, t] corresponds to argmax at position t.

    Returns:
        correct_len: int32 tensor [bs], number of accepted *draft* tokens (excluding current token and bonus token).
        bonus: int64 tensor [bs], the target-predicted token at index correct_len (the "bonus" token to append).

    Notes:
        Matches the reference implementation rule:
          accept while candidates[:, 1:] == target_predict[:, :-1] consecutively.
    """
    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={tuple(candidates.shape)}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates. "
            f"candidates.shape={tuple(candidates.shape)}, target_predict.shape={tuple(target_predict.shape)}"
        )

    bs, block_size = candidates.shape
    if bs <= 0:
        raise ValueError(f"batch size must be positive, got {bs}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if candidates.is_cuda:
        num_correct_drafts = torch.empty(
            bs, dtype=torch.int32, device=candidates.device
        )
        bonus_tokens = torch.empty(bs, dtype=torch.int64, device=candidates.device)
        _fused_correct_drafts_and_bonus_kernel[(bs,)](
            candidates.contiguous(),
            target_predict.contiguous(),
            num_correct_drafts,
            bonus_tokens,
            block_size,
            BLOCK=triton.next_power_of_2(max(block_size - 1, 1)),
        )
        return num_correct_drafts, bonus_tokens

    matches = candidates[:, 1:] == target_predict[:, :-1]
    correct_len = matches.to(torch.int32).cumprod(dim=1).sum(dim=1)
    bonus = target_predict[torch.arange(bs, device=target_predict.device), correct_len]
    return correct_len.to(torch.int32), bonus.to(torch.int64)
