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


@triton.jit
def _dspark_vocab_argmax_partial_kernel(
    base_logits_ptr,
    bias_ptr,
    local_token_ids_ptr,
    partial_scores_ptr,
    partial_tokens_ptr,
    base_row_stride,
    bias_row_stride,
    partial_row_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size

    token_ids = tl.load(local_token_ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (token_ids >= 0)
    base = tl.load(
        base_logits_ptr + row * base_row_stride + offs,
        mask=mask,
        other=-float("inf"),
    )
    bias = tl.load(
        bias_ptr + row * bias_row_stride + offs,
        mask=mask,
        other=0.0,
    )
    scores = base.to(tl.float32) + bias.to(tl.float32)
    scores = tl.where(valid, scores, -float("inf"))

    max_score = tl.max(scores, axis=0)
    large_token = tl.full((), 2147483647, tl.int64)
    best_token = tl.min(
        tl.where((scores == max_score) & valid, token_ids, large_token), axis=0
    )
    best_token = tl.where(best_token == large_token, -1, best_token)

    out = partial_scores_ptr + row * partial_row_stride + chunk
    tl.store(out, max_score)
    tl.store(partial_tokens_ptr + row * partial_row_stride + chunk, best_token)


@triton.jit
def _dspark_vocab_argmax_reduce_kernel(
    partial_scores_ptr,
    partial_tokens_ptr,
    scores_out_ptr,
    tokens_out_ptr,
    partial_row_stride,
    num_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < num_chunks

    scores = tl.load(
        partial_scores_ptr + row * partial_row_stride + offs,
        mask=mask,
        other=-float("inf"),
    )
    tokens = tl.load(
        partial_tokens_ptr + row * partial_row_stride + offs, mask=mask, other=-1
    )
    valid = mask & (tokens >= 0)
    scores = tl.where(valid, scores, -float("inf"))

    max_score = tl.max(scores, axis=0)
    large_token = tl.full((), 2147483647, tl.int64)
    best_token = tl.min(
        tl.where((scores == max_score) & valid, tokens, large_token), axis=0
    )
    best_token = tl.where(best_token == large_token, -1, best_token)

    tl.store(scores_out_ptr + row, max_score)
    tl.store(tokens_out_ptr + row, best_token)


def _compute_dspark_vocab_argmax_triton_unchecked(
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    local_token_ids: torch.Tensor,
    partial_scores: torch.Tensor,
    partial_tokens: torch.Tensor,
    scores_out: torch.Tensor,
    tokens_out: torch.Tensor,
) -> None:
    if base_logits.ndim != 2 or bias.ndim != 2:
        raise ValueError("DSPARK Triton vocab argmax requires 2D logits and bias.")
    if base_logits.shape != bias.shape:
        raise ValueError("DSPARK Triton vocab argmax requires matching logits and bias.")
    if base_logits.stride(-1) != 1 or bias.stride(-1) != 1:
        raise ValueError(
            "DSPARK Triton vocab argmax requires contiguous vocab dimension."
        )
    if local_token_ids.ndim != 1 or local_token_ids.shape[0] != base_logits.shape[1]:
        raise ValueError("DSPARK Triton vocab argmax got invalid token-id mapping.")
    if local_token_ids.dtype != torch.int64:
        raise ValueError("DSPARK Triton vocab argmax requires int64 token ids.")

    batch_size, vocab_size = base_logits.shape
    if batch_size == 0:
        return

    block = 1024
    num_chunks = triton.cdiv(vocab_size, block)
    if partial_scores.shape[0] < batch_size or partial_scores.shape[1] < num_chunks:
        raise ValueError("DSPARK Triton vocab argmax partial score buffer is too small.")
    if partial_tokens.shape[0] < batch_size or partial_tokens.shape[1] < num_chunks:
        raise ValueError("DSPARK Triton vocab argmax partial token buffer is too small.")
    if scores_out.shape[0] < batch_size or tokens_out.shape[0] < batch_size:
        raise ValueError("DSPARK Triton vocab argmax output buffer is too small.")

    _dspark_vocab_argmax_partial_kernel[(batch_size, num_chunks)](
        base_logits,
        bias,
        local_token_ids,
        partial_scores,
        partial_tokens,
        base_logits.stride(0),
        bias.stride(0),
        partial_scores.stride(0),
        vocab_size,
        BLOCK_SIZE=block,
        num_warps=8,
    )

    reduce_block = triton.next_power_of_2(num_chunks)
    _dspark_vocab_argmax_reduce_kernel[(batch_size,)](
        partial_scores,
        partial_tokens,
        scores_out,
        tokens_out,
        partial_scores.stride(0),
        num_chunks,
        BLOCK_SIZE=reduce_block,
        num_warps=_pick_num_warps(reduce_block),
    )
