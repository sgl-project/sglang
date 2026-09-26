import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, next_power_of_2

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import fill_accept_out_cache_loc_cpu, fill_bonus_tokens_cpu


@triton.jit
def fill_bonus_tokens(
    accept_tokens,
    accept_lens,
    bonus_tokens_ptr,
    accept_stride: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    # `accept_lens` includes the bonus token; the last accepted slot is at -1.
    accept_len = tl.load(accept_lens + pid)

    # accept_stride = per-req width of accept_tokens (= accept_index.shape[1]).
    bonus_token_idx = accept_stride * pid + accept_len - 1
    bonus_token = tl.load(accept_tokens + bonus_token_idx)
    tl.store(bonus_tokens_ptr + pid, bonus_token)


def fill_bonus_tokens_func(
    accept_tokens: torch.Tensor,
    accept_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,  # mutable
    accept_stride: int,
    batch_size: int,
):
    if _is_cpu:
        fill_bonus_tokens_cpu(
            accept_tokens,
            accept_lens,
            bonus_tokens,
            accept_stride,
        )
        return
    fill_bonus_tokens[(batch_size,)](
        accept_tokens,
        accept_lens,
        bonus_tokens,
        accept_stride,
    )


@triton.jit
def fill_accept_out_cache_loc(
    accept_index,
    out_cache_loc,
    accept_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accept_out_cache_loc + dst, value)


def fill_accept_out_cache_loc_func(
    accept_index: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_out_cache_loc: torch.Tensor,  # mutable
    size: int,
):
    if _is_cpu:
        fill_accept_out_cache_loc_cpu(
            accept_index,
            out_cache_loc,
            accept_out_cache_loc,
        )
        return
    fill_accept_out_cache_loc[(size,)](
        accept_index,
        out_cache_loc,
        accept_out_cache_loc,
        next_power_of_2(size),
    )


@triton.jit
def _verify_commit_outputs_kernel(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    new_seq_lens,
    bonus,
    BS: tl.constexpr,
    PREDICT_SIZE: tl.constexpr,
    PREDICT_STRIDE: tl.constexpr,
    INDEX_STRIDE_0: tl.constexpr,
    INDEX_STRIDE_1: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    last_steps,
    track_steps,
    correct,
    select,
    ids,
    DRAFT_WIDTH: tl.constexpr,
    TRACK_INTERVAL: tl.constexpr,
):
    row = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = row < BS
    accepted = tl.load(accept_lens + row * ACCEPT_STRIDE, valid, other=1).to(tl.int64)
    length = tl.load(seq_lens + row * SEQ_STRIDE, valid, other=0).to(tl.int64)
    idx = tl.load(
        accept_index + row * INDEX_STRIDE_0 + (accepted - 1) * INDEX_STRIDE_1,
        valid,
        other=0,
    ).to(tl.int64)
    if DRAFT_WIDTH > 0:
        tl.store(last_steps + row, idx - row * DRAFT_WIDTH, valid)
        tl.store(correct + row, accepted - 1, valid)
        tl.store(select + row, row * DRAFT_WIDTH + accepted - 1, valid)
        output_token = tl.load(
            predict + row * PREDICT_STRIDE, row < PREDICT_SIZE, other=0
        )
        tl.store(ids + row, output_token.to(tl.int64), row < PREDICT_SIZE)
        if TRACK_INTERVAL > 0:
            post = length + accepted
            crossing = length // TRACK_INTERVAL != post // TRACK_INTERVAL
            tracking_position = tl.maximum(
                post // TRACK_INTERVAL * TRACK_INTERVAL - length - 1, 0
            )
            track = (
                tl.load(
                    accept_index
                    + row * INDEX_STRIDE_0
                    + tracking_position * INDEX_STRIDE_1,
                    valid & crossing,
                    other=0,
                ).to(tl.int64)
                - row * DRAFT_WIDTH
            )
            tl.store(track_steps + row, tl.where(crossing, track, -1), valid)
    idx = tl.where(idx < 0, idx + PREDICT_SIZE, idx)
    token = tl.load(predict + idx * PREDICT_STRIDE, valid, other=0)
    tl.store(new_seq_lens + row, length + accepted, valid)
    tl.store(bonus + row, token, valid)


def prepare_verify_commit_outputs(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    num_draft_tokens=0,
    mamba_track_interval=0,
):
    bs = accept_lens.numel()
    new_seq_lens = torch.empty(
        (bs,),
        device=seq_lens.device,
        dtype=torch.promote_types(seq_lens.dtype, accept_lens.dtype),
    )
    bonus = torch.empty((bs,), device=predict.device, dtype=torch.int32)
    last_steps = track_steps = correct = select = ids = None
    if num_draft_tokens:
        last_steps = torch.empty((bs,), device=seq_lens.device, dtype=torch.int64)
        if mamba_track_interval:
            track_steps = torch.empty_like(last_steps)
        correct = torch.empty_like(accept_lens, memory_format=torch.contiguous_format)
        select = torch.empty_like(last_steps)
        ids = torch.empty((predict.numel(),), device=predict.device, dtype=torch.int64)
    if bs:
        size = max(bs, predict.numel()) if num_draft_tokens else bs
        _verify_commit_outputs_kernel[(triton.cdiv(size, 128),)](
            predict,
            accept_index,
            accept_lens,
            seq_lens,
            new_seq_lens,
            bonus,
            bs,
            predict.numel(),
            predict.stride(0),
            accept_index.stride(0),
            accept_index.stride(1),
            accept_lens.stride(0),
            seq_lens.stride(0),
            BLOCK=128,
            last_steps=last_steps,
            track_steps=track_steps,
            correct=correct,
            select=select,
            ids=ids,
            DRAFT_WIDTH=num_draft_tokens,
            TRACK_INTERVAL=mamba_track_interval,
        )
    if num_draft_tokens:
        return new_seq_lens, bonus, (last_steps, track_steps), (correct, select, ids)
    return new_seq_lens, bonus


@triton.jit
def _draft_extend_inputs_kernel(
    accept_lens,
    predict,
    correct,
    select,
    ids,
    BS: tl.constexpr,
    TOKENS: tl.constexpr,
    WIDTH: tl.constexpr,
    ACCEPT_STRIDE: tl.constexpr,
    PREDICT_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    accepted = tl.load(accept_lens + offset * ACCEPT_STRIDE, offset < BS, other=1).to(
        tl.int64
    )
    tl.store(correct + offset, accepted - 1, offset < BS)
    tl.store(select + offset, offset * WIDTH + accepted - 1, offset < BS)
    tokens = tl.load(predict + offset * PREDICT_STRIDE, offset < TOKENS, other=0)
    tl.store(ids + offset, tokens.to(tl.int64), offset < TOKENS)


def prepare_draft_extend_inputs(accept_lens, predict, num_draft_tokens):
    bs, tokens = accept_lens.numel(), predict.numel()
    correct = torch.empty((bs,), device=accept_lens.device, dtype=accept_lens.dtype)
    select = torch.empty((bs,), device=accept_lens.device, dtype=torch.int64)
    ids = torch.empty((tokens,), device=predict.device, dtype=torch.int64)
    if max(bs, tokens):
        _draft_extend_inputs_kernel[(triton.cdiv(max(bs, tokens), 128),)](
            accept_lens,
            predict,
            correct,
            select,
            ids,
            bs,
            tokens,
            num_draft_tokens,
            accept_lens.stride(0),
            predict.stride(0),
            BLOCK=128,
        )
    return correct, select, ids


@triton.jit
def _prepare_draft_extend_lengths_kernel(
    seq_lens,
    prefix_lens,
    extend_lens,
    post_lens,
    BS: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    FRONT: tl.constexpr,
    BLOCK: tl.constexpr,
    positions=None,
    start_locs=None,
    mrope=None,
    MAKE_POSITIONS: tl.constexpr = False,
    MAKE_MROPE: tl.constexpr = False,
):
    row = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    base = tl.load(seq_lens + row * STRIDE, row < BS, other=0).to(tl.int64)
    tl.store(prefix_lens + row, tl.maximum(base - FRONT, 0), row < BS)
    tl.store(extend_lens + row, WIDTH + FRONT, row < BS)
    tl.store(post_lens + row, base + WIDTH, row < BS)
    if MAKE_POSITIONS:
        tl.store(start_locs + row, row * WIDTH, row < BS)
        req = row // WIDTH
        token_base = tl.load(seq_lens + req * STRIDE, row < BS * WIDTH, other=0).to(
            tl.int64
        )
        prefix = tl.maximum(token_base - FRONT, 0).to(tl.int32)
        position = prefix + (row % WIDTH).to(tl.int32)
        tl.store(positions + row, position.to(tl.int64), row < BS * WIDTH)
        if MAKE_MROPE:
            for axis in tl.static_range(3):
                tl.store(
                    mrope + axis * BS * WIDTH + row,
                    position.to(tl.int64),
                    row < BS * WIDTH,
                )


def prepare_draft_extend_lengths(seq_lens, num_draft_tokens, front_offset):
    bs = seq_lens.numel()
    prefix = torch.empty((bs,), dtype=torch.int32, device=seq_lens.device)
    extend = torch.empty_like(prefix)
    post = torch.empty((bs,), dtype=seq_lens.dtype, device=seq_lens.device)
    if bs:
        _prepare_draft_extend_lengths_kernel[(triton.cdiv(bs, 128),)](
            seq_lens,
            prefix,
            extend,
            post,
            bs,
            seq_lens.stride(0),
            num_draft_tokens,
            front_offset,
            BLOCK=128,
        )
    return prefix, extend, post


def prepare_draft_extend_layout(seq_lens, num_draft_tokens, with_mrope):
    bs = seq_lens.numel()
    prefix = torch.empty((bs,), dtype=torch.int32, device=seq_lens.device)
    extend = torch.empty_like(prefix)
    post = torch.empty((bs,), dtype=seq_lens.dtype, device=seq_lens.device)
    positions = torch.empty(
        (bs * num_draft_tokens,), dtype=torch.int64, device=seq_lens.device
    )
    starts = torch.empty_like(prefix)
    mrope = (
        torch.empty(
            (3, bs * num_draft_tokens), dtype=torch.int64, device=seq_lens.device
        )
        if with_mrope
        else None
    )
    if bs:
        _prepare_draft_extend_lengths_kernel[
            (triton.cdiv(bs * num_draft_tokens, 128),)
        ](
            seq_lens,
            prefix,
            extend,
            post,
            bs,
            seq_lens.stride(0),
            num_draft_tokens,
            0,
            BLOCK=128,
            positions=positions,
            start_locs=starts,
            mrope=mrope,
            MAKE_POSITIONS=True,
            MAKE_MROPE=with_mrope,
        )
    return prefix, extend, post, (positions, starts, mrope)


@triton.jit
def _build_chain_tree_kernel(
    bonus,
    draft,
    seq,
    mask,
    positions,
    retrieve,
    tokens,
    BS: tl.constexpr,
    WIDTH: tl.constexpr,
    BONUS_STRIDE: tl.constexpr,
    DRAFT_ROW: tl.constexpr,
    DRAFT_COL: tl.constexpr,
    SEQ_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    req_indices,
    req_to_token,
    out_cache_loc,
    mrope,
    REQ_STRIDE: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    WRITE_MROPE: tl.constexpr,
):
    offset = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    row = offset // WIDTH
    col = offset % WIDTH
    valid = offset < BS * WIDTH
    first = tl.load(bonus + row * BONUS_STRIDE, valid & (col == 0), other=0)
    rest = tl.load(
        draft + row * DRAFT_ROW + (col - 1) * DRAFT_COL, valid & (col > 0), other=0
    )
    tl.store(tokens + offset, tl.where(col == 0, first, rest), valid)
    length = tl.load(seq + row * SEQ_STRIDE, valid, other=0).to(tl.int64)
    tl.store(positions + offset, length + col, valid)
    if WRITE_CACHE:
        req = tl.load(req_indices + row * REQ_STRIDE, valid, other=0).to(tl.int64)
        loc = tl.load(
            req_to_token + req * TABLE_ROW + (length + col) * TABLE_COL, valid, other=0
        )
        tl.store(out_cache_loc + offset, loc, valid)
    if WRITE_MROPE:
        for axis in tl.static_range(3):
            tl.store(mrope + axis * BS * WIDTH + offset, length + col, valid)
    tl.store(retrieve + offset, offset, valid)
    tl.store(
        retrieve + BS * WIDTH + offset, tl.where(col + 1 < WIDTH, col + 1, -1), valid
    )
    tl.store(retrieve + 2 * BS * WIDTH + offset, -1, valid)
    mask_col = offset % WIDTH
    mask_row = (offset // WIDTH) % WIDTH
    tl.store(mask + offset, mask_col <= mask_row, offset < BS * WIDTH * WIDTH)


def build_chain_tree(
    bonus_tokens,
    draft_tokens,
    seq_lens,
    tree_mask_buf=None,
    req_pool_indices=None,
    req_to_token=None,
    with_mrope=False,
):
    bs, depth = draft_tokens.shape
    width = depth + 1
    assert bonus_tokens.numel() == seq_lens.numel() == bs
    mask = tree_mask_buf
    if mask is None:
        mask = torch.empty(
            (bs * width * width,), dtype=torch.bool, device=seq_lens.device
        )
    assert mask.is_contiguous() and mask.numel() >= bs * width * width
    positions = torch.empty((bs * width,), dtype=torch.int64, device=seq_lens.device)
    retrieve = torch.empty((3, bs, width), dtype=torch.int64, device=seq_lens.device)
    tokens = torch.empty(
        (bs * width,),
        dtype=torch.promote_types(bonus_tokens.dtype, draft_tokens.dtype),
        device=seq_lens.device,
    )
    out_cache_loc = torch.empty_like(positions) if req_to_token is not None else None
    mrope = (
        torch.empty((3, bs * width), dtype=torch.int64, device=seq_lens.device)
        if with_mrope
        else None
    )
    if req_to_token is not None:
        assert req_pool_indices is not None and req_pool_indices.numel() == bs
    if bs:
        _build_chain_tree_kernel[(triton.cdiv(bs * width * width, 128),)](
            bonus_tokens,
            draft_tokens,
            seq_lens,
            mask,
            positions,
            retrieve,
            tokens,
            bs,
            width,
            bonus_tokens.stride(0),
            draft_tokens.stride(0),
            draft_tokens.stride(1),
            seq_lens.stride(0),
            BLOCK=128,
            req_indices=req_pool_indices,
            req_to_token=req_to_token,
            out_cache_loc=out_cache_loc,
            mrope=mrope,
            REQ_STRIDE=(
                req_pool_indices.stride(0) if req_pool_indices is not None else 0
            ),
            TABLE_ROW=req_to_token.stride(0) if req_to_token is not None else 0,
            TABLE_COL=req_to_token.stride(1) if req_to_token is not None else 0,
            WRITE_CACHE=req_to_token is not None,
            WRITE_MROPE=with_mrope,
        )
    return (
        mask,
        positions,
        retrieve[0],
        retrieve[1],
        retrieve[2],
        tokens,
        out_cache_loc,
        mrope,
    )
