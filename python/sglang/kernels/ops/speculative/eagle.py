from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.speculative import perfll_spec_index
from sglang.srt.environ import envs
from sglang.srt.utils import is_cpu, next_power_of_2, perfll_host

_is_cpu = is_cpu()
# SGLANG_PERFLL_GLUE_LINEAR_ACCEPT, read once
_LINEAR_ACCEPT = envs.SGLANG_PERFLL_GLUE_LINEAR_ACCEPT.get()

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


@triton.jit(do_not_specialize=["BS", "PREDICT_SIZE"])
def _verify_commit_outputs_kernel(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    new_seq_lens,
    bonus,
    BS,
    PREDICT_SIZE,
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
    prefix=None,
    extend=None,
    post_lens=None,
    positions=None,
    start_locs=None,
    mrope=None,
    MROPE_STRIDE=0,
    EXTEND_LAYOUT: tl.constexpr = False,
    EXTEND_MROPE: tl.constexpr = False,
    req_pool_indices=None,
    runner_write_window=None,
    track_indices=None,
    num_accept_tokens=None,
    runner_req_pool_indices=None,
    runner_out_cache_loc=None,
    runner_track_indices=None,
    published_seq_lens=None,
    RUNNER_INPUTS: tl.constexpr = False,
    RUNNER_TRACK: tl.constexpr = False,
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
    if EXTEND_LAYOUT:
        # prepare_draft_extend_layout of the pre-verify lengths (the draft extend's window)
        tl.store(prefix + row, tl.maximum(length, 0), valid)
        tl.store(extend + row, DRAFT_WIDTH, valid)
        tl.store(post_lens + row, length + DRAFT_WIDTH, valid)
        tl.store(start_locs + row, row * DRAFT_WIDTH, valid)
        tokens = row < BS * DRAFT_WIDTH
        base = tl.load(seq_lens + (row // DRAFT_WIDTH) * SEQ_STRIDE, tokens, other=0)
        position = tl.maximum(base.to(tl.int64), 0).to(tl.int32) + (
            row % DRAFT_WIDTH
        ).to(tl.int32)
        tl.store(positions + row, position.to(tl.int64), tokens)
        if EXTEND_MROPE:
            for axis in tl.static_range(3):
                tl.store(mrope + axis * MROPE_STRIDE + row, position.to(tl.int64), tokens)
    if RUNNER_INPUTS:
        # the draft extend runner's other inputs and the FutureMap entries (SGLANG_PERFLL_RS_D1)
        req = tl.load(req_pool_indices + row, valid, other=0).to(tl.int64)
        tl.store(num_accept_tokens + row, accepted, valid)
        tl.store(runner_req_pool_indices + row, req, valid)
        tl.store(published_seq_lens + req, length + accepted, valid)
        if RUNNER_TRACK:
            tl.store(
                runner_track_indices + row,
                tl.load(track_indices + row, valid, other=0),
                valid,
            )
        # Copy the immutable write window from this iteration's KVLocPlan. Re-indexing
        # the mutable global req_to_token table here can select a different allocation.
        tokens = row < BS * DRAFT_WIDTH
        loc = tl.load(runner_write_window + row, tokens, other=0).to(tl.int64)
        tl.store(runner_out_cache_loc + row, loc, tokens)
    idx = tl.where(idx < 0, idx + PREDICT_SIZE, idx)
    token = tl.load(predict + idx * PREDICT_STRIDE, valid, other=0)
    tl.store(new_seq_lens + row, length + accepted, valid)
    tl.store(bonus + row, token.to(bonus.dtype.element_ty), valid)


def prepare_verify_commit_outputs(
    predict,
    accept_index,
    accept_lens,
    seq_lens,
    num_draft_tokens=0,
    mamba_track_interval=0,
    extend_layout=False,
    extend_mrope=False,
    bonus_dtype=torch.int32,
    runner_inputs=None,
    req_pool_indices=None,
    track_indices=None,
    runner_write_window=None,
):
    """``extend_layout`` (SGLANG_PST_SPEC_EXTEND_LAYOUT, with ``num_draft_tokens``): the same
    launch also writes ``prepare_draft_extend_layout(seq_lens, num_draft_tokens, extend_mrope)``,
    appended to the draft-extend inputs. ``bonus_dtype``: int64 for SGLANG_PERFLL_STEP5A_BONUS_I64,
    the dtype the relay and the next draft read. ``runner_inputs`` (SGLANG_PERFLL_RS_D1, with
    ``extend_layout``): a perfll_step2.DraftExtendInputs, whose views of the draft extend runner's
    static inputs the launch writes in place of fresh outputs, with the exact immutable
    ``runner_write_window`` from this iteration's KVLocPlan, and the new lengths published
    to its FutureMap buffer at ``req_pool_indices``."""
    bs = accept_lens.numel()
    assert runner_inputs is None or (
        num_draft_tokens
        and extend_layout
        and req_pool_indices is not None
        and runner_write_window is not None
        and runner_write_window.numel() == bs * num_draft_tokens
    )
    new_seq_lens = torch.empty(
        (bs,),
        device=seq_lens.device,
        dtype=torch.promote_types(seq_lens.dtype, accept_lens.dtype),
    )
    bonus = torch.empty((bs,), device=predict.device, dtype=bonus_dtype)
    last_steps = track_steps = correct = select = ids = None
    if num_draft_tokens:
        last_steps = torch.empty((bs,), device=seq_lens.device, dtype=torch.int64)
        if mamba_track_interval:
            track_steps = torch.empty_like(last_steps)
        if runner_inputs is None:
            correct = torch.empty_like(
                accept_lens, memory_format=torch.contiguous_format
            )
            select = torch.empty_like(last_steps)
            ids = torch.empty(
                (predict.numel(),), device=predict.device, dtype=torch.int64
            )
        else:
            correct = runner_inputs.num_correct_drafts
            select = runner_inputs.select_index
            ids = runner_inputs.input_ids
    layout = None
    if num_draft_tokens and extend_layout:
        tokens = bs * num_draft_tokens
        device = seq_lens.device
        runner = runner_inputs
        layout = (
            torch.empty((bs,), dtype=torch.int32, device=device),
            (
                torch.empty((bs,), dtype=torch.int32, device=device)
                if runner is None
                else runner.extend_seq_lens
            ),
            (
                torch.empty((bs,), dtype=seq_lens.dtype, device=device)
                if runner is None
                else runner.seq_lens
            ),
            (
                (
                    torch.empty((tokens,), dtype=torch.int64, device=device)
                    if runner is None
                    else runner.positions
                ),
                torch.empty((bs,), dtype=torch.int32, device=device),
                (
                    None
                    if not extend_mrope
                    else (
                        torch.empty((3, tokens), dtype=torch.int64, device=device)
                        if runner is None
                        else runner.mrope_positions
                    )
                ),
            ),
        )
    if bs:
        size = max(bs, predict.numel()) if num_draft_tokens else bs
        if layout is not None:
            size = max(size, bs * num_draft_tokens)
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
            **(
                dict(
                    prefix=layout[0],
                    extend=layout[1],
                    post_lens=layout[2],
                    positions=layout[3][0],
                    start_locs=layout[3][1],
                    mrope=layout[3][2],
                    MROPE_STRIDE=(
                        bs * num_draft_tokens
                        if layout[3][2] is None
                        else layout[3][2].stride(0)
                    ),
                    EXTEND_LAYOUT=True,
                    EXTEND_MROPE=layout[3][2] is not None,
                )
                if layout is not None
                else {}
            ),
            **(
                dict(
                    req_pool_indices=req_pool_indices,
                    runner_write_window=runner_write_window,
                    track_indices=track_indices,
                    num_accept_tokens=runner_inputs.num_accept_tokens,
                    runner_req_pool_indices=runner_inputs.req_pool_indices,
                    runner_out_cache_loc=runner_inputs.out_cache_loc,
                    runner_track_indices=runner_inputs.mamba_track_indices,
                    published_seq_lens=runner_inputs.published_seq_lens,
                    RUNNER_INPUTS=True,
                    RUNNER_TRACK=runner_inputs.mamba_track_indices is not None,
                )
                if runner_inputs is not None
                else {}
            ),
        )
    if num_draft_tokens:
        draft = (correct, select, ids) + (() if layout is None else (layout,))
        return new_seq_lens, bonus, (last_steps, track_steps), draft
    return new_seq_lens, bonus


@triton.jit
def _chain_walk(
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    candidates,
    target_predict,
    base,
    cols,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # VerifyTreeGreedy over the request's tree, in registers
    last = tl.load(retrieve_index + base).to(tl.int64)
    first = last
    acc = tl.where(cols == 0, last, -1)
    pred = tl.zeros([BLOCK], dtype=tl.int64)
    num = last * 0
    cur = last * 0
    done = last != last
    for _ in tl.static_range(1, WIDTH):
        active = ~done
        cur = tl.load(retrieve_next_token + base + cur, mask=active, other=-1)
        cur = cur.to(tl.int64)
        target = tl.load(target_predict + last).to(tl.int64)
        matched = last != last
        for _ in tl.static_range(WIDTH):
            searching = active & (cur != -1) & ~matched
            node = tl.load(retrieve_index + base + cur, mask=searching, other=0)
            draft = tl.load(candidates + base + cur, mask=searching, other=0)
            hit = searching & (draft.to(tl.int64) == target)
            pred = tl.where(hit & (cols == last - base), target, pred)
            num = num + hit.to(tl.int64)
            acc = tl.where(hit & (cols == num), node.to(tl.int64), acc)
            last = tl.where(hit, node.to(tl.int64), last)
            matched = matched | hit
            sibling = tl.load(
                retrieve_next_sibling + base + cur, mask=searching & ~hit, other=-1
            )
            cur = tl.where(searching & ~hit, sibling.to(tl.int64), cur)
        done = done | (active & ~matched)
    target = tl.load(target_predict + last).to(tl.int64)
    pred = tl.where(cols == last - base, target, pred)
    return first, acc, pred, num


@triton.jit
def _chain_linear(candidates, target_predict, base, cols, WIDTH: tl.constexpr):
    # The walk's result on a canonical chain: the longest prefix of drafts that equal the
    # previous node's target, from one vector compare.
    in_row = cols < WIDTH
    tgt = tl.load(target_predict + base + cols, mask=in_row, other=0).to(tl.int64)
    drf = tl.load(candidates + base + cols + 1, mask=cols < WIDTH - 1, other=0)
    hit = (cols < WIDTH - 1) & (drf.to(tl.int64) == tgt)
    num = tl.min(tl.where(hit, WIDTH - 1, cols), axis=0).to(tl.int64)
    acc = tl.where(cols <= num, base + cols, -1)
    pred = tl.where(cols <= num, tgt, 0)
    return base, acc, pred, num


@triton.jit
def _chain_is_canonical(
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    base,
    cols,
    WIDTH: tl.constexpr,
):
    # node i is row base + i, its only child node i + 1 (none for the last)
    in_row = cols < WIDTH
    nxt = tl.load(retrieve_next_token + base + cols, mask=in_row, other=-1).to(tl.int64)
    sib = tl.load(retrieve_next_sibling + base + cols, mask=in_row, other=-1).to(
        tl.int64
    )
    idx = tl.load(retrieve_index + base + cols, mask=in_row, other=0).to(tl.int64)
    want = tl.where(cols < WIDTH - 1, cols + 1, -1)
    bad = in_row & ((nxt != want) | (sib != -1) | (idx != base + cols))
    return tl.sum(bad.to(tl.int32), axis=0) == 0


@triton.jit
def _chain_accept_kernel(
    predict,
    accept_index,
    num_correct_drafts,
    accept_lens,
    sim_accept_index,
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    bonus_target,
    simulated_acc_len,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    VERIFY: tl.constexpr,
    FINISH: tl.constexpr,
    SIMULATE: tl.constexpr,
    REAL_DRAFT_TOKENS: tl.constexpr,
    LINEAR: tl.constexpr = False,
):
    # One program per request; its WIDTH nodes are rows [b * WIDTH, (b + 1) * WIDTH) of
    # predict / candidates / retrieve_*, and accept_index is [bs, WIDTH].
    b = tl.program_id(0).to(tl.int64)
    base = b * WIDTH
    cols = tl.arange(0, BLOCK)
    in_row = cols < WIDTH
    if VERIFY:
        if LINEAR:
            if _chain_is_canonical(
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                base,
                cols,
                WIDTH,
            ):
                first, acc, pred, num = _chain_linear(
                    candidates, target_predict, base, cols, WIDTH
                )
            else:
                first, acc, pred, num = _chain_walk(
                    retrieve_index,
                    retrieve_next_token,
                    retrieve_next_sibling,
                    candidates,
                    target_predict,
                    base,
                    cols,
                    WIDTH,
                    BLOCK,
                )
        else:
            first, acc, pred, num = _chain_walk(
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                candidates,
                target_predict,
                base,
                cols,
                WIDTH,
                BLOCK,
            )
        tl.store(accept_index + base + cols, acc.to(tl.int32), mask=in_row)
        if not (FINISH and SIMULATE):
            tl.store(predict + base + cols, pred.to(tl.int32), mask=in_row)
            tl.store(num_correct_drafts + b, num.to(tl.int32))
    else:
        pred = tl.load(predict + base + cols, mask=in_row, other=0).to(tl.int64)
        first = tl.load(accept_index + base).to(tl.int64)
        num = tl.load(num_correct_drafts + b).to(tl.int64)
    if FINISH:
        if SIMULATE:
            # spec_utils.generate_simulated_accept_index_device for a length on the device
            length = tl.load(simulated_acc_len).to(tl.int64)
            sim = tl.where(cols < length, first + cols, -1)
            tl.store(sim_accept_index + base + cols, sim.to(tl.int32), mask=in_row)
            num = length - 1
            if REAL_DRAFT_TOKENS:
                nxt = tl.minimum(cols + 1, WIDTH - 1)
                drafted = tl.load(candidates + base + nxt, mask=in_row, other=0)
                bonus = tl.load(bonus_target + base + cols, mask=in_row, other=0)
                pred = tl.where(
                    cols < length - 1,
                    drafted.to(tl.int32).to(tl.int64),
                    tl.where(cols == length - 1, bonus.to(tl.int32).to(tl.int64), pred),
                )
            else:
                pred = tl.full([BLOCK], 100, dtype=tl.int64)
            tl.store(predict + base + cols, pred.to(tl.int32), mask=in_row)
            tl.store(num_correct_drafts + b, num.to(tl.int32))
        tl.store(accept_lens + b, (num + 1).to(tl.int32))


def chain_accept(
    *,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: Optional[torch.Tensor],
    bonus_target: Optional[torch.Tensor],
    simulated_acc_len: Optional[torch.Tensor],
    real_draft_tokens: bool,
    verify: bool,
    finish: bool,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """eagle_sample's tail for a chain draft whose tree width equals the accept width,
    bitwise: with verify, verify_tree_greedy over target_predict into the (unset)
    predict / accept_index / num_correct_drafts; with finish, the injected acceptance
    of simulated_acc_len (device int64 [1], None when off) and the accept lengths,
    returned with the accept index eagle_sample returns. Contiguous [bs, width] rows."""
    bs, width = accept_index.shape
    accept_lens = sim_accept_index = None
    if finish:
        accept_lens = torch.empty((bs,), dtype=torch.int32, device=predict.device)
        if simulated_acc_len is not None:
            sim_accept_index = torch.empty_like(accept_index)
    if bs:
        _chain_accept_kernel[(bs,)](
            predict,
            accept_index,
            num_correct_drafts,
            accept_lens,
            sim_accept_index,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            bonus_target,
            simulated_acc_len,
            WIDTH=width,
            BLOCK=next_power_of_2(width),
            VERIFY=verify,
            FINISH=finish,
            SIMULATE=simulated_acc_len is not None,
            REAL_DRAFT_TOKENS=real_draft_tokens,
            LINEAR=_LINEAR_ACCEPT,
            num_warps=1,
        )
    if not finish:
        return None
    return accept_lens, (accept_index if sim_accept_index is None else sim_accept_index)


@triton.jit(do_not_specialize=["BS", "TOKENS"])
def _draft_extend_inputs_kernel(
    accept_lens,
    predict,
    correct,
    select,
    ids,
    BS,
    TOKENS,
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


@triton.jit(do_not_specialize=["BS"])
def _prepare_draft_extend_lengths_kernel(
    seq_lens,
    prefix_lens,
    extend_lens,
    post_lens,
    BS,
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
        tl.static_assert(FRONT == 0)
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
    device = seq_lens.device
    empty = perfll_host.round_empty if perfll_host.OUT_POOL else perfll_host.torch_empty
    prefix = empty((bs,), torch.int32, device)
    extend = empty((bs,), torch.int32, device)
    post = empty((bs,), seq_lens.dtype, device)
    positions = empty((bs * num_draft_tokens,), torch.int64, device)
    starts = empty((bs,), torch.int32, device)
    mrope = empty((3, bs * num_draft_tokens), torch.int64, device) if with_mrope else None
    if bs and not (
        perfll_spec_index.ENABLED
        and perfll_spec_index.extend_layout(
            seq_lens, num_draft_tokens, prefix, extend, post, positions, starts, mrope
        )
    ):
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


@triton.jit(do_not_specialize=["BS"])
def _build_chain_tree_kernel(
    bonus,
    draft,
    seq,
    mask,
    positions,
    retrieve,
    tokens,
    BS,
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
    device = seq_lens.device
    empty = perfll_host.round_empty if perfll_host.OUT_POOL else perfll_host.torch_empty
    mask = tree_mask_buf
    if mask is None:
        mask = empty((bs * width * width,), torch.bool, device)
    assert mask.is_contiguous() and mask.numel() >= bs * width * width
    positions = empty((bs * width,), torch.int64, device)
    retrieve = empty((3, bs, width), torch.int64, device)
    tokens = empty(
        (bs * width,),
        torch.promote_types(bonus_tokens.dtype, draft_tokens.dtype),
        device,
    )
    out_cache_loc = (
        empty((bs * width,), torch.int64, device) if req_to_token is not None else None
    )
    mrope = empty((3, bs * width), torch.int64, device) if with_mrope else None
    if req_to_token is not None:
        assert req_pool_indices is not None and req_pool_indices.numel() == bs
    if bs and not (
        perfll_spec_index.ENABLED
        and perfll_spec_index.chain_tree(
            bonus_tokens,
            draft_tokens,
            seq_lens,
            mask,
            positions,
            retrieve,
            tokens,
            req_pool_indices,
            req_to_token,
            out_cache_loc,
            mrope,
        )
    ):
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
