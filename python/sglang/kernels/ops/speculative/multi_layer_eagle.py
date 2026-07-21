# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu, is_npu

_is_cpu = is_cpu()
_is_npu = is_npu()

if _is_cpu:
    from sgl_kernel import rotate_input_ids_cpu


@triton.jit
def rotate_input_ids_kernel(
    input_ids_ptr,
    extend_start_loc_ptr,
    extend_seq_lens_ptr,
    topk_index_ptr,
    select_index_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    start_loc = tl.load(extend_start_loc_ptr + pid)
    seq_len = tl.load(extend_seq_lens_ptr + pid)
    new_token = tl.load(topk_index_ptr + pid)

    num_elements_to_shift = seq_len - 1

    for off in range(0, num_elements_to_shift, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_to_shift

        read_ptr = input_ids_ptr + start_loc + offsets + 1
        val = tl.load(read_ptr, mask=mask)
        tl.debug_barrier()

        write_ptr = input_ids_ptr + start_loc + offsets
        tl.store(write_ptr, val, mask=mask)
        tl.debug_barrier()

    if seq_len > 0:
        if select_index_ptr is not None:
            last_pos_ptr = input_ids_ptr + tl.load(select_index_ptr + pid)
        else:
            last_pos_ptr = input_ids_ptr + start_loc + seq_len - 1
        tl.store(last_pos_ptr, new_token)


def rotate_input_ids(
    input_ids, extend_start_loc, extend_seq_lens, topk_index, select_index=None
):
    if _is_cpu:
        rotate_input_ids_cpu(
            input_ids,
            extend_start_loc,
            extend_seq_lens,
            topk_index,
            select_index,
        )
        return input_ids

    batch_size = extend_seq_lens.shape[0]

    # rotate_input_ids kernel skipped: batch_size=0 (empty extend_seq_lens).
    # This is expected when a DP rank has no requests.
    # TODO: @iforgetmyname Remove NPU-specific guard after triton-ascend fixes zero-sized grid kernel launch abort
    if batch_size == 0 and _is_npu:
        return input_ids

    BLOCK_SIZE = 4096 if select_index is not None else 8
    grid = (batch_size,)

    rotate_input_ids_kernel[grid](
        input_ids,
        extend_start_loc,
        extend_seq_lens,
        topk_index,
        select_index,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return input_ids


@triton.jit
def stash_append_boundary_state_kernel(
    # flat sources (decode: predict + verify FULL hiddens; prefill: rotated
    # input_ids + target FULL hiddens)
    src_tokens_ptr,
    src_hiddens_ptr,  # [num_src_rows, hidden]
    src_row_ends_ptr,  # [bs] exclusive end row of each request's source segment
    num_available_ptr,  # [bs] committed rows at the segment tail (accept_lens / extend len)
    req_pool_indices_ptr,  # [bs]
    # stash (per request, rolling last `front` committed (token, base-hidden)
    # pairs; slot j of a request at boundary B holds position B - front + j)
    stash_tokens_ptr,  # [req_pool_size, front] int64
    stash_hiddens_ptr,  # [req_pool_size, front, hidden]
    stash_valid_lens_ptr,  # [req_pool_size] int32, count of valid tail slots
    front: tl.constexpr,
    hidden_dim: tl.constexpr,
    SET_VALID: tl.constexpr,  # prefill: valid = m; decode: valid = min(valid + m, front)
    BLOCK_H: tl.constexpr,
):
    """Roll the per-request boundary stash forward by m = min(available, front)
    newly committed (token, base-hidden) pairs taken from the source tail
    rows [end - m, end). Kept old pairs shift down (reads stay ahead of
    writes, ascending order)."""
    pid = tl.program_id(0)
    rpi = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    end = tl.load(src_row_ends_ptr + pid).to(tl.int64)
    avail = tl.load(num_available_ptr + pid).to(tl.int64)
    m = tl.minimum(avail, front)
    keep = front - m

    h_off = tl.arange(0, BLOCK_H)

    # 1) Shift the kept tail of the old stash to the front: new[i] = old[i + m]
    for i in range(0, keep):
        src_t = tl.load(stash_tokens_ptr + rpi * front + i + m)
        tl.store(stash_tokens_ptr + rpi * front + i, src_t)
        for hb in range(0, hidden_dim, BLOCK_H):
            hmask = (hb + h_off) < hidden_dim
            src_h = tl.load(
                stash_hiddens_ptr + (rpi * front + i + m) * hidden_dim + hb + h_off,
                mask=hmask,
            )
            tl.store(
                stash_hiddens_ptr + (rpi * front + i) * hidden_dim + hb + h_off,
                src_h,
                mask=hmask,
            )

    # 2) Append the m newest committed pairs from the source tail.
    for i in range(0, m):
        row = end - m + i
        dst = rpi * front + keep + i
        tok = tl.load(src_tokens_ptr + row)
        tl.store(stash_tokens_ptr + dst, tok)
        for hb in range(0, hidden_dim, BLOCK_H):
            hmask = (hb + h_off) < hidden_dim
            src_h = tl.load(src_hiddens_ptr + row * hidden_dim + hb + h_off, mask=hmask)
            tl.store(
                stash_hiddens_ptr + dst * hidden_dim + hb + h_off, src_h, mask=hmask
            )

    if SET_VALID:
        valid = m
    else:
        valid = tl.minimum(tl.load(stash_valid_lens_ptr + rpi).to(tl.int64) + m, front)
    tl.store(stash_valid_lens_ptr + rpi, valid.to(tl.int32))


def stash_append_boundary_state_triton(
    src_tokens,
    src_hiddens,
    src_row_ends,
    num_available,
    req_pool_indices,
    stash_tokens,
    stash_hiddens,
    stash_valid_lens,
    set_valid: bool,
):
    """Append newly committed (token, base-hidden) pairs to the rolling
    boundary stash (see kernel docstring). Decode: sources are (predict,
    verify FULL hiddens) with ends = i*W + accept_lens. Prefill: sources are
    (post-rotation input_ids, target FULL hiddens) with ends = start + len."""
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return
    stash_append_boundary_state_kernel[(bs,)](
        src_tokens,
        src_hiddens,
        src_row_ends,
        num_available,
        req_pool_indices,
        stash_tokens,
        stash_hiddens,
        stash_valid_lens,
        front=stash_tokens.shape[1],
        hidden_dim=stash_hiddens.shape[2],
        SET_VALID=set_valid,
        BLOCK_H=1024,
    )


@triton.jit
def fill_widened_draft_extend_inputs_kernel(
    # outputs: the widened per-request window buffers, width = W + front
    input_ids_ptr,  # [bs * width]
    hidden_ptr,  # [bs * width, hidden]
    # sources
    predict_ptr,  # [bs * W] verify-sampled successor per verify row
    verify_hidden_ptr,  # [bs * W, hidden] target verify hiddens (FULL capture)
    stash_tokens_ptr,  # [req_pool_size, front]
    stash_hiddens_ptr,  # [req_pool_size, front, hidden]
    stash_valid_lens_ptr,  # [req_pool_size]
    seq_lens_ptr,  # [bs] PRE-verify seq_lens (window base = seq_lens - front)
    req_pool_indices_ptr,  # [bs]
    draft_token_num: tl.constexpr,  # W
    front: tl.constexpr,  # F_total
    hidden_dim: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Materialize the widened depth-0 window's input tokens and hiddens: front
    rows (j < front) source from stash slot j, original rows (j >= front) from
    predict/verify hiddens; data-invalid front rows are zeroed. Locs/positions
    are computed separately by compute_widened_draft_extend_locs_positions."""
    pid = tl.program_id(0)
    rpi = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int64)
    valid_len = tl.load(stash_valid_lens_ptr + rpi).to(tl.int64)

    # Rows below this hold no usable stash data (unseeded slot or p < 0).
    first_valid = tl.maximum(tl.maximum(front - valid_len, front - seq_len), 0)

    h_off = tl.arange(0, BLOCK_H)
    width = draft_token_num + front

    for j in range(0, width):
        row = pid * width + j
        if j >= front:
            src = pid * draft_token_num + j - front
            tok = tl.load(predict_ptr + src)
            tl.store(input_ids_ptr + row, tok)
            for hb in range(0, hidden_dim, BLOCK_H):
                hmask = (hb + h_off) < hidden_dim
                src_h = tl.load(
                    verify_hidden_ptr + src * hidden_dim + hb + h_off, mask=hmask
                )
                tl.store(hidden_ptr + row * hidden_dim + hb + h_off, src_h, mask=hmask)
        else:
            if j >= first_valid:
                tok = tl.load(stash_tokens_ptr + rpi * front + j)
                tl.store(input_ids_ptr + row, tok)
                for hb in range(0, hidden_dim, BLOCK_H):
                    hmask = (hb + h_off) < hidden_dim
                    src_h = tl.load(
                        stash_hiddens_ptr + (rpi * front + j) * hidden_dim + hb + h_off,
                        mask=hmask,
                    )
                    tl.store(
                        hidden_ptr + row * hidden_dim + hb + h_off, src_h, mask=hmask
                    )
            else:
                tl.store(input_ids_ptr + row, 0)
                for hb in range(0, hidden_dim, BLOCK_H):
                    hmask = (hb + h_off) < hidden_dim
                    tl.store(
                        hidden_ptr + row * hidden_dim + hb + h_off, 0.0, mask=hmask
                    )


def fill_widened_draft_extend_inputs_triton(
    input_ids,
    hidden_states,
    predict,
    verify_hiddens,
    stash_tokens,
    stash_hiddens,
    stash_valid_lens,
    seq_lens,
    req_pool_indices,
    draft_token_num: int,
):
    """Fill the widened window's input tokens and hiddens in place (see kernel
    docstring). Must run AFTER verify sampling (reads predict / hiddens) and
    BEFORE the stash update for this iteration (the stash is still based at
    the pre-verify boundary)."""
    bs = req_pool_indices.shape[0]
    if bs == 0:
        return
    fill_widened_draft_extend_inputs_kernel[(bs,)](
        input_ids,
        hidden_states,
        predict,
        verify_hiddens,
        stash_tokens,
        stash_hiddens,
        stash_valid_lens,
        seq_lens,
        req_pool_indices,
        draft_token_num=draft_token_num,
        front=stash_tokens.shape[1],
        hidden_dim=stash_hiddens.shape[2],
        BLOCK_H=1024,
    )


@triton.jit
def _wide_row_softmax_partials_kernel(
    logits_ptr,  # [bs, vocab] fp32
    temperatures_ptr,  # [bs, 1] fp32 (dummy when HAS_TEMPS is False)
    partial_max_ptr,  # [bs, nblocks] fp32
    partial_sum_ptr,  # [bs, nblocks] fp32
    vocab,
    nblocks,
    HAS_TEMPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    z = tl.load(
        logits_ptr + row.to(tl.int64) * vocab + offs, mask=mask, other=-float("inf")
    )
    if HAS_TEMPS:
        z = z / tl.load(temperatures_ptr + row)
    m = tl.max(z, axis=0)
    s = tl.sum(tl.exp(z - m), axis=0)
    tl.store(partial_max_ptr + row * nblocks + blk, m)
    tl.store(partial_sum_ptr + row * nblocks + blk, s)


@triton.jit
def _wide_row_softmax_finalize_kernel(
    partial_max_ptr,
    partial_sum_ptr,
    row_max_ptr,  # [bs] fp32
    row_sum_ptr,  # [bs] fp32
    nblocks,
    NBLOCK_POW2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NBLOCK_POW2)
    mask = offs < nblocks
    m = tl.load(partial_max_ptr + row * nblocks + offs, mask=mask, other=-float("inf"))
    s = tl.load(partial_sum_ptr + row * nblocks + offs, mask=mask, other=0.0)
    gm = tl.max(m, axis=0)
    gs = tl.sum(s * tl.exp(m - gm), axis=0)
    tl.store(row_max_ptr + row, gm)
    tl.store(row_sum_ptr + row, gs)


@triton.jit
def _wide_row_softmax_write_kernel(
    logits_ptr,
    temperatures_ptr,
    row_max_ptr,
    row_sum_ptr,
    out_ptr,  # [bs, out_row_stride] fp32; row i written at i * out_row_stride
    vocab,
    out_row_stride,
    HAS_TEMPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    mask = offs < vocab
    z = tl.load(
        logits_ptr + row.to(tl.int64) * vocab + offs, mask=mask, other=-float("inf")
    )
    if HAS_TEMPS:
        z = z / tl.load(temperatures_ptr + row)
    gm = tl.load(row_max_ptr + row)
    gs = tl.load(row_sum_ptr + row)
    q = tl.exp(z - gm) / gs
    tl.store(out_ptr + row.to(tl.int64) * out_row_stride + offs, q, mask=mask)


def wide_row_softmax_triton(
    logits: torch.Tensor,
    temperatures,
    out: torch.Tensor,
) -> torch.Tensor:
    """Column-parallel softmax over very wide fp32 rows, optionally with
    per-row temperature (q = softmax(logits / T)), written into ``out``
    (any row stride >= vocab). torch.softmax gives one block per row, which
    serializes a single wide draft-vocab row onto one SM."""
    bs, vocab = logits.shape
    BLOCK = 4096
    nblocks = triton.cdiv(vocab, BLOCK)
    partial_max = torch.empty((bs, nblocks), dtype=torch.float32, device=logits.device)
    partial_sum = torch.empty((bs, nblocks), dtype=torch.float32, device=logits.device)
    row_max = torch.empty((bs,), dtype=torch.float32, device=logits.device)
    row_sum = torch.empty((bs,), dtype=torch.float32, device=logits.device)
    has_temps = temperatures is not None
    dummy = row_max
    _wide_row_softmax_partials_kernel[(bs, nblocks)](
        logits,
        temperatures if has_temps else dummy,
        partial_max,
        partial_sum,
        vocab,
        nblocks,
        HAS_TEMPS=has_temps,
        BLOCK=BLOCK,
    )
    _wide_row_softmax_finalize_kernel[(bs,)](
        partial_max,
        partial_sum,
        row_max,
        row_sum,
        nblocks,
        NBLOCK_POW2=triton.next_power_of_2(nblocks),
    )
    _wide_row_softmax_write_kernel[(bs, nblocks)](
        logits,
        temperatures if has_temps else dummy,
        row_max,
        row_sum,
        out,
        vocab,
        out.stride(0),
        HAS_TEMPS=has_temps,
        BLOCK=BLOCK,
    )
    return out


@triton.jit
def compute_widened_draft_extend_locs_positions_kernel(
    seq_lens_ptr,
    req_pool_indices_ptr,
    req_to_token_ptr,
    stash_valid_lens_ptr,
    locs_ptr,  # [bs * width] int64
    positions_ptr,  # [bs * width] int64
    req_to_token_stride,
    front,
    num_warmup_tokens,
    width,
    WIDTH_BLOCK: tl.constexpr,
):
    """Per-request widened-window locs + positions: pos = seq_len - front + j;
    rows below first_valid = max(front - stash_valid, front - seq_len, 0) hold
    no stash data (positions zeroed), and the first num_warmup_tokens valid
    front rows write to sacrificial loc 0."""
    pid = tl.program_id(0)
    offs = tl.arange(0, WIDTH_BLOCK)
    wmask = offs < width
    offs64 = offs.to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int64)
    rpi = tl.load(req_pool_indices_ptr + pid).to(tl.int64)
    valid_len = tl.load(stash_valid_lens_ptr + rpi).to(tl.int64)

    pos = seq_len - front + offs64
    first_valid = tl.maximum(tl.maximum(front - valid_len, front - seq_len), 0)
    data_valid = offs64 >= first_valid
    write_real = offs64 >= tl.minimum(first_valid + num_warmup_tokens, front)

    tok = tl.load(
        req_to_token_ptr + rpi * req_to_token_stride + tl.maximum(pos, 0),
        mask=wmask,
        other=0,
    ).to(tl.int64)
    locs = tl.where(write_real, tok, 0)
    positions = tl.where(data_valid, pos, 0)

    base = pid.to(tl.int64) * width
    tl.store(locs_ptr + base + offs, locs, mask=wmask)
    tl.store(positions_ptr + base + offs, positions, mask=wmask)


def compute_widened_draft_extend_locs_positions_triton(
    seq_lens,
    req_pool_indices,
    req_to_token,
    stash_valid_lens,
    draft_token_num: int,
    num_front_tokens: int,
    num_warmup_tokens: int,
):
    width = draft_token_num + num_front_tokens
    bs = seq_lens.shape[0]
    locs = torch.empty((bs * width,), dtype=torch.int64, device=seq_lens.device)
    positions = torch.empty((bs * width,), dtype=torch.int64, device=seq_lens.device)
    if bs > 0:
        compute_widened_draft_extend_locs_positions_kernel[(bs,)](
            seq_lens,
            req_pool_indices,
            req_to_token,
            stash_valid_lens,
            locs,
            positions,
            req_to_token.stride(0),
            num_front_tokens,
            num_warmup_tokens,
            width,
            WIDTH_BLOCK=triton.next_power_of_2(width),
        )
    return locs, positions


@triton.jit
def fill_draft_extend_prepare_buffers_kernel(
    # persistent per-token buffers (length max_num_token, int64)
    input_ids_ptr,
    positions_ptr,
    out_cache_loc_ptr,
    # per-token sources (length num_tokens)
    src_input_ids_ptr,
    src_positions_ptr,
    src_out_cache_loc_ptr,
    # persistent per-request buffers (length max_bs)
    seq_lens_ptr,  # int32
    req_pool_indices_ptr,  # int64
    num_correct_drafts_ptr,  # int32
    num_accept_tokens_ptr,  # int32
    select_index_ptr,  # int64
    temperatures_ptr,  # float32 [max_bs, 1] (dummy when HAS_TEMPS is False)
    # per-request sources (length raw_bs)
    src_seq_lens_ptr,
    src_req_pool_indices_ptr,
    src_num_correct_drafts_ptr,
    src_num_accept_tokens_ptr,
    src_temperatures_ptr,  # dummy when HAS_TEMPS is False
    # chain hidden window, flat [num_tokens * hidden] (dummies when HAS_HIDDEN
    # is False)
    hidden_states_ptr,
    src_hidden_states_ptr,
    # gathered-buffer mirrors (dummies when HAS_GATHERED is False)
    global_num_tokens_ptr,
    global_num_tokens_for_logprob_ptr,
    # scalars
    num_tokens,
    max_num_token,
    raw_bs,
    bs,
    max_bs,
    num_tokens_per_bs,
    num_front_tokens,
    seq_len_fill_value,
    hidden_numel,
    num_global,
    num_token_programs,
    HAS_TEMPS: tl.constexpr,
    HAS_HIDDEN: tl.constexpr,
    HAS_GATHERED: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    GLOBAL_BLOCK: tl.constexpr,
):
    """The whole draft-extend prepare() buffer population in one launch;
    program roles split by flat program id:

    - [0, num_token_programs): input_ids / positions / out_cache_loc; rows
      < num_tokens take the source, the tail up to max_num_token is zeroed.
    - [num_token_programs, +max_bs): one program per request row. Real rows
      [0, raw_bs) take source values; padded rows [raw_bs, bs) take the pad
      sentinels the graphs rely on (seq_len fill value, num_accept_tokens = -1,
      temperatures = 1.0); rows >= bs are untouched except seq_lens, which is
      fully reset. select_index = i*window + front + num_correct_drafts
      (padded rows keep their stale num_correct_drafts, whose gather result
      is discarded).
    - num_token_programs + max_bs: the DP gathered-buffer fills.
    - the rest: flat copy of the chain hidden window's real rows (the padded
      tail is never read).
    """
    pid = tl.program_id(0)

    if pid < num_token_programs:
        tok_offs = pid * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
        store_mask = tok_offs < max_num_token
        copy_mask = tok_offs < num_tokens

        tok = tl.load(src_input_ids_ptr + tok_offs, mask=copy_mask, other=0).to(
            tl.int64
        )
        tl.store(input_ids_ptr + tok_offs, tok, mask=store_mask)
        tok = tl.load(src_positions_ptr + tok_offs, mask=copy_mask, other=0).to(
            tl.int64
        )
        tl.store(positions_ptr + tok_offs, tok, mask=store_mask)
        tok = tl.load(src_out_cache_loc_ptr + tok_offs, mask=copy_mask, other=0).to(
            tl.int64
        )
        tl.store(out_cache_loc_ptr + tok_offs, tok, mask=store_mask)
    elif pid < num_token_programs + max_bs:
        i = pid - num_token_programs
        is_real = i < raw_bs
        is_pad = (i >= raw_bs) & (i < bs)
        in_bs = i < bs

        sl = tl.load(src_seq_lens_ptr + i, mask=is_real, other=seq_len_fill_value)
        tl.store(seq_lens_ptr + i, sl.to(tl.int32))

        rpi = tl.load(src_req_pool_indices_ptr + i, mask=is_real, other=0).to(tl.int64)
        tl.store(req_pool_indices_ptr + i, rpi, mask=is_real)

        # The stale count must be read BEFORE the real-row store below.
        ncd_stale = tl.load(num_correct_drafts_ptr + i, mask=is_pad, other=0).to(
            tl.int64
        )
        ncd_src = tl.load(src_num_correct_drafts_ptr + i, mask=is_real, other=0).to(
            tl.int64
        )
        tl.store(num_correct_drafts_ptr + i, ncd_src.to(tl.int32), mask=is_real)
        ncd = tl.where(is_real, ncd_src, ncd_stale)

        # Padded rows get -1 so the sconv commit skips their live mamba slots.
        nat = tl.load(src_num_accept_tokens_ptr + i, mask=is_real, other=-1).to(
            tl.int32
        )
        tl.store(num_accept_tokens_ptr + i, nat, mask=in_bs)

        si = i.to(tl.int64) * num_tokens_per_bs + num_front_tokens + ncd
        tl.store(select_index_ptr + i, si, mask=in_bs)

        if HAS_TEMPS:
            t = tl.load(src_temperatures_ptr + i, mask=is_real, other=1.0)
            tl.store(temperatures_ptr + i, t, mask=in_bs)
    elif pid == num_token_programs + max_bs:
        if HAS_GATHERED:
            g_offs = tl.arange(0, GLOBAL_BLOCK)
            g_mask = g_offs < num_global
            g_vals = tl.zeros((GLOBAL_BLOCK,), dtype=tl.int32) + bs * num_tokens_per_bs
            tl.store(global_num_tokens_ptr + g_offs, g_vals, mask=g_mask)
            tl.store(global_num_tokens_for_logprob_ptr + g_offs, g_vals, mask=g_mask)
    else:
        if HAS_HIDDEN:
            h_base = pid - num_token_programs - max_bs - 1
            h_offs = h_base.to(tl.int64) * BLOCK_HIDDEN + tl.arange(0, BLOCK_HIDDEN)
            h_mask = h_offs < hidden_numel
            h_vals = tl.load(src_hidden_states_ptr + h_offs, mask=h_mask)
            tl.store(hidden_states_ptr + h_offs, h_vals, mask=h_mask)


def fill_draft_extend_prepare_buffers_triton(
    input_ids,
    positions,
    out_cache_loc,
    src_input_ids,
    src_positions,
    src_out_cache_loc,
    seq_lens,
    req_pool_indices,
    num_correct_drafts,
    num_accept_tokens,
    select_index,
    temperatures,
    src_seq_lens,
    src_req_pool_indices,
    src_num_correct_drafts,
    src_num_accept_tokens,
    src_temperatures,
    hidden_states,
    src_hidden_states,
    global_num_tokens,
    global_num_tokens_for_logprob,
    raw_bs,
    bs,
    num_tokens_per_bs,
    num_front_tokens,
    seq_len_fill_value,
):
    max_num_token = input_ids.shape[0]
    num_tokens = src_input_ids.shape[0]
    max_bs = seq_lens.shape[0]
    has_temps = temperatures is not None
    has_hidden = src_hidden_states is not None
    has_gathered = global_num_tokens is not None

    BLOCK_TOK = 1024
    BLOCK_HIDDEN = 2048
    num_token_programs = triton.cdiv(max_num_token, BLOCK_TOK)

    if has_hidden:
        hidden_numel = num_tokens * hidden_states.shape[1]
        num_hidden_programs = triton.cdiv(hidden_numel, BLOCK_HIDDEN)
    else:
        hidden_numel = 0
        num_hidden_programs = 0

    if has_gathered:
        num_global = global_num_tokens.shape[0]
        global_block = triton.next_power_of_2(num_global)
    else:
        num_global = 0
        global_block = 1

    grid = (num_token_programs + max_bs + 1 + num_hidden_programs,)
    fill_draft_extend_prepare_buffers_kernel[grid](
        input_ids,
        positions,
        out_cache_loc,
        src_input_ids,
        src_positions,
        src_out_cache_loc,
        seq_lens,
        req_pool_indices,
        num_correct_drafts,
        num_accept_tokens,
        select_index,
        temperatures if has_temps else seq_lens,
        src_seq_lens,
        src_req_pool_indices,
        src_num_correct_drafts,
        src_num_accept_tokens,
        src_temperatures if has_temps else seq_lens,
        hidden_states if has_hidden else seq_lens,
        src_hidden_states if has_hidden else seq_lens,
        global_num_tokens if has_gathered else seq_lens,
        global_num_tokens_for_logprob if has_gathered else seq_lens,
        num_tokens,
        max_num_token,
        raw_bs,
        bs,
        max_bs,
        num_tokens_per_bs,
        num_front_tokens,
        seq_len_fill_value,
        hidden_numel,
        num_global,
        num_token_programs,
        HAS_TEMPS=has_temps,
        HAS_HIDDEN=has_hidden,
        HAS_GATHERED=has_gathered,
        BLOCK_TOK=BLOCK_TOK,
        BLOCK_HIDDEN=BLOCK_HIDDEN,
        GLOBAL_BLOCK=global_block,
    )
