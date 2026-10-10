"""Triton n-gram hash ids for the DeepSeek-V4.1 engram, one launch per forward.

Each token t needs its n predecessors: shift 0 is the token itself, shift s an
earlier token of the same request. Shifts that reach past the request's first token
of this forward come from a per-request history row (oldest first). A shift that runs
off the sequence start, or (with vision) reaches an image token, is PAD, and so is
every older shift. The compressed ids are multiplied per (layer, shift), XOR-folded
one shift at a time and bucketed by that n-gram size's per-head primes.

The arithmetic must remain bit-identical to EngramHasher's torch path
in sglang.srt.layers.engram.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

MODE_DECODE = 0
MODE_VERIFY = 1
MODE_EXTEND = 2


@triton.jit
def _engram_commit_history_kernel(
    history_ptr,
    tokens_ptr,
    slots_ptr,
    commit_ptr,
    HISTORY_STRIDE: tl.constexpr,
    HISTORY_WIDTH: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    commit = tl.load(commit_ptr + row)
    col = tl.arange(0, BLOCK)
    source_col = commit + col
    valid = col < HISTORY_WIDTH
    old = tl.load(
        history_ptr + slot * HISTORY_STRIDE + source_col,
        mask=valid & (source_col < HISTORY_WIDTH),
        other=0,
    )
    new = tl.load(
        tokens_ptr + row * TOKEN_STRIDE + source_col - HISTORY_WIDTH,
        mask=valid & (source_col >= HISTORY_WIDTH),
        other=0,
    )
    values = tl.where(source_col < HISTORY_WIDTH, old, new)
    # Every lane must finish reading the old row before any lane overwrites it.
    tl.debug_barrier()
    tl.store(history_ptr + slot * HISTORY_STRIDE + col, values, mask=valid)


def engram_commit_history(
    history: torch.Tensor,
    verify_ids: torch.Tensor,
    req_slots: torch.Tensor,
    commit_lens: torch.Tensor,
) -> None:
    """Update distinct live request slots with anchor + correct drafts, not bonus."""
    bs = req_slots.numel()
    width = history.shape[1]
    if bs == 0 or width == 0:
        return
    assert history.stride(1) == 1 and verify_ids.stride(1) == 1
    assert req_slots.stride(0) == 1 and commit_lens.stride(0) == 1
    _engram_commit_history_kernel[(bs,)](
        history,
        verify_ids,
        req_slots,
        commit_lens,
        HISTORY_STRIDE=history.stride(0),
        HISTORY_WIDTH=width,
        TOKEN_STRIDE=verify_ids.stride(0),
        BLOCK=triton.next_power_of_2(width),
        num_warps=4,
    )


@triton.jit
def _engram_hash_kernel(
    ids_ptr,
    pos_ptr,
    row_ptr,
    starts_ptr,
    slots_ptr,
    hist_ptr,
    token_map_ptr,
    mult_ptr,
    primes_ptr,
    offsets_ptr,
    out_loc_ptr,
    tokens_out_ptr,
    out_ptr,
    num_tokens,
    num_real,
    pad_id,
    image_token_id,
    mm_pad_shift,
    MODE: tl.constexpr,
    BLOCK: tl.constexpr,
    HIST_VIA_SLOTS: tl.constexpr,
    HAS_IMAGE: tl.constexpr,
    COMMIT: tl.constexpr,
    WRITE_TOKENS: tl.constexpr,
    N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EXTEND_BS: tl.constexpr = 0,
    SEARCH_STEPS: tl.constexpr = 0,
):
    COLS: tl.constexpr = (N - 1) * H
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < num_tokens
    real = t < num_real
    # Request row r and the token's offset inside its run for this forward.
    if MODE == 0:
        r = t.to(tl.int64)
        off = t * 0
    elif MODE == 1:
        r = (t // BLOCK).to(tl.int64)
        off = t - (t // BLOCK) * BLOCK
    else:
        if EXTEND_BS == 1:
            r = tl.full((BLOCK_T,), 0, tl.int64)
        elif EXTEND_BS > 1:
            # Upper bound handles adjacent starts from zero-length requests.
            # Resolve rows in registers instead of materializing repeat_interleave.
            lo = tl.full((BLOCK_T,), 0, tl.int32)
            hi = tl.full((BLOCK_T,), EXTEND_BS, tl.int32)
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                start = tl.load(
                    starts_ptr + mid, mask=real & (mid < EXTEND_BS), other=0
                )
                right = (mid < EXTEND_BS) & (start <= t)
                lo = tl.where(right, mid + 1, lo)
                hi = tl.where(right, hi, mid)
            r = tl.maximum(lo - 1, 0).to(tl.int64)
        else:
            r = tl.load(row_ptr + t, mask=real, other=0).to(tl.int64)
        off = t - tl.load(starts_ptr + r, mask=real, other=0).to(tl.int32)
    if HIST_VIA_SLOTS:
        hrow = tl.load(slots_ptr + r, mask=real, other=0).to(tl.int64)
    else:
        hrow = r
    pos = tl.load(pos_ptr + t, mask=tmask, other=0).to(tl.int32)

    t2 = t[:, None]
    s = tl.arange(0, N)[None, :]
    tmask2 = tmask[:, None] & (s >= 0)
    real2 = real[:, None] & (s >= 0)
    off2 = off[:, None]
    # Predecessor at shift s: an earlier token of the same run when s <= off, else
    # history[row, n - 2 - (s - off - 1)]; the history is oldest first.
    from_batch = tl.load(ids_ptr + tl.maximum(t2 - s, 0), mask=tmask2, other=0).to(
        tl.int64
    )
    hcol = tl.minimum(tl.maximum(N - 2 - (s - off2 - 1), 0), N - 2)
    from_hist = tl.load(
        hist_ptr + hrow[:, None] * (N - 1) + hcol, mask=tmask2, other=0
    ).to(tl.int64)
    tok = tl.where(s <= off2, from_batch, from_hist)
    tok = tl.where(real2, tok, 0)
    blk = (pos[:, None] < s) | (real2 == 0)
    if HAS_IMAGE:
        # Scheduler-provided history still carries the multimodal pad ids.
        tok = tl.where(tok >= mm_pad_shift, image_token_id, tok)
        blk = blk | (tok == image_token_id)
    # Once a shift is blocked every older shift is too (cummax along s).
    blocked = tl.cumsum(blk.to(tl.int32), axis=1) > 0
    if WRITE_TOKENS:
        tl.store(tokens_out_ptr + t2 * N + s, tok.to(tl.int32), mask=tmask2)
    if COMMIT:
        # Decode: the token and its n - 2 newest predecessors become the request's
        # history, oldest first. Graph-padded rows (out_cache_loc 0) write nothing.
        live = tl.load(out_loc_ptr + t, mask=real, other=0) != 0
        cmask = real2 & live[:, None] & (s <= N - 2)
        tl.store(
            hist_ptr + hrow[:, None] * (N - 1) + (N - 2 - s),
            tok.to(tl.int32),
            mask=cmask,
        )
    mapped = tl.load(token_map_ptr + tok, mask=tmask2, other=0).to(tl.int64)
    comp = tl.where(blocked, pad_id, mapped)

    h = tl.arange(0, H)[None, :]
    omask = tmask[:, None] & (h >= 0)
    for l in tl.static_range(L):
        mult = tl.load(mult_ptr + l * N + s)
        prod = comp * mult
        for i in tl.static_range(1, N):
            # (i + 1)-gram hash: XOR of the first i + 1 shifts' products.
            rolling = tl.xor_sum(tl.where(s <= i, prod, 0), axis=1)
            primes = tl.load(primes_ptr + l * COLS + (i - 1) * H + h)
            offsets = tl.load(offsets_ptr + l * COLS + (i - 1) * H + h)
            val = rolling[:, None] % primes + offsets
            tl.store(
                out_ptr + t2 * (L * COLS) + l * COLS + (i - 1) * H + h, val, mask=omask
            )


def _launch_hash_kernel(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    mode: int,
    history: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    num_real: Optional[int],
    req_slots: Optional[torch.Tensor],
    block: int,
    row: Optional[torch.Tensor],
    starts: Optional[torch.Tensor],
    image_token_id: Optional[int],
    mm_pad_shift: int,
    out_cache_loc: Optional[torch.Tensor],
    write_tokens: bool,
    block_t: int,
    extend_bs: int = 0,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_tokens = input_ids.shape[0]
    L, N = multipliers.shape
    H = primes.shape[-1]
    assert N & (N - 1) == 0 and H & (H - 1) == 0, (N, H)
    assert primes.shape == (L, N - 1, H) and offsets.shape == (L, (N - 1) * H)
    assert history.dim() == 2 and history.shape[1] == N - 1, history.shape
    if mode == MODE_EXTEND:
        assert (row is not None or extend_bs > 0) and starts is not None
    if out_cache_loc is not None:
        assert mode == MODE_DECODE and req_slots is not None, "commit is decode-only"
        assert out_cache_loc.shape[0] == num_tokens, out_cache_loc.shape
    if num_real is None:
        num_real = num_tokens
    device = input_ids.device
    out = torch.empty(num_tokens, L, (N - 1) * H, dtype=torch.int64, device=device)
    tokens = (
        torch.empty(num_tokens, N, dtype=torch.int32, device=device)
        if write_tokens
        else None
    )
    if num_tokens == 0:
        return out, tokens
    dummy = out  # unused pointer slots; never dereferenced under their constexprs
    _engram_hash_kernel[(triton.cdiv(num_tokens, block_t),)](
        input_ids,
        positions,
        row if row is not None else dummy,
        starts if starts is not None else dummy,
        req_slots if req_slots is not None else dummy,
        history,
        token_map,
        multipliers,
        primes,
        offsets,
        out_cache_loc if out_cache_loc is not None else dummy,
        tokens if tokens is not None else dummy,
        out,
        num_tokens,
        num_real,
        pad_id,
        image_token_id if image_token_id is not None else -1,
        mm_pad_shift,
        MODE=mode,
        BLOCK=block,
        HIST_VIA_SLOTS=req_slots is not None,
        HAS_IMAGE=image_token_id is not None,
        COMMIT=out_cache_loc is not None,
        WRITE_TOKENS=write_tokens,
        N=N,
        L=L,
        H=H,
        BLOCK_T=block_t,
        EXTEND_BS=extend_bs,
        SEARCH_STEPS=extend_bs.bit_length(),
        num_warps=4,
    )
    return out, tokens


@triton.jit
def _engram_commit_extend_history_kernel(
    history_ptr,
    tokens_ptr,
    slots_ptr,
    starts_ptr,
    lens_ptr,
    out_loc_ptr,
    NUM_TOKENS: tl.constexpr,
    N: tl.constexpr,
    HAS_OUT_LOC: tl.constexpr,
):
    r = tl.program_id(0)
    length = tl.load(lens_ptr + r)
    start = tl.load(starts_ptr + r)
    last = tl.minimum(tl.maximum(start + length - 1, 0), NUM_TOKENS - 1)
    live = length > 0
    if HAS_OUT_LOC:
        live = live & (tl.load(out_loc_ptr + last) != 0)
    slot = tl.load(slots_ptr + r).to(tl.int64)
    col = tl.arange(0, N)
    value = tl.load(
        tokens_ptr + last * N + (N - 2 - col),
        mask=live & (col < N - 1),
        other=0,
    )
    tl.store(history_ptr + slot * (N - 1) + col, value, mask=live & (col < N - 1))


def engram_hash_extend_and_commit(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    history: torch.Tensor,
    commit_history: torch.Tensor,
    req_slots: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    num_real: int,
    history_via_slots: bool,
    out_cache_loc: Optional[torch.Tensor],
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    image_token_id: Optional[int] = None,
    mm_pad_shift: int = 0,
) -> torch.Tensor:
    """Eager extend: hash without a row-map tensor, then commit in one launch.

    Hashing must finish before history is updated: different token programs may
    still read a request's old history. Keep these as two ordered kernels.
    Empty/padded requests leave live history untouched; the spare padding row is
    intentionally not updated. Request slots must be distinct for live requests.
    """
    bs = req_slots.numel()
    assert 0 <= num_real <= input_ids.numel()
    assert starts.numel() == lengths.numel() == bs
    assert history.is_contiguous() and commit_history.is_contiguous()
    assert all(t.ndim == 1 and t.stride(0) == 1 for t in (req_slots, starts, lengths))
    out, tokens = _launch_hash_kernel(
        input_ids,
        positions,
        mode=MODE_EXTEND,
        history=history,
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        pad_id=pad_id,
        num_real=num_real,
        req_slots=req_slots if history_via_slots else None,
        block=1,
        row=None,
        starts=starts,
        image_token_id=image_token_id,
        mm_pad_shift=mm_pad_shift,
        out_cache_loc=None,
        write_tokens=True,
        block_t=32,
        extend_bs=bs,
    )
    if input_ids.numel() and bs:
        _engram_commit_extend_history_kernel[(bs,)](
            commit_history,
            tokens,
            req_slots,
            starts,
            lengths,
            out_cache_loc if out_cache_loc is not None else req_slots,
            NUM_TOKENS=input_ids.numel(),
            N=multipliers.shape[1],
            HAS_OUT_LOC=out_cache_loc is not None,
            num_warps=4,
        )
    return out


def engram_hash_ids(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    mode: int,
    history: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    num_real: Optional[int] = None,
    req_slots: Optional[torch.Tensor] = None,
    block: int = 1,
    row: Optional[torch.Tensor] = None,
    starts: Optional[torch.Tensor] = None,
    image_token_id: Optional[int] = None,
    mm_pad_shift: int = 0,
    block_t: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hash ids [T, L, (n - 1) * heads] int64 and the predecessor table [T, n] int32.
    Reads ``history`` and never writes it.

    ``mode``: MODE_DECODE (row = t, offset 0), MODE_VERIFY (row = t // block),
    MODE_EXTEND (``row`` [num_real] and ``starts`` [bs] give each token's request and
    the run's first token). ``history`` is [rows, n - 1] oldest first; with
    ``req_slots`` given it is indexed by ``req_slots[row]``, else by ``row``.
    Tokens at or past ``num_real`` are padding: PAD ids, zero predecessors.
    """
    out, tokens = _launch_hash_kernel(
        input_ids,
        positions,
        mode=mode,
        history=history,
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        pad_id=pad_id,
        num_real=num_real,
        req_slots=req_slots,
        block=block,
        row=row,
        starts=starts,
        image_token_id=image_token_id,
        mm_pad_shift=mm_pad_shift,
        out_cache_loc=None,
        write_tokens=True,
        block_t=block_t,
    )
    return out, tokens


def engram_hash_ids_and_commit(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    history: torch.Tensor,
    req_slots: torch.Tensor,
    out_cache_loc: torch.Tensor,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    image_token_id: Optional[int] = None,
    mm_pad_shift: int = 0,
    block_t: int = 32,
) -> torch.Tensor:
    """One decode step: hash ids [T, L, (n - 1) * heads] for the T = bs tokens, and
    ``history[req_slots[t]]`` advanced in place to the token and its n - 2 newest
    predecessors (oldest first). Rows whose ``out_cache_loc`` is 0 are CUDA-graph
    padding and leave the table untouched.
    """
    out, _ = _launch_hash_kernel(
        input_ids,
        positions,
        mode=MODE_DECODE,
        history=history,
        token_map=token_map,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        pad_id=pad_id,
        num_real=None,
        req_slots=req_slots,
        block=1,
        row=None,
        starts=None,
        image_token_id=image_token_id,
        mm_pad_shift=mm_pad_shift,
        out_cache_loc=out_cache_loc,
        write_tokens=False,
        block_t=block_t,
    )
    return out
