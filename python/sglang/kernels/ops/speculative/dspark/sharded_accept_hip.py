"""
DSpark chain rejection sampling over draft logits held as TP vocab shards.

The rule is ``AcceptSampling``'s without block verification: draft token x_i is
accepted when u_i * q_i(x_i) < p_i(x_i), p and q being the target's and the
draft's temperature softmax; the first rejection draws from max(p_i - q_i, 0),
and a fully accepted chain from p at its last row. Each rank holds only its
vocab slice of the draft logits, so

  1. row stats  each rank reduces every verify row over its slice to the max
                and sum-exp of the target and draft logits, plus both logits of
                the row's draft token when its slice holds it; the ranks
                exchange these.
  2. accept     every rank merges the same stats into p_i(x_i) and q_i(x_i) and
                tests the chain with the same Philox uniforms, so all reach one
                decision; it caps at the verify length and picks the bonus row.
  3. bonus      a Gumbel race over that row's residual (or p) on each slice,
                whose block candidates the ranks exchange as a draft step does;
                the winner is an exact draw from the normalized residual.

Neither p nor q is materialized: outcomes match AcceptSampling in distribution,
not draw for draw.
"""

from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.ops.speculative.dspark.markov_sample import (
    _markov_sample_finish_kernel,
)

_BLOCK = 4096
_PHILOX_ROUNDS = 10
# 2**-32: maps a 32-bit Philox word to a uniform in (0, 1].
_UNIFORM_SCALE = tl.constexpr(2.3283064365386963e-10)
# The finish kernel's index of a block without a candidate.
_NO_INDEX = tl.constexpr(2147483647)
# Six stats a row, padded to 32 bytes for the vectorized gathers.
_STATS = 8
# The draft block already advanced the noise counter, and draft steps count up
# from 0, so draws at these steps repeat none of the draft's.
_ACCEPT_STEP = 0xFFFFFFFE
_BONUS_STEP = 0xFFFFFFFF


@triton.jit
def _fold_logsumexp(m, s, x):
    """Fold block ``x`` into the running (max, sum of exp(v - max))."""
    new_m = tl.maximum(m, tl.max(x, 0))
    ref = tl.where(new_m == -float("inf"), 0.0, new_m)
    return new_m, s * tl.exp(m - ref) + tl.sum(tl.exp(x - ref), 0)


@triton.jit
def _merge_logsumexp(m, s):
    """Merge the ranks' (max, sum-exp) pairs along axis 1."""
    big = tl.max(m, 1)
    ref = tl.where(big == -float("inf"), 0.0, big)
    return big, tl.sum(s * tl.exp(m - ref[:, None]), 1)


@triton.jit
def _row_stats_kernel(
    TARGET,
    DRAFT,
    CAND,
    TARGET_T,
    DRAFT_T,
    STATS,
    TARGET_S: tl.constexpr,
    DRAFT_S: tl.constexpr,
    CAND_S: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB_START: tl.constexpr,
    SLOTS: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """STATS[r] = (target max, target sum-exp, target logit of x, then the same
    three for the draft) of verify row r over this slice, logits / T; the
    logits of x are -inf unless the slice holds the row's draft token x."""
    r = tl.program_id(0)
    b, i = r // SLOTS, r % SLOTS
    has_draft = i < SLOTS - 1
    inv_tt = 1.0 / tl.maximum(tl.load(TARGET_T + b).to(tl.float32), 1e-5)
    inv_dt = 1.0 / tl.maximum(tl.load(DRAFT_T + b).to(tl.float32), 1e-5)
    target = TARGET + r.to(tl.int64) * TARGET_S + VOCAB_START
    draft = DRAFT + (b * GAMMA + i).to(tl.int64) * DRAFT_S
    t_max, t_sum = -float("inf"), 0.0
    d_max, d_sum = -float("inf"), 0.0
    for start in range(0, WIDTH, BLOCK):
        j = start + tl.arange(0, BLOCK)
        valid = j < WIDTH
        t = tl.load(target + j, valid, -float("inf")).to(tl.float32) * inv_tt
        t_max, t_sum = _fold_logsumexp(t_max, t_sum, t)
        d = tl.load(draft + j, valid & has_draft, -float("inf")).to(tl.float32)
        d_max, d_sum = _fold_logsumexp(d_max, d_sum, d * inv_dt)
    local = tl.load(CAND + b * CAND_S + i + 1, has_draft, -1).to(tl.int64)
    local -= VOCAB_START
    own = has_draft & (local >= 0) & (local < WIDTH)
    local = tl.where(own, local, 0)
    t_x = tl.load(target + local, own, -float("inf")).to(tl.float32) * inv_tt
    d_x = tl.load(draft + local, own, -float("inf")).to(tl.float32) * inv_dt
    out = STATS + r * 8
    tl.store(out, t_max)
    tl.store(out + 1, t_sum)
    tl.store(out + 2, t_x)
    tl.store(out + 3, d_max)
    tl.store(out + 4, d_sum)
    tl.store(out + 5, d_x)


@triton.jit
def _accept_kernel(
    STATS,
    CAND,
    VERIFY_LENS,
    OFFSET,
    NORM,
    CORRECT,
    TRIM,
    FIXED,
    BONUS_ROW,
    CAND_S,
    ROWS,
    KEY,
    SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_P2: tl.constexpr,
    STEP: tl.constexpr,
    HAS_CUTOFF: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    """Per request: merge the ranks' stats (keeping each row's normalizers in
    NORM), run the chain test, cap at the verify length, and record the row the
    bonus is drawn from: the first rejected row, or the last."""
    b = tl.program_id(0)
    i = tl.arange(0, SLOTS_P2)
    w = tl.arange(0, WORLD_P2)
    in_chain = i < SLOTS
    mask = in_chain[:, None] & (w < WORLD)[None, :]
    stats = STATS + (w[None, :] * ROWS + b * SLOTS + i[:, None]) * 8
    t_max, t_sum = _merge_logsumexp(
        tl.load(stats, mask, -float("inf")), tl.load(stats + 1, mask, 0.0)
    )
    d_max, d_sum = _merge_logsumexp(
        tl.load(stats + 3, mask, -float("inf")), tl.load(stats + 4, mask, 0.0)
    )
    norm = NORM + (b * SLOTS + i) * 4
    tl.store(norm, t_max, in_chain)
    tl.store(norm + 1, t_sum, in_chain)
    tl.store(norm + 2, d_max, in_chain)
    tl.store(norm + 3, d_sum, in_chain)
    t_x = tl.max(tl.load(stats + 2, mask, -float("inf")), 1)
    d_x = tl.max(tl.load(stats + 5, mask, -float("inf")), 1)
    p = tl.where(t_x == -float("inf"), 0.0, tl.exp(t_x - t_max) / t_sum)
    q = tl.where(d_x == -float("inf"), 0.0, tl.exp(d_x - d_max) / d_sum)
    lane = tl.zeros([SLOTS_P2], tl.int64)
    bits, _, _, _ = tl.philox(
        KEY,
        i.to(tl.uint32),
        (lane + b).to(tl.uint32),
        (lane + STEP).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        PHILOX_ROUNDS,
    )
    u = tl.uint_to_uniform_float(bits)
    ok = (q > 0.0) & (q <= 1.0) & (u * q < p)
    accepted = tl.min(tl.where((i < SLOTS - 1) & ~ok, i, SLOTS - 1), 0)
    capped = accepted
    fixed = tl.full((), -1, tl.int64)
    if HAS_CUTOFF:
        capped = tl.minimum(accepted, tl.load(VERIFY_LENS + b).to(tl.int32) - 1)
        # A trimmed chain ends on an accepted draft token, not on a draw.
        trimmed = capped < accepted
        fixed = tl.load(CAND + b * CAND_S + capped + 1, trimmed, -1).to(tl.int64)
    tl.store(CORRECT + b, capped)
    tl.store(TRIM + b, accepted - capped)
    tl.store(FIXED + b, fixed)
    tl.store(BONUS_ROW + b, accepted)


@triton.jit
def _bonus_partial_kernel(
    TARGET,
    DRAFT,
    TARGET_T,
    DRAFT_T,
    NORM,
    BONUS_ROW,
    OFFSET,
    P,
    PARTS,
    KEY,
    TARGET_S: tl.constexpr,
    DRAFT_S: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB_START: tl.constexpr,
    SLOTS: tl.constexpr,
    GAMMA: tl.constexpr,
    STEP: tl.constexpr,
    BLOCK: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    """This slice's block candidates for the bonus: a Gumbel race over
    max(p - q, 0) on a rejected row, over p on the last row."""
    b, part = tl.program_id(0), tl.program_id(1)
    i = tl.load(BONUS_ROW + b)
    r = b * SLOTS + i
    residual = i < SLOTS - 1
    j = part * BLOCK + tl.arange(0, BLOCK)
    valid = j < WIDTH
    norm = NORM + r * 4
    inv_tt = 1.0 / tl.maximum(tl.load(TARGET_T + b).to(tl.float32), 1e-5)
    t_ptr = TARGET + r.to(tl.int64) * TARGET_S + VOCAB_START + j
    t = tl.load(t_ptr, valid, -float("inf")).to(tl.float32)
    p = tl.exp(t * inv_tt - tl.load(norm)) / tl.load(norm + 1)
    inv_dt = 1.0 / tl.maximum(tl.load(DRAFT_T + b).to(tl.float32), 1e-5)
    d_ptr = DRAFT + (b * GAMMA + i).to(tl.int64) * DRAFT_S + j
    d = tl.load(d_ptr, valid & residual, -float("inf")).to(tl.float32) * inv_dt
    q = tl.exp(d - tl.load(norm + 2)) / tl.load(norm + 3)
    q = tl.where((q >= 0.0) & (q <= 1.0), q, 0.0)
    mass = tl.where(residual, tl.maximum(p - q, 0.0), p)
    # One Philox call yields four words; word k goes to token 4 * q + k.
    quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
    lane = tl.zeros([BLOCK // 4], tl.int64)
    w0, w1, w2, w3 = tl.philox(
        KEY,
        quad.to(tl.uint32),
        (lane + b).to(tl.uint32),
        (lane + STEP).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
        PHILOX_ROUNDS,
    )
    bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    exp_noise = -libdevice.log1p(-((bits.to(tl.float32) + 0.5) * _UNIFORM_SCALE))
    live = valid & (mass > 0.0)
    key = tl.where(live, tl.log(mass) - tl.log(exp_noise), -float("inf"))
    top = tl.max(key, 0)
    idx = tl.min(tl.where(valid & (key == top), j + VOCAB_START, _NO_INDEX), 0)
    tl.store(P + (b * PARTS + part) * 2, top)
    tl.store(P + (b * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


def accept_sampling_sharded(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    target_temperatures: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_temperatures: torch.Tensor,
    vocab_start: int,
    width: int,
    gather_stacked: Callable[[torch.Tensor], torch.Tensor],
    key: int,
    offset: torch.Tensor,
    verify_lens: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (correct_len int32, bonus int64, cap_trim_lens int32), each [bs].

    candidates     [bs, slots]: the anchor, then the draft tokens
    target_logits  [bs * slots, vocab], whole on every rank
    draft_logits   [bs, gamma, padded]: this rank's vocab columns
                   [vocab_start, vocab_start + width), then padding
    key, offset    the draft's Philox key and its block counter
    """
    bs, slots = candidates.shape
    _, gamma, padded = draft_logits.shape
    assert slots - 1 <= gamma
    assert target_logits.shape[0] == bs * slots and target_logits.stride(1) == 1
    assert draft_logits.is_contiguous() and candidates.stride(1) == 1
    # Philox words go to tokens in quads of global ids.
    assert vocab_start % 4 == 0, vocab_start
    rows = bs * slots
    device = candidates.device

    stats = torch.empty((rows, _STATS), dtype=torch.float32, device=device)
    _row_stats_kernel[(rows,)](
        target_logits,
        draft_logits,
        candidates,
        target_temperatures,
        draft_temperatures,
        stats,
        target_logits.stride(0),
        draft_logits.stride(1),
        candidates.stride(0),
        width,
        vocab_start,
        slots,
        gamma,
        _BLOCK,
        num_warps=8,
    )
    stats = gather_stacked(stats)
    world = stats.shape[0] // rows

    norm = torch.empty((rows, 4), dtype=torch.float32, device=device)
    correct_len = torch.empty((bs,), dtype=torch.int32, device=device)
    cap_trim_lens = torch.empty_like(correct_len)
    fixed_bonus = torch.empty((bs,), dtype=torch.int64, device=device)
    bonus_row = torch.empty_like(correct_len)
    _accept_kernel[(bs,)](
        stats,
        candidates,
        correct_len if verify_lens is None else verify_lens,
        offset,
        norm,
        correct_len,
        cap_trim_lens,
        fixed_bonus,
        bonus_row,
        candidates.stride(0),
        rows,
        key,
        slots,
        triton.next_power_of_2(slots),
        world,
        triton.next_power_of_2(world),
        _ACCEPT_STEP,
        verify_lens is not None,
        _PHILOX_ROUNDS,
        num_warps=1,
    )

    parts = triton.cdiv(padded, _BLOCK)
    partial = torch.empty((bs, parts, 2), dtype=torch.float32, device=device)
    _bonus_partial_kernel[(bs, parts)](
        target_logits,
        draft_logits,
        target_temperatures,
        draft_temperatures,
        norm,
        bonus_row,
        offset,
        partial,
        parts,
        key,
        target_logits.stride(0),
        draft_logits.stride(1),
        width,
        vocab_start,
        slots,
        gamma,
        _BONUS_STEP,
        _BLOCK,
        _PHILOX_ROUNDS,
        num_warps=8,
    )
    gathered = gather_stacked(partial.view(bs, parts * 2))
    drawn = torch.empty((bs,), dtype=torch.int64, device=device)
    _markov_sample_finish_kernel[(bs,)](
        gathered,
        drawn,
        bs,
        parts,
        world,
        triton.next_power_of_2(parts * world),
        num_warps=4,
    )
    return correct_len, torch.where(fixed_bonus >= 0, fixed_bonus, drawn), cap_trim_lens
