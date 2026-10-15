"""DSpark rejection sampling with the draft logits on the TP vocab shards.

The rule is ``AcceptSampling``'s without block verification: draft token x is
accepted when u * q(x) < p(x), p and q being the target's and the draft's
temperature softmax; the first rejection draws the bonus from max(p - q, 0), a
fully accepted chain from the last row's p. No rank holds the whole draft vocab:

  1. ``_stats_kernel``: each rank reduces its slice of every verify row to the
     max and sum-exp of the target and draft logits, plus both logits of the
     row's draft token when the slice holds it. The ranks exchange these.
  2. ``_accept_kernel``: every rank merges them into p(x) and q(x) and tests the
     chain with the same Philox uniforms, so all ranks decide alike; then it
     races log(mass) + Gumbel noise over its slice of the bonus row. The ranks
     exchange the block winners, and the draft's finish kernel picks the bonus.

The bonus follows AcceptSampling's distribution, not its draws.
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
# Philox steps the draft never reaches, so these draws repeat none of its own.
_ACCEPT_STEP, _BONUS_STEP = tl.constexpr(0xFFFFFFFE), tl.constexpr(0xFFFFFFFF)


@triton.jit
def _fold(m, s, x):
    """Fold block ``x`` into the running (max, sum of exp(v - max))."""
    new_m = tl.maximum(m, tl.max(x, 0))
    ref = tl.where(new_m == -float("inf"), 0.0, new_m)
    return new_m, s * tl.exp(m - ref) + tl.sum(tl.exp(x - ref), 0)


@triton.jit
def _merge(m, s):
    """Merge the ranks' (max, sum-exp) pairs along axis 1."""
    big = tl.max(m, 1)
    ref = tl.where(big == -float("inf"), 0.0, big)
    return big, tl.sum(s * tl.exp(m - ref[:, None]), 1)


@triton.jit
def _philox(KEY, c0, row, step, OFFSET):
    """Philox words at (c0, row, step, block counter), the draft's counters."""
    lane = tl.zeros(c0.shape, tl.int64)
    return tl.philox(
        KEY,
        c0.to(tl.uint32),
        (lane + row).to(tl.uint32),
        (lane + step).to(tl.uint32),
        ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
    )


@triton.jit
def _stats_kernel(
    TARGET,
    DRAFT,
    CAND,
    TEMP,
    STATS,
    TARGET_S,
    START,
    WIDTH,
    PADDED,
    SLOTS: tl.constexpr,
    GAMMA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """STATS[r] = (max, sum-exp, logit of x) of the target, then the same of the
    draft, over this slice of verify row r at temperature; x is the row's draft
    token, its logits -inf off the slice. The last row has no draft token."""
    r = tl.program_id(0)
    b, i = r // SLOTS, r % SLOTS
    has_x = i < SLOTS - 1
    inv_t = 1.0 / tl.maximum(tl.load(TEMP + b), 1e-5)
    target = TARGET + r.to(tl.int64) * TARGET_S + START
    draft = DRAFT + (b * GAMMA + i).to(tl.int64) * PADDED
    t_m, t_s, d_m, d_s = -float("inf"), 0.0, -float("inf"), 0.0
    for k in range(0, WIDTH, BLOCK):
        j = k + tl.arange(0, BLOCK)
        t = tl.load(target + j, j < WIDTH, -float("inf")).to(tl.float32)
        d = tl.load(draft + j, has_x & (j < WIDTH), -float("inf")).to(tl.float32)
        t_m, t_s = _fold(t_m, t_s, t * inv_t)
        d_m, d_s = _fold(d_m, d_s, d * inv_t)
    x = tl.load(CAND + r + 1, has_x, -1) - START
    own = has_x & (x >= 0) & (x < WIDTH)
    out = STATS + r * 6
    tl.store(out, t_m)
    tl.store(out + 1, t_s)
    tl.store(out + 2, tl.load(target + x, own, -float("inf")).to(tl.float32) * inv_t)
    tl.store(out + 3, d_m)
    tl.store(out + 4, d_s)
    tl.store(out + 5, tl.load(draft + x, own, -float("inf")).to(tl.float32) * inv_t)


@triton.jit
def _accept_kernel(
    STATS,
    TARGET,
    DRAFT,
    CAND,
    TEMP,
    VERIFY_LENS,
    KEY,
    OFFSET,
    LENS,
    PARTIAL,
    BS,
    TARGET_S,
    START,
    WIDTH,
    PADDED,
    PARTS,
    SLOTS: tl.constexpr,
    SLOTS_P2: tl.constexpr,
    GAMMA: tl.constexpr,
    WORLD: tl.constexpr,
    WORLD_P2: tl.constexpr,
    HAS_CUTOFF: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """For request b and vocab block ``part``: merge the ranks' stats and test
    the chain (block 0 stores correct_len, trim and a capped chain's bonus in
    LENS), then race this block of the bonus row, whose mass is max(p - q, 0)
    after a rejection and p after a full chain."""
    b, part = tl.program_id(0), tl.program_id(1)
    slot = tl.arange(0, SLOTS_P2)
    rank = tl.arange(0, WORLD_P2)
    mask = (slot < SLOTS)[:, None] & (rank < WORLD)[None, :]
    stats = STATS + (rank[None, :] * BS * SLOTS + b * SLOTS + slot[:, None]) * 6
    t_m, t_s = _merge(
        tl.load(stats, mask, -float("inf")), tl.load(stats + 1, mask, 0.0)
    )
    d_m, d_s = _merge(
        tl.load(stats + 3, mask, -float("inf")), tl.load(stats + 4, mask, 0.0)
    )
    p_x = tl.exp(tl.max(tl.load(stats + 2, mask, -float("inf")), 1) - t_m) / t_s
    q_x = tl.exp(tl.max(tl.load(stats + 5, mask, -float("inf")), 1) - d_m) / d_s
    u, _, _, _ = _philox(KEY, slot, b, _ACCEPT_STEP, OFFSET)
    ok = (q_x > 0.0) & (tl.uint_to_uniform_float(u) * q_x < p_x)
    accepted = tl.min(tl.where((slot < SLOTS - 1) & ~ok, slot, SLOTS - 1), 0)
    capped = accepted
    if HAS_CUTOFF:
        capped = tl.minimum(accepted, tl.load(VERIFY_LENS + b).to(tl.int32) - 1)
    if part == 0:
        # A trimmed chain ends on its accepted draft token, not on a draw.
        fixed = tl.load(CAND + b * SLOTS + capped + 1, capped < accepted, -1)
        tl.store(LENS + b, capped)
        tl.store(LENS + BS + b, accepted - capped)
        tl.store(LENS + 2 * BS + b, fixed.to(tl.int32))

    at = slot == accepted
    t_m, t_s = tl.sum(tl.where(at, t_m, 0.0), 0), tl.sum(tl.where(at, t_s, 0.0), 0)
    d_m, d_s = tl.sum(tl.where(at, d_m, 0.0), 0), tl.sum(tl.where(at, d_s, 0.0), 0)
    residual = accepted < SLOTS - 1
    j = part * BLOCK + tl.arange(0, BLOCK)
    valid = j < WIDTH
    inv_t = 1.0 / tl.maximum(tl.load(TEMP + b), 1e-5)
    t_row = TARGET + (b * SLOTS + accepted).to(tl.int64) * TARGET_S + START
    t = tl.load(t_row + j, valid, -float("inf")).to(tl.float32)
    d_row = DRAFT + (b * GAMMA + accepted).to(tl.int64) * PADDED
    d = tl.load(d_row + j, valid & residual, -float("inf")).to(tl.float32)
    p = tl.exp(t * inv_t - t_m) / t_s
    q = tl.exp(d * inv_t - d_m) / d_s
    mass = tl.where(residual, tl.maximum(p - q, 0.0), p)
    # Word k of a Philox call goes to token 4 * quad + k, keyed by global ids.
    quad = (part * BLOCK + START) // 4 + tl.arange(0, BLOCK // 4)
    w0, w1, w2, w3 = _philox(KEY, quad, b, _BONUS_STEP, OFFSET)
    bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
    # Uniform in (0, 1]: a zero word would give zero noise and win outright.
    noise = -libdevice.log1p(-(bits.to(tl.float32) + 0.5) * 2.3283064365386963e-10)
    key = tl.where(valid & (mass > 0.0), tl.log(mass) - tl.log(noise), -float("inf"))
    top = tl.max(key, 0)
    # 0x7FFFFFFF is the finish kernel's "no candidate", for an all-padding block.
    idx = tl.min(tl.where(valid & (key == top), j + START, 0x7FFFFFFF), 0)
    tl.store(PARTIAL + (b * PARTS + part) * 2, top)
    tl.store(PARTIAL + (b * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


def accept_sampling_sharded(
    *,
    candidates: torch.Tensor,
    target_logits: torch.Tensor,
    draft_logits: torch.Tensor,
    temperatures: torch.Tensor,
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
    temperatures   [bs] float32 sampling temperatures
    key, offset    the draft's Philox key and its block counter
    """
    bs, slots = candidates.shape
    _, gamma, padded = draft_logits.shape
    assert slots - 1 <= gamma and target_logits.stride(1) == 1
    assert candidates.is_contiguous() and draft_logits.is_contiguous()
    # Philox words go to tokens in quads of global ids.
    assert vocab_start % 4 == 0, vocab_start
    device = candidates.device
    parts = triton.cdiv(padded, _BLOCK)

    stats = torch.empty((bs * slots, 6), dtype=torch.float32, device=device)
    _stats_kernel[(bs * slots,)](
        target_logits,
        draft_logits,
        candidates,
        temperatures,
        stats,
        target_logits.stride(0),
        vocab_start,
        width,
        padded,
        slots,
        gamma,
        _BLOCK,
        num_warps=8,
    )
    stats = gather_stacked(stats)
    world = stats.shape[0] // (bs * slots)

    lens = torch.empty((3, bs), dtype=torch.int32, device=device)
    partial = torch.empty((bs, parts * 2), dtype=torch.float32, device=device)
    _accept_kernel[(bs, parts)](
        stats,
        target_logits,
        draft_logits,
        candidates,
        temperatures,
        lens if verify_lens is None else verify_lens,  # unread without a cutoff
        key,
        offset,
        lens,
        partial,
        bs,
        target_logits.stride(0),
        vocab_start,
        width,
        padded,
        parts,
        slots,
        triton.next_power_of_2(slots),
        gamma,
        world,
        triton.next_power_of_2(world),
        verify_lens is not None,
        _BLOCK,
        num_warps=8,
    )
    gathered = gather_stacked(partial)
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
    correct_len, cap_trim_lens, fixed = lens
    return correct_len, torch.where(fixed >= 0, fixed.long(), drawn), cap_trim_lens
