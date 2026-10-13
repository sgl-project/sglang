# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 68506823: atom/models/deepseek_v41/mono/index_plan.py
# (the FP4 plane's parts: no FP8 ``score_tasks``)
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every per-step decision of which mailbox cells the indexer writes and reads,
defined once .

Each function takes a token's ``bound`` -- how many columns its scorer may read:
``step.visible[ratio]``, or the candidate table's context on a REINDEX layer --
and is called both by the kernel (on traced Int32) and by the CPU property test
(on Python ints), so the test checks what the kernel runs. Only non-negative
operands reach ``//`` and ``%``, where Python's floor and the device's
truncation agree; ``i32`` asserts the device's range on the CPU side.
"""

import numpy as np

from sglang.srt.layers.mono.common.plan.arith import sel

CHUNK = 128  # columns a score task: 16 tiles of 8 rows
SEG = 8192  # values a select task ranks: its LDS-resident keys (K2a's LDS budget)
CHUNKS_PER_SEG = SEG // CHUNK
TOPK = 512
TOPK_BLOCKS = 2048
BLOCK_ROWS = 8  # a candidate block, and an index tile (an FP4 plane's page)
STEP = 64  # keys an FP4 walk step: a wave's 4 MFMA tiles, 8 pages
WALK_ROWS = 6  # query rows an FP4 walk scores off each key load
HEADS = 32  # the indexer's heads (``index_n_heads``)
DIM = 128  # an index head's dims (``index_head_dim``)
# the two selections: a token's top rows, and on a candidate-producing layer its
# top blocks -- each a tree over its own bound (``kind_bound``)
KINDS = {"sel": TOPK, "blk": TOPK_BLOCKS}


def imin(a, b):
    return sel(a < b, a, b)


def imax(a, b):
    return sel(a > b, a, b)


def i32(x):
    """``x``, asserted inside the device's Int32 range when it is a Python int."""
    if isinstance(x, int):
        assert -(2**31) <= x < 2**31, x
    return x


def cdiv(a, b):
    """ceil(a / b) for a >= 0, b > 0."""
    return i32((a + b - 1) // b)


def chunks(bound):
    """Chunks of a token's columns: chunk ch covers [CHUNK ch, CHUNK (ch + 1))."""
    return cdiv(imax(bound, 0), CHUNK)


def scored(bound):
    """Does the score stage run for a token? Not while its selection is every
    column it sees (bound <= TOPK: the top-k takes them all, in column order,
    and so does the candidate-block top-k, its blocks <= TOPK_BLOCKS)."""
    return bound > TOPK


def score_chunks(bound):
    """The chunks the score stage scores for a token."""
    return sel(scored(bound), chunks(bound), 0)


def chunk_written(bound, ch):
    """Does the score stage write chunk ch's flag (and logits)?"""
    return scored(bound) & (ch * CHUNK < bound)


def segments(bound):
    """Select segments of a token: segment s covers columns [SEG s, SEG (s + 1))."""
    return cdiv(imax(bound, 0), SEG)


def seg_len(bound, s):
    """Columns in segment s."""
    return imax(imin(bound - s * SEG, SEG), 0)


def seg_chunks(bound, s):
    """The chunk flags segment s polls: [CHUNKS_PER_SEG s, + this)."""
    return sel(scored(bound), cdiv(seg_len(bound, s), CHUNK), 0)


def winners(n, k):
    """Valid entries a top-k of n values emits (the rest are -1)."""
    return imin(imax(n, 0), k)


# The selection is a tree. Level 0: segment s of SEG values -> its top-k list.
# Level l > 0: list j merges the level l - 1 lists [FAN j, FAN (j + 1)), FAN =
# SEG // k, laid end to end: every list but a level's last holds exactly k
# valid entries (its inputs held >= k), so the merged input is dense and in
# index order. A global winner is in its own list's top-k (fewer than k beat it
# there), so each level keeps every winner. The tree ends at a level of one list.


def fan(k):
    return SEG // k


def lists(bound, level, k):
    """Lists at ``level`` (static unroll: callers loop over levels in Python)."""
    n = segments(bound)
    for _ in range(level):
        n = cdiv(n, fan(k))
    return n


def list_inputs(bound, level, j, k):
    """Values list j of ``level`` ranks: a segment's columns at level 0, else the
    valid entries of the lists it merges."""
    if level == 0:
        return seg_len(bound, j)
    below = lists(bound, level - 1, k)
    first = j * fan(k)
    count = imin(below - first, fan(k))
    last = first + count - 1
    return i32(imax(count - 1, 0) * k + list_valid(bound, level - 1, last, k))


def list_valid(bound, level, j, k):
    """Valid entries of list j at ``level``."""
    return winners(list_inputs(bound, level, j, k), k)


def levels(bound_max, k):
    """Merge levels a build needs for bounds up to ``bound_max`` (Python ints):
    the depth at which one list remains."""
    level, n = 0, segments(bound_max)
    while n > 1:
        n = cdiv(n, fan(k))
        level += 1
    return level


def is_final(bound, level, k):
    """Is ``level``'s list the token's answer (one list or none): the selection
    publishes IRDY from it."""
    return lists(bound, level, k) <= 1


def final_level(bound, depth, k):
    """The level whose single list is the token's answer: the first that
    ``is_final`` (a token below the build's maximum ends early)."""
    at = depth
    for level in range(depth, -1, -1):
        at = sel(is_final(bound, level, k), level, at)
    return at


def selected(bound, k=TOPK):
    """Valid ids in the final selection, which the attention reads as its first
    keys (the rest of the ``k`` slots are -1)."""
    return winners(bound, k)


def blocks(bound):
    """Candidate blocks a candidate-producing layer ranks: the blocks holding
    visible rows, the newest pinned."""
    return cdiv(imax(bound, 0), BLOCK_ROWS)


def block_chunks(bound, j):
    """(first chunk, chunk count) whose logits hold the rows of block segment j
    (blocks [SEG j, SEG (j + 1))): the flags that segment's task polls."""
    rows = SEG * BLOCK_ROWS
    first = j * (rows // CHUNK)
    count = cdiv(imax(imin(bound, (j + 1) * rows) - j * rows, 0), CHUNK)
    return first, sel(scored(bound), count, 0)


def flat_total(counts):
    """Tasks of a stage whose token t has ``counts[t]``: laid token after token."""
    total = 0
    for n in counts:
        total = total + n
    return total


def flat_task(counts, task):
    """(token, its index) of flat task ``task`` < ``flat_total(counts)``."""
    # task's own type for the accumulators: a select needs a traced operand
    t, start, running = task * 0, task * 0, 0
    for i, n in enumerate(counts[:-1]):
        running = running + n
        past = task >= running
        t = sel(past, i + 1, t)
        start = sel(past, running, start)
    return t, task - start


def fill_walk_plan(bounds, owners, plan):
    """The FP4 plane's FULL-layer score tasks of a step, on the host: ``plan``
    [ctas, 4] int32 gets a (lead, span, ch0, ch1) a CTA -- chunks [ch0, ch1)
    of the ``span`` tokens from ``lead``, a piece: at most WALK_ROWS adjacent
    tokens of one owner (one page table), scored off each key load -- and
    zeros past the last task. ``bounds`` and ``owners`` are the step's rows
    (int numpy). Every piece gets ceil(need / width) tasks sharing its chunks
    evenly, at the smallest width that fits them all in ``ctas``: a width one
    chunk too wide left a wave per CTA a step behind the rest."""
    n, ctas = bounds.shape[0], plan.shape[0]
    plan.fill(0)
    t = np.arange(n)
    run_start = np.ones(n, dtype=bool)
    run_start[1:] = owners[1:] != owners[:-1]
    run_first = np.maximum.accumulate(np.where(run_start, t, 0))
    lead = np.flatnonzero(run_start | ((t - run_first) % WALK_ROWS == 0))
    span = np.diff(lead, append=n)
    need = np.maximum.reduceat(np.where(scored(bounds), cdiv(bounds, CHUNK), 0), lead)
    live = need > 0
    lead, span, need = lead[live], span[live], need[live]
    if not need.size:
        return
    assert need.size < ctas, (need.size, ctas)
    # the width search's range: ceil(total / ctas), the least that could fit,
    # to ceil(total / (ctas - pieces)), which does (a piece's ceil adds under
    # one task); feasibility is monotone, so the first width that fits
    total = int(need.sum())
    lo = max(cdiv(total, ctas), 1)
    widths = np.arange(lo, max(cdiv(total, ctas - need.size), lo) + 1)[:, None]
    width = lo + int(np.argmax(cdiv(need, widths).sum(1) <= ctas))
    k = cdiv(need, width)
    piece = np.repeat(np.arange(need.size), k)
    i = np.arange(piece.size) - np.repeat(np.cumsum(k) - k, k)
    tasks = plan[: piece.size]
    tasks[:, 0] = lead[piece]
    tasks[:, 1] = span[piece]
    tasks[:, 2] = i * need[piece] // k[piece]
    tasks[:, 3] = (i + 1) * need[piece] // k[piece]


# a REINDEX token's chunks at most: its candidate blocks' rows
REINDEX_CHUNKS = TOPK_BLOCKS * BLOCK_ROWS // CHUNK


def reindex_split(tokens, ctas, least):
    """FP4 plane, a REINDEX layer: tasks a token (each its own piece on its own
    candidate table), REINDEX_CHUNKS shared over them -- fixed by the step's
    width, so a CTA finds its task from its index alone. At least ``least``
    chunks a task."""
    return max(1, min(ctas // tokens, REINDEX_CHUNKS // least))


def reindex_chunks(i, per):
    """Chunks [ch0, ch1) of a REINDEX token's task i of ``per``."""
    return i * REINDEX_CHUNKS // per, (i + 1) * REINDEX_CHUNKS // per


def task_writes(bound, ch0, width, m):
    """Does a score task over chunks [ch0, ch0 + width) write its chunk ch0 + m
    for a token whose scorer reads ``bound`` columns?"""
    return (m < width) & chunk_written(bound, ch0 + m)


def list_count(bound, level, k):
    """Lists a token runs at ``level``: level 0 always one (so a token that sees
    nothing still emits its -1s), a higher level only while the one below had
    more than one."""
    if level == 0:
        return imax(lists(bound, 0, k), 1)
    return sel(lists(bound, level - 1, k) > 1, lists(bound, level, k), 0)


def kind_bound(kind, bound):
    """The values a ``kind`` selection ranks for a token whose scorer reads
    ``bound`` columns: the columns, or the blocks holding them."""
    return bound if kind == "sel" else blocks(bound)


def top_level(kind, bound_max):
    """A ``kind`` selection tree's last level for bounds up to ``bound_max``."""
    return levels(kind_bound(kind, bound_max), KINDS[kind])


def list_slots(kind, level, bound_max):
    """Lists a token can have at ``level`` (a region's per-token size)."""
    return lists(kind_bound(kind, bound_max), level, KINDS[kind])
