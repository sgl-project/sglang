# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shapes the mono kernels are compiled for: one MiniMax-M3 TP4 rank.

The host op validates every tensor it is given against these values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# the shared execution model: one CTA per MI355X CU, every CTA co-resident
from sglang.kernels.ops.mono.gfx950.common.plan.execution import (  # noqa: F401
    BLOCKS,
    THREADS,
    WAVES,
)

HIDDEN = 6144
HEAD_DIM = 128
ROTARY_DIM = 64

# The model's totals, and this rank's share. TP defaults to 4, so a build with
# SGLANG_MONO_TP unset is bit-identical to the TP4-only original: at TP = 4
# LOCAL_KV_HEADS is 1, so `2 * LOCAL_KV_HEADS + 1` is the original `3`, and
# `2 * LOCAL_KV_HEADS` in the offsets below is the original `2`.
#
# This is read at import because the constants are bound by value in about
# nine modules; there is no later point at which changing TP would propagate.
TOTAL_Q_HEADS = 64
TOTAL_KV_HEADS = 4
TOTAL_INDEX_HEADS = 4
TOTAL_INTER = 3072

# RFC #43949 decision 1: "There are no other user knobs and no per-model env
# vars; test hooks use the SGLANG_TEST_MONO_ prefix." So this is a TEST HOOK
# only, for GPU-free compiles of a TP this process is not running at. The
# production mechanism the RFC specifies is the spec's
# `devices: Mapping[str, DeviceLimits]  # per ISA: widths, TP sizes`, read at
# attach() -- not an environment variable. Wiring that requires these constants
# to stop being module-level (they are bound by value across ~9 modules), which
# is tracked separately.
TP = int(os.environ.get("SGLANG_TEST_MONO_TP", "4"))
LOCAL_Q_HEADS = TOTAL_Q_HEADS // TP
LOCAL_KV_HEADS = max(1, TOTAL_KV_HEADS // TP)
LOCAL_INDEX_HEADS = max(1, TOTAL_INDEX_HEADS // TP)


def qkv_rows(idx_heads: int) -> int:
    """Rows of a rank's fused projection, q | k | v | index_q | index_k:
    ``LOCAL_Q_HEADS`` q heads, ``LOCAL_KV_HEADS`` k and v heads, ``idx_heads``
    index q heads (the rank's own, or every one under indexer context
    parallelism) and the one replicated index k head."""
    return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + 1 + idx_heads) * HEAD_DIM


O_K = LOCAL_Q_HEADS * HEAD_DIM

N_ROUTED = 128
TOP_K = 4
SHARED_EXPERT = N_ROUTED  # the fused shared expert's id
MOE_SLOTS = TOP_K + 1
INTER = TOTAL_INTER // TP  # expert intermediate per rank (768 at TP4)
# aiter's e8m0_shuffle pads an MXFP4 scale's columns up to a multiple of 8
# (measured over 12 column counts, tools/probe_e8m0_rule.py: 12 -> 16, 24 -> 24,
# 48 -> 48). The w2 scale is [HIDDEN, INTER // 32], so at TP8 its 12 columns
# become 16 and the stored tensor is wider than INTER // 32 implies. Anything
# indexing that tensor must use the padded stride, not INTER // 32.
SCALE_COLS = INTER // 32
PADDED_SCALE_COLS = -(-SCALE_COLS // 8) * 8

SPARSE_BLOCK = 128
TOPK_BLOCKS = 16
MAX_SPARSE_KEYS = SPARSE_BLOCK * TOPK_BLOCKS
# the longest context served: the score region spans it
MAX_CONTEXT = 1 << 20
MAX_INDEX_BLOCKS = MAX_CONTEXT // SPARSE_BLOCK
PAGE16 = 16

MAX_TOKENS = 16  # tokens one mono step serves (the MFMA B operand holds 16)

# The widest step the 160 KB LDS budget admits at this TP. The budget scales
# with H (TOTAL_Q_HEADS // TP) and INTER (TOTAL_INTER // TP), so a *smaller* TP
# buys *fewer* tokens per step: TP2 carries 32 q heads and INTER 1536 against
# TP4's 16 and 768.
#
# TP2 measured by COMPILE_ONLY builds of the own-mode kernel (ih2), and
# re-measured after the pool was resized by INTER: S = 12, 13 and 14 compile
# clean, S = 15 reports local memory 172,960 and S = 16 182,304, against the
# 163,840 limit. The numbers did not move, because every width from WIDE_FROM up
# takes the expert-grouped pool, which that fix did not touch.
#
# The cap is 12 rather than 14 because 12 is the widest step a GPU run has
# actually checked (mono-check, widths 1-4 and 12); 13 and 14 are known only to
# compile. Raising it is a measurement away, not a code change.
#
# Do NOT take these from tools/lds_budget_pr44319.py: that mirror models
# count = 1 and is low by 256 B per token in own mode, where qs collapses to a
# single row and K1's index q moves to the pool. It predicted S = 14 to
# overflow; the compiler says it fits.
#
# A step wider than this routes to the modules (srt/layers/mono/routing.py),
# which is the existing supported fallback rather than a failure. The LDS
# overflow it avoids is a NON-RAISING compiler diagnostic, so an uncapped TP2
# build would silently produce over-budget kernels.
_LDS_MAX_WIDTH = {2: 12}.get(TP, MAX_TOKENS)
# Test hook (RFC #43949 decision 1 permits the SGLANG_TEST_MONO_ prefix), not a
# user knob: every declared width is JIT-built during eager warm-up, and one
# build takes minutes, so a GPU correctness check that only needs a couple of
# widths would otherwise spend an hour compiling ones it never exercises. It can
# only narrow the range -- it is clamped to the LDS-safe cap above.
MAX_WIDTH = min(
    int(os.environ.get("SGLANG_TEST_MONO_MAX_WIDTH", _LDS_MAX_WIDTH)),
    _LDS_MAX_WIDTH,
)
assert MAX_WIDTH >= 1, "SGLANG_TEST_MONO_MAX_WIDTH must leave at least one width"

# The bottom of the range, for the same reason MAX_WIDTH caps the top: a step
# outside it routes to the modules, which is the supported fallback.
#
# TP8's NARROW MoE path is broken. Measured 2026-10-16 on an 8-rank decode
# graph capture: widths 16, 12 and 8 captured on mono cleanly, then width 4
# took `Memory access fault ... on address (nil) / 0x1000 / 0x2000` on all
# eight ranks at once. Those are unpopulated pointers, not an LDS overflow, so
# the compile gate cannot see it -- every TP8 width builds clean.
#
# 4 is the first width below ``layout.WIDE_FROM`` (5 at TP8, 3 at TP2), i.e.
# the first step that takes the narrow path, and the narrow path is the one
# whose sizing has already needed TP-specific repair once (it was sized by
# HIDDEN rather than INTER, which broke TP2 at widths 3-4). Treat this as the
# same class, unfixed at TP8.
#
# TP2 and TP4 are NOT capped: both have run every width from 1 in anger.
_NARROW_BROKEN_AT_TP = {8: 5}
_MIN_WIDTH = _NARROW_BROKEN_AT_TP.get(TP, 1)
# Unlike MAX_WIDTH's hook this one may also LOWER the floor, so the fix can be
# exercised without a rebuild. Lowering it at TP8 re-enables a fault that takes
# down every rank on the node -- do it only on a node you have to yourself.
MIN_WIDTH = int(os.environ.get("SGLANG_TEST_MONO_MIN_WIDTH", _MIN_WIDTH))
assert 1 <= MIN_WIDTH <= MAX_WIDTH, (
    f"MIN_WIDTH {MIN_WIDTH} must be in [1, MAX_WIDTH={MAX_WIDTH}]"
)
# The widest fused projection any build asks for. ``IndexHeads.__post_init__``
# pins count to TOTAL_INDEX_HEADS under CP and LOCAL_INDEX_HEADS otherwise, and
# LOCAL is never the larger, so CP's every-index-head rows are the bound.
#
# This was ``qkv_rows(TP)``, reading "a rank computes every index q head" -- true
# only because TOTAL_INDEX_HEADS happens to be 4: qkv_rows is injective, so
# qkv_rows(TP) and qkv_rows(TOTAL_INDEX_HEADS) coincide at TP4 and nowhere else.
# It is too small at TP1 (9472 against 9856) and at TP2 for a CP build (4992
# against 5248). TP4 is byte-identical either way, so this cannot move the
# working build.
MAX_QKV_ROWS = qkv_rows(TOTAL_INDEX_HEADS)
# indexer context parallelism serves a request past this many index blocks: its
# selection costs a fixed overhead that the one-head path's O(n^2) ranking only
# exceeds beyond here
INDEX_CP_FROM_BLOCKS = 288


@dataclass(frozen=True)
class IndexHeads:
    """The index q heads in a rank's fused projection, a build parameter.

    Two modes, which must not be inferred from ``count`` -- at TP2 they have
    the same count and different meanings:

    * **own** (``cp=False``): the rank's own ``LOCAL_INDEX_HEADS`` heads, each
      scoring every block for the KV head it pairs with. The model gives a rank
      as many index heads as KV heads (``num_idx_heads == num_kv_heads``,
      probed: 1/1 at TP4, 2/2 at TP2), so index head h selects for KV head h
      and the split stage's pass h reads its blocks.
    * **cp** (``cp=True``): indexer context parallelism -- all
      ``TOTAL_INDEX_HEADS`` in head order, a rank scoring only its own 1/TP
      stripe of blocks (``own``) and exchanging candidates with its peers.

    ``count > 1`` used to select CP on its own, which is why TP2 was refused:
    its own pair is not context parallelism.

    **No caller builds ``cp=True`` today, at any TP.** The host passes
    ``ONE_INDEX_HEAD``; the only other constructions are
    ``IndexHeads(heads.count, 0)`` (pre_attn) and the jit boundary's
    round-trip, both of which carry the flag they were given. So every
    ``heads.cp`` branch in the kernels -- the striped block list, the ``// TP``
    block index and sub-region size, and ``sel_cp``'s peer exchange -- is
    unreachable in this build. They are kept as the upstream author's lineage,
    not because they run: do not infer from their presence that CP is a tested
    path. CP also cannot serve TP2 as written, since its peer exchange indexes
    ``peer_addr(part // 2)`` for ``part < N_SPLIT``, i.e. up to peer 3.
    """

    count: int = LOCAL_INDEX_HEADS
    own: int = 0
    cp: bool = False

    def __post_init__(self):
        want = TOTAL_INDEX_HEADS if self.cp else LOCAL_INDEX_HEADS
        assert self.count == want, (
            f"{'cp' if self.cp else 'own'} mode takes {want} index heads, "
            f"got {self.count}"
        )
        assert 0 <= self.own < max(self.count, 1)
        # pre_attn.py's SCRATCH_X8 is sized from MAX_QKV_ROWS, so a build whose
        # projection is wider than that bound overruns the norm tasks' x8 rows
        # silently.
        assert self.rows <= MAX_QKV_ROWS, (
            f"{self.count} index heads need {self.rows} projection rows, past "
            f"the MAX_QKV_ROWS bound of {MAX_QKV_ROWS}"
        )

    @property
    def rows(self) -> int:
        return qkv_rows(self.count)

    @property
    def score_blocks(self) -> int:
        """Blocks one head's score region holds. Under CP a rank scores only
        its 1/TP stripe; in own mode a head scores every block."""
        return MAX_INDEX_BLOCKS // TP if self.cp else MAX_INDEX_BLOCKS

    @property
    def score_words(self) -> int:
        """A token's whole score region, every head. Equal to MAX_INDEX_BLOCKS
        in both modes wherever a rank has one index head, which is why TP4 and
        TP8 see no change: CP spreads TOTAL heads over 1/TP blocks each."""
        return self.count * self.score_blocks

    @property
    def iq_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.own) * HEAD_DIM

    @property
    def ik_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.count) * HEAD_DIM


# without indexer context parallelism: the rank's own index q head only
ONE_INDEX_HEAD = IndexHeads()


LAYER_SLOTS = 128  # mailbox epochs: step * LAYER_SLOTS + layer + 1
