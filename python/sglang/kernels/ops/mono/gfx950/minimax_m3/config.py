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
# TP2 measured by COMPILE_ONLY builds of the own-mode kernel (ih2): S = 13 and
# S = 14 compile clean, S = 15 reports local memory 172,960 and S = 16 182,304,
# against the 163,840 limit. The cap is 12, well inside that, for margin and
# because the MoE down stage -- the second overflow carrier, which scales with
# INTER rather than H -- stays out of the binding budget there.
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
# indexer context parallelism: a rank computes every index q head
MAX_QKV_ROWS = qkv_rows(TP)
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
