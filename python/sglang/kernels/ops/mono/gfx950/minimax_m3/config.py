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
# indexer context parallelism: a rank computes every index q head
MAX_QKV_ROWS = qkv_rows(TP)
# indexer context parallelism serves a request past this many index blocks: its
# selection costs a fixed overhead that the one-head path's O(n^2) ranking only
# exceeds beyond here
INDEX_CP_FROM_BLOCKS = 288


@dataclass(frozen=True)
class IndexHeads:
    """The index q heads in a rank's fused projection, a build parameter:
    ``count`` of them (``LOCAL_INDEX_HEADS``, or all TP in head order under
    indexer context parallelism) and ``own``, the one this rank's selection
    scores.

    WARNING -- unresolved at TP != 4. ``count > 1`` is currently also the
    switch that selects the indexer-context-parallel path
    (``index_score.py:107,154``, ``post_attn.py:639``), whose meaning is "all
    ranks' heads in head order, rank r scoring head r". At TP2 SGLang instead
    gives a rank its OWN contiguous pair (``idx_head_rank == rank``,
    ``num_idx_heads=2``, probed against the checkpoint). The counts coincide at
    2 and the semantics do not, so ``count=2`` here compiles but does NOT yet
    mean what TP2 needs. Separating "my own heads" from "CP over all heads"
    into an explicit mode is required before any TP2 number is trustworthy.
    """

    count: int = LOCAL_INDEX_HEADS
    own: int = 0

    def __post_init__(self):
        assert self.count in (1, TP, LOCAL_INDEX_HEADS)
        assert 0 <= self.own < max(self.count, 1)

    @property
    def rows(self) -> int:
        return qkv_rows(self.count)

    @property
    def iq_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.own) * HEAD_DIM

    @property
    def ik_off(self) -> int:
        return (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS + self.count) * HEAD_DIM


# without indexer context parallelism: the rank's own index q head only
ONE_INDEX_HEAD = IndexHeads()


LAYER_SLOTS = 128  # mailbox epochs: step * LAYER_SLOTS + layer + 1
