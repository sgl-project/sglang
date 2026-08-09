# Copyright (c) 2026 NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""KeySplit -- training-free block-sparse attention routing for video DiTs.

Vendored from the standalone KeySplit repository (``keysplit/router.py`` and
``keysplit/kernels.py``, unmodified apart from this package wrapper).

The routing score for a 64x64 block splits **only the key block** into ``n_k``
sub-blocks and combines them with a log-sum-exp::

    score(i, j) = log sum_{b < n_k} exp( mean(Q_i) . mean(K_{j,b}) / sqrt(d) )

which estimates the block's un-normalised softmax mass -- exactly how much
attention is lost by skipping it. Splitting the query side instead is measurably
worthless: a block's mass sums over all its query rows, so query differences
average out while key differences decide which key wins.

The plan feeds FlashInfer's ``bsa_attn_blk64_fwd`` (SM100, bf16, head_dim 128).
"""

from .router import (
    BLOCK,
    RoutingPlan,
    SubBlockRouter,
    block_sizes,
    load_bsa_attn_blk64_fwd,
    subblock_sparse_attention,
)

KeySplitRouter = SubBlockRouter
keysplit_attention = subblock_sparse_attention

__all__ = [
    "BLOCK",
    "KeySplitRouter",
    "RoutingPlan",
    "SubBlockRouter",
    "block_sizes",
    "keysplit_attention",
    "load_bsa_attn_blk64_fwd",
    "subblock_sparse_attention",
]
