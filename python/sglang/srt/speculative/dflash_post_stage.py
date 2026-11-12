# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""DFlash verify-token staging inside the draft graph (SGLANG_OPT_DFLASH_POST_IN_GRAPH).

Without it, five eager launches sit on the main stream between the draft graph and the target verify graph:
the worker builds the verify tokens [anchor, draft tokens] per row (two copies), clones the prefix seq lens for
the mamba / KDA commit, the verify runner's plan copies the tokens into its static input_ids (fill_from) and its
replay copies input_ids and positions again. With it, the draft graph's captured tail writes the tokens once,
into a static buffer the accept reads and into the target verify runner's static input_ids, the worker takes the
seq lens snapshot before the draft launch, and the verify runner skips both copies of rows it already holds.
Every value is a copy of the same source, so outputs are unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _post_stage_kernel(
    anchors,
    anchor_stride,
    tokens,
    out,
    verify_ids,
    B: tl.constexpr,
    BP: tl.constexpr,
    HAS_VERIFY: tl.constexpr,
):
    # One program per row: [anchor, tokens[row, 0 : B - 1]] -> out[row, :] (and verify_ids[row * B : row * B + B]).
    # verify_ids may alias the buffer the anchors come from: each program writes back its own anchor value only.
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BP)
    anchor = tl.load(anchors + row * anchor_stride)
    tok = tl.load(tokens + row * (B - 1) + (col - 1), mask=(col >= 1) & (col < B), other=0)
    val = tl.where(col == 0, anchor, tok)
    tl.store(out + row * B + col, val, mask=col < B)
    if HAS_VERIFY:
        tl.store(verify_ids + row * B + col, val, mask=col < B)


def post_stage_verify_tokens(
    *,
    anchors: torch.Tensor,
    tokens: torch.Tensor,
    out: torch.Tensor,
    verify_ids: Optional[torch.Tensor],
) -> None:
    """anchors [bs] (any stride), tokens [bs, B - 1] contiguous, out [bs, B] contiguous, verify_ids [bs * B] or None;
    all int64. Capture-safe (one launch, no host sync)."""
    bs, gamma = tokens.shape
    block = gamma + 1
    assert anchors.dim() == 1 and anchors.shape[0] == bs, (anchors.shape, bs)
    assert out.shape == (bs, block) and out.is_contiguous(), out.shape
    assert tokens.is_contiguous()
    assert anchors.dtype == tokens.dtype == out.dtype == torch.int64
    if verify_ids is not None:
        assert verify_ids.dtype == torch.int64 and verify_ids.is_contiguous()
        assert verify_ids.numel() == bs * block, (verify_ids.numel(), bs, block)
    if bs == 0:
        return
    _post_stage_kernel[(bs,)](
        anchors,
        anchors.stride(0),
        tokens,
        out,
        verify_ids if verify_ids is not None else out,
        B=block,
        BP=triton.next_power_of_2(block),
        HAS_VERIFY=verify_ids is not None,
        num_warps=1,
    )
