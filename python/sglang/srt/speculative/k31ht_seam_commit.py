# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""K31HT_SEAM_COMMIT_FUSE (default off): the sampling accept's rank-0 broadcast and the committed-block build in three
launches instead of about eight.

Served path (knob off), eager post-verify, per decode round:
SpecTpSync broadcast(accept_len) and broadcast(bonus) (two NCCL calls on the attn-TP group), then _commit_accept: out_tokens[:, :-1] = candidates[:, 1:], out_tokens[:, -1] = 0, accept_len to
int64, scatter of the bonus at column accept_len, commit_lens = accept_len (int32) + 1.

Knob on: one pack kernel writes [accept_len; bonus] as int64 into a [2, bs] buffer, ONE broadcast of that buffer from
rank 0 (only when the site is enabled, as served), and one commit kernel builds out_tokens (int64 [bs, block]),
commit_lens (int32), accept_len (int32) and bonus (int64) from the broadcast values. Integer moves only: the outputs are
bitwise those of the served path by construction. Valid accept_len is 0..block-1, as served.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

KNOB = "K31HT_SEAM_COMMIT_FUSE"


def k31ht_seam_commit_fuse_requested() -> bool:
    """Fail-closed parse: unset / '' / '0' = off, '1' = on, anything else raises."""
    v = os.environ.get(KNOB, "")
    if v in ("", "0"):
        return False
    if v == "1":
        return True
    raise ValueError(f"{KNOB} must be unset, '', '0' or '1', got {v!r}")


@triton.jit
def _k31ht_pack_accept_bonus_kernel(acc_ptr, bonus_ptr, packed_ptr, bs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < bs
    a = tl.load(acc_ptr + offs, mask=m, other=0).to(tl.int64)
    b = tl.load(bonus_ptr + offs, mask=m, other=0).to(tl.int64)
    tl.store(packed_ptr + offs, a, mask=m)
    tl.store(packed_ptr + bs + offs, b, mask=m)


@triton.jit
def _k31ht_commit_block_kernel(
    cand_ptr,
    cand_s0,
    cand_s1,
    packed_ptr,
    bs,
    out_tok_ptr,
    commit_ptr,
    acc32_ptr,
    bonus_ptr,
    NCOL: tl.constexpr,
    NCOL_P2: tl.constexpr,
):
    r = tl.program_id(0)
    a = tl.load(packed_ptr + r)
    b = tl.load(packed_ptr + bs + r)
    j = tl.arange(0, NCOL_P2)
    # out[j] = candidates[j + 1] for j < NCOL - 1, 0 at j = NCOL - 1; then the bonus at column a.
    src = tl.load(cand_ptr + r * cand_s0 + (j + 1) * cand_s1, mask=j < NCOL - 1, other=0).to(tl.int64)
    val = tl.where(j == a, b, src)
    tl.store(out_tok_ptr + r * NCOL + j, val, mask=j < NCOL)
    tl.store(commit_ptr + r, (a + 1).to(tl.int32))
    tl.store(acc32_ptr + r, a.to(tl.int32))
    tl.store(bonus_ptr + r, b)


def k31ht_fused_sync_and_commit(
    candidates: torch.Tensor,
    accept_len: torch.Tensor,
    bonus: torch.Tensor,
    tp_sync: Any,
    site: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(accept_len int32, bonus int64, out_tokens int64 [bs, block], commit_lens int32), bitwise those of
    tp_sync.sync(site, accept_len); tp_sync.sync(site, bonus); _commit_accept(candidates, accept_len, bonus)."""
    bs, ncol = int(candidates.shape[0]), int(candidates.shape[1])
    dev = candidates.device
    packed = torch.empty((2, bs), dtype=torch.int64, device=dev)
    if bs > 0:
        _k31ht_pack_accept_bonus_kernel[(triton.cdiv(bs, 256),)](accept_len, bonus, packed, bs, BLOCK=256)
    tp_sync.sync(site, packed)
    out_tokens = torch.empty((bs, ncol), dtype=torch.int64, device=dev)
    commit_lens = torch.empty((bs,), dtype=torch.int32, device=dev)
    acc32 = torch.empty((bs,), dtype=torch.int32, device=dev)
    bonus64 = torch.empty((bs,), dtype=torch.int64, device=dev)
    if bs > 0:
        _k31ht_commit_block_kernel[(bs,)](
            candidates,
            candidates.stride(0),
            candidates.stride(1),
            packed,
            bs,
            out_tokens,
            commit_lens,
            acc32,
            bonus64,
            NCOL=ncol,
            NCOL_P2=triton.next_power_of_2(ncol),
        )
    return acc32, bonus64, out_tokens, commit_lens


_logged = False


def k31ht_seam_commit_fuse_engaged() -> bool:
    """The knob, read once per call site; logs the engagement line the first time it is on."""
    global _logged
    on = k31ht_seam_commit_fuse_requested()
    if on and not _logged:
        _logged = True
        logger.info(
            "%s ENGAGED: sampling accept broadcast packed into one call and the committed block built by one kernel",
            KNOB,
        )
    return on
