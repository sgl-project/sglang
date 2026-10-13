# SPDX-License-Identifier: MIT
# Vendored from ROCm/ATOM @ 454d0e3023: atom/mono/runtime/step_fence.py
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step's peer fence (``DESIGN_v3`` I1): no rank pushes a step's first
pair into a peer's buffer before that peer has cleared it for the step.

A runner clears its mailboxes between steps, and a peer's push of the next
step must land after the clear. A TP collective between the two was taken for
that fence, and it is not one: aiter's custom all-reduce skips the
communication in a graph capture's warmup forward (``custom_all_reduce``
returns zeros there), and a V4.1 DSpark draft warmup of two requests found a
peer's pushes cleared by the late rank's clear. The fence is its own: after
the clear, each rank writes its step epoch into its slot at every rank's
buffer tail (never cleared, the epoch only grows) and waits until all of its
own slots reach it (``PeerBuffer.fence``).
"""

import torch
import triton
import triton.language as tl

# int64 epoch slots at a peer buffer's tail, one a source rank
FENCE_SLOTS = 8
FENCE_BYTES = FENCE_SLOTS * 8
# a debug build's wait gives up after this many polls (``ATOM_MONO_DEBUG``)
DEBUG_POLLS = 1 << 26


@triton.jit
def _step_fence_kernel(
    epoch, peers, fence_off, rank, missing, TP: tl.constexpr, MAX_POLLS: tl.constexpr
):
    e = tl.load(epoch) + 1
    tl.store(epoch, e)
    for p in tl.static_range(TP):
        tail = tl.load(peers + p) + fence_off
        slot = (tail + rank * 8).to(tl.pointer_type(tl.int64))
        tl.atomic_xchg(slot, e, sem="release", scope="sys")
    own = (tl.load(peers + rank) + fence_off).to(tl.pointer_type(tl.int64))
    for p in tl.static_range(TP):
        seen = tl.atomic_add(own + p, 0, sem="acquire", scope="sys")
        polls = 0
        while (seen < e) & ((MAX_POLLS == 0) | (polls < MAX_POLLS)):
            seen = tl.atomic_add(own + p, 0, sem="acquire", scope="sys")
            polls += 1
        if MAX_POLLS > 0:
            # a debug build: the rank that never arrived (its last epoch seen)
            tl.store(missing + p, seen + 1, mask=seen < e)


def step_fence(
    epoch: torch.Tensor,
    addresses: torch.Tensor,
    fence_off: int,
    rank: int,
    missing: torch.Tensor | None = None,
):
    """This step's fence, after this rank's clear on the same stream:
    ``epoch`` (int64 [1]) this runner's step count, ``addresses`` (int64 [TP])
    every rank's peer buffer, ``fence_off`` its slots' byte offset.
    ``missing`` (int64 [TP], a debug build): bounded waits; a rank that never
    arrived leaves its last seen epoch + 1 there (0: it arrived)."""
    tp = addresses.numel()
    assert tp <= FENCE_SLOTS
    _step_fence_kernel[(1,)](
        epoch,
        addresses,
        fence_off,
        rank,
        # never touched without a debug build: MAX_POLLS 0 compiles its stores out
        epoch if missing is None else missing,
        TP=tp,
        MAX_POLLS=0 if missing is None else DEBUG_POLLS,
        num_warps=1,
    )
