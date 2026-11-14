# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Lamport mailboxes: BF16 rows published in 16-byte fragments; a reader polls until no
fragment holds the -0.0 sentinel, then re-arms."""

import cutlass
import cutlass.cute as cute
from cutlass import Uint32

from sglang.kernels.jit.kimi_k3.primitives import (
    load_words,
    published,
    store_relaxed,
    wait_deadline_check,
    wait_deadline_start,
)

SENTINEL = 0x80000000


def rearm(pointer):
    store_relaxed(pointer, (Uint32(SENTINEL),) * 4)


@cute.jit
def fetch(values, i: cutlass.Constexpr[int], pointer):
    """Load fragment ``i`` into ``values[i]``; returns its dirty flag at bit ``i``."""
    got = load_words(pointer, count=4, volatile=True)
    for k in cutlass.range_constexpr(4):
        values[i, k] = got[k]
    return Uint32(not published(got)) << i


@cute.jit
def poll(
    base,
    stride,
    count: cutlass.Constexpr[int],
    deadline_ns: cutlass.Constexpr[int],
):
    """Lamport-poll ``count`` 16-byte fragments at base + i*stride (elements); all
    loads issue before any check and only dirty fragments re-poll."""
    values = cute.make_rmem_tensor((count, 4), Uint32)
    pending = Uint32(0)
    for i in cutlass.range_constexpr(count):
        pending = pending | fetch(values, i, base + i * stride)
    started, polls = cutlass.Int64(0), cutlass.Int32(0)
    if pending != 0:
        started = wait_deadline_start(deadline_ns)
    while pending != 0:
        polls = wait_deadline_check(started, polls, deadline_ns)
        again = Uint32(0)
        for i in cutlass.range_constexpr(count):
            if (pending >> i) & 1:
                again = again | fetch(values, i, base + i * stride)
        pending = again
    return values
