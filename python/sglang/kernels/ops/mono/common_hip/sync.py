# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
"""The tagged mailbox between CTAs (device scope) and GPUs (system scope)."""

from dataclasses import dataclass
from enum import Enum

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sglang.kernels.ops.mono.common_hip import guard
from sglang.kernels.ops.mono.common_hip.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    rsrc,
    traced,
)

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)
CHECK_POLLS = 64  # a pending wait checks its deadline and the failure flags this often


@traced
def _spin(load_all, pending):
    """Re-issue ``load_all`` until ``pending`` clears.

    Under a guard (``guard.scope``) the wait also stops once its deadline passes
    or once this GPU or a peer failed: a timeout sets the error word and the
    peers' flags, a failure seen elsewhere sets the ``peer`` word, and then the
    launch stops (``guard.stop_launch``): it never runs on with words that did not
    come."""
    g = guard.current()
    if const_expr(g is None):
        v = load_all()
        while pending(v):
            rocdl.s_sleep(1)
            v = load_all()
        return v
    site = g.next_site()
    # bound before the traced loops: a local whose method a loop calls becomes loop state
    deadline_of, stopped, fail, note_peer = g.deadline, g.stopped, g.fail, g.note_peer
    v = load_all()
    stop = fx.Int32(0)  # 1: this wait timed out; 2: a failure elsewhere
    if const_expr(g.fault_site == site):
        # SGLANG_TEST_MONO_FAULT_INJECT: once armed, this rank stalls here until
        # the deadline
        armed = g.armed()
        t0 = guard.clock()
        deadline = deadline_of()
        while armed & (stop == 0):
            rocdl.s_sleep(1)
            v = load_all()
            late = (guard.clock() - t0) > deadline
            stop = late.select(fx.Int32(1), stopped().select(fx.Int32(2), fx.Int32(0)))
    # A satisfied wait costs nothing more, and so does a pending one that the data
    # reaches within CHECK_POLLS polls. A longer one checks the failure flags every
    # CHECK_POLLS polls, and its deadline from the first of those checks.
    t0 = fx.Int64(0)
    polls = fx.Int32(0)
    while pending(v) & (stop == 0):
        rocdl.s_sleep(1)
        v = load_all()
        polls = polls + 1
        if (polls & (CHECK_POLLS - 1)) == 0:
            now = guard.clock()
            t0 = (polls == CHECK_POLLS).select(now, t0)
            late = (now - t0) > deadline_of()
            stop = late.select(fx.Int32(1), stopped().select(fx.Int32(2), fx.Int32(0)))
    if stop == 1:
        fail(site)
    if stop == 2:
        note_peer()
    return v


class Space(Enum):
    SCRATCH = "scratch"  # this GPU's memory: device-scope hand-off between CTAs
    PEER = "peer"  # a peer-symmetric region: system-scope hand-off between GPUs


@dataclass(frozen=True)
class Addr:
    """A mailbox region's name, space and traced address; the space sets the scope."""

    region: str
    space: Space
    value: object


def sreg(base, offset, region):
    """Scratch region ``region`` at byte ``offset`` of ``base``."""
    return Addr(region, Space.SCRATCH, base + fx.Int64(offset))


def preg(base, offset, region):
    """Return the address of peer-symmetric region ``region`` at byte ``offset`` of buffer ``base``."""
    return Addr(region, Space.PEER, base + fx.Int64(offset))


# the store / load cache policy of a hand-off in each space
_SCOPE_CM = {Space.SCRATCH: CM_DEV, Space.PEER: CM_SYS}


class Mailbox:
    """Tagged-pair hand-off between CTAs or GPUs: each value is stored next to the launch tag."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, a, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_bf(self, a, i, vs):
        """Store elements i .. i + len(vs) as packed bf16 pairs in one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(a.value),
            i,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def put_words(self, a, i, words):
        """Store raw 32-bit words at pairs i..; integer payloads must use this path."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(a.value),
            i * 2,
            cache_modifier=_SCOPE_CM[a.space],
        )

    def poll(self, specs, batch=POLL_MAX):
        """Poll the pairs in ``specs`` (one space) until every tag is current; return the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        spaces = {a.space for a, _, _ in specs}
        assert len(spaces) == 1, f"one poll batch spans {spaces}"
        cm = _SCOPE_CM[spaces.pop()]
        bases = [a.value for a, _, _ in specs]
        tag = self.tag

        def load_all():
            words = []
            for b, (_, i, n) in zip(bases, specs):
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=cm,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


@traced
def publish(put, a, i, v, who):
    """Drain this CTA's plain stores, then store mailbox pair (``a``, i) := v from thread ``who``."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if who:
        put(a, i, v)
