# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/sync.py
"""The tagged mailbox between CTAs (device scope) and GPUs (system scope), and the guarded wait every
mailbox poll and the epoch hand-off go through (RFC #43949 section 8)."""

from dataclasses import dataclass
from enum import Enum

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from sglang.experimental.monokernels.hip.common.guard import (
    EPOCH_GUARD,
    G_DEADLINE,
    G_MIRROR,
    G_OWN,
    G_PEERS,
    G_RANK,
    MIRROR_ARMED,
    MIRROR_DEADLINE,
    MIRROR_PEER,
    REASON_DEADLINE,
    REASON_PEER,
    STOP_FLAG,
    STOP_REASON,
    site_id,
)
from sglang.experimental.monokernels.hip.common.ops import (
    CM_DEV,
    CM_SYS,
    bf16_pair,
    claim_i32,
    gstore,
    load_ptr64,
    memrealtime,
    rsrc,
    traced,
)

POLL_MAX = 12  # mailbox specs polled per batch (bounds live registers)
GUARD_POLLS = 64  # polls between two looks at the clock and the stop word; the first poll reads neither


class WaitGuard:
    """The guard of one launch's waits: the epoch buffer (whose words EPOCH_GUARD.. hold the guard block's
    address), this launch's epoch, the TP size, and the build's injected fault (site, after launches) or None."""

    def __init__(self, epoch, ep, npes, fault=None):
        self.epoch, self.ep, self.npes, self.fault = epoch, ep, npes, fault

    def stalls(self, site):
        """For the armed site of a fault-injection build: true from launch ``after`` on (the wait then never
        completes, so its deadline stops the launch). None everywhere else."""
        if self.fault is None or self.fault[0] != site:
            return None
        return self.ep >= self.fault[1]

    @traced
    def check(self, site, t0):
        """Stop the launch if this wait passed its deadline or a peer stopped; return the wait's start time."""
        now = memrealtime()
        t0 = (t0 == 0).select(now, t0)
        block = load_ptr64(self.epoch, EPOCH_GUARD // 2)
        own = load_ptr64(block, G_OWN // 2)
        flagged = fx.Int32(
            bo.buffer_load(
                rsrc(own), STOP_FLAG, vec_width=1, dtype=T.i32, cache_modifier=CM_SYS
            )
        )
        late = (now - t0) > load_ptr64(block, G_DEADLINE // 2)
        if const_expr(self.stalls(site) is not None):
            mirror = load_ptr64(block, G_MIRROR // 2)
            if self.stalls(site) & (mirror != 0):
                bo.buffer_store(
                    fx.Int32(site_id(site)),
                    rsrc(mirror),
                    MIRROR_ARMED,
                    cache_modifier=CM_SYS,
                )
        if late | (flagged != 0):
            self.stop(site, late, block, own)
        return t0

    @traced
    def stop(self, site, late, block, own):
        """End the launch with a store to address 0. A wait past its own deadline claims this rank's stop reason
        with a compare-and-swap from 0 (so the first deadline is the one recorded, even when several workgroups
        expire together), waits for that store, then flags every peer; a wait that saw a flag only stops."""
        rank = fx.Int32(
            bo.buffer_load(
                rsrc(block), G_RANK, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        sid = fx.Int32(site_id(site) << 8)
        mirror = load_ptr64(block, G_MIRROR // 2)
        if late:
            why = fx.Int32(REASON_DEADLINE) | sid | (rank << 16)
            won = claim_i32(own, STOP_REASON, why)
            # the claim must land before the peer flags, so a peer reads a recorded reason here
            rocdl.s_waitcnt(vmcnt=0)
            flag = (rank + 1) | sid
            for p in range_constexpr(self.npes):
                peer = load_ptr64(block, G_PEERS // 2 + p)
                bo.buffer_store(flag, rsrc(peer), STOP_FLAG, cache_modifier=CM_SYS)
            if (mirror != 0) & won:
                # only the winning workgroup records the mirror reason, so the test reads the first deadline too;
                # `&`, not `and`: `and` would call bool() on a traced value at trace time
                bo.buffer_store(
                    why, rsrc(mirror), MIRROR_DEADLINE, cache_modifier=CM_SYS
                )
        else:
            if mirror != 0:
                bo.buffer_store(
                    fx.Int32(REASON_PEER) | sid | (rank << 16),
                    rsrc(mirror),
                    MIRROR_PEER,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0)
        gstore(fx.Int64(0), rank, words=1)


def _waiting(pending, stall):
    """The wait's loop condition: ``pending``, and at an armed fault site also the injected stall."""
    if stall is None:
        return pending
    return lambda v: pending(v) | stall


def _checker(guard, site):
    return lambda t0: guard.check(site, t0)


@traced
def _spin(load_all, pending, guard, site):
    """Re-issue ``load_all`` until ``pending`` clears. Every GUARD_POLLS polls the guard looks at the clock and
    the stop word: past the deadline, or once a peer stopped, the launch stops."""
    if guard is None:
        raise ValueError(f"wait {site!r} without a guard: every wait needs a deadline")
    v = load_all()
    # the loop carries only traced values and callables
    waiting = _waiting(pending, guard.stalls(site))
    check = _checker(guard, site)
    busy = waiting(v)
    n = fx.Int32(0)
    t0 = fx.Int64(0)
    while busy:
        rocdl.s_sleep(1)
        v = load_all()
        n = n + 1
        if (n & (GUARD_POLLS - 1)) == 0:
            t0 = check(t0)
        busy = waiting(v)
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
    """Tagged-pair hand-off between CTAs or GPUs: each value is stored next to the launch tag. A poll is a
    guarded wait named after its first region."""

    def __init__(self, tag, guard):
        self.tag = tag
        self.guard = guard

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
        if len(spaces) != 1:
            raise ValueError(f"one poll batch spans {spaces}")
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

        v = _spin(load_all, pending, self.guard, specs[0][0].region)
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
