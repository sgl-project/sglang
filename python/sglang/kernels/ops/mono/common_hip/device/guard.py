# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail-stop inside the kernel (RFC #43949 section 7), the device half. A wait that misses its deadline, or sees
that another rank stopped, records why, flags every rank, fences, and stops its launch with a store to address 0
(a GPU fault): a launch never runs on after a failed wait. A satisfied wait costs nothing extra -- the clock and the
stop flag are read only once the first poll comes back stale. Word layout: ``common_hip.guard``."""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.kernels.ops.mono.common_hip.device.ops import (
    CM_SYS,
    ld_i32,
    load_ptr64,
    memrealtime,
    traced,
)
from sglang.kernels.ops.mono.common_hip.guard import (
    ERROR,
    REASON_DEADLINE,
    REASON_PEER,
    SITE,
    STOP,
    ZERO,
    site_id,
)


@dataclass(frozen=True)
class Guard:
    """A launch's view of the guard words: ``own`` (this rank's, traced address), ``peers`` (every rank's,
    this one included), ``ticks`` (the deadline, traced Int64). ``fault_site`` / ``fault_tag``: a build's
    injected stall (tests): the waits on that region stay pending from that hand-off tag on."""

    own: object
    peers: tuple
    ticks: object
    fault_site: str = ""
    fault_tag: int = 0


def guard_from_args(
    own, table, tp: int, ticks, fault_site: str = "", fault_after: int = 0
):
    """``table``: the i64 addresses of every rank's guard words. A launch pair ``e`` tags its hand-offs 2 e + 1
    (K1) and 2 e + 2 (K2), so pair ``fault_after`` on is tag 2 * fault_after + 1 on."""
    peers = tuple(load_ptr64(table, p) for p in range_constexpr(tp))
    return Guard(own, peers, fx.Int64(ticks), fault_site, 2 * fault_after + 1)


def _word_ptr(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), addr)


@traced
def _stop(g: Guard, site: int, reason):
    fx.llvm.atomic_or(_word_ptr(g.own + 4 * ERROR), fx.Int32(reason))
    fx.llvm.atomic_cas(_word_ptr(g.own + 4 * SITE), fx.Int32(0), fx.Int32(site))
    for p in range_constexpr(len(g.peers)):
        fx.llvm.atomic_xchg(_word_ptr(g.peers[p] + 4 * STOP), fx.Int32(1))
    fx.llvm.memory_fence()
    rocdl.s_waitcnt(vmcnt=0)
    # the ZERO word is 0 at run time, so the store goes to address 0 and faults; read, it cannot be folded away
    null = fx.Int64(ld_i32(g.own, ZERO, cm=CM_SYS))
    ptr = llvm.IntToPtrOp(ir.Type.parse("!llvm.ptr<1>"), as_ir_value(null))
    llvm.StoreOp(as_ir_value(fx.Int32(reason)), ptr.result, alignment=4, volatile_=True)


def stall_of(g, region: str, tag):
    """The injected stall of a wait on ``region`` at hand-off ``tag`` (traced bool), or None."""
    if g is None or not g.fault_site or g.fault_site != region:
        return None
    return fx.Int32(tag) >= fx.Int32(g.fault_tag)


@traced
def guarded_spin(load_all, pending, g: Guard, region: str, stall=None):
    """``load_all()`` until ``pending`` clears, within ``g``'s deadline; a stop otherwise (see the module doc)."""

    def busy(v):
        return pending(v) if stall is None else (pending(v) | stall)

    v = load_all()
    t0 = fx.Int64(0)
    stop = fx.Int32(0)
    while busy(v) & (stop == 0):
        rocdl.s_sleep(1)
        v = load_all()
        now = memrealtime()
        t0 = (t0 == 0).select(now, t0)
        peer = ld_i32(g.own, STOP, cm=CM_SYS) != 0
        late = (now - t0 > g.ticks).select(fx.Int32(REASON_DEADLINE), fx.Int32(0))
        stop = peer.select(fx.Int32(REASON_PEER), late)
    if stop != 0:
        _stop(g, site_id(region), stop)
    return v
