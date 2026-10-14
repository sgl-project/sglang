# SPDX-License-Identifier: Apache-2.0
"""Deadlines, sticky error words and the stop of a failed launch (RFC #43949).

Each wait on another workgroup or rank has a deadline below the server's
watchdog timeout. On a timeout the wait sets a sticky error word on its GPU,
writes an error flag to every peer, fences, and stops the launch (a store to
the null page: a GPU fault at address 0x0), and with it the process. A wait
that sees a failure (its own GPU's words or a peer's flag) stops too. A launch
never runs on after a failed wait: its data-dependent addresses would come from
words that never arrived. A rank finishes a step only when each of its waits got
that step's data, so its tokens are correct; nothing is checked on the host.

The deadline comes from ``--watchdog-timeout`` (half of it, at most 60 s), and
one under 5 s is refused, so that a normal delay on one rank is not a failure
(``kernels/ops/mono/hooks.py``). A satisfied wait costs nothing
more, and so does a pending one that the data reaches within ``CHECK_POLLS``
polls; a longer one checks the failure flags every ``CHECK_POLLS`` polls and its
deadline on the 100 MHz ``s_memrealtime`` clock from the first of those checks.
The guard loads nothing at launch entry: the deadline and the peers' addresses
are read only on those checks and on a failure.

The guard block lives in the launch's epoch buffer (int32 words):

====  ======================================================================
word  meaning
====  ======================================================================
0     the epoch (``epoch.py``)
1-2   the deadline in clock ticks (int64), written by the host at build
3     ``deadline``: 1 + the site of the first wait on this GPU that timed out
4     ``peer``: set when a wait saw a failure elsewhere
5     ``fault``: set by the test fault injection (``SGLANG_TEST_MONO_FAULT_INJECT``)
====  ======================================================================

A peer's error flags are ``FLAG_WORDS`` words at ``flags_offset`` of each
rank's peer buffer: rank r writes word r of every rank's flags, its own too.

The guard is set for the duration of a kernel's trace (``scope``); the waits
pick it up there, so a kernel only builds its guard once at the top.
"""

import contextlib
from typing import Optional

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.kernels.ops.mono.common_hip.ops import CM_DEV, CM_SYS, load_ptr64, rsrc

GUARD_DEADLINE = 1  # int64 at words 1-2
ERR_DEADLINE = 3
ERR_PEER = 4
ERR_FAULT = 5
ERROR_WORDS = (ERR_DEADLINE, ERR_PEER, ERR_FAULT)
ERROR_NAMES = ("deadline", "peer", "fault")
GUARD_WORDS = 8  # the epoch buffer's words before the per-CTA marks

FLAG_WORDS = 8  # one a rank; TP 8 at most
FLAG_BYTES = 256  # the peer buffer's flags region, padded to a cache line pair

CLOCK_HZ = 100_000_000  # s_memrealtime: the constant 100 MHz reference clock

# The SGLang names this module reads (see hooks.SGLANG_READS)
SGLANG_READS = {"srt/environ.py": ("envs", "SGLANG_TEST_MONO_FAULT_INJECT")}


def deadline_ticks(seconds: float) -> int:
    return int(seconds * CLOCK_HZ)


def fault_injection(rank: int) -> Optional[tuple[int, int]]:
    """(site, after): the wait this rank stalls at once ``after`` launches have run,
    from ``SGLANG_TEST_MONO_FAULT_INJECT`` (``rank:site[:after]``), or None. A test
    hook: the stalled wait times out."""
    from sglang.srt.environ import envs

    spec = envs.SGLANG_TEST_MONO_FAULT_INJECT.get()
    if not spec:
        return None
    r, site, after = (spec.split(":") + ["0", "0"])[:3]
    return (int(site or 0), int(after or 0)) if int(r) == rank else None


def clock():
    """The 100 MHz reference clock (s_memrealtime)."""
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def stop_launch():
    """Stop the launch: a volatile store to the null page faults (address 0x0, so a
    deliberate stop is told apart from a real fault), and the runtime tears the
    queue down. Unlike ``llvm.trap`` it asks nothing of the kernel's ABI, so a
    launch that never fails pays nothing for it."""
    null = llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(0))
    ).result
    llvm.StoreOp(as_ir_value(fx.Int32(1)), null, alignment=4, volatile_=True)


class Guard:
    """One launch's deadline and error words, and optionally the peers' flags.

    ``epoch``: the launch's epoch buffer address. ``own``: this rank's peer
    buffer base and ``peers`` the table of every rank's (traced), with ``rank``,
    ``npes`` and ``flags_offset``; without them a failure stays on this GPU (a
    launch that never waits on peers). ``stop=False`` keeps a failed launch
    running (the one-GPU test reads the words afterwards).
    """

    def __init__(
        self,
        epoch,
        own=None,
        peers=None,
        rank=None,
        npes=1,
        flags_offset=0,
        fault_site=None,
        fault_after=0,
        stop=True,
    ):
        self.epoch = epoch
        self.own = own
        self.peers = peers
        self.rank = rank
        self.npes = npes
        self.flags_offset = flags_offset
        self.fault_site = fault_site
        self.fault_after = fault_after
        self.stop = stop
        self._sites = 0

    def armed(self):
        """The fault injection's stall is on: ``fault_after`` launches have run."""
        ep = fx.Int32(
            bo.buffer_load(
                rsrc(self.epoch), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        return ep >= fx.Int32(self.fault_after)

    def deadline(self):
        """The deadline in clock ticks, loaded where a check needs it."""
        r = rsrc(self.epoch)
        lo = fx.Int32(
            bo.buffer_load(
                r, GUARD_DEADLINE, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        hi = fx.Int32(
            bo.buffer_load(
                r, GUARD_DEADLINE + 1, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )
        return (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))

    def next_site(self) -> int:
        self._sites += 1
        return self._sites

    def _word(self, i):
        return fx.Int32(
            bo.buffer_load(
                rsrc(self.epoch), i, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV
            )
        )

    def flags(self):
        """This rank's flags as its peers wrote them (system scope); 0 when clean."""
        if self.own is None:
            return fx.Int32(0)
        v = fx.Vector(
            bo.buffer_load(
                rsrc(self.own + fx.Int64(self.flags_offset)),
                0,
                vec_width=self.npes,
                dtype=T.i32,
                cache_modifier=CM_SYS,
            )
        )
        acc = fx.Int32(v[0])
        for p in range_constexpr(1, self.npes):
            acc = acc | v[p]
        return acc

    def stopped(self):
        """Whether this GPU already failed or saw a peer fail: later waits return at once."""
        own = self._word(ERR_DEADLINE) | self._word(ERR_PEER) | self._word(ERR_FAULT)
        return (own | self.flags()) != 0

    def fail(self, site):
        """This GPU's wait at ``site`` passed its deadline: set the error word, flag
        every peer, fence, and stop the launch."""
        bo.buffer_store(
            fx.Int32(site + 1), rsrc(self.epoch), ERR_DEADLINE, cache_modifier=CM_DEV
        )
        if self.peers is not None:
            for p in range_constexpr(self.npes):
                bo.buffer_store(
                    fx.Int32(1),
                    rsrc(load_ptr64(self.peers, p) + fx.Int64(self.flags_offset)),
                    self.rank,
                    cache_modifier=CM_SYS,
                )
        rocdl.s_waitcnt(vmcnt=0)
        if const_expr(self.stop):
            stop_launch()

    def note_peer(self):
        """A wait saw a failure elsewhere: stop the launch (the failed rank flagged
        everyone), or without the stop keep a sticky ``peer`` word."""
        if const_expr(self.stop):
            stop_launch()
        else:
            old = self._word(ERR_PEER)
            bo.buffer_store(
                (old == 0).select(fx.Int32(1), old),
                rsrc(self.epoch),
                ERR_PEER,
                cache_modifier=CM_DEV,
            )


_CURRENT: list[Guard] = []


@contextlib.contextmanager
def scope(guard: Optional[Guard]):
    """Make ``guard`` the one the waits traced inside this block take."""
    _CURRENT.append(guard)
    try:
        yield guard
    finally:
        _CURRENT.pop()


def current() -> Optional[Guard]:
    return _CURRENT[-1] if _CURRENT else None


def barrier_all():
    gpu.barrier()
