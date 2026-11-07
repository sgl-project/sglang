"""Randomized stress harness for the page safety of the unified KV pool.

A `ChainAllocator` is built over a real arena (an owner end, one or two
floating window bands and a state end, with mixed entry sizes and page sizes,
optionally widened by decode context parallelism). Random sequences of
scheduler-level operations run against it: allocations for forwards that are
launched and retired out of step with the host (overlap), frees under running
forwards, window retirement, host-cache loads and backups, network sends and
receives, flushes, reliefs and float relocations, state slot copies and the
zeroing of handed-out pages. A page freed while a launched forward is
unprocessed stays pending under the newest recorded forward (the chain's
forward steps, launched, recorded and processed as the scheduler does) and
reaches the free list when that forward is processed, or earlier when a take
or an urgent recovery settles it behind a stream wait on the forward; window
retirement, proven below every running window, frees at once.

The device is modeled. Every operation that touches the arena is an *actor* on
a stream: forwards on the forward stream, page moves, zeroing and state copies
on the schedule stream (the stream the allocator issues on), host-cache copies
on their own streams, network transfers on no stream at all. An actor reads
and writes byte ranges of the arena; a forward's, a copy's or a transfer's
writes land only when it completes, which the harness decides (``Tick``), so
two simulated ranks run the same operations with different completion timing.
A stream wait orders the waiting stream after the actor; the host learning an
actor finished (an event query that fired, a synchronize, a network wait)
orders everything it issues later after it. These edges give a happens-before
relation, and the checks read it:

- ``lost_write``: every live virtual page holds what its last writer wrote,
  and every transfer reads what it was handed;
- ``unordered_overlap``: no actor writes bytes another actor still reads or
  writes, or reads bytes another still writes, unless one is ordered after the
  other;
- ``host_read_after_wait``: no host read of device state follows, within one
  scheduler step, a stream wait queued on an actor that had not finished;
- ``rank_divergence``: both ranks make the same layout decisions (pages taken
  and their order, frontiers, free lists, capacity answers, admission, page
  moves and network waits);
- ``cross_dtype_bytes``: bytes last written as FP32 state never reach an
  attention page a running forward reads;
- ``ledger_leak``: once everything retired and was released, the page ledger
  keeps no record of it, and a reset leaves it empty.

A failing sequence is shrunk to a minimal one (`minimize`). The device model
runs on the CPU (`SimDevice`); `CudaDevice` runs the same operations on real
CUDA streams with delayed kernels, where only the payload, divergence and
ledger checks apply.
"""

from __future__ import annotations

import contextlib
import dataclasses
import random
import re
import traceback
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.kv.alloc import band as band_module
from sglang.srt.mem_cache.kv.alloc import signals as signals_module
from sglang.srt.mem_cache.kv.alloc.band import Band
from sglang.srt.mem_cache.kv.alloc.chain import ChainAllocator, ChainRoles
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.srt.mem_cache.kv.alloc.page_usage import VACATED
from sglang.srt.mem_cache.kv.alloc.signals import NetworkTransfer
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.srt.mem_cache.kv.storage import KVStorage
from sglang.srt.runtime_context import get_parallel
from sglang.test.unified_allocator_fixtures import arena_of, setup_allocator_context

INF = float("inf")
KIND_ZERO, KIND_ATTN, KIND_STATE = 0, 1, 2
UNDEF = -1

CHECKS = (
    "lost_write",
    "unordered_overlap",
    "host_read_after_wait",
    "rank_divergence",
    "cross_dtype_bytes",
    "ledger_leak",
    "crash",
)


class Violation(Exception):
    def __init__(self, check: str, message: str):
        assert check in CHECKS, check
        super().__init__(f"[{check}] {message}")
        self.check = check
        self.message = message


# ---------------------------------------------------------------------------
# Configuration and chain construction
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StressConfig:
    page_size: int = 2
    dcp: int = 1
    owner_grows: str = "down"
    full_layers: int = 2
    window_layers: Tuple[int, ...] = (1,)
    window_lens: Tuple[int, ...] = (8,)
    full_pages: int = 24
    window_pages: Tuple[int, ...] = (12,)
    state_slots: int = 8
    max_reqs: int = 6
    max_context_pages: int = 6

    @property
    def windows(self) -> int:
        return len(self.window_lens)

    @classmethod
    def from_dict(cls, fields: dict) -> StressConfig:
        """A config from its JSON form (lists for tuples); fields this
        version does not know are ignored."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(
            **{
                k: tuple(v) if isinstance(v, list) else v
                for k, v in fields.items()
                if k in known
            }
        )


def random_config(rng: random.Random) -> StressConfig:
    page_size = rng.choice([1, 2, 4])
    windows = rng.choice([1, 2])
    dcp = rng.choice([1, 1, 2])
    psw = page_size * dcp
    window_lens = tuple(sorted(rng.sample([psw, 2 * psw, 3 * psw, 4 * psw], windows)))
    return StressConfig(
        page_size=page_size,
        dcp=dcp,
        owner_grows=rng.choice(["down", "up"]),
        full_layers=rng.choice([2, 3]),
        window_layers=tuple(rng.choice([1, 2]) for _ in range(windows)),
        window_lens=window_lens,
        full_pages=rng.randint(16, 30),
        window_pages=tuple(rng.randint(8, 16) for _ in range(windows)),
        state_slots=rng.randint(5, 9),
        max_reqs=rng.randint(4, 7),
        max_context_pages=rng.randint(4, 7),
    )


WINDOW_NAMES = ("win_a", "win_b")


def build_chain(cfg: StressConfig, device: str):
    """The chain: the owner and the state at the two ends, the windows
    floating between them; returns ``(arena, allocator)``."""
    owner_dir = cfg.owner_grows
    state_dir = "up" if owner_dir == "down" else "down"
    full = MHASubPoolSpec(
        name="full",
        layer_num=cfg.full_layers,
        head_num=1,
        head_dim=4,
        store_dtype=torch.float16,
        grow_direction=owner_dir,
    )
    windows = [
        MHASubPoolSpec(
            name=WINDOW_NAMES[i],
            layer_num=cfg.window_layers[i],
            head_num=1,
            head_dim=4,
            store_dtype=torch.float16,
            grow_direction="float",
        )
        for i in range(cfg.windows)
    ]
    state = MambaSubPoolSpec(
        name="state",
        layer_num=1,
        conv_state_shapes=((3, 8),),
        conv_dtype=torch.bfloat16,
        temporal_state_shape=(1, 2, 4),
        temporal_dtype=torch.float32,
        grow_direction=state_dir,
    )
    ps = cfg.page_size
    sink = max(
        full.entry_bytes() * ps,
        state.entry_bytes(),
        *(w.entry_bytes() * ps for w in windows),
    )
    total = (
        sink
        + cfg.full_pages * ps * full.entry_bytes()
        + sum(n * ps * w.entry_bytes() for n, w in zip(cfg.window_pages, windows))
        + cfg.state_slots * state.entry_bytes()
    )
    # Every band sizes its entries over the whole arena: keep it a multiple
    # of every unit so that no band's last unit is cut.
    lcm = 1
    for unit in (
        full.entry_bytes() * ps,
        state.entry_bytes(),
        *(w.entry_bytes() * ps for w in windows),
    ):
        lcm = np.lcm(lcm, unit)
    total = int(-(-total // lcm) * lcm)
    arena = arena_of(
        [full, *windows, state], total_bytes=total, device=device, page_size=ps
    )
    names = tuple(w.name for w in windows)
    roles = ChainRoles(owner="full", window=names[0], windows=names, state="state")
    views = {name: SimpleNamespace() for name in ("full", *names, "state")}
    allocator = ChainAllocator(
        roles=roles,
        arena=arena,
        kvcache=SimpleNamespace(),
        views=views,
        device=device,
        page_size=ps,
        need_sort=False,
    )
    return arena, allocator


# ---------------------------------------------------------------------------
# Operations: what the generator emits and both ranks replay
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Op:
    kind: str
    args: Tuple = ()

    def __repr__(self) -> str:
        return f"{self.kind}{self.args!r}"


# ---------------------------------------------------------------------------
# Happens-before bookkeeping shared by both devices
# ---------------------------------------------------------------------------


class Actor:
    """One operation on the device: what it reads and writes, on which
    stream, and what it is ordered after."""

    __slots__ = (
        "sim",
        "id",
        "kind",
        "label",
        "stream",
        "seq",
        "vc",
        "issue",
        "observed_at",
        "done",
        "preds",
        "r_lo",
        "r_hi",
        "w_lo",
        "w_hi",
        "effect",
        "attn_reads",
        "handle",
    )

    def __repr__(self) -> str:
        return f"<{self.kind}#{self.id} {self.label} on {self.stream}>"


def _ranges(pairs) -> Tuple[np.ndarray, np.ndarray]:
    if not pairs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    arr = np.asarray(pairs, dtype=np.int64)
    return arr[:, 0].copy(), arr[:, 1].copy()


def _first_overlap(a_lo, a_hi, b_lo, b_hi):
    if not len(a_lo) or not len(b_lo):
        return None
    m = (a_lo[:, None] < b_hi[None, :]) & (b_lo[None, :] < a_hi[:, None])
    if not m.any():
        return None
    i, j = np.argwhere(m)[0]
    return int(max(a_lo[i], b_lo[j])), int(min(a_hi[i], b_hi[j]))


class SimEvent:
    """A CUDA event stand-in: ``query`` says whether its actor completed; a
    query that fired means the host has seen it finish."""

    __slots__ = ("actor",)

    def __init__(self, actor: Actor):
        self.actor = actor

    def query(self) -> bool:
        a = self.actor
        if a.done:
            a.sim.observe(a)
        return a.done

    def synchronize(self) -> None:
        self.actor.sim.host_sync(self.actor)

    def __repr__(self) -> str:
        return f"SimEvent({self.actor!r})"


class _SimStream:
    def __init__(self, sim: SimDevice):
        self.sim = sim

    def wait_event(self, event) -> None:
        self.sim.schedule_wait(event.actor)


# Host reads of device tensors, guarded while the allocator runs.
_READ_GUARD: Optional[SimDevice] = None
_PATCHED = False


def _install_read_guard() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    T = torch.Tensor

    def guard(name, fn):
        def wrapped(*args, **kwargs):
            sim = _READ_GUARD
            if sim is not None:
                sim.on_host_read(name, args[0] if args else None)
            return fn(*args, **kwargs)

        wrapped.__name__ = getattr(fn, "__name__", name)
        return wrapped

    for name in (
        "tolist",
        "item",
        "__bool__",
        "__int__",
        "__float__",
        "cpu",
        "numpy",
        "nonzero",
        "unique",
    ):
        setattr(T, name, guard(name, getattr(T, name)))
    orig_getitem = T.__getitem__

    def getitem(self, idx):
        sim = _READ_GUARD
        if (
            sim is not None
            and isinstance(idx, torch.Tensor)
            and idx.dtype == torch.bool
        ):
            sim.on_host_read("boolean-mask index", self)
        return orig_getitem(self, idx)

    T.__getitem__ = getitem
    torch.unique = guard("torch.unique", torch.unique)
    torch.nonzero = guard("torch.nonzero", torch.nonzero)


@contextlib.contextmanager
def _unguarded():
    global _READ_GUARD
    saved, _READ_GUARD = _READ_GUARD, None
    try:
        yield
    finally:
        _READ_GUARD = saved


_REAL_WAIT_SIGNALS = signals_module.wait_signals
_ACTIVE_SIM: Optional[SimDevice] = None


def _sim_wait_signals(key, stream=None, *, network_waiter=None):
    sim = _ACTIVE_SIM
    assert sim is not None, "a band waited outside the harness"
    _REAL_WAIT_SIGNALS(key, _SimStream(sim), network_waiter=network_waiter)


@contextlib.contextmanager
def sim_patches(sim: SimDevice):
    """Route the allocator's stream waits into ``sim``."""
    global _ACTIVE_SIM
    saved = _ACTIVE_SIM
    _ACTIVE_SIM = sim
    band_module.wait_signals = _sim_wait_signals
    try:
        yield
    finally:
        band_module.wait_signals = _REAL_WAIT_SIGNALS
        _ACTIVE_SIM = saved


class SimDevice:
    """The device of one rank, modeled on the host."""

    def __init__(self, arena, rank: int):
        self.rank = rank
        self.arena = arena
        self.raw = arena.raw.numpy()
        self.kind = np.zeros(arena.total_bytes, np.uint8)
        self.clock = 0
        self.next_id = 0
        self.seq: Dict[str, int] = {}
        self.tails: Dict[str, Actor] = {}
        self.svc: Dict[str, int] = {}
        self.vcs: Dict[str, Dict[str, int]] = {"F": {}, "L": {}, "B": {}}
        self.live: List[Actor] = []
        self.pending: List[Actor] = []
        self.transfers: Dict[int, Actor] = {}
        self.transfer_ids: Dict[int, int] = {}
        # Per step: the queue log, the waits, and host-read discipline.
        self.queue_log: List[tuple] = []
        self.decisions: List[tuple] = []
        self.unfinished_wait: Optional[Tuple[str, Actor]] = None
        self.in_call: Optional[str] = None
        self.call_index = 0
        self.violations: List[Violation] = []
        self.describe = None  # byte offset -> text, set by the model
        self.step_reads: Dict[tuple, int] = {}
        self.stats = {
            "waits": 0,
            "unfinished_waits": 0,
            "copies": 0,
            "zeroes": 0,
            "network_waits": 0,
        }

    # -- happens-before --

    def hb(self, a: Actor, x: Actor) -> bool:
        return a.observed_at < x.issue or x.vc.get(a.stream, 0) >= a.seq

    def _new(self, kind, stream, label, vc, preds, reads, writes, effect) -> Actor:
        self.clock += 1
        a = Actor()
        a.sim = self
        a.id = self.next_id
        self.next_id += 1
        a.kind = kind
        a.label = label
        a.stream = stream
        self.seq[stream] = self.seq.get(stream, 0) + 1
        a.seq = self.seq[stream]
        a.vc = dict(vc)
        a.vc[stream] = a.seq
        a.issue = self.clock
        a.observed_at = INF
        a.done = False
        a.preds = [p for p in preds if p is not None and not p.done]
        a.r_lo, a.r_hi = _ranges(reads)
        a.w_lo, a.w_hi = _ranges(writes)
        a.effect = effect
        a.attn_reads = ()
        a.handle = SimEvent(a)
        self._check_conflicts(a)
        self.tails[stream] = a
        self.live.append(a)
        self.queue_log.append((a.id, kind, stream, label))
        return a

    def _check_conflicts(self, x: Actor) -> None:
        for a in self.live:
            if a is x or self.hb(a, x):
                continue
            hit = _first_overlap(x.w_lo, x.w_hi, a.w_lo, a.w_hi)
            how = "writes bytes it also writes"
            if hit is None:
                hit = _first_overlap(x.w_lo, x.w_hi, a.r_lo, a.r_hi)
                how = "writes bytes it still reads"
            if hit is None:
                hit = _first_overlap(x.r_lo, x.r_hi, a.w_lo, a.w_hi)
                how = "reads bytes it still writes"
            if hit is not None:
                self.fail(
                    "unordered_overlap",
                    f"rank {self.rank}: {x!r} {how}, unordered after unfinished "
                    f"{a!r} (done={a.done}): bytes [{hit[0]}, {hit[1]}) "
                    f"{self.describe(hit[0]) if self.describe else ''}",
                )

    def check_state_landing(self, lo: int, hi: int, writer: str) -> None:
        """FP32 state bytes just landed on ``[lo, hi)``: no running forward
        may read them as attention bytes."""
        for a in self.pending:
            if a.kind != "forward" or a.done or not a.attn_reads:
                continue
            for r_lo, r_hi in a.attn_reads:
                if r_lo < hi and lo < r_hi:
                    self.fail(
                        "cross_dtype_bytes",
                        f"rank {self.rank}: {writer} put FP32 state bytes on "
                        f"[{max(lo, r_lo)}, {min(hi, r_hi)}) "
                        f"{self.describe(max(lo, r_lo)) if self.describe else ''}, "
                        f"an attention page running {a!r} reads",
                    )

    def fail(self, check: str, message: str) -> None:
        v = Violation(check, message)
        self.violations.append(v)
        raise v

    def observe(self, a: Actor) -> None:
        """The host learned ``a`` finished: everything ordered before it too."""
        if a.observed_at <= self.clock:
            return
        assert a.done, a
        now = self.clock
        keep = []
        for b in self.live:
            if b is a or a.vc.get(b.stream, 0) >= b.seq:
                assert b.done or b is a, (b, a)
                b.observed_at = min(b.observed_at, now)
            if b.observed_at > now:
                keep.append(b)
        a.observed_at = min(a.observed_at, now)
        self.live = keep

    def complete(self, a: Actor) -> None:
        if a.done:
            return
        for p in a.preds:
            self.complete(p)
        if a.effect is not None:
            a.effect(a)
        a.done = True
        if a in self.pending:
            self.pending.remove(a)

    def host_sync(self, a: Actor) -> None:
        self.complete(a)
        self.observe(a)

    def tick(self, f: float) -> None:
        if not self.pending:
            return
        a = self.pending[min(int(f * len(self.pending)), len(self.pending) - 1)]
        self.complete(a)

    # -- the schedule stream (the allocator's current stream) --

    def schedule_wait(self, a: Actor) -> None:
        self.stats["waits"] += 1
        if not a.done:
            self.stats["unfinished_waits"] += 1
            if self.unfinished_wait is None:
                self.unfinished_wait = (self.in_call or "?", a, self.call_index)
            self.complete(a)
        for stream, seq in a.vc.items():
            if self.svc.get(stream, 0) < seq:
                self.svc[stream] = seq
        self.queue_log.append(("wait", a.id, a.label))

    def schedule_op(self, kind, label, reads, writes, effect, copy=None) -> Actor:
        a = self._new(kind, "S", label, self.svc, [], reads, writes, effect)
        self.svc = dict(a.vc)
        if effect is not None:
            effect(a)
        a.effect = None
        a.done = True
        return a

    def observe_schedule(self) -> None:
        """A host readback ordered after the schedule stream (a publish or a
        send reading the rows' ids): it observes the stream and every actor
        the stream waited."""
        now = self.clock
        for a in self.live:
            if a.stream == "S" or self.svc.get(a.stream, 0) >= a.seq:
                assert a.done, a
                a.observed_at = min(a.observed_at, now)
        self.live = [a for a in self.live if a.observed_at > now]

    def on_host_read(self, what: str, tensor=None) -> None:
        if self.unfinished_wait is None:
            return
        if (
            not isinstance(self, SimDevice)
            and isinstance(tensor, torch.Tensor)
            and not tensor.is_cuda
        ):
            return
        where, actor, call = self.unfinished_wait
        stack = [
            f"{f.filename.rsplit('/', 1)[-1]}:{f.lineno}:{f.name}"
            for f in traceback.extract_stack()[-9:-2]
            if "unified_page_safety_harness" not in f.filename
        ]
        if call != self.call_index:
            # A wait queued by an earlier allocator call of the same step: the
            # host blocks on it (a stall), recorded per call site.
            site = (what, self.in_call, where, stack[-1] if stack else "?")
            self.step_reads[site] = self.step_reads.get(site, 0) + 1
            return
        same = "the same allocator call"
        self.fail(
            "host_read_after_wait",
            f"rank {self.rank}: {what} in {self.in_call} after a stream wait on "
            f"unfinished {actor!r} queued in {where} ({same}); at {' <- '.join(reversed(stack))}",
        )

    # -- actors other than schedule-stream ops --

    def host_raw(self) -> np.ndarray:
        return self.raw

    def launch(
        self,
        kind,
        stream,
        label,
        joins,
        reads,
        writes,
        effect,
        payload=None,
        expect=None,
    ) -> Actor:
        vc: Dict[str, int] = {}
        for j in joins:
            for s, q in j.items():
                if vc.get(s, 0) < q:
                    vc[s] = q
        preds = []
        if stream in ("F", "L", "B"):
            preds.append(self.tails.get(stream))
            vc_stream = self.vcs[stream]
            for s, q in vc_stream.items():
                if vc.get(s, 0) < q:
                    vc[s] = q
        if any("F" in j for j in joins) or stream in ("L", "B"):
            preds.append(self.tails.get("F"))
        a = self._new(kind, stream, label, vc, preds, reads, writes, effect)
        if stream in self.vcs:
            self.vcs[stream] = dict(a.vc)
        self.pending.append(a)
        return a


class CudaActor:
    """A device operation on real CUDA streams, done when its event fired."""

    __slots__ = (
        "id",
        "kind",
        "label",
        "stream",
        "handle",
        "expect",
        "result",
        "keep",
        "attn_reads",
        "effect",
    )

    @property
    def done(self) -> bool:
        return self.handle.query()

    def __repr__(self) -> str:
        return f"<{self.kind}#{self.id} {self.label} on {self.stream}>"


class CudaDevice:
    """The device of one rank on real CUDA streams: forwards, host-cache
    copies and network transfers are kernels delayed by a random sleep, so
    they finish when the GPU gets to them; the allocator issues on its own
    schedule stream and waits with the real `wait_signals`."""

    # The longest delay of a kernel, in GPU clock cycles.
    SLEEP_CYCLES = 40_000_000

    def __init__(self, arena, rank: int, seed: int):
        self.rank = rank
        self.arena = arena
        self.raw32 = arena.raw.view(torch.int32)
        self.rng = random.Random(seed * 7919 + rank)
        self.S = torch.cuda.Stream()
        self.streams = {k: torch.cuda.Stream() for k in ("F", "L", "B")}
        self.net = [torch.cuda.Stream() for _ in range(4)]
        self.kind = None
        self.clock = 0
        self.next_id = 0
        self.tails: Dict[str, CudaActor] = {}
        self.vcs: Dict[str, Dict[str, int]] = {"F": {}, "L": {}, "B": {}}
        self.svc: Dict[str, int] = {}
        self.pending: List[CudaActor] = []
        self.transfers: Dict[int, CudaActor] = {}
        self.transfer_ids: Dict[int, int] = {}
        self.decisions: List[tuple] = []
        self.unfinished_wait = None
        self.in_call: Optional[str] = None
        self.call_index = 0
        self.step_reads: Dict[tuple, int] = {}
        self.violations: List[Violation] = []
        self.describe = None
        self.stats = {
            "waits": 0,
            "unfinished_waits": 0,
            "copies": 0,
            "zeroes": 0,
            "network_waits": 0,
        }

    def _words(self, items):
        idx, val = [], []
        for off, eb, v in items:
            n = eb // 4
            idx.extend(range(off // 4, off // 4 + n))
            val.extend([v] * n)
        host_idx = torch.tensor(idx, dtype=torch.int64).pin_memory()
        host_val = torch.tensor(
            np.asarray(val, dtype=np.uint32).view(np.int32)
        ).pin_memory()
        return host_idx, host_val

    def _actor(self, kind, stream, label, event) -> CudaActor:
        self.clock += 1
        a = CudaActor()
        a.id = self.next_id
        self.next_id += 1
        a.kind, a.label, a.stream, a.handle = kind, label, stream, event
        a.expect = a.result = None
        a.keep = ()
        self.tails[stream] = a
        return a

    def launch(
        self,
        kind,
        stream,
        label,
        joins,
        reads,
        writes,
        effect,
        payload=None,
        expect=None,
    ) -> CudaActor:
        if stream in self.streams:
            s = self.streams[stream]
        else:
            s = self.net[self.next_id % len(self.net)]
        keep = []
        result = None
        with torch.cuda.stream(s):
            if stream == "F":
                s.wait_stream(self.S)
            elif stream in ("L", "B"):
                s.wait_stream(self.S)
                s.wait_stream(self.streams["F"])
            if kind != "marker":
                # Up to tens of milliseconds: longer than the host takes
                # to issue the next operations, so work stays in flight.
                torch.cuda._sleep(self.rng.randint(0, self.SLEEP_CYCLES))
            if payload:
                hi, hv = self._words(payload)
                di, dv = (
                    hi.to("cuda", non_blocking=True),
                    hv.to("cuda", non_blocking=True),
                )
                self.raw32.index_put_((di,), dv)
                keep += [hi, hv, di, dv]
            if expect:
                hi, hv = self._words(expect)
                di = hi.to("cuda", non_blocking=True)
                result = self.raw32[di]
                keep += [hi, di]
            event = torch.cuda.Event()
            event.record(s)
        a = self._actor(kind, stream, label, event)
        a.keep = tuple(keep)
        if expect:
            a.expect = expect
            a.result = result
        self.pending.append(a)
        return a

    def schedule_op(self, kind, label, reads, writes, effect, copy=None) -> CudaActor:
        with torch.cuda.stream(self.S):
            if copy is not None:
                src, dst, n = copy
                raw = self.arena.raw
                raw[dst : dst + n].copy_(raw[src : src + n])
            event = torch.cuda.Event()
            event.record(self.S)
        return self._actor(kind, "S", label, event)

    def _verify(self, a: CudaActor) -> None:
        if a.expect is None:
            return
        got = a.result.cpu().numpy().view(np.uint32)
        _, want = self._words(a.expect)
        want = want.numpy().view(np.uint32)
        if not (got == want).all():
            bad = int(np.argmax(got != want))
            raise Violation(
                "lost_write",
                f"rank {self.rank}: {a!r} read {int(got[bad])} where {int(want[bad])} was written",
            )
        a.expect = None

    def complete(self, a: CudaActor) -> None:
        a.handle.synchronize()
        self._verify(a)
        if a in self.pending:
            self.pending.remove(a)

    host_sync = complete

    def observe(self, a) -> None:
        pass

    def observe_schedule(self) -> None:
        self.S.synchronize()

    def tick(self, f: float) -> None:
        pass

    def host_raw(self) -> np.ndarray:
        torch.cuda.synchronize()
        for a in list(self.pending):
            self.complete(a)
        return self.arena.raw.cpu().numpy()

    def check_state_landing(self, lo, hi, writer) -> None:
        pass

    on_host_read = SimDevice.on_host_read
    fail = SimDevice.fail


def _cuda_wait_signals(key, stream=None, *, network_waiter=None):
    dev = _ACTIVE_SIM
    for s in signals_module.signals_of(key):
        if isinstance(s, signals_module.TransferRelease):
            continue
        if not s.query():
            dev.stats["waits"] += 1
            dev.stats["unfinished_waits"] += 1
            if dev.unfinished_wait is None:
                dev.unfinished_wait = (dev.in_call or "?", s, dev.call_index)
    _REAL_WAIT_SIGNALS(key, stream, network_waiter=network_waiter)


@contextlib.contextmanager
def cuda_patches(dev: CudaDevice):
    global _ACTIVE_SIM
    saved = _ACTIVE_SIM
    _ACTIVE_SIM = dev
    band_module.wait_signals = _cuda_wait_signals
    try:
        with torch.cuda.stream(dev.S):
            yield
    finally:
        band_module.wait_signals = _REAL_WAIT_SIGNALS
        _ACTIVE_SIM = saved


# ---------------------------------------------------------------------------
# The model: requests, expected payload, operations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Req:
    rid: int
    row: int
    pages: List[int]
    seq: int
    status: str = "running"  # running, loading, held, published, sending
    slot: Optional[int] = None
    track: List[int] = dataclasses.field(default_factory=list)
    inflight: int = 0
    locks: int = 0
    hold: Optional[object] = None
    transfer: Optional[NetworkTransfer] = None
    engine: Optional[Actor] = None
    load: Optional[Actor] = None
    backups: List[Actor] = dataclasses.field(default_factory=list)
    alloc_clock: int = 0
    inline: bool = False
    ready: Optional[Actor] = None
    last_take: int = 0
    last_retired_launch: int = -1
    held_rows: Optional[Dict[str, List[int]]] = None
    held_epoch: int = 0
    alloc_tail: Optional[Actor] = None
    # The positions from which each unprocessed forward that includes the
    # request reads its window rows.
    window_reads_from: List[int] = dataclasses.field(default_factory=list)

    @property
    def inflight_steps(self) -> int:
        """Unprocessed forwards that include the request, as the scheduler
        counts them."""
        return self.inflight


@dataclasses.dataclass
class Forward:
    actor: Actor
    event: object
    reqs: List[Req]
    spec: Dict[int, int]
    launch_clock: int
    # The step's launch record, which its processing hands back.
    launch: object = None

    @property
    def rids(self) -> List[int]:
        return [r.rid for r in self.reqs]


class Model:
    """One rank: the allocator, its device, the requests and the payload each
    virtual page must hold."""

    def __init__(
        self, cfg: StressConfig, rank: int, device: str = "cpu", seed: int = 0
    ):
        self.cfg = cfg
        self.rank = rank
        self.device = device
        self.arena, self.alloc = build_chain(cfg, device)
        self.dev = (
            SimDevice(self.arena, rank)
            if device == "cpu"
            else CudaDevice(self.arena, rank, seed)
        )
        self.dev.describe = self.describe
        self.bands = {b.sub_pool_name: b for b in self.alloc.bands}
        self.owner = self.alloc.full_attn_allocator
        self.windows = list(self.alloc._windows)
        self.state = self.alloc.mamba_allocator
        self.psw = self.owner.page_size
        self.dcp = self.psw // self.owner.pool_page_size
        self.alloc.set_network_waiter(self._network_waiter)
        self.reqs: Dict[int, Req] = {}
        self.forwards: List[Forward] = []
        self.free_rows = list(range(cfg.max_reqs + 4))[::-1]
        max_tokens = (cfg.max_context_pages + 2) * self.psw
        self.table = torch.full(
            (cfg.max_reqs + 4, max_tokens), -1, dtype=torch.int64, device=device
        )
        # Expected rows per band: virtual page -> [value], and the actor whose
        # write is still landing, per row.
        self.exp: Dict[str, Dict[int, List[int]]] = {n: {} for n in self.bands}
        self.pend: Dict[str, Dict[int, List[Optional[Actor]]]] = {
            n: {} for n in self.bands
        }
        self.stamp = 0
        self.n_transfers = 0
        self.counts: Dict[str, int] = {}
        self.max_dead_fences = 0
        self.decisions: List[tuple] = []
        self.op_trace: List[tuple] = []
        # The storage's receive-row bookkeeping runs on the harness's tables.
        self.state_of_row = torch.full(
            (cfg.max_reqs + 4, 1), -1, dtype=torch.int64, device=device
        )
        self.storage = SimpleNamespace(
            _receive_holds={},
            _allocator=self.alloc,
            requests=SimpleNamespace(
                req_to_token=self.table,
                req_index_to_mamba_index_mapping=self.state_of_row,
            ),
        )
        self.storage._member = KVStorage._member.__get__(self.storage)

    # -- geometry --

    def rows(self, band: Band) -> int:
        return band.pool_page_size

    def epp(self, band: Band) -> int:
        return band.entry_bytes_per_page

    def describe(self, off: int) -> str:
        parts = []
        for name, band in self.bands.items():
            page = off // band.entry_bytes_per_page
            v = (
                int(band.ids.physical_to_virtual[page])
                if page < band.num_pages
                else None
            )
            parts.append(f"{name}:p{page}(v{v})")
        return "= " + " ".join(parts)

    def t(self, values) -> torch.Tensor:
        return torch.tensor(list(values), dtype=torch.int64, device=self.device)

    def next_stamp(self) -> int:
        self.stamp += 1
        return self.stamp

    def v2p(self, band: Band, vpage: int) -> int:
        return int(band.ids.virtual_to_physical[vpage])

    def tokens(self, req: Req, start: int, end: int) -> torch.Tensor:
        return self.table[req.row, start:end]

    def _set_table(self, req: Req) -> None:
        psw = self.psw
        n = len(req.pages) * psw
        ids = (
            self.t(req.pages)[:, None] * psw
            + torch.arange(psw, device=self.device)[None, :]
        ).reshape(-1)
        self.table[req.row, :n] = ids
        self.table[req.row, n:] = -1

    # -- allocator calls, under the host-read discipline --

    @contextlib.contextmanager
    def call(self, name: str):
        global _READ_GUARD
        dev = self.dev
        dev.in_call = name
        dev.call_index += 1
        saved = _READ_GUARD
        _READ_GUARD = dev
        try:
            yield
        except Violation:
            raise
        except Exception as e:  # noqa: BLE001
            raise Violation(
                "crash",
                f"rank {self.rank}: {name} raised {type(e).__name__}: {e}\n"
                + "".join(traceback.format_exc().splitlines(True)[-12:]),
            )
        finally:
            _READ_GUARD = saved
            dev.in_call = None

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    # -- the arena's copies and zeroing, as schedule-stream actors --

    def install_arena_hooks(self) -> None:
        arena = self.arena
        real_move, real_zero = arena.move_units, arena.zero_units
        dev = self.dev
        if not isinstance(dev, SimDevice):
            # Real streams: the copies run as issued; record the decisions.
            # Counts only: reading the ids back would synchronize the stream.
            def move_units_cuda(name, dst, src):
                dev.decisions.append(("copy", name, int(dst.numel())))
                self.count("page_copies", int(dst.numel()))
                real_move(name, dst, src)

            def zero_units_cuda(name, units):
                dev.decisions.append(("zero", name, int(units.numel())))
                self.count("page_zeroes", int(units.numel()))
                real_zero(name, units)

            arena.move_units = move_units_cuda
            arena.zero_units = zero_units_cuda
            return

        def move_units(name, dst, src):
            with _unguarded():
                band = self.bands[name]
                epp = band.entry_bytes_per_page
                d, s = dst.tolist(), src.tolist()
                self.count("page_copies", len(d))
                dev.stats["copies"] += len(d)

                def effect(a):
                    real_move(name, dst, src)
                    for sp, dp in zip(s, d):
                        dev.kind[dp * epp : (dp + 1) * epp] = dev.kind[
                            sp * epp : (sp + 1) * epp
                        ]
                        if (dev.kind[dp * epp : (dp + 1) * epp] == KIND_STATE).any():
                            dev.check_state_landing(dp * epp, (dp + 1) * epp, repr(a))

                dev.decisions.append(("copy", name, tuple(s), tuple(d)))
                dev.schedule_op(
                    "copy",
                    f"{name} copy {s}->{d}",
                    [(p * epp, (p + 1) * epp) for p in s],
                    [(p * epp, (p + 1) * epp) for p in d],
                    effect,
                )

        def zero_units(name, units):
            with _unguarded():
                band = self.bands[name]
                epp = band.entry_bytes_per_page
                u = units.tolist()
                self.count("page_zeroes", len(u))
                dev.stats["zeroes"] += len(u)

                def effect(_a):
                    real_zero(name, units)
                    for p in u:
                        dev.kind[p * epp : (p + 1) * epp] = KIND_ZERO

                dev.decisions.append(("zero", name, tuple(u)))
                dev.schedule_op(
                    "zero",
                    f"{name} zero {u}",
                    [],
                    [(p * epp, (p + 1) * epp) for p in u],
                    effect,
                )

        arena.move_units = move_units
        arena.zero_units = zero_units

    def _network_waiter(self, transfers) -> None:
        dev = self.dev
        ids = tuple(sorted(dev.transfer_ids[id(t)] for t in transfers))
        dev.decisions.append(("network_wait", ids))
        dev.stats["network_waits"] += 1
        self.count("network_waits")
        with _unguarded():
            for t in transfers:
                actor = dev.transfers.get(id(t))
                if actor is not None:
                    dev.host_sync(actor)

    # -- expected payload --

    def _register(self, band: Band, vpages: Sequence[int], value: int) -> None:
        rows = self.rows(band)
        for v in vpages:
            self.exp[band.sub_pool_name][v] = [value] * rows
            self.pend[band.sub_pool_name][v] = [None] * rows

    def _forget(self, band: Band, vpages: Sequence[int]) -> None:
        for v in vpages:
            self.exp[band.sub_pool_name].pop(v, None)
            self.pend[band.sub_pool_name].pop(v, None)

    def _row_off(self, band: Band, vpage: int, row: int) -> int:
        phys = self.v2p(band, vpage)
        if phys < band.min_page_index:
            raise Violation(
                "lost_write",
                f"rank {self.rank}: {band.sub_pool_name} virtual page {vpage} is "
                f"unbound (v2p={phys}) while the harness holds it",
            )
        return phys * band.entry_bytes_per_page + row * band.entry_bytes

    def window_bound(self, band: Band, vpage: int) -> bool:
        return vpage in self.exp[band.sub_pool_name]

    def token_rows(self, req: Req, t: int):
        """``(band, vpage, row)`` for every row token ``t`` of ``req`` writes
        on this rank."""
        psw, dcp = self.psw, self.dcp
        j = t % psw
        if j % dcp:
            return []
        v = req.pages[t // psw]
        row = j // dcp
        out = [(self.owner, v, row)]
        for w in self.windows:
            if self.window_bound(w, v):
                out.append((w, v, row))
        return out

    def check_payload(self) -> None:
        """Every live virtual page holds what its last finished writer wrote."""
        raw = self.dev.host_raw()
        for name, pages in self.exp.items():
            band = self.bands[name]
            eb = band.entry_bytes
            v2p = band.ids.virtual_to_physical.cpu().numpy()
            p2v = band.ids.physical_to_virtual.cpu().numpy()
            pend = self.pend[name]
            for v, vals in pages.items():
                phys = int(v2p[v])
                if phys < band.min_page_index or int(p2v[phys]) != v:
                    raise Violation(
                        "lost_write",
                        f"rank {self.rank}: {name} virtual page {v} maps to {phys}, "
                        f"whose p2v is {int(p2v[phys]) if phys >= 0 else None}",
                    )
                base = phys * band.entry_bytes_per_page
                writers = pend[v]
                for row, want in enumerate(vals):
                    w = writers[row]
                    if w is not None:
                        if not w.done:
                            continue
                        writers[row] = None
                    if want == UNDEF:
                        continue
                    got = raw[base + row * eb : base + (row + 1) * eb].view(np.uint32)
                    if not (got == want).all():
                        raise Violation(
                            "lost_write",
                            f"rank {self.rank}: {name} virtual page {v} (physical "
                            f"{phys}) row {row} holds {sorted(set(got.tolist()))[:4]}, "
                            f"its last writer wrote {want}",
                        )

    def _write_effect(self, writes):
        dev = self.dev
        if not isinstance(dev, SimDevice):
            return None
        raw, kind = dev.raw, dev.kind

        def effect(a):
            for off, eb, stamp, k in writes:
                raw[off : off + eb].view(np.uint32)[:] = stamp
                kind[off : off + eb] = k
                if k == KIND_STATE:
                    dev.check_state_landing(off, off + eb, repr(a))

        return effect

    def _read_check(self, reads, label):
        if not isinstance(self.dev, SimDevice):
            return None
        raw = self.dev.raw

        def effect(actor):
            for off, eb, want in reads:
                got = raw[off : off + eb].view(np.uint32)
                if not (got == want).all():
                    raise Violation(
                        "lost_write",
                        f"rank {self.rank}: {label} read {sorted(set(got.tolist()))[:4]} "
                        f"at byte {off} {self.describe(off)}, expected {want}",
                    )

        return effect

    def _attn_read_check(self, actor, pages):
        kind = self.dev.kind
        for lo, hi in pages:
            if (kind[lo:hi] == KIND_STATE).any():
                raise Violation(
                    "cross_dtype_bytes",
                    f"rank {self.rank}: {actor!r} reads attention bytes [{lo}, {hi}) "
                    f"{self.describe(lo)} that hold FP32 state bytes",
                )

    # -- requests --

    def _new_req(self, rid: int, pages: List[int], seq: int, status: str) -> Req:
        row = self.free_rows.pop()
        req = Req(rid=rid, row=row, pages=list(pages), seq=seq, status=status)
        req.alloc_clock = self.dev.clock
        req.last_take = self.dev.clock
        self.reqs[rid] = req
        self._set_table(req)
        return req

    def _drop_req(self, req: Req) -> None:
        del self.reqs[req.rid]
        # The row is reused: what the ledger kept of it must be its own copy.
        self.table[req.row].fill_(-7)
        self.free_rows.append(req.row)

    def _pages_from_alloc(self, out: torch.Tensor) -> List[int]:
        with _unguarded():
            return (out[:: self.psw] // self.psw).tolist()

    def held(self, req: Req):
        """A release of ``req``'s pages: freed logically at once, its physical
        pages pending while a recorded forward is unprocessed, until that
        forward is processed. Counts the releases made pending."""
        steps = self.alloc.forward_steps
        if steps is not None and steps.in_flight():
            self.count("pending_releases")
        return contextlib.nullcontext()

    def _free_req(self, req: Req, why: str) -> None:
        with self.held(req):
            if req.pages:
                ids = self.tokens(req, 0, len(req.pages) * self.psw)
                with self.call(f"free({why})"):
                    self.alloc.free(ids)
                for band in (self.owner, *self.windows):
                    self._forget(band, req.pages)
            slots = ([req.slot] if req.slot is not None else []) + req.track
            if slots:
                with self.call(f"free_state({why})"):
                    self.state.free(self.t(slots))
                self._forget(self.state, slots)
        self.count(f"free_{why}")
        self._drop_req(req)

    # -- operations --

    def apply(self, op: Op) -> None:
        dev = self.dev
        dev.unfinished_wait = None
        dev.decisions = []
        getattr(self, "op_" + op.kind)(*op.args)
        self.count("op_" + op.kind)
        self.decisions.append(
            (op.kind, _normalized(dev.decisions), self.digest(), tuple(dev.decisions))
        )

    def digest(self) -> tuple:
        with _unguarded():
            out = []
            for name, band in self.bands.items():
                out.append(
                    (
                        name,
                        band._span_range(),
                        tuple(sorted(band._free_phys_pages.tolist())),
                        tuple(sorted(band.usage.pending_pages)),
                        tuple(sorted(band.fixed_pages())),
                        band.ids.virtual_to_physical.cpu().numpy().tobytes(),
                        band.ids.physical_to_virtual.cpu().numpy().tobytes(),
                        None
                        if band.free_virtual_ids is None
                        else band.free_virtual_ids.cpu().numpy().tobytes(),
                    )
                )
            out.append(self.alloc.pricing_snapshot())
            out.append(tuple(self.alloc.units_available(n) for n in self.bands))
            out.append(self.alloc.available_size())
            out.append(
                tuple(band._pending_until_retire_pages for band in self.bands.values())
            )
            return tuple(out)

    def op_tick(self, rank: int, f: float) -> None:
        if rank == self.rank:
            with _unguarded():
                self.dev.tick(f)

    def _running(self, rid) -> Optional[Req]:
        req = self.reqs.get(rid)
        return req if req is not None and req.status == "running" else None

    def op_step(self, items, spec_draft: int, state_new: Tuple, cow: Tuple) -> None:
        """One scheduler step: retire window pages out of reach, take state
        slots for new requests (with copies into them), take token pages for
        every item in one take, then launch the forward."""
        psw = self.psw
        cfg = self.cfg
        plan = []
        for rid, ntok in items:
            req = self.reqs.get(rid)
            if req is None:
                if len(self.reqs) >= cfg.max_reqs or rid in self.done_rids:
                    continue
                plan.append((rid, None, ntok))
            elif req.status == "running":
                plan.append((rid, req, ntok))
        if not plan:
            return
        # Window retirement of the stepping requests, before the take.
        for rid, req, ntok in plan:
            if req is not None:
                self._retire_windows(req)
        # State slots for new requests.
        new = [rid for rid, req, _ in plan if req is None]
        slots: Dict[int, int] = {}
        if new:
            with self.call("alloc_state_slots"):
                out = self.alloc.alloc_state_slots(len(new))
            self.dev.decisions.append(
                ("state_slots", None if out is None else tuple(out.tolist()))
            )
            if out is None:
                self.count("admission_refused")
                plan = [p for p in plan if p[1] is not None]
            else:
                ids = out.tolist()
                self._register(self.state, ids, UNDEF)
                slots = dict(zip(new, ids))
        # Token pages for the whole step, in one take.
        need = []
        for rid, req, ntok in plan:
            have = 0 if req is None else len(req.pages) * psw
            seq = 0 if req is None else req.seq
            want = seq + ntok + (spec_draft if req is not None else 0)
            if (want + psw - 1) // psw > cfg.max_context_pages:
                want = cfg.max_context_pages * psw
                ntok = max(
                    0, min(ntok, want - seq - (spec_draft if req is not None else 0))
                )
            extra = max(0, -(-want // psw) - (have // psw))
            need.append(extra)
        total = sum(need)
        new_pages: List[int] = []
        if total:
            with self.call("alloc"):
                out = self.alloc.alloc(total * psw)
            self.dev.decisions.append(
                ("alloc", total, None if out is None else self._pages_from_alloc(out))
            )
            if out is None:
                self.count("alloc_refused")
                # Admission refused: new requests give their state back.
                for rid, slot in slots.items():
                    with self.call("free_state(refused)"):
                        self.state.free(self.t([slot]))
                    self._forget(self.state, [slot])
                return
            new_pages = self._pages_from_alloc(out)
            for band in (self.owner, *self.windows):
                self._register(band, new_pages, 0)
        cursor = 0
        batch = []
        for (rid, req, ntok), extra in zip(plan, need):
            pages = new_pages[cursor : cursor + extra]
            cursor += extra
            if req is None:
                req = self._new_req(rid, pages, 0, "running")
                req.slot = slots[rid]
                self._cow_into(req, cow)
            else:
                req.pages.extend(pages)
                self._set_table(req)
            if pages or req.seq == 0:
                req.last_take = self.dev.clock
            if ntok > 0 or req.seq == 0:
                batch.append((req, max(ntok, 1)))
        if batch:
            self._launch(batch, spec_draft)

    done_rids: set = set()

    def _cow_into(self, req: Req, cow: Tuple) -> None:
        """Copy a donor's state into ``req``'s new slot on the schedule stream
        (a copy-on-write from a retired request's state)."""
        donors = [
            r
            for r in self.reqs.values()
            if r is not req
            and r.status == "running"
            and r.inflight == 0
            and r.slot is not None
        ]
        if not cow or not donors:
            return
        donor = donors[int(cow[0] * len(donors)) % len(donors)]
        band = self.state
        src = self.v2p(band, donor.slot)
        dst = self.v2p(band, req.slot)
        eb = band.entry_bytes_per_page
        dev = self.dev

        def effect(_a):
            dev.raw[dst * eb : (dst + 1) * eb] = dev.raw[src * eb : (src + 1) * eb]
            dev.kind[dst * eb : (dst + 1) * eb] = dev.kind[src * eb : (src + 1) * eb]

        dev.schedule_op(
            "state_copy",
            f"state copy {donor.slot}->{req.slot}",
            [(src * eb, (src + 1) * eb)],
            [(dst * eb, (dst + 1) * eb)],
            effect,
            copy=(src * eb, dst * eb, eb),
        )
        self.exp["state"][req.slot] = list(self.exp["state"][donor.slot])
        self.count("state_copies")

    def _retire_windows(self, req: Req) -> None:
        if req.locks:
            # A host copy reads the rows: their owner keeps them until it ends.
            return
        psw = self.psw
        # Rows below the window of every forward that includes the request
        # and is unprocessed: what the scheduler's window retirement frees.
        reads_from = min([req.seq, *req.window_reads_from])
        for w, wlen in zip(self.windows, self.cfg.window_lens):
            keep_from = max(0, reads_from - wlen)
            out = [
                v
                for i, v in enumerate(req.pages)
                if (i + 1) * psw <= keep_from and self.window_bound(w, v)
            ]
            if not out:
                continue
            ids = torch.cat(
                [
                    self.tokens(
                        req, req.pages.index(v) * psw, (req.pages.index(v) + 1) * psw
                    )
                    for v in out
                ]
            )
            # Held like any free while a forward is in flight, as the
            # storage's advance does.
            with self.call(f"free_window({w.sub_pool_name})"):
                self.alloc.free_window(w.sub_pool_name, ids)
            self._forget(w, out)
            self.count("window_retirements", len(out))

    def _launch(self, batch, spec_draft: int) -> None:
        dev = self.dev
        writes, reads, attn_reads = [], [], []
        written_tokens, state_written = [], []
        spec = {}
        for req, ntok in batch:
            start = req.seq
            end = req.seq + ntok + (spec_draft if spec_draft else 0)
            end = min(end, len(req.pages) * self.psw)
            for t in range(start, end):
                stamp = self.next_stamp()
                for band, v, row in self.token_rows(req, t):
                    off = self._row_off(band, v, row)
                    writes.append(
                        (
                            off,
                            band.entry_bytes,
                            stamp,
                            KIND_ATTN,
                            band.sub_pool_name,
                            v,
                            row,
                        )
                    )
            written_tokens.append(self.tokens(req, start, end))
            # The forward reads every page its page table names.
            for band in (self.owner, *self.windows):
                for i, v in enumerate(req.pages):
                    if band is not self.owner:
                        wlen = self.cfg.window_lens[self.windows.index(band)]
                        if (i + 1) * self.psw <= max(
                            0, req.seq - wlen
                        ) or not self.window_bound(band, v):
                            continue
                    p = self.v2p(band, v)
                    lo = p * band.entry_bytes_per_page
                    rng = (lo, lo + band.entry_bytes_per_page)
                    reads.append(rng)
                    attn_reads.append(rng)
            for slot in [req.slot, *req.track]:
                p = self.v2p(self.state, slot)
                eb = self.state.entry_bytes_per_page
                stamp = self.next_stamp()
                writes.append((p * eb, eb, stamp, KIND_STATE, "state", slot, 0))
                reads.append((p * eb, (p + 1) * eb))
                state_written.append(slot)
            req.inflight += 1
            req.window_reads_from.append(start)
            spec[req.rid] = end - start
        unknown = spec_draft > 0
        label = f"forward rids={[r.rid for r, _ in batch]}{' spec' if unknown else ''}"
        attn = list(attn_reads)
        actor = dev.launch(
            "forward",
            "F",
            label,
            [dev.svc],
            reads,
            [(off, off + eb) for off, eb, *_ in writes],
            None,
            payload=[(off, eb, st) for off, eb, st, *_ in writes],
        )
        write_effect = self._write_effect(
            [(off, eb, s, k) for off, eb, s, k, *_ in writes]
        )

        def effect(a):
            self._attn_read_check(a, attn)
            write_effect(a)

        if write_effect is not None:
            actor.effect = effect
        actor.attn_reads = tuple(attn)
        for off, eb, stamp, k, name, v, row in writes:
            self.exp[name][v][row] = stamp
            self.pend[name][v][row] = actor
        if len(written_tokens) == 1:
            tokens = written_tokens[0]  # a view of the request's table row
        else:
            tokens = torch.cat(written_tokens)
        state = self.t(state_written)
        # The scheduler's step launch, then its forward's event.
        steps = self.alloc.forward_steps
        launch = steps.forward_launched()
        steps.forward_recorded(actor.handle)
        with self.call("record_forward"):
            self.alloc.set_latest_forward_done_event(actor.handle)
            self.alloc.set_inflight_forward(
                actor.handle,
                tokens,
                state,
                token_writes_unknown=unknown,
                # A speculative batch's tracking slots are not readable
                # before it ends either.
                state_writes_unknown=unknown,
            )
        self.forwards.append(
            Forward(
                actor,
                actor.handle,
                [r for r, _ in batch],
                spec if unknown else {},
                dev.clock,
                launch,
            )
        )
        self.count("forwards")
        if len(self.forwards) > 1:
            self.count("overlapped_forwards")
        # Logical sequence advance (the accepted length of a spec step is
        # settled at its retirement).
        for req, ntok in batch:
            req.seq = min(req.seq + ntok, len(req.pages) * self.psw)

    def op_retire(self, finish: Tuple, accept: float) -> None:
        """Process the oldest forward's result: after the host saw it end, its
        spec over-allocation and the requests it finishes are released (their
        pages pending while a recorded forward is unprocessed, this one
        included), then the step is processed, as the scheduler does: the
        forward retires and the pages pending under it are released."""
        if not self.forwards:
            return
        fwd = self.forwards.pop(0)
        dev = self.dev
        with _unguarded():
            dev.host_sync(fwd.actor)
        for req in fwd.reqs:
            req.last_retired_launch = max(req.last_retired_launch, fwd.launch_clock)
            if req.rid in fwd.spec and self.reqs.get(req.rid) is req:
                # Release over-allocated pages past the accepted tokens.
                keep_pages = -(-req.seq // self.psw)
                extra = req.pages[keep_pages:]
                if extra and req.status == "running":
                    start = keep_pages * self.psw
                    ids = self.tokens(req, start, len(req.pages) * self.psw)
                    with self.held(req), self.call("free(overallocated)"):
                        self.alloc.free(ids.clone())
                    for band in (self.owner, *self.windows):
                        self._forget(band, extra)
                    del req.pages[keep_pages:]
                    self._set_table(req)
                    self.count("spec_overalloc_frees")
        for rid in finish:
            req = self.reqs.get(rid)
            if req is not None and req.status == "running" and req.locks == 0:
                self.done_rids = self.done_rids | {rid}
                self._free_req(req, "finish")
        for req in fwd.reqs:
            req.inflight -= 1
            req.window_reads_from.pop(0)
        with self.call("forward_processed"):
            self.alloc.forward_steps.forward_processed(fwd.launch)

    def op_free(self, rid: int) -> None:
        """Retract or abort a running request, possibly under its own running
        forward."""
        req = self._running(rid)
        if req is None or req.locks:
            return
        self.done_rids = self.done_rids | {rid}
        self._free_req(req, "retract" if req.inflight else "abort")

    def op_window_retire(self, rid: int) -> None:
        req = self._running(rid)
        if req is not None:
            self._retire_windows(req)

    def op_flush(self) -> None:
        with self.call("flush_opportunistic"):
            n = self.alloc.flush_opportunistic()
        self.dev.decisions.append(("flush", n))

    def op_relief(self, band_index: int, frac: float) -> None:
        names = list(self.bands)
        name = names[band_index % len(names)]
        band = self.bands[name]
        units = max(1, int(frac * max(1, band.num_pages - band.min_page_index)))
        with self.call(f"lay_out_for({name})"):
            ok = self.alloc.lay_out_for({name: units})
        self.dev.decisions.append(("relief", name, units, ok))
        self.count("reliefs")

    def op_state_take(self, rid: int) -> None:
        req = self._running(rid)
        if req is None or len(req.track) >= 2:
            return
        with self.call("alloc_state_slots(track)"):
            out = self.alloc.alloc_state_slots(1)
        self.dev.decisions.append(
            ("track", None if out is None else tuple(out.tolist()))
        )
        if out is not None:
            slot = out.tolist()[0]
            self._register(self.state, [slot], UNDEF)
            req.track.append(slot)
            req.last_take = self.dev.clock
            self.count("state_takes")

    def op_state_free(self, rid: int) -> None:
        req = self._running(rid)
        if req is None or not req.track or req.locks:
            return
        slot = req.track.pop(0)
        with self.held(req), self.call("free_state(track)"):
            self.state.free(self.t([slot]))
        self._forget(self.state, [slot])
        self.count("state_frees")

    # -- host cache --

    def _host_ids(self, req: Req, band: Band, vpages: Sequence[int]) -> torch.Tensor:
        """The ids a host copy records for ``vpages`` of ``band``: the
        owner's token ids (a view of the request's row) when a virtual page is
        one physical page, otherwise one id per page naming its first row."""
        if band is self.owner and band.page_size == band.pool_page_size:
            first = req.pages.index(vpages[0])
            if req.pages[first : first + len(vpages)] == list(vpages):
                return self.tokens(
                    req, first * self.psw, (first + len(vpages)) * self.psw
                )
        return self.t(list(vpages)) * band.pool_page_size

    def _page_writes(self, req: Req, stamps: bool = True):
        """Whole-page writes of every row ``req`` holds, at the physical pages
        its virtual pages map to now: what a copy or a transfer that fills
        them writes."""
        writes = []
        for band in (self.owner, *self.windows, self.state):
            k = KIND_STATE if band is self.state else KIND_ATTN
            for v in self._req_pages(req, band):
                for row in range(self.rows(band)):
                    off = self._row_off(band, v, row)
                    writes.append(
                        (
                            off,
                            band.entry_bytes,
                            self.next_stamp(),
                            k,
                            band.sub_pool_name,
                            v,
                            row,
                        )
                    )
        return writes

    def _expect(self, writes, actor) -> None:
        for off, eb, stamp, k, name, v, row in writes:
            self.exp[name][v][row] = stamp
            self.pend[name][v][row] = actor

    def op_load(self, rid: int, npages: int) -> None:
        """A request whose prefix loads from the host cache: the owner's
        pages, each window's tail and a state slot, taken under one
        reservation, then written by a copy that runs behind the forwards."""
        if (
            rid in self.reqs
            or rid in self.done_rids
            or len(self.reqs) >= self.cfg.max_reqs
        ):
            return
        npages = max(1, min(npages, self.cfg.max_context_pages - 1))
        psw = self.psw
        seq = npages * psw
        tails = {}
        for w, wlen in zip(self.windows, self.cfg.window_lens):
            tails[w.sub_pool_name] = min(npages, -(-wlen // psw))
        demand = KVDemand(seq, {n: t * psw for n, t in tails.items()}, 1)
        dev = self.dev
        scope = self.alloc.scoped_physical_takes(demand)
        with self.call("scoped_physical_takes"):
            try:
                scope.__enter__()
            except RuntimeError as e:
                if "reclamation" not in str(e):
                    raise
                scope = None
        if scope is None:
            dev.decisions.append(("load", "refused"))
            self.count("load_refused")
            return
        try:
            with self.call("load owner alloc"):
                out = self.owner.alloc(seq)
            if out is None:
                raise Violation(
                    "crash",
                    f"rank {self.rank}: a reserved load take of the owner was refused",
                )
            pages = self._pages_from_alloc(out)
            window_phys = {}
            for w in self.windows:
                n = tails[w.sub_pool_name]
                if not n:
                    continue
                with self.call(f"load alloc_physical({w.sub_pool_name})"):
                    phys = w.alloc_physical(n * psw)
                if phys is None:
                    raise Violation(
                        "crash",
                        f"rank {self.rank}: a reserved load take of {w.sub_pool_name} was refused",
                    )
                with _unguarded():
                    window_phys[w.sub_pool_name] = (
                        phys[:: w.page_size] // w.page_size
                    ).tolist()
            with self.call("load state slot"):
                slot_out = self.alloc.alloc_state_slots(1)
            if slot_out is None:
                raise Violation(
                    "crash",
                    f"rank {self.rank}: a reserved load take of a state slot was refused",
                )
        finally:
            if scope is not None:
                with self.call("scope exit"):
                    scope.__exit__(None, None, None)
        slot = slot_out.tolist()[0]
        req = self._new_req(rid, pages, seq, "loading")
        req.slot = slot
        self._register(self.owner, pages, 0)
        self._register(self.state, [slot], UNDEF)
        for w in self.windows:
            pp = window_phys.get(w.sub_pool_name)
            if not pp:
                continue
            tail_v = pages[len(pages) - len(pp) :]
            # The tree binds the window rows to the owner's ids (as
            # `set_full_to_swa_mapping` does).
            w.ids.bind(self.t(tail_v), self.t(pp))
            self._register(w, tail_v, UNDEF)
        dev.decisions.append(
            ("load", tuple(pages), tuple(sorted(window_phys.items())), slot)
        )
        # The copy fills whole pages, translated at its submission.
        writes = self._page_writes(req)
        actor = dev.launch(
            "load",
            "L",
            f"load rid={rid}",
            [dev.svc, dev.vcs["F"]],
            [],
            [(off, off + eb) for off, eb, *_ in writes],
            self._write_effect([(off, eb, s, k) for off, eb, s, k, *_ in writes]),
            payload=[(off, eb, st) for off, eb, st, *_ in writes],
        )
        self._expect(writes, actor)
        req.load = actor
        with self.call("record_host_transfer(load)"):
            for band in (self.owner, *self.windows):
                vp = self._req_pages(req, band)
                if vp:
                    band.record_transfer(
                        actor.handle, self._host_ids(req, band, vp), writes=True
                    )
            self.state.record_transfer(actor.handle, self.t([slot]), writes=True)
            self.alloc.end_load_reservations()
        self.count("loads")

    def _req_pages(self, req: Req, band: Band) -> List[int]:
        if band is self.state:
            return [s for s in [req.slot, *req.track] if s is not None]
        return [v for v in req.pages if v in self.exp[band.sub_pool_name]]

    def op_load_done(self, rid: int) -> None:
        req = self.reqs.get(rid)
        if req is None or req.status != "loading":
            return
        with _unguarded():
            self.dev.host_sync(req.load)
        req.status = "running"
        req.load = None

    def op_backup(self, rid: int) -> None:
        """Back a retired request's full pages up to the host: a copy that
        reads them behind the forwards; the rows stay locked until it is
        acknowledged."""
        req = self._running(rid)
        if req is None or req.inflight:
            return
        full = req.seq // self.psw
        if not full:
            return
        dev = self.dev
        reads, ranges = [], []
        backed = {}
        for band in (self.owner, *self.windows):
            vp = [v for v in req.pages[:full] if v in self.exp[band.sub_pool_name]]
            backed[band.sub_pool_name] = vp
            for v in vp:
                for row, want in enumerate(self.exp[band.sub_pool_name][v]):
                    off = self._row_off(band, v, row)
                    if want != UNDEF:
                        reads.append((off, band.entry_bytes, want))
                    ranges.append((off, off + band.entry_bytes))
        actor = dev.launch(
            "backup",
            "B",
            f"backup rid={rid}",
            [dev.svc, dev.vcs["F"]],
            ranges,
            [],
            self._read_check(reads, f"backup rid={rid}"),
            expect=reads,
        )
        with self.call("record_host_transfer(backup)"):
            for band in (self.owner, *self.windows):
                vp = backed[band.sub_pool_name]
                if vp:
                    band.record_transfer(
                        actor.handle, self._host_ids(req, band, vp), writes=False
                    )
        req.locks += 1
        req.backups.append(actor)
        self.count("backups")

    def op_backup_done(self, rid: int) -> None:
        req = self.reqs.get(rid)
        if req is None or not req.backups:
            return
        actor = req.backups.pop(0)
        with _unguarded():
            self.dev.host_sync(actor)
        req.locks -= 1

    # -- network --

    def _new_transfer(self) -> NetworkTransfer:
        transfer = NetworkTransfer(peer=None)
        self.n_transfers += 1
        self.dev.transfer_ids[id(transfer)] = self.n_transfers
        return transfer

    def _physical_rows(self, req: Req) -> Dict[str, List[int]]:
        return {
            band.sub_pool_name: [self.v2p(band, v) for v in self._req_pages(req, band)]
            for band in (self.owner, *self.windows, self.state)
        }

    def _record_network(
        self, transfer, rows: Dict[str, List[int]], writes: bool
    ) -> None:
        """Record one send or receive by the physical rows it addresses."""
        with self.call("record_network_transfer"):
            for name, pages in rows.items():
                if pages:
                    band = self.bands[name]
                    band.record_transfer(
                        transfer,
                        self.t(pages),
                        writes=writes,
                        release=transfer,
                        unit_tokens=band.pool_page_size,
                    )

    def op_send(self, rid: int) -> None:
        """A prefill node sends a retired request's rows: a network read that
        runs outside every stream until its logical release."""
        req = self._running(rid)
        if req is None or req.inflight or req.locks:
            return
        if req.last_retired_launch < req.last_take:
            # A prefill send follows the retirement of a forward that wrote
            # every row it sends.
            return
        dev = self.dev
        with _unguarded():
            # The sender reads the rows' ids after the schedule stream.
            dev.observe_schedule()
        transfer = self._new_transfer()
        reads, ranges = [], []
        for band in (self.owner, *self.windows, self.state):
            for v in self._req_pages(req, band):
                for row, want in enumerate(self.exp[band.sub_pool_name][v]):
                    off = self._row_off(band, v, row)
                    if want != UNDEF:
                        reads.append((off, band.entry_bytes, want))
                    ranges.append((off, off + band.entry_bytes))
        actor = dev.launch(
            "send",
            f"N{self.n_transfers}",
            f"send rid={rid} t{self.n_transfers}",
            [],
            ranges,
            [],
            self._read_check(reads, f"send rid={rid}"),
            expect=reads,
        )
        dev.transfers[id(transfer)] = actor
        self._record_network(transfer, self._physical_rows(req), writes=False)
        req.status = "sending"
        req.transfer = transfer
        req.engine = actor
        self.count("sends")

    def op_send_release(self, rid: int, free: bool) -> None:
        req = self.reqs.get(rid)
        if req is None or req.status != "sending":
            return
        with _unguarded():
            self.dev.host_sync(req.engine)
        with self.call("end_network_transfer(send)"):
            req.transfer.released = True
        req.status = "running"
        req.transfer = req.engine = None
        if free:
            self.done_rids = self.done_rids | {rid}
            self._free_req(req, "sent")

    def move_epoch(self) -> int:
        return sum(band.held_row_moves for band in self.alloc.bands)

    def op_receive(self, rid: int, npages: int, inline: bool) -> None:
        """A decode node takes rows for a request a prefill node will send,
        and holds them until their ids are published (`hold_receive_rows`)."""
        if (
            rid in self.reqs
            or rid in self.done_rids
            or len(self.reqs) >= self.cfg.max_reqs
        ):
            return
        npages = max(1, min(npages, self.cfg.max_context_pages - 1))
        psw = self.psw
        with self.call("alloc_state_slots(receive)"):
            slot_out = self.alloc.alloc_state_slots(1)
        if slot_out is None:
            self.dev.decisions.append(("receive", "refused"))
            return
        with self.call("alloc(receive)"):
            out = self.alloc.alloc(npages * psw)
        slot = slot_out.tolist()[0]
        if out is None:
            with self.call("free_state(refused)"):
                self.state.free(slot_out)
            self.dev.decisions.append(("receive", "refused"))
            return
        pages = self._pages_from_alloc(out)
        self.dev.decisions.append(("receive", tuple(pages), slot))
        req = self._new_req(rid, pages, npages * psw, "held")
        req.slot = slot
        req.inline = inline
        for band in (self.owner, *self.windows):
            self._register(band, pages, 0)
        self._register(self.state, [slot], UNDEF)
        dev = self.dev
        if inline:
            # The inline path sends once a forward-stream event recorded now
            # fired, and holds the rows with it.
            marker = dev.launch("marker", "F", f"ready rid={rid}", [], [], [], None)
        else:
            marker = dev.schedule_op("marker", f"allocated rid={rid}", [], [], None)
        signal = marker.handle
        req.ready = marker
        # The storage's own hold: the request row's token ids for the owner
        # and every window, and the state slot, until the publish.
        self.state_of_row[req.row, 0] = slot
        handle = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=req.row, holds_mamba=True)
        )
        with self.call("hold_receive_rows"):
            KVStorage.hold_receive_rows(
                self.storage,
                handle,
                token_start=0,
                token_end=npages * psw,
                signal=signal,
            )
        _, req.hold = self.storage._receive_holds.pop(id(handle))
        req.alloc_clock = dev.clock
        # The schedule-stream work up to the allocation: the publish reads
        # the rows' ids back after it.
        req.alloc_tail = dev.tails.get("S")
        req.held_rows = self._physical_rows(req)
        req.held_epoch = self.move_epoch()
        self.count("receives")

    def op_publish(self, rid: int) -> None:
        """Publish held rows: their ids reach the peer, which writes them by
        physical id. The ids are those read at the hold unless a page move
        since then bumped the move epoch (the reader orders itself after the
        schedule stream only then)."""
        req = self.reqs.get(rid)
        if req is None or req.status != "held":
            return
        dev = self.dev
        if req.inline:
            # The inline path sends once the forward-stream event recorded at
            # the allocation fired: the host waits for it here.
            if req.ready is not None:
                with _unguarded():
                    dev.host_sync(req.ready)
        elif any(f.launch_clock <= req.alloc_clock for f in self.forwards):
            # Deferred publish: only once every forward launched before the
            # allocation was retired.
            return
        if req.alloc_tail is not None:
            with _unguarded():
                dev.observe(req.alloc_tail)
        if self.move_epoch() == req.held_epoch:
            rows = req.held_rows
            stale = rows != self._physical_rows(req)
        else:
            with _unguarded():
                dev.observe_schedule()
            rows = self._physical_rows(req)
            stale = False
        if stale:
            self.count("publish_stale_ids")
        with self.call("end_receive_hold"):
            req.hold.released = True
        transfer = self._new_transfer()
        writes = []
        for name, pages in rows.items():
            band = self.bands[name]
            k = KIND_STATE if band is self.state else KIND_ATTN
            vpages = self._req_pages(req, band)
            for v, p in zip(vpages, pages):
                for row in range(self.rows(band)):
                    off = p * band.entry_bytes_per_page + row * band.entry_bytes
                    writes.append(
                        (off, band.entry_bytes, self.next_stamp(), k, name, v, row)
                    )
        actor = dev.launch(
            "receive",
            f"N{self.n_transfers}",
            f"receive rid={rid} t{self.n_transfers}{' inline' if req.inline else ''}",
            [],
            [],
            [(off, off + eb) for off, eb, *_ in writes],
            self._write_effect([(off, eb, s, k) for off, eb, s, k, *_ in writes]),
            payload=[(off, eb, st) for off, eb, st, *_ in writes],
        )
        self._expect(writes, actor)
        dev.transfers[id(transfer)] = actor
        self._record_network(transfer, rows, writes=True)
        dev.decisions.append(("publish", rid, stale))
        req.status = "published"
        req.transfer = transfer
        req.engine = actor
        self.count("publishes")

    def op_receive_commit(self, rid: int) -> None:
        req = self.reqs.get(rid)
        if req is None or req.status != "published":
            return
        with _unguarded():
            self.dev.host_sync(req.engine)
        with self.call("end_network_transfer(receive)"):
            req.transfer.released = True
        req.status = "running"
        req.transfer = req.engine = None

    # -- quiescence --

    def op_quiesce(self, reset: bool) -> None:
        """Retire everything, end every transfer, then check the ledger keeps
        nothing of what ended; with ``reset``, free every request and clear
        the allocator, which must leave the ledger empty."""
        dev = self.dev
        with _unguarded():
            for a in list(dev.pending):
                dev.host_sync(a)
        while self.forwards:
            self.op_retire((), 0.0)
        for req in list(self.reqs.values()):
            if req.status == "loading":
                self.op_load_done(req.rid)
            while req.backups:
                self.op_backup_done(req.rid)
            if req.status == "sending":
                self.op_send_release(req.rid, False)
            if req.status == "held":
                self.op_publish(req.rid)
            if req.status == "published":
                self.op_receive_commit(req.rid)
        with _unguarded():
            for a in list(dev.pending):
                dev.host_sync(a)
        with self.call("flush_opportunistic(quiesce)"):
            self.alloc.flush_opportunistic()
        self.check_ledger()
        if reset:
            for req in list(self.reqs.values()):
                self._free_req(req, "reset")
            with self.call("flush_opportunistic(reset)"):
                self.alloc.flush_opportunistic()
            self.check_ledger()
            with self.call("clear"):
                self.alloc.clear()
            for band in self.bands.values():
                u = band.usage
                if u.uses or u.fixed or u.pending_pages or u.event_seq:
                    raise Violation(
                        "ledger_leak",
                        f"rank {self.rank}: {band.sub_pool_name} keeps {len(u.uses)} uses, "
                        f"{len(u.fixed)} fixed keys, {len(u.pending_pages)} pending pages, "
                        f"{len(u.event_seq)} events after clear()",
                    )
            for n in self.exp:
                self.exp[n].clear()
                self.pend[n].clear()
            self.count("resets")
        self.count("quiesces")

    def check_ledger(self) -> None:
        for name, band in self.bands.items():
            u = band.usage
            with _unguarded():
                u.prune()
            kinds = {}
            for use in u.uses:
                kinds[use.kind] = kinds.get(use.kind, 0) + 1
            bad = []
            if u.forwards():
                bad.append(f"{len(u.forwards())} forward records")
            if u.transfers():
                bad.append(f"{len(u.transfers())} transfer records")
            if u.pending_pages:
                bad.append(f"{len(u.pending_pages)} pending pages")
            fixed = u.fixed_pages()
            if fixed:
                bad.append(f"fixed pages {sorted(fixed)} under {list(u.fixed)}")
            held = set()
            for use in u.uses:
                if use.kind == VACATED:
                    held.update(id(k) for k in signals_module.keys_of(use.signal))
            # A retirement re-filters the event map: an event a vacated use
            # held at the last one may linger until the next.
            lingering = [e for e in u.event_seq if id(e) not in held]
            self.count("lingering_events", len(lingering))
            with _unguarded():
                u._retire_to(u.retired_seq)
            stray = [e for e in u.event_seq if id(e) not in held]
            if stray:
                bad.append(f"{len(stray)} retired events still mapped")
            dead = sum(1 for use in u.uses if use.kind == VACATED and use.pages is None)
            self.max_dead_fences = max(self.max_dead_fences, dead)
            limit = 4 * band.num_pages
            if dead > limit:
                bad.append(f"{dead} fences of ended work (> {limit})")
            if band._pending_until_retire:
                bad.append(
                    f"{band._pending_until_retire_pages} pages pending with no "
                    "forward in flight"
                )
            if bad:
                raise Violation(
                    "ledger_leak",
                    f"rank {self.rank}: {name} keeps, with nothing running: "
                    + "; ".join(bad),
                )


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------


class Generator:
    """Emits operations from the logical state of rank 0 (both ranks hold
    the same logical state while they agree)."""

    def __init__(self, rng: random.Random, cfg: StressConfig, ranks: int):
        self.rng = rng
        self.cfg = cfg
        self.ranks = ranks
        self.next_rid = 0
        self.weights = {
            "step": 24,
            "retire": 14,
            "tick": 22,
            "free": 3,
            "window_retire": 3,
            "flush": 4,
            "relief": 3,
            "state_take": 2,
            "state_free": 2,
            "load": 3,
            "load_done": 3,
            "backup": 3,
            "backup_done": 3,
            "send": 2,
            "send_release": 3,
            "receive": 3,
            "publish": 3,
            "receive_commit": 3,
            "quiesce": 0.3,
            "reset": 0.1,
        }

    def rid(self) -> int:
        self.next_rid += 1
        return self.next_rid

    def pick(self, xs):
        return xs[self.rng.randrange(len(xs))] if xs else None

    def next(self, m: Model) -> Op:
        rng = self.rng
        kinds = list(self.weights)
        while True:
            kind = rng.choices(kinds, [self.weights[k] for k in kinds])[0]
            op = getattr(self, "gen_" + kind)(m)
            if op is not None:
                return op

    def running(self, m: Model):
        return [r.rid for r in m.reqs.values() if r.status == "running"]

    def gen_step(self, m: Model):
        rng = self.rng
        if len(m.forwards) >= 3:
            return None
        items = []
        for rid in self.running(m):
            if rng.random() < 0.7:
                items.append(
                    (rid, 1 if rng.random() < 0.7 else rng.randint(1, 2 * m.psw))
                )
        if len(m.reqs) < m.cfg.max_reqs and rng.random() < 0.5:
            items.append((self.rid(), rng.randint(1, 3 * m.psw)))
        if not items:
            return None
        spec = rng.choice([0, 0, 0, 1, 2]) if rng.random() < 0.3 else 0
        return Op(
            "step",
            (tuple(items), spec, (), (rng.random(),) if rng.random() < 0.3 else ()),
        )

    def gen_retire(self, m: Model):
        if not m.forwards or (len(m.forwards) == 1 and self.rng.random() < 0.3):
            return None
        rids = m.forwards[0].rids
        finish = tuple(r for r in rids if self.rng.random() < 0.25)
        return Op("retire", (finish, self.rng.random()))

    def gen_tick(self, m: Model):
        return Op("tick", (self.rng.randrange(self.ranks), self.rng.random()))

    def gen_free(self, m: Model):
        rid = self.pick(self.running(m))
        return None if rid is None else Op("free", (rid,))

    def gen_window_retire(self, m: Model):
        rid = self.pick(self.running(m))
        return None if rid is None else Op("window_retire", (rid,))

    def gen_flush(self, m: Model):
        return Op("flush")

    def gen_relief(self, m: Model):
        return Op("relief", (self.rng.randrange(8), self.rng.random() * 0.6))

    def gen_state_take(self, m: Model):
        rid = self.pick(self.running(m))
        return None if rid is None else Op("state_take", (rid,))

    def gen_state_free(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.track])
        return None if rid is None else Op("state_free", (rid,))

    def gen_load(self, m: Model):
        if len(m.reqs) >= m.cfg.max_reqs:
            return None
        return Op("load", (self.rid(), self.rng.randint(1, 3)))

    def gen_load_done(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.status == "loading"])
        return None if rid is None else Op("load_done", (rid,))

    def gen_backup(self, m: Model):
        rid = self.pick(
            [r.rid for r in m.reqs.values() if r.status == "running" and not r.inflight]
        )
        return None if rid is None else Op("backup", (rid,))

    def gen_backup_done(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.backups])
        return None if rid is None else Op("backup_done", (rid,))

    def gen_send(self, m: Model):
        rid = self.pick(
            [
                r.rid
                for r in m.reqs.values()
                if r.status == "running" and not r.inflight and not r.locks
            ]
        )
        return None if rid is None else Op("send", (rid,))

    def gen_send_release(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.status == "sending"])
        return (
            None if rid is None else Op("send_release", (rid, self.rng.random() < 0.7))
        )

    def gen_receive(self, m: Model):
        if len(m.reqs) >= m.cfg.max_reqs:
            return None
        return Op(
            "receive", (self.rid(), self.rng.randint(1, 3), self.rng.random() < 0.3)
        )

    def gen_publish(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.status == "held"])
        return None if rid is None else Op("publish", (rid,))

    def gen_receive_commit(self, m: Model):
        rid = self.pick([r.rid for r in m.reqs.values() if r.status == "published"])
        return None if rid is None else Op("receive_commit", (rid,))

    def gen_quiesce(self, m: Model):
        return Op("quiesce", (False,))

    def gen_reset(self, m: Model):
        return Op("quiesce", (True,))


# ---------------------------------------------------------------------------
# Running, comparing ranks, minimizing
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RunResult:
    seed: int
    cfg: StressConfig
    ops: List[Op]
    failure: Optional[Violation]
    failed_at: Optional[int]
    counts: Dict[str, int]
    stats: Dict[str, int]
    max_dead_fences: int = 0
    step_reads: Dict[tuple, int] = dataclasses.field(default_factory=dict)


# Whether two ranks that move the same pages in different launches fail.
STRICT_BATCHING = False


def _normalized(decisions) -> tuple:
    """The decisions with page moves and zeroing flattened per band: how
    many launches a move takes is not a layout decision, which pages move
    where, and in what order, is."""
    out, moves, zeroes = [], {}, {}
    for d in decisions:
        if d[0] == "copy" and len(d) == 4:
            moves.setdefault(d[1], []).extend(zip(d[2], d[3]))
        elif d[0] == "zero" and len(d) == 3 and isinstance(d[2], tuple):
            zeroes.setdefault(d[1], []).extend(d[2])
        else:
            out.append(d)
    out.append(("moves", tuple((k, tuple(v)) for k, v in sorted(moves.items()))))
    out.append(("zeroes", tuple((k, tuple(v)) for k, v in sorted(zeroes.items()))))
    return tuple(out)


def _compare(m0: Model, m1: Model, i: int, op: Op) -> None:
    d0, d1 = m0.decisions[-1], m1.decisions[-1]
    if d0[3] != d1[3]:
        # Same pages, split into different launches: recorded, not a failure
        # unless asked for (to shrink a case that shows one).
        m0.count("batching_differences")
        if STRICT_BATCHING:
            n = next(k for k, (a, b) in enumerate(zip(d0[3], d1[3])) if a != b)
            raise Violation(
                "rank_divergence",
                f"after op {i} {op!r}: move batches differ: rank0 {d0[3][n]} vs rank1 {d1[3][n]}",
            )
    if d0[:3] == d1[:3]:
        return
    kind, dec0, dig0, _ = d0
    _, dec1, dig1, _ = d1
    if dec0 != dec1:
        n = next(
            (k for k, (a, b) in enumerate(zip(dec0, dec1)) if a != b),
            min(len(dec0), len(dec1)),
        )
        detail = f"decision {n}: rank0 {dec0[n] if n < len(dec0) else None} vs rank1 {dec1[n] if n < len(dec1) else None}"
    else:
        n = next(k for k, (a, b) in enumerate(zip(dig0, dig1)) if a != b)
        a, b = dig0[n], dig1[n]
        if isinstance(a, tuple) and isinstance(b, tuple):
            j = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
            detail = (
                f"state {a[0] if a else ''} field {j}: {str(a[j])[:200]} vs {str(b[j])[:200]}"
                if j is not None
                else f"{a} vs {b}"
            )
        else:
            detail = f"{str(a)[:300]} vs {str(b)[:300]}"
    raise Violation("rank_divergence", f"after op {i} {op!r}: {detail}")


def run_ops(
    cfg: StressConfig,
    ops: Sequence[Op],
    *,
    ranks: int = 2,
    gen: Optional[Generator] = None,
    n_ops: int = 0,
    device: str = "cpu",
    seed: int = 0,
    check_every: int = 1,
) -> RunResult:
    """Replay ``ops`` (or generate ``n_ops`` with ``gen``) on ``ranks``
    models in lockstep, checking after every operation."""
    _install_read_guard()
    record: List[Op] = []
    failure, failed_at = None, None
    setup_allocator_context()
    with get_parallel().override(attn_dcp_size=cfg.dcp):
        models = [Model(cfg, r, device=device, seed=seed) for r in range(ranks)]
        for m in models:
            m.install_arena_hooks()
        i = 0
        try:
            while True:
                if gen is not None:
                    if i >= n_ops:
                        break
                    op = gen.next(models[0])
                else:
                    if i >= len(ops):
                        break
                    op = ops[i]
                record.append(op)
                for m in models:
                    with _patches(m, cfg):
                        try:
                            m.apply(op)
                        except Violation:
                            raise
                        except Exception as e:  # noqa: BLE001
                            raise Violation(
                                "crash",
                                f"rank {m.rank}: harness op {op!r} raised {type(e).__name__}: {e}\n"
                                + "".join(
                                    traceback.format_exc().splitlines(True)[-14:]
                                ),
                            )
                        if i % check_every == 0 or op.kind == "quiesce":
                            m.check_payload()
                if ranks > 1:
                    _compare(models[0], models[1], i, op)
                i += 1
            # Drain at the end: every write lands and is checked.
            if gen is not None or ops:
                op = Op("quiesce", (False,))
                record.append(op)
                for m in models:
                    with _patches(m, cfg):
                        m.apply(op)
                        m.check_payload()
                if ranks > 1:
                    _compare(models[0], models[1], i, op)
        except Violation as v:
            failure, failed_at = v, len(record) - 1
    counts: Dict[str, int] = {}
    stats: Dict[str, int] = {}
    for m in models:
        for k, n in m.counts.items():
            counts[k] = counts.get(k, 0) + n
        for k, n in m.dev.stats.items():
            stats[k] = stats.get(k, 0) + n
    return RunResult(
        seed=-1,
        cfg=cfg,
        ops=record,
        failure=failure,
        failed_at=failed_at,
        counts=counts,
        stats=stats,
        max_dead_fences=max(m.max_dead_fences for m in models),
        step_reads={k: n for m in models for k, n in m.dev.step_reads.items()},
    )


def _patches(m: Model, cfg: StressConfig):
    if isinstance(m.dev, SimDevice):
        return sim_patches(m.dev)
    return cuda_patches(m.dev)


def run_seed_cuda(
    seed: int, n_ops: int, *, ranks: int = 2, check_every: int = 16
) -> RunResult:
    """The same operations on real CUDA streams with delayed kernels; the
    payload is checked every ``check_every`` operations, after a device
    synchronize."""
    rng = random.Random(seed)
    cfg = random_config(rng)
    gen = Generator(rng, cfg, ranks)
    result = run_ops(
        cfg,
        [],
        ranks=ranks,
        gen=gen,
        n_ops=n_ops,
        device="cuda",
        seed=seed,
        check_every=check_every,
    )
    result.seed = seed
    return result


def run_seed(
    seed: int, n_ops: int, *, ranks: int = 2, cfg: Optional[StressConfig] = None
) -> RunResult:
    rng = random.Random(seed)
    cfg = cfg or random_config(rng)
    gen = Generator(rng, cfg, ranks)
    result = run_ops(cfg, [], ranks=ranks, gen=gen, n_ops=n_ops)
    result.seed = seed
    return result


def signature(v: Optional[Violation]) -> Optional[str]:
    """What a shrunk sequence must keep failing with: the check and the shape
    of its message, without ids, pages or stamps."""
    if v is None:
        return None
    text = v.message.split("\n", 1)[0]
    text = re.sub(r"rank \d+: ", "", text)
    text = re.sub(r"[#\d\[\](),=-]+", " ", text)
    text = re.sub(r"<\S+", "<", text)
    return v.check + ":" + " ".join(text.split()[:6])


def minimize(
    cfg: StressConfig,
    ops: Sequence[Op],
    sig: str,
    *,
    ranks: int = 2,
    budget: int = 3000,
) -> List[Op]:
    """Shrink ``ops`` (delta debugging, then single removals) while the run
    still fails with signature ``sig``."""
    ops = list(ops)
    if ops and ops[-1].kind == "quiesce":
        ops = ops[:-1]

    def fails(candidate) -> bool:
        nonlocal budget
        budget -= 1
        r = run_ops(cfg, candidate, ranks=ranks)
        return signature(r.failure) == sig

    if not fails(ops):
        return ops
    n = 2
    while len(ops) >= 2 and budget > 0:
        chunk = max(1, len(ops) // n)
        reduced = False
        for start in range(0, len(ops), chunk):
            candidate = ops[:start] + ops[start + chunk :]
            if candidate and fails(candidate):
                ops = candidate
                n = max(n - 1, 2)
                reduced = True
                break
            if budget <= 0:
                break
        if not reduced:
            if chunk == 1:
                break
            n = min(len(ops), n * 2)
    # Simplify the arguments that remain: ticks of one rank only.
    return ops


# ---------------------------------------------------------------------------
# Cases: a configuration and an operation list, as JSON
# ---------------------------------------------------------------------------


def op_to_json(op: Op) -> list:
    def plain(x):
        return [plain(e) for e in x] if isinstance(x, tuple) else x

    return [op.kind, plain(op.args)]


def op_from_json(item) -> Op:
    kind, args = item

    def tup(x):
        return tuple(tup(e) for e in x) if isinstance(x, list) else x

    return Op(kind, tup(args))


def case_to_json(cfg: StressConfig, ops: Sequence[Op]) -> dict:
    return {
        "cfg": {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in dataclasses.asdict(cfg).items()
        },
        "ops": [op_to_json(op) for op in ops],
    }


def replay_case(case: dict, *, ranks: int = 2) -> Optional[Violation]:
    """Replay a case (``{"cfg": ..., "ops": [[kind, args], ...]}``, the form
    the stress CLI writes) on the CPU device model; the violation it ends
    with, or None when every check passes."""
    cfg = StressConfig.from_dict(case["cfg"])
    ops = [op_from_json(item) for item in case["ops"]]
    return run_ops(cfg, ops, ranks=ranks).failure
