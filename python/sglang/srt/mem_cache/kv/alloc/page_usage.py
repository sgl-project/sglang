"""Who may still touch a band's physical pages, and until when.

One record per use, in one list: a forward that writes the band, a transfer
that reads or writes it, and the pages a move vacated while some reader could
still read them. (Pages their owner frees while a forward runs are kept in the
band's pending batches instead; see `forward_steps`.) A copy waits the writers
of its sources, a vacated page returns to the free list once its readers
retired, and a write into vacated bytes outside the band waits their readers.
Decisions read retirement (a forward the scheduler processed, a transfer
released where every rank sees it), never whether an event has fired, so every
rank makes the same ones; streams wait on the events themselves, and a network
transfer is waited for on the host (`signals.NetworkTransfer`).

Whether a page may be written is one rule, `CLEARANCE`: what its next writer
(a `Consumer`) must wait for, given each `Hazard` the ledger records on it. A
forward that takes a page needs nothing from a forward before it, since both
run in order on one stream; a copy on the schedule stream waits every hazard.
"""

import bisect
import enum
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.kv.alloc.signals import (
    ReceiveHold,
    StreamTransfer,
    TransferRelease,
    keys_of,
    signal_fired,
    unreleased_transfers,
)

FORWARD = "forward"
TRANSFER = "transfer"
VACATED = "vacated"

# The keys of transfers, which are never a forward's event.
_TRANSFER_KEYS = (StreamTransfer, TransferRelease)


class Consumer(enum.Enum):
    """Who writes a page once it is handed out, and how that write is ordered."""

    # A forward, on the forward stream after every forward launched before it.
    FORWARD = "forward"
    # The schedule stream: a page move, or a state copy into a new slot.
    SCHEDULE_COPY = "schedule_copy"
    # The schedule stream zeroing the page's envelope so that bytes nobody
    # wrote read as finite values.
    ZERO = "zero"
    # A host load, on a copy stream that first waits the forward stream.
    HOST_LOAD = "host_load"
    # A network engine, once the page's id is published to the peer.
    NETWORK_RECEIVE = "network_receive"


class Hazard(enum.Enum):
    """Unfinished work that may still touch a page."""

    # An unretired forward may still read its old bytes: a move vacated it
    # while the forward, whose page table names it, ran. (A page its owner
    # freed while a forward ran waits in the band's pending batches instead,
    # under that forward's key, with the same meaning.)
    FORWARD = "forward"
    # A host copy may still read it.
    COPY_READ = "copy_read"
    # A host copy may still write it.
    COPY_WRITE = "copy_write"
    # A network transfer may still read or write it.
    NETWORK = "network"
    # It must stay where it is (a load reservation, rows a relief holds).
    FIXED = "fixed"


class Wait(enum.Enum):
    """What clears one hazard for one consumer."""

    NONE = "none"
    # The consumer's stream waits the hazard's event (on the GPU).
    STREAM = "stream"
    # The host waits until the network engine is done (`NetworkWaiter`).
    HOST = "host"
    # The consumer may start only once the hazard's forward retired: the page's
    # id must not reach a network peer before then.
    AFTER_RETIRE = "after_retire"
    # The page cannot be handed out.
    EXCLUDED = "excluded"


def _clearance_row(**waits: Wait) -> Dict[Hazard, Wait]:
    row = {
        Hazard.FORWARD: Wait.STREAM,
        Hazard.COPY_READ: Wait.STREAM,
        Hazard.COPY_WRITE: Wait.STREAM,
        Hazard.NETWORK: Wait.HOST,
        Hazard.FIXED: Wait.EXCLUDED,
    }
    row.update({Hazard[name]: wait for name, wait in waits.items()})
    return row


# The one rule for when a page may be written. A host copy and a network
# engine are ordered by nothing the consumer runs on, so every consumer waits
# them. The one forward hazard, a running forward reading the old bytes of a
# page a move vacated, differs by consumer:
# - a forward runs after every forward launched before it;
# - a schedule-stream copy or zeroing waits the forward on its stream;
# - a host load's copy stream waits the forward stream before it starts;
# - a network receive cannot be ordered on a stream, so the id is published
#   only after the forward retired.
CLEARANCE: Dict[Consumer, Dict[Hazard, Wait]] = {
    Consumer.FORWARD: _clearance_row(FORWARD=Wait.NONE),
    Consumer.SCHEDULE_COPY: _clearance_row(),
    Consumer.ZERO: _clearance_row(),
    Consumer.HOST_LOAD: _clearance_row(FORWARD=Wait.NONE),
    Consumer.NETWORK_RECEIVE: _clearance_row(FORWARD=Wait.AFTER_RETIRE),
}


def must_wait(consumers: Sequence[Consumer], hazard: Hazard) -> bool:
    """Whether any of ``consumers`` must clear ``hazard`` before it writes."""
    return any(CLEARANCE[c][hazard] is not Wait.NONE for c in consumers)


class Clearance:
    """What a write into some pages must wait for: ``stream`` events on its
    stream, ``host`` network transfers on the host, ``after`` forwards to
    retire before the pages' ids reach a peer; ``excluded`` when one of the
    pages cannot be handed out at all."""

    __slots__ = ("stream", "host", "after", "excluded")

    def __init__(self):
        self.stream: List = []
        self.host: List = []
        self.after: List = []
        self.excluded = False

    def add(self, wait: Wait, key) -> None:
        """Add ``key`` to what ``wait`` asks; a key two consumers wait the
        same way is waited once."""
        if wait is Wait.STREAM:
            keys = self.stream
        elif wait is Wait.HOST:
            keys = self.host
        elif wait is Wait.AFTER_RETIRE:
            keys = self.after
        else:
            self.excluded = self.excluded or wait is Wait.EXCLUDED
            return
        if not any(held is key for held in keys):
            keys.append(key)

    def merge(self, other: "Clearance") -> "Clearance":
        self.stream += other.stream
        self.host += other.host
        self.after += other.after
        self.excluded = self.excluded or other.excluded
        return self

    @property
    def quiet(self) -> bool:
        return not (self.stream or self.host or self.after or self.excluded)

    def waits(self) -> tuple:
        """The signals to wait, network transfers and events alike
        (`signals.wait_signals` waits each where it can be waited)."""
        return (*self.host, *self.stream)


def _owned(ids):
    """The ledger's own copy of ids a caller hands it: a caller may reuse its
    tensor (a request-table row, a slice of one) while the record lives, and a
    record must keep naming the pages it was made for. A device copy runs on
    the current stream, so record on the stream that page moves read the
    ledger on. A NumPy array stays one, copied: its pages are compact."""
    if ids is None:
        return None
    if isinstance(ids, torch.Tensor):
        return ids.clone()
    if isinstance(ids, np.ndarray):
        return ids.copy()
    return list(ids)


class PageUse:
    """One use of a band's pages.

    ``signal``: what a wait for the use to end waits for (a CUDA event, a
    network transfer, or a tuple of them). ``ids``: controller ids
    translated to pages when read, or with ``unit_tokens`` physical units the
    engine was handed; None for a forward whose writes are unknown.
    ``widened``: the ids are the alloc surface's (DCP-widened under DCP: a
    forward's write ids, a request's token rows), not rank-local rows.
    ``host_pages``: the physical pages ``ids`` name, read on the host once by
    the band that keeps the ledger, before a stream wait could make that read
    block, and kept current as pages move. A vacated use holds byte range
    ``[lo, hi)``, and, when the pages return to this band's free list, their
    ids on the host and on the device (``pages``, ``batch``). Every id tensor
    a use holds is the ledger's own copy (`_owned`)."""

    __slots__ = (
        "kind",
        "signal",
        "writes",
        "ids",
        "widened",
        "unit_tokens",
        "release",
        "lo",
        "hi",
        "pages",
        "batch",
        "host_pages",
    )

    def __init__(
        self,
        kind: str,
        signal,
        *,
        writes: bool = False,
        ids=None,
        widened: bool = False,
        unit_tokens: Optional[int] = None,
        release: Optional[TransferRelease] = None,
        lo: int = 0,
        hi: int = 0,
        pages: Optional[List[int]] = None,
        batch: Optional[torch.Tensor] = None,
    ):
        self.kind = kind
        self.signal = signal
        self.writes = writes
        self.ids = ids
        self.widened = widened
        self.unit_tokens = unit_tokens
        self.release = release
        self.lo = lo
        self.hi = hi
        self.pages = pages
        self.batch = batch
        self.host_pages: Optional[Set[int]] = None


class PageUsage:
    """The ledger of one band (module docstring)."""

    def __init__(self):
        self.forward_seq = 0
        self.reset()

    def reset(self) -> None:
        self.uses: List[PageUse] = []
        # Pages of vacated uses that return to the free list, for O(1)
        # membership in the survivor walk.
        self.pending_pages: Set[int] = set()
        self.latest_forward = None
        self.retired_seq = self.forward_seq
        self.event_seq: Dict[Hashable, int] = {}
        # Physical pages that must stay where they are, by the key that fixed
        # them.
        self.fixed: Dict[Hashable, Tuple[int, ...]] = {}

    # -- forwards and retirement --

    def register_forward(self, event) -> None:
        if event is not None and event not in self.event_seq:
            self.forward_seq += 1
            self.event_seq[event] = self.forward_seq

    def forward_retired(self, event) -> bool:
        """Whether the forward recorded with ``event`` retired: the scheduler
        processed it, or a stream wait ran past it (a forgotten event is a
        retired one). Logical, alike on every rank. A forward's event only: a
        transfer's key, or a key holding one, is asked with `ended`."""
        if event is None:
            return True
        if isinstance(event, (tuple, *_TRANSFER_KEYS)):
            raise TypeError(f"not a forward's event: {event!r}")
        return self.event_seq.get(event, 0) <= self.retired_seq

    @staticmethod
    def transfer_ended(key) -> bool:
        """Whether a transfer has ended for the decisions that read it: a
        host-observed transfer once it is released, where every rank sees it
        end; a copy on a stream always, since whatever writes its pages next
        waits its events after the decision is made."""
        if isinstance(key, TransferRelease):
            return key.released
        if isinstance(key, StreamTransfer):
            return True
        raise TypeError(f"not a transfer's key: {key!r}")

    def ended(self, key) -> bool:
        """Whether all the work behind a release ``key`` ended for the
        decisions that read it: each forward retired (`forward_retired`) and
        each transfer ended (`transfer_ended`). A tuple ends with all of its
        members; None holds nothing."""
        return all(
            self.transfer_ended(k)
            if isinstance(k, _TRANSFER_KEYS)
            else self.forward_retired(k)
            for k in keys_of(key)
        )

    def retire_forward(self, event) -> None:
        """The scheduler processed this forward: it and every forward recorded
        before it retire. Its event may already have retired by a stream wait;
        it is forgotten either way."""
        seq = self.event_seq.get(event)
        if seq is not None:
            self._retire_to(seq)

    def retire_through(self, key) -> None:
        """Retire on a stream wait for ``key``: everything recorded up to it ran
        before it on the forward stream."""
        if isinstance(key, tuple):
            for k in key:
                self.retire_through(k)
            return
        seq = self.event_seq.get(key)
        if seq is not None:
            self._retire_to(seq)

    def retire_all(self) -> None:
        """Every recorded forward ran before a wait on the latest one."""
        self._retire_to(self.forward_seq)
        self.uses = [u for u in self.uses if u.kind != FORWARD]

    def _retire_to(self, seq: int) -> None:
        """Retire every forward recorded up to ``seq`` and forget their events,
        except those a vacated use still waits on. Every retirement goes
        through here, so the event map stays bounded by unretired work."""
        self.retired_seq = max(self.retired_seq, seq)
        # Fences whose readers retired and whose signals completed decide
        # nothing more (`take_clearance_runs`): drop them now, so the walks
        # of a take stay short. A fence a host copy still reads keeps.
        done: Dict[int, bool] = {}
        self.uses = [
            use
            for use in self.uses
            if not (
                use.kind == VACATED
                and use.pages is None
                and self._fence_done(use, done)
            )
        ]
        held = self._vacated_keys()
        self.event_seq = {
            e: s for e, s in self.event_seq.items() if s > self.retired_seq or e in held
        }
        self.prune()

    def _vacated_keys(self) -> Set:
        keys = set()
        for use in self.uses:
            if use.kind == VACATED:
                keys.update(keys_of(use.signal))
        return keys

    # -- recording --

    def record_forward(self, event, ids, *, writes_unknown: bool) -> None:
        self.register_forward(event)
        self.prune()
        if writes_unknown:
            self.uses.append(PageUse(FORWARD, event, writes=True))
        elif ids is not None and ids.numel() > 0:
            self.uses.append(
                PageUse(FORWARD, event, writes=True, ids=_owned(ids), widened=True)
            )

    def record_transfer(
        self,
        signal,
        ids,
        *,
        writes: bool,
        release: Optional[TransferRelease],
        unit_tokens: Optional[int],
        widened: bool = False,
        copy: bool = True,
    ) -> None:
        """Record a transfer that ends with ``signal``: a host-observed
        transfer's release, or a stream copy's event(s), keyed as a
        `StreamTransfer`. Without ``copy`` the caller hands over its own copy."""
        self.prune()
        if not isinstance(signal, _TRANSFER_KEYS):
            signal = StreamTransfer(signal)
        self.uses.append(
            PageUse(
                TRANSFER,
                signal,
                writes=writes,
                ids=_owned(ids) if copy else ids,
                widened=widened,
                unit_tokens=unit_tokens,
                release=release,
            )
        )

    def vacate(
        self,
        key,
        page_runs: Sequence[Tuple[int, int]],
        *,
        page_bytes: int,
        pages: Optional[List[int]] = None,
        batch: Optional[torch.Tensor] = None,
        writes: bool = False,
    ) -> None:
        """Pages left their place while ``key``'s readers could still read them.
        With ``pages`` and ``batch`` they return to this band's free list once
        ``key`` retired; without, a write into their bytes clears ``key`` first
        (a fence, `clearance`). ``writes``: a transfer ``key`` may still write
        the bytes rather than read them."""
        if pages is not None:
            self.uses.append(
                PageUse(VACATED, key, pages=list(pages), batch=_owned(batch))
            )
            self.pending_pages.update(pages)
            return
        for first, end in page_runs:
            self.uses.append(
                PageUse(
                    VACATED,
                    key,
                    writes=writes,
                    lo=first * page_bytes,
                    hi=end * page_bytes,
                )
            )

    # -- reading --

    def prune(self) -> None:
        """Drop forwards that retired, and transfers that were released or,
        without a release, whose signal completed: they only decided waits."""
        self.uses = [use for use in self.uses if not self._ended(use)]

    def _ended(self, use: PageUse) -> bool:
        if use.kind == FORWARD:
            return self.forward_retired(use.signal)
        if use.kind == TRANSFER:
            if use.release is not None:
                return self.transfer_ended(use.release)
            return signal_fired(use.signal)
        return False

    def forwards(self) -> List[PageUse]:
        return [use for use in self.uses if use.kind == FORWARD]

    def transfers(self) -> List[PageUse]:
        return [use for use in self.uses if use.kind == TRANSFER]

    def holds_receive_rows(self) -> bool:
        """Whether receive rows are held for a publish still to come."""
        return any(
            use.kind == TRANSFER
            and isinstance(use.release, ReceiveHold)
            and not use.release.released
            for use in self.uses
        )

    def writes_unknown(self) -> bool:
        return any(use.kind == FORWARD and use.ids is None for use in self.uses)

    def has_pending(self) -> bool:
        return bool(self.pending_pages)

    def pending_by_key(self) -> Dict:
        """Pending pages grouped by the signal they wait: key -> (pages on the
        host, their device batches), in recording order."""
        out: Dict = {}
        for use in self.uses:
            if use.kind == VACATED and use.pages is not None:
                pages, batches = out.setdefault(use.signal, ([], []))
                pages.extend(use.pages)
                batches.append(use.batch)
        return out

    def take_ready_vacated(
        self, *, urgent: bool, keep: Set[int] = frozenset()
    ) -> List[PageUse]:
        """Remove and return the vacated uses whose pages may return to the
        free list: those whose readers retired, or every one when urgent;
        never one holding a page of ``keep`` (pages that must stay put)."""
        ready, kept = [], []
        for use in self.uses:
            if (
                use.kind == VACATED
                and use.pages is not None
                and (urgent or self.ended(use.signal))
                and keep.isdisjoint(use.pages)
            ):
                ready.append(use)
            else:
                kept.append(use)
        if ready:
            self.uses = kept
            for use in ready:
                self.pending_pages.difference_update(use.pages)
        return ready

    def hazards(self, use: PageUse) -> List[Tuple[Hazard, object]]:
        """What a fence (a vacated byte range) still guards against, one entry
        per key: a network transfer, a forward of this ledger, or a host
        copy."""
        out = []
        for key in keys_of(use.signal):
            if isinstance(key, TransferRelease):
                out.append((Hazard.NETWORK, key))
            elif not isinstance(key, StreamTransfer):
                out.append((Hazard.FORWARD, key))
            else:
                hazard = Hazard.COPY_WRITE if use.writes else Hazard.COPY_READ
                out.append((hazard, key))
        return out

    def _clear_hazard(self, consumers, hazard: Hazard, key, out: Clearance) -> bool:
        """Add what ``consumers`` wait for ``hazard`` on ``key`` to ``out``;
        whether they all clear it by waiting (so the fence may go)."""
        waited = True
        for consumer in consumers:
            wait = CLEARANCE[consumer][hazard]
            if wait is Wait.AFTER_RETIRE and self.forward_retired(key):
                # Retired: the scheduler processed the forward.
                wait = Wait.NONE
            out.add(wait, key)
            waited = waited and wait in (Wait.STREAM, Wait.HOST)
        return waited

    def _fence_done(self, use: PageUse, done: Dict[int, bool]) -> bool:
        """Whether a fence's readers retired and its signals completed (it is
        then bookkeeping only); memoized per signal in ``done`` for one call."""
        key = id(use.signal)
        if key not in done:
            done[key] = self.ended(use.signal) and signal_fired(use.signal)
        return done[key]

    def take_clearance(
        self, lo: int, hi: int, consumers: Sequence[Consumer]
    ) -> Clearance:
        """What a write by ``consumers`` into bytes ``[lo, hi)`` must wait for
        (`take_clearance_runs` over one run)."""
        return self.take_clearance_runs(((lo, hi),), consumers)

    def take_clearance_runs(
        self, runs: Sequence[Tuple[int, int]], consumers: Sequence[Consumer]
    ) -> Clearance:
        """What a write by ``consumers`` into the disjoint byte runs ``runs``
        must wait for, by the fences meeting them, in one walk of the uses.

        A fence every consumer clears by waiting is removed over the runs and
        kept over its other bytes; one some consumer does not wait stays whole
        for the next writer. Done fences are dropped: whether their signal
        completed, which ranks see at different times, only skips a wait."""
        runs = sorted((lo, hi) for lo, hi in runs if lo < hi)
        starts = [lo for lo, _ in runs]
        out = Clearance()
        done: Dict[int, bool] = {}
        kept = []
        for use in self.uses:
            if use.kind == VACATED and use.pages is None:
                if self._fence_done(use, done):
                    continue
                i = max(0, bisect.bisect_right(starts, use.lo) - 1)
                hits = []
                while i < len(runs) and runs[i][0] < use.hi:
                    if use.lo < runs[i][1]:
                        hits.append(runs[i])
                    i += 1
                if hits:
                    waited = True
                    for hazard, key in self.hazards(use):
                        waited &= self._clear_hazard(consumers, hazard, key, out)
                    if waited:
                        at = use.lo
                        for lo, hi in hits:
                            if at < lo:
                                kept.append(
                                    PageUse(
                                        VACATED,
                                        use.signal,
                                        writes=use.writes,
                                        lo=at,
                                        hi=lo,
                                    )
                                )
                            at = max(at, hi)
                        if at < use.hi:
                            kept.append(
                                PageUse(
                                    VACATED,
                                    use.signal,
                                    writes=use.writes,
                                    lo=at,
                                    hi=use.hi,
                                )
                            )
                        continue
            kept.append(use)
        self.uses = kept
        return out

    def clearance(
        self,
        page_runs: Sequence[Tuple[int, int]],
        consumers: Sequence[Consumer],
        *,
        page_bytes: int,
    ) -> Clearance:
        """What ``consumers`` must clear to write the pages of ``page_runs``
        of this ledger's band, removing nothing: the fences meeting them, and
        exclusion when one of them is fixed. One walk of the uses."""
        out = Clearance()
        runs = sorted(
            (first * page_bytes, end * page_bytes)
            for first, end in page_runs
            if first < end
        )
        starts = [lo for lo, _ in runs]
        done: Dict[int, bool] = {}
        for use in self.uses:
            if use.kind != VACATED or use.pages is not None:
                continue
            i = max(0, bisect.bisect_right(starts, use.lo) - 1)
            meets = False
            while i < len(runs) and runs[i][0] < use.hi:
                if use.lo < runs[i][1]:
                    meets = True
                    break
                i += 1
            if meets and not self._fence_done(use, done):
                for hazard, key in self.hazards(use):
                    self._clear_hazard(consumers, hazard, key, out)
        fixed = self.fixed_pages()
        if fixed and any(
            first <= page < end for first, end in page_runs for page in fixed
        ):
            for consumer in consumers:
                out.add(CLEARANCE[consumer][Hazard.FIXED], None)
        return out

    def take_fences_meeting(self, lo: int, hi: int) -> List:
        """The signals a copy into bytes ``[lo, hi)`` waits (`take_clearance`
        for a schedule-stream copy)."""
        return list(self.take_clearance(lo, hi, (Consumer.SCHEDULE_COPY,)).waits())

    def _orders(self, use: PageUse, consumers: Sequence[Consumer]) -> bool:
        """Whether fence ``use`` holds an unretired forward or an unreleased
        network transfer that one of ``consumers`` must clear: what orders a
        take's choice of holes."""
        for hazard, key in self.hazards(use):
            if hazard in (Hazard.COPY_READ, Hazard.COPY_WRITE):
                continue
            if not self.ended(key) and must_wait(consumers, hazard):
                return True
        return False

    def fences_need(self, lo: int, hi: int, consumers: Sequence[Consumer]) -> bool:
        """Whether a fence meeting bytes ``[lo, hi)`` holds an unretired forward
        or an unreleased network transfer that one of ``consumers`` must clear;
        removes nothing. It picks which pages a take prefers, so it reads
        retirement and releases only, alike on every rank: a host copy, which
        has no logical release, only ever adds a wait."""
        return bool(self.ordering_spans(lo, hi, consumers))

    def ordering_spans(
        self, lo: int, hi: int, consumers: Sequence[Consumer]
    ) -> List[Tuple[int, int]]:
        """The parts of bytes ``[lo, hi)`` that fences holding an unretired
        forward or an unreleased network transfer which one of ``consumers``
        must clear meet (`fences_need`), in one walk of the uses; removes
        nothing."""
        out = []
        for use in self.uses:
            if (
                use.kind == VACATED
                and use.pages is None
                and use.lo < hi
                and lo < use.hi
                and self._orders(use, consumers)
            ):
                out.append((max(use.lo, lo), min(use.hi, hi)))
        return out

    def fences_pending(self, lo: int, hi: int, consumers: Sequence[Consumer]) -> bool:
        """Whether a fence meeting bytes ``[lo, hi)`` may still make one of
        ``consumers`` wait (`take_clearance`); removes nothing. It reads
        whether signals fired, which ranks see at different times, so it
        decides only where a take makes its choice (on the host, where the
        taken pages' waits are read), never which pages it takes."""
        return any(
            use.kind == VACATED
            and use.pages is None
            and use.lo < hi
            and lo < use.hi
            and not (self.ended(use.signal) and signal_fired(use.signal))
            and any(must_wait(consumers, hazard) for hazard, _ in self.hazards(use))
            for use in self.uses
        )

    def fence_meets(self, lo: int, hi: int) -> bool:
        """Whether a fence whose readers have not retired meets bytes
        ``[lo, hi)``; removes nothing. It stops an opportunistic walk, so it
        reads retirement, never whether a signal fired."""
        return any(
            use.kind == VACATED
            and use.pages is None
            and use.lo < hi
            and lo < use.hi
            and not self.ended(use.signal)
            for use in self.uses
        )

    def network_held_pending(self) -> Set[int]:
        """Pending pages an unreleased network transfer still reads; a layout
        that does not need them treats them as occupied."""
        held: Set[int] = set()
        for use in self.uses:
            if use.kind == VACATED and use.pages is not None:
                keys = use.signal if isinstance(use.signal, tuple) else (use.signal,)
                if any(isinstance(k, TransferRelease) and not k.released for k in keys):
                    held.update(use.pages)
        return held

    def network_fences(self, lo: int = 0, hi: Optional[int] = None) -> List[PageUse]:
        """Fences of unreleased network transfers meeting bytes ``[lo, hi)``:
        bytes a move took out of the band while a send still read them. A write
        into them waits on the host for the send; logical releases only, alike
        on every rank."""
        return [
            use
            for use in self.uses
            if use.kind == VACATED
            and use.pages is None
            and (hi is None or (use.lo < hi and lo < use.hi))
            and unreleased_transfers(use.signal)
        ]

    def has_fences(self) -> bool:
        return any(use.kind == VACATED and use.pages is None for use in self.uses)

    def fence_outstanding(self, lo: int, hi: int) -> bool:
        """Whether a fence meeting bytes ``[lo, hi)`` is not yet done: its
        readers have not all retired or a signal of it has not fired. A done
        fence is bookkeeping only until clearance drops it; it makes no take
        wait or choose. Reads whether signals fired, so it decides only what
        the host keeps, never which pages a take takes."""
        done: Dict[int, bool] = {}
        return any(
            use.kind == VACATED
            and use.pages is None
            and use.lo < hi
            and lo < use.hi
            and not self._fence_done(use, done)
            for use in self.uses
        )

    # -- fixed pages --

    def fix(self, key: Hashable, pages: Sequence[int]) -> None:
        fixed = dict(self.fixed)
        fixed[key] = fixed.get(key, ()) + tuple(int(p) for p in pages)
        self.fixed = fixed

    def unfix(self, key: Hashable, pages: Optional[Sequence[int]] = None) -> None:
        if key not in self.fixed:
            return
        fixed = dict(self.fixed)
        if pages is None:
            del fixed[key]
        else:
            rest = list(fixed[key])
            for page in pages:
                rest.remove(int(page))
            if rest:
                fixed[key] = tuple(rest)
            else:
                del fixed[key]
        self.fixed = fixed

    def fixed_pages(self) -> Set[int]:
        return {page for pages in self.fixed.values() for page in pages}
