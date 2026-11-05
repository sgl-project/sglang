"""Who may still touch a band's physical pages, and until when.

One record per use, in one list: a forward that writes the band, a transfer
that reads or writes it, the pages a move vacated while some reader could
still read them, and the pages freed while a forward that may still touch them
runs. Every move, reuse and write decision of the band reads this
ledger: a copy waits the writers of its sources, a vacated page returns to the
free list once its readers retired, and a write into vacated bytes outside the
band waits their readers. Decisions read retirement (a forward the scheduler
processed), never whether an event has fired, so every rank makes the same
ones. A stream waits on the events themselves.

Whether a page may be written is one rule, `CLEARANCE`: what its next writer
(a `Consumer`) must wait for, given each `Hazard` the ledger records on it. A
forward that takes a page needs nothing from a forward before it, since both
run in order on one stream; a copy on the schedule stream waits every hazard.
"""

import enum
from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

import torch

from sglang.srt.mem_cache.kv.alloc.signals import signal_fired, signals_of

FORWARD = "forward"
TRANSFER = "transfer"
VACATED = "vacated"
RECENT = "recent"


class Consumer(enum.Enum):
    """Who writes a page once it is handed out, and how that write is ordered."""

    # A forward, on the forward stream after every forward launched before it.
    FORWARD = "forward"
    # The schedule stream: a page move, or a state copy into a new slot.
    SCHEDULE_COPY = "schedule_copy"
    # The schedule stream zeroing the page's envelope so that bytes nobody
    # wrote read as finite values. Any later KV written over it is finite.
    ZERO = "zero"
    # A host load, on a copy stream that first waits the forward stream.
    HOST_LOAD = "host_load"
    # A network engine, once the page's id is published to the peer.
    NETWORK_RECEIVE = "network_receive"


class Hazard(enum.Enum):
    """Unfinished work that may still touch a page."""

    # A forward may still write it: it was freed while the forward ran.
    FORWARD_WRITE = "forward_write"
    # A forward may still read its old bytes: a move vacated it.
    FORWARD_READ = "forward_read"
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
    # The host waits until the network engine is done.
    HOST = "host"
    # The consumer may start only once the hazard's forward retired: the page's
    # id must not reach a network peer before then.
    AFTER_RETIRE = "after_retire"
    # The page cannot be handed out.
    EXCLUDED = "excluded"


def _clearance_row(**waits: Wait) -> Dict[Hazard, Wait]:
    row = {
        Hazard.FORWARD_WRITE: Wait.STREAM,
        Hazard.FORWARD_READ: Wait.STREAM,
        Hazard.COPY_READ: Wait.STREAM,
        Hazard.COPY_WRITE: Wait.STREAM,
        Hazard.NETWORK: Wait.HOST,
        Hazard.FIXED: Wait.EXCLUDED,
    }
    row.update({Hazard[name]: wait for name, wait in waits.items()})
    return row


# The one rule for when a page may be written. A host copy and a network
# engine are ordered by nothing the consumer runs on, so every consumer waits
# them. Forward hazards differ by consumer:
# - a forward runs after every forward launched before it;
# - zeroing races only a forward's write, which leaves finite bytes too, but
#   must not overwrite bytes a forward still reads;
# - a host load's copy stream waits the forward stream before it starts;
# - a network receive cannot be ordered on a stream, so the id is published
#   only after the forward retired.
CLEARANCE: Dict[Consumer, Dict[Hazard, Wait]] = {
    Consumer.FORWARD: _clearance_row(FORWARD_WRITE=Wait.NONE, FORWARD_READ=Wait.NONE),
    Consumer.SCHEDULE_COPY: _clearance_row(),
    Consumer.ZERO: _clearance_row(FORWARD_WRITE=Wait.NONE),
    Consumer.HOST_LOAD: _clearance_row(FORWARD_WRITE=Wait.NONE, FORWARD_READ=Wait.NONE),
    Consumer.NETWORK_RECEIVE: _clearance_row(
        FORWARD_WRITE=Wait.AFTER_RETIRE, FORWARD_READ=Wait.AFTER_RETIRE
    ),
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
        if wait is Wait.STREAM:
            self.stream.append(key)
        elif wait is Wait.HOST:
            self.host.append(key)
        elif wait is Wait.AFTER_RETIRE:
            self.after.append(key)
        elif wait is Wait.EXCLUDED:
            self.excluded = True

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


class PageUse:
    """One use of a band's pages.

    ``signal``: what a wait for the use to end waits for (a CUDA event, or a
    tuple of them). ``ids``: controller ids translated to pages when read;
    None for a forward whose writes are unknown. A vacated
    use holds byte range ``[lo, hi)``, and, when the pages return to this
    band's free list, their ids on the host and on the device (``pages``,
    ``batch``). A recent use holds, in ``batch``, device page ids freed while
    the forward ``signal`` ran, which may still write or read them."""

    __slots__ = (
        "kind",
        "signal",
        "writes",
        "ids",
        "lo",
        "hi",
        "pages",
        "batch",
    )

    def __init__(
        self,
        kind: str,
        signal,
        *,
        writes: bool = False,
        ids=None,
        lo: int = 0,
        hi: int = 0,
        pages: Optional[List[int]] = None,
        batch: Optional[torch.Tensor] = None,
    ):
        self.kind = kind
        self.signal = signal
        self.writes = writes
        self.ids = ids
        self.lo = lo
        self.hi = hi
        self.pages = pages
        self.batch = batch


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

    def retired(self, key) -> bool:
        """Whether the work behind ``key`` retired: a forward once the
        scheduler processed it (a forgotten event is a retired one). A tuple
        retires with all of its members."""
        if key is None:
            return True
        if isinstance(key, tuple):
            return all(self.retired(k) for k in key)
        return self.event_seq.get(key, 0) <= self.retired_seq

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
        held = self._vacated_keys()
        self.event_seq = {
            e: s for e, s in self.event_seq.items() if s > self.retired_seq or e in held
        }
        self.prune()

    def _vacated_keys(self) -> Set:
        keys = set()
        for use in self.uses:
            if use.kind == VACATED:
                signal = use.signal
                keys.update(signal if isinstance(signal, tuple) else (signal,))
        return keys

    # -- recording --

    def record_forward(self, event, ids, *, writes_unknown: bool) -> None:
        self.register_forward(event)
        self.prune()
        if writes_unknown:
            self.uses.append(PageUse(FORWARD, event, writes=True))
        elif ids is not None and ids.numel() > 0:
            self.uses.append(PageUse(FORWARD, event, writes=True, ids=ids))

    def record_transfer(self, signal, ids, *, writes: bool) -> None:
        self.prune()
        self.uses.append(PageUse(TRANSFER, signal, writes=writes, ids=ids))

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
        (a fence, `clearance`). ``writes``: ``key`` may still write the bytes
        rather than read them (pages freed while a forward ran)."""
        if pages is not None:
            self.uses.append(PageUse(VACATED, key, pages=list(pages), batch=batch))
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

    def record_recent_holes(
        self, key, batch: torch.Tensor, virtual: torch.Tensor
    ) -> None:
        """Physical pages ``batch``, which held virtual pages ``virtual``, went
        back to the free list while the forward ``key`` and those before it
        ran: one of them may still write a page it was handed (a forward
        launched before a request's row was released writes the row's next
        slot). Kept on the device; `resolve_recent_holes` fences those a
        running forward writes."""
        self.uses = [u for u in self.uses if u.kind != RECENT or not self._ended(u)]
        self.uses.append(PageUse(RECENT, key, batch=batch, ids=virtual))

    def resolve_recent_holes(
        self, page_bytes: int, written: Optional[Set[int]]
    ) -> None:
        """Fence the recent holes that a running forward still writes, keyed by
        the newest forward at their free: a copy or take that writes one waits
        it first. ``written``: the virtual pages the running forwards write,
        None when some forward's writes are unknown. Reads the device once;
        call it before any stream wait."""
        recent = [u for u in self.uses if u.kind == RECENT and not self._ended(u)]
        self.uses = [u for u in self.uses if u.kind != RECENT]
        if not recent:
            return
        pages = torch.cat([u.batch.reshape(-1) for u in recent]).tolist()
        virtual = torch.cat([u.ids.reshape(-1) for u in recent]).tolist()
        start = 0
        for use in recent:
            n = use.batch.numel()
            chunk = sorted(
                {
                    page
                    for page, v in zip(
                        pages[start : start + n], virtual[start : start + n]
                    )
                    if written is None or v in written
                }
            )
            start += n
            runs, first, prev = [], None, None
            for page in chunk:
                if first is None:
                    first = prev = page
                elif page == prev + 1:
                    prev = page
                else:
                    runs.append((first, prev + 1))
                    first = prev = page
            if first is not None:
                runs.append((first, prev + 1))
            self.vacate(use.signal, runs, page_bytes=page_bytes, writes=True)

    def fence_recent_holes_on(
        self, page_runs: Sequence[Tuple[int, int]], page_bytes: int
    ) -> None:
        """The pages of ``page_runs`` leave this band with every recent hole
        among them unresolved: fence all of their bytes, as written, on the
        forwards the recent holes wait. No device read; a superset of what
        `resolve_recent_holes` would fence."""
        keys = tuple(
            {
                id(u.signal): u.signal
                for u in self.uses
                if u.kind == RECENT and not self._ended(u)
            }.values()
        )
        if keys and page_runs:
            key = keys[0] if len(keys) == 1 else keys
            self.vacate(key, page_runs, page_bytes=page_bytes, writes=True)

    def has_recent_holes(self) -> bool:
        return any(use.kind == RECENT for use in self.uses)

    def recent_holes(self) -> List[PageUse]:
        """The recent holes still unresolved, whose forward has not retired."""
        return [u for u in self.uses if u.kind == RECENT and not self._ended(u)]

    def drop_recent_holes(self) -> None:
        self.uses = [u for u in self.uses if u.kind != RECENT]

    # -- reading --

    def prune(self) -> None:
        """Drop forwards that retired, and transfers whose signal completed:
        they only decided waits."""
        self.uses = [use for use in self.uses if not self._ended(use)]

    def _ended(self, use: PageUse) -> bool:
        if use.kind == RECENT:
            # A recent hole decides which bytes are fenced, and a fence stops a
            # walk: it ends with its forward's retirement, alike on every rank.
            return self.retired(use.signal)
        if use.kind == FORWARD:
            return self.retired(use.signal)
        if use.kind == TRANSFER:
            return signal_fired(use.signal)
        return False

    def forwards(self) -> List[PageUse]:
        return [use for use in self.uses if use.kind == FORWARD]

    def transfers(self) -> List[PageUse]:
        return [use for use in self.uses if use.kind == TRANSFER]

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
                and (urgent or self.retired(use.signal))
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
        per signal: a forward of this ledger, or a host copy."""
        out = []
        for key in signals_of(use.signal):
            if key in self.event_seq:
                hazard = Hazard.FORWARD_WRITE if use.writes else Hazard.FORWARD_READ
                out.append((hazard, key))
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
            if wait is Wait.AFTER_RETIRE and self.retired(key):
                # Retired: the scheduler processed the forward.
                wait = Wait.NONE
            out.add(wait, key)
            waited = waited and wait in (Wait.STREAM, Wait.HOST)
        return waited

    def take_clearance(
        self, lo: int, hi: int, consumers: Sequence[Consumer]
    ) -> Clearance:
        """What a write by ``consumers`` into bytes ``[lo, hi)`` must wait for,
        by the fences meeting them. A fence that every consumer clears by
        waiting is removed, as later work on the waiting stream is ordered
        after it; one some consumer does not wait (a forward needs nothing from
        a forward before it) stays for the next writer. Fences whose readers
        retired and whose signals completed are dropped as bookkeeping only: a
        retired fence decides nothing (`fence_meets`), so whether its signal
        completed, which ranks see at different times, only skips a wait."""
        out = Clearance()
        kept = []
        for use in self.uses:
            if use.kind == VACATED and use.pages is None:
                if self.retired(use.signal) and signal_fired(use.signal):
                    continue
                if use.lo < hi and lo < use.hi:
                    waited = True
                    for hazard, key in self.hazards(use):
                        waited &= self._clear_hazard(consumers, hazard, key, out)
                    if waited:
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
        exclusion when one of them is fixed."""
        out = Clearance()
        fixed = self.fixed_pages()
        for first, end in page_runs:
            lo, hi = first * page_bytes, end * page_bytes
            for use in self.uses:
                if (
                    use.kind == VACATED
                    and use.pages is None
                    and use.lo < hi
                    and lo < use.hi
                    and not (self.retired(use.signal) and signal_fired(use.signal))
                ):
                    for hazard, key in self.hazards(use):
                        self._clear_hazard(consumers, hazard, key, out)
            if any(first <= page < end for page in fixed):
                for consumer in consumers:
                    out.add(CLEARANCE[consumer][Hazard.FIXED], None)
        return out

    def take_fences_meeting(self, lo: int, hi: int) -> List:
        """The signals a copy into bytes ``[lo, hi)`` waits (`take_clearance`
        for a schedule-stream copy)."""
        return list(self.take_clearance(lo, hi, (Consumer.SCHEDULE_COPY,)).waits())

    def fences_need(self, lo: int, hi: int, consumers: Sequence[Consumer]) -> bool:
        """Whether a fence meeting bytes ``[lo, hi)`` holds an unretired forward
        that one of ``consumers`` must clear; removes nothing. It picks which
        pages a take prefers, so it reads retirement only, alike on every rank:
        a host copy, which has no logical release, only ever adds a wait."""
        for use in self.uses:
            if not (
                use.kind == VACATED
                and use.pages is None
                and use.lo < hi
                and lo < use.hi
            ):
                continue
            for hazard, key in self.hazards(use):
                if hazard in (Hazard.COPY_READ, Hazard.COPY_WRITE):
                    continue
                if not self.retired(key) and must_wait(consumers, hazard):
                    return True
        return False

    def fences_pending(self, lo: int, hi: int, consumers: Sequence[Consumer]) -> bool:
        """Whether a fence meeting bytes ``[lo, hi)`` may still make one of
        ``consumers`` wait (`take_clearance`); removes nothing. It decides only
        whether a take looks closer, never which pages it takes."""
        return any(
            use.kind == VACATED
            and use.pages is None
            and use.lo < hi
            and lo < use.hi
            and not (self.retired(use.signal) and signal_fired(use.signal))
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
            and not self.retired(use.signal)
            for use in self.uses
        )

    def has_fences(self) -> bool:
        return any(use.kind == VACATED and use.pages is None for use in self.uses)

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
