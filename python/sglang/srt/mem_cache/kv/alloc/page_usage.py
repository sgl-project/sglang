"""Who may still touch a band's physical pages, and until when.

One record per use, in one list: a forward that writes the band, a transfer
that reads or writes it, the pages a move vacated while some reader could
still read them, and the pages freed while a forward that may still touch them
runs. Every move, reuse and write decision of the band reads this
ledger: a copy waits the writers of its sources, a vacated page returns to the
free list once its readers retired, and a write into vacated bytes outside the
band waits their readers. Decisions read retirement (a forward the scheduler
processed, a transfer released where every rank sees it), never whether an
event has fired, so every rank makes the same ones; the GPU waits on the
signals themselves.
"""

from typing import Dict, Hashable, List, Optional, Sequence, Set, Tuple

import torch

from sglang.srt.mem_cache.kv.alloc.signals import TransferRelease, signal_fired

FORWARD = "forward"
TRANSFER = "transfer"
VACATED = "vacated"
RECENT = "recent"


class PageUse:
    """One use of a band's pages.

    ``signal``: what a GPU waits for the use to end (a CUDA event, a completion
    flag, or a tuple of them and transfer releases). ``ids``: controller ids
    translated to pages when read, or with ``unit_tokens`` physical units the
    engine was handed; None for a forward whose writes are unknown. A vacated
    use holds byte range ``[lo, hi)``, and, when the pages return to this
    band's free list, their ids on the host and on the device (``pages``,
    ``batch``). A recent use holds, in ``batch``, device page ids freed while
    the forward ``signal`` ran, which may still write or read them."""

    __slots__ = (
        "kind",
        "signal",
        "writes",
        "ids",
        "unit_tokens",
        "release",
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
        self.unit_tokens = unit_tokens
        self.release = release
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
        scheduler processed it (a forgotten event is a retired one), a
        host-observed transfer once released. A tuple retires with all of its
        members."""
        if key is None:
            return True
        if isinstance(key, tuple):
            return all(self.retired(k) for k in key)
        if isinstance(key, TransferRelease):
            return key.released
        return self.event_seq.get(key, 0) <= self.retired_seq

    def retire_forward(self, event) -> None:
        """The scheduler processed this forward: it and every forward recorded
        before it retire."""
        seq = self.event_seq.get(event)
        if seq is None or seq <= self.retired_seq:
            return
        self.retired_seq = seq
        held = self._vacated_keys()
        self.event_seq = {
            e: s for e, s in self.event_seq.items() if s > seq or e in held
        }
        self.prune()

    def retire_through(self, key) -> None:
        """Retire on a stream wait for ``key``: everything recorded up to it ran
        before it on the forward stream."""
        if isinstance(key, tuple):
            for k in key:
                self.retire_through(k)
            return
        seq = self.event_seq.get(key)
        if seq is not None and seq > self.retired_seq:
            self.retired_seq = seq
            self.prune()

    def retire_all(self) -> None:
        """Every recorded forward ran before a wait on the latest one."""
        self.retired_seq = self.forward_seq
        self.uses = [u for u in self.uses if u.kind != FORWARD]

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

    def record_transfer(
        self,
        signal,
        ids,
        *,
        writes: bool,
        release: Optional[TransferRelease],
        unit_tokens: Optional[int],
    ) -> None:
        self.prune()
        self.uses.append(
            PageUse(
                TRANSFER,
                signal,
                writes=writes,
                ids=ids,
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
    ) -> None:
        """Pages left their place while ``key``'s readers could still read them.
        With ``pages`` and ``batch`` they return to this band's free list once
        ``key`` retired; without, they left the band and a write into their
        bytes waits ``key`` first."""
        if pages is not None:
            self.uses.append(PageUse(VACATED, key, pages=list(pages), batch=batch))
            self.pending_pages.update(pages)
            return
        for first, end in page_runs:
            self.uses.append(
                PageUse(VACATED, key, lo=first * page_bytes, hi=end * page_bytes)
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
            self.vacate(use.signal, runs, page_bytes=page_bytes)

    def has_recent_holes(self) -> bool:
        return any(use.kind == RECENT for use in self.uses)

    # -- reading --

    def prune(self) -> None:
        """Drop forwards that retired, and transfers that were released or,
        without a release, whose signal completed: they only decided waits."""
        self.uses = [use for use in self.uses if not self._ended(use)]

    def _ended(self, use: PageUse) -> bool:
        if use.kind == RECENT:
            # Only decides whether a later write waits: a fired event needs none.
            return signal_fired(use.signal)
        if use.kind == FORWARD:
            return self.retired(use.signal)
        if use.kind == TRANSFER:
            if use.release is not None:
                return self.retired(use.release)
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

    def take_fences_meeting(self, lo: int, hi: int) -> List:
        """Remove the out-of-band vacated uses meeting bytes ``[lo, hi)`` and
        return their signals to wait; drop, as bookkeeping only, those whose
        signals completed."""
        waits, kept = [], []
        for use in self.uses:
            if use.kind == VACATED and use.pages is None:
                if signal_fired(use.signal):
                    continue
                if use.lo < hi and lo < use.hi:
                    waits.append(use.signal)
                    continue
            kept.append(use)
        self.uses = kept
        return waits

    def fence_meets(self, lo: int, hi: int) -> bool:
        """Whether an unfinished fence meets bytes ``[lo, hi)``; removes
        nothing."""
        return any(
            use.kind == VACATED
            and use.pages is None
            and use.lo < hi
            and lo < use.hi
            and not signal_fired(use.signal)
            for use in self.uses
        )

    def network_held_pending(self) -> Set[int]:
        """Pending pages a network transfer still reads, until its logical
        release: they return to the free list only after the release, so a
        layout that does not need them treats them as occupied."""
        held: Set[int] = set()
        for use in self.uses:
            if use.kind == VACATED and use.pages is not None:
                keys = use.signal if isinstance(use.signal, tuple) else (use.signal,)
                if any(isinstance(k, TransferRelease) and not k.released for k in keys):
                    held.update(use.pages)
        return held

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
