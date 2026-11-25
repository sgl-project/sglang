"""The host copy of a band's free list.

A band keeps its free pages on the device and, while a later step may choose
holes on the host, a copy on the host in the same order. `HostFreeList` owns
that copy: one backing list consumed from both ends by offsets, so a take of a
short prefix or suffix touches only the pages it takes, never the rest.
"""

from __future__ import annotations

from typing import Iterator, List, Sequence


class HostFreeList:
    """The free pages on the host, in the device list's order.

    ``ascending`` says whether the pages are known to be in ascending order,
    as the device list is once sorted; a band sorts both together
    (`sort`). Frees append (`extend`); takes consume either end. The backing
    list is never handed out: `to_list` copies, so no caller sees a later
    take change what it read. ``touched`` counts the pages every call copied,
    scanned or sorted, the host work the list's own operations cost."""

    __slots__ = ("_pages", "_start", "_end", "ascending", "touched")

    def __init__(self, pages: List[int], *, ascending: bool = False):
        # Owned from here on: the caller must not keep mutating ``pages``.
        self._pages = pages
        self._start = 0
        self._end = len(pages)
        self.ascending = ascending or len(pages) <= 1
        self.touched = 0

    def __len__(self) -> int:
        return self._end - self._start

    def to_list(self) -> List[int]:
        """A copy of the pages, in order."""
        self.touched += len(self)
        return self._pages[self._start : self._end]

    def iter_from(self, end: str) -> Iterator[int]:
        """The pages from the ``end`` ("low": first; "high": last) inward,
        lazily: a caller that stops early touches only what it read."""
        if end == "low":
            for i in range(self._start, self._end):
                self.touched += 1
                yield self._pages[i]
        else:
            for i in range(self._end - 1, self._start - 1, -1):
                self.touched += 1
                yield self._pages[i]

    def peek(self, n: int, end: str) -> List[int]:
        """The ``n`` pages at ``end``, from that end inward."""
        assert 0 <= n <= len(self), f"peek {n} of {len(self)} free pages"
        self.touched += n
        if end == "low":
            return self._pages[self._start : self._start + n]
        return self._pages[self._end - n : self._end][::-1]

    def take(self, n: int, end: str, *, keep: Sequence[int] = ()) -> None:
        """Remove the ``n`` pages at ``end``, then put ``keep`` (pages among
        them, in ascending order) back at that end: what a walk from ``end``
        passed over without taking."""
        assert 0 <= len(keep) <= n <= len(self), (
            f"take {n} (keeping {len(keep)}) of {len(self)} free pages"
        )
        k = len(keep)
        self.touched += k
        if end == "low":
            self._start += n - k
            self._pages[self._start : self._start + k] = keep
        else:
            self._end -= n - k
            self._pages[self._end - k : self._end] = keep
        self._shrink()

    def extend(self, pages: Sequence[int]) -> None:
        """Append freed ``pages``; the list stays ascending only if they are
        and follow every page it holds."""
        if not pages:
            return
        self.touched += len(pages)
        if self.ascending:
            last = self._pages[self._end - 1] if len(self) else None
            self.ascending = (last is None or last < pages[0]) and all(
                a < b for a, b in zip(pages, pages[1:])
            )
        if self._end < len(self._pages):
            del self._pages[self._end :]
        self._pages.extend(pages)
        self._end = len(self._pages)

    def sort(self) -> None:
        """Make the pages ascending; nothing to do when they already are."""
        if self.ascending:
            return
        self.touched += len(self)
        self._pages = sorted(self._pages[self._start : self._end])
        self._start, self._end = 0, len(self._pages)
        self.ascending = True

    def _shrink(self) -> None:
        # Drop the consumed ends once they outweigh what is left, so the
        # backing list stays within twice the pages held (amortized O(1) per
        # page consumed: what a compaction copies, the takes since the last
        # one consumed at least as much).
        held = len(self)
        if self._start + (len(self._pages) - self._end) > max(held, 1):
            self.touched += held
            self._pages = self._pages[self._start : self._end]
            self._start, self._end = 0, held
