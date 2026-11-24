"""Explicit CPU provenance for physical paged KV, with no device readbacks.

The scheduler passes these immutable page-granular spans alongside tensors.
They are never inferred from tensor identity, storage pointers, or CUDA data.
"""

from dataclasses import dataclass

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator


@dataclass(frozen=True)
class PageFree:
    """Explicit tensor + CPU provenance carried through existing free actions."""

    indices: object
    page_span: "PageSpan"


def with_page_span(indices, page_span):
    return indices if page_span is None else PageFree(indices, page_span)


def page_span_kwargs(page_span):
    # Storages and allocators without CPU page identity take no page_span.
    return {} if page_span is None else {"page_span": page_span}


def split_page_free(indices):
    """Return ``(indices, page_span)``; the span is None for a bare tensor."""
    if isinstance(indices, PageFree):
        return indices.indices, indices.page_span
    return indices, None


def slice_page_span(page_span, start, end):
    return None if page_span is None else page_span.slice(start, end)


def cpu_page_identity(allocator):
    # A tree cache may be built without an allocator (None).
    if not isinstance(allocator, BaseTokenToKVPoolAllocator):
        return None
    return allocator.cpu_page_identity()


@dataclass(frozen=True)
class PageSpan:
    page_ids: tuple[int, ...]
    page_size: int
    token_offset: int
    token_length: int

    def __post_init__(self):
        if self.page_size <= 0 or not 0 <= self.token_offset < self.page_size:
            raise ValueError("invalid page geometry")
        if self.token_length < 0:
            raise ValueError("negative span length")
        needed = -(-(self.token_offset + self.token_length) // self.page_size)
        if not self.token_length:
            needed = 0
        if len(self.page_ids) != needed:
            raise ValueError("page identities do not cover the token span exactly")

    def slice(self, start: int, end: int):
        if not 0 <= start <= end <= self.token_length:
            raise ValueError("CPU page identity slice is outside the span")
        if start == end:
            return PageSpan((), self.page_size, 0, 0)
        lo = (self.token_offset + start) // self.page_size
        hi = -(-(self.token_offset + end) // self.page_size)
        return PageSpan(
            self.page_ids[lo:hi],
            self.page_size,
            (self.token_offset + start) % self.page_size,
            end - start,
        )

    @classmethod
    def concat(cls, spans, *, page_size=None):
        spans = tuple(spans)
        if not spans:
            if page_size is None:
                raise ValueError("empty concat needs page size")
            return cls((), page_size, 0, 0)
        ps = spans[0].page_size
        ids, length, offset = [], 0, None
        for span in spans:
            if span.page_size != ps:
                raise ValueError("incompatible page sizes")
            if not span.token_length:
                continue
            if offset is None:
                offset = span.token_offset
            else:
                seam = (offset + length) % ps
                if seam != span.token_offset:
                    raise ValueError("incompatible span boundary")
                if seam:
                    if ids[-1] != span.page_ids[0]:
                        raise ValueError(
                            "partial page identity changed across boundary"
                        )
                    ids.pop()
            ids.extend(span.page_ids)
            length += span.token_length
        return cls(tuple(ids), ps, offset or 0, length)


class PageRowTable:
    """Sparse CPU request rows in decoder token positions; encoder-decoder rows
    are unsupported. O(pages written/read), never O(tokens)."""

    def __init__(self, page_size, *, reset_preflight=None):
        self.page_size = page_size
        self._rows = {}
        self._lengths = {}
        self._reset_preflight = reset_preflight

    def assert_reset_safe(self):
        if self._reset_preflight is not None:
            self._reset_preflight()

    def check_writes(self, writes):
        """Validate a batch of (row, start, span) writes, one per row, before
        any is applied."""
        for row, start, span in writes:
            self._check_write(row, start, span)

    def _check_write(self, row, start, span):
        if (
            start < 0
            or span.page_size != self.page_size
            or span.token_offset != start % self.page_size
        ):
            raise ValueError("row write page alignment mismatch")
        length = self._lengths.get(row, 0)
        if start > length:
            raise ValueError("row write leaves missing CPU page identities")
        ids = self._rows.get(row, [])
        end = start + span.token_length
        # A page keeps its identity while any of its recorded tokens stay unwritten.
        if span.token_offset and ids[start // self.page_size] != span.page_ids[0]:
            raise ValueError("row write changes an existing partial page")
        if (
            end % self.page_size
            and end < length
            and ids[(end - 1) // self.page_size] != span.page_ids[-1]
        ):
            raise ValueError("row write changes an existing partial page")

    def write(self, row: int, start: int, span: PageSpan):
        self._check_write(row, start, span)
        ids = self._rows.setdefault(row, [])
        lo = start // self.page_size
        ids[lo : lo + len(span.page_ids)] = span.page_ids
        self._lengths[row] = max(self._lengths.get(row, 0), start + span.token_length)

    def get(self, row: int, start: int, end: int):
        if not 0 <= start <= end <= self._lengths.get(row, 0):
            raise ValueError("request row lacks CPU page identity coverage")
        if start == end:
            return PageSpan((), self.page_size, 0, 0)
        lo, hi = start // self.page_size, -(-end // self.page_size)
        return PageSpan(
            tuple(self._rows[row][lo:hi]),
            self.page_size,
            start % self.page_size,
            end - start,
        )

    def clear(self, row=None):
        """Without ``row``, the caller has run ``assert_reset_safe`` first."""
        if row is None:
            self._rows.clear()
            self._lengths.clear()
        else:
            self._rows.pop(row, None)
            self._lengths.pop(row, None)
