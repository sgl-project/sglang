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
    """Sparse CPU request rows; O(pages written/read), never O(tokens)."""

    def __init__(self, page_size, *, reset_preflight=None):
        self.page_size = page_size
        self._rows = {}
        self._lengths = {}
        self._reset_preflight = reset_preflight

    def assert_reset_safe(self):
        if self._reset_preflight is not None:
            self._reset_preflight()

    def assert_empty(self):
        if self._rows or self._lengths:
            raise ValueError("CPU request page rows must be empty")

    def set(self, row: int, span: PageSpan):
        if span.page_size != self.page_size or span.token_offset:
            raise ValueError("row must begin at a page boundary")
        self._rows[row] = list(span.page_ids)
        self._lengths[row] = span.token_length

    def check_writes(self, writes):
        """Validate a batch of (row, start, span) writes before any is applied."""
        if len({row for row, _, _ in writes}) != len(writes):
            raise ValueError("a row write batch must name each row once")
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
        if row is None:
            self.assert_reset_safe()
            self._rows.clear()
            self._lengths.clear()
        else:
            self._rows.pop(row, None)
            self._lengths.pop(row, None)

    def last_page(self, row, token_length):
        if not 0 < token_length <= self._lengths.get(row, 0):
            raise ValueError("request row lacks its last CPU page identity")
        return self._rows[row][(token_length - 1) // self.page_size]
