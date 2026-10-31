"""Batch-by-batch bookkeeping for a multi-window Unlimited-OCR document plan."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import msgspec


class DocumentBatchLayout(msgspec.Struct, frozen=True):
    """Page range of one document window; page ids are global document ids."""

    input_start: int
    input_end: int
    emit_start: int
    new_pages: Tuple[int, ...]


class DocumentKVState(msgspec.Struct):
    """Validated ``[(lo, hi, emit_from), ...]`` plan, advanced one batch at a time."""

    page_count: int
    overlap_pages: int
    plan: Tuple[Tuple[int, int, int], ...]
    next_batch_index: int = 0
    active_layout: Optional[DocumentBatchLayout] = None

    def __post_init__(self) -> None:
        if self.page_count < 0:
            raise ValueError("page_count must be non-negative")
        if self.overlap_pages < 0:
            raise ValueError("overlap_pages must be non-negative")
        if not self.plan:
            raise ValueError("document KV state requires a non-empty plan")
        self._validate_plan()

    @classmethod
    def from_plan(
        cls,
        *,
        page_count: int,
        overlap_pages: int,
        plan: Sequence[Tuple[int, int, int]],
    ) -> DocumentKVState:
        """Build the state for a ``[(lo, hi, emit_from), ...]`` split plan."""
        return cls(
            page_count=page_count,
            overlap_pages=overlap_pages,
            plan=tuple(tuple(int(x) for x in batch) for batch in plan),
        )

    def _validate_plan(self) -> None:
        if self.plan[0] != (0, self.plan[0][1], 0):
            raise ValueError("the first document batch must start at page zero")
        previous_emit = 0
        for index, (lo, hi, emit_from) in enumerate(self.plan):
            if not 0 <= lo < hi <= self.page_count:
                raise ValueError(f"invalid page range in batch {index}: {(lo, hi)}")
            if not lo <= emit_from < hi:
                raise ValueError(
                    f"emit_from must be inside the batch in batch {index}: "
                    f"{(lo, hi, emit_from)}"
                )
            if index == 0 and emit_from != 0:
                raise ValueError("the first batch must emit from page zero")
            if index > 0:
                if emit_from != previous_emit:
                    raise ValueError("document batches must emit contiguously")
                if emit_from - lo != self.overlap_pages:
                    raise ValueError(
                        f"batch {index} overlap does not match configured value: "
                        f"expected {self.overlap_pages}, got {emit_from - lo}"
                    )
            previous_emit = hi
        if previous_emit != self.page_count:
            raise ValueError("document plan does not cover all pages")

    def begin_batch(self) -> DocumentBatchLayout:
        """Advance to the next planned batch and return its page layout."""
        index = self.next_batch_index
        if index >= len(self.plan):
            raise ValueError("all document batches have already started")
        lo, hi, emit_from = self.plan[index]
        layout = DocumentBatchLayout(
            input_start=lo,
            input_end=hi,
            emit_start=emit_from,
            new_pages=tuple(range(emit_from, hi)),
        )
        self.active_layout = layout
        self.next_batch_index += 1
        return layout

    def complete_batch(self, emitted_pages: Iterable[int]) -> None:
        """Commit output ownership for the active batch."""
        if self.active_layout is None:
            raise ValueError("cannot complete a document batch before begin_batch")
        emitted = tuple(int(page) for page in emitted_pages)
        expected = self.active_layout.new_pages
        if emitted != expected:
            raise ValueError(f"batch emitted pages {emitted}, expected {expected}")
        self.active_layout = None

    def abort_batch(self) -> None:
        """Discard the active layout after an error or incomplete response."""
        self.active_layout = None
