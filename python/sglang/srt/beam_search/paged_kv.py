"""Request-owned compact beam pages; actual K/V stays in the shared KV pool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator


@dataclass(frozen=True)
class GRPagedForwardMetadata:
    """One forward's addressing snapshot, without mutable request/group objects.

    ``topk_indices[b, s]`` selects beam column b's ancestor at decode step s.
    It includes the current beam column at its last entry. Page IDs and tensor
    contents are read-only for consumers; a later selection creates new tensors.
    """

    page_size: int
    context_len: int
    beam_width: int
    step: int
    context_locs: torch.Tensor
    context_pages: torch.Tensor
    beam_pages: torch.Tensor
    topk_indices: torch.Tensor
    out_cache_loc: torch.Tensor

    def beam_locations(self) -> torch.Tensor:
        """Resolve [beam, history step] to physical token slots, excluding padding."""
        steps = torch.arange(self.step + 1, device=self.beam_pages.device)
        pages = self.beam_pages[steps[None, :], self.topk_indices // self.page_size]
        return pages * self.page_size + self.topk_indices % self.page_size


@dataclass
class GRPagedKVState:
    """Own a whole-page arena for one fixed-width, fixed-depth GR group.

    Context pages remain owned by the ordinary request/radix-cache path. Each
    decode forward writes a fresh page-aligned row of beam slots. Pruned slots
    and unused future rows stay reserved until terminal cleanup. Only metadata,
    not actual K/V tensors, is stored here.
    """

    context_len: int
    beam_width: int
    output_len: int
    page_size: int
    step: int = field(default=0, init=False)
    has_reservation: bool = field(default=False, init=False)
    released: bool = field(default=False, init=False)
    beam_pages: torch.Tensor | None = field(default=None, init=False, repr=False)
    history: torch.Tensor | None = field(default=None, init=False, repr=False)
    _prepared_topk_indices: torch.Tensor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self):
        if min(self.context_len, self.beam_width, self.output_len, self.page_size) <= 0:
            raise ValueError(
                "GR paged KV requires positive context, width, depth and page size"
            )

    @property
    def pages_per_step(self) -> int:
        return (self.beam_width + self.page_size - 1) // self.page_size

    @property
    def arena_slots(self) -> int:
        return (self.output_len - 1) * self.pages_per_step * self.page_size

    @property
    def reserved_slots(self) -> int:
        return self.arena_slots if self.has_reservation else 0

    def reserve(self, allocator: BaseTokenToKVPoolAllocator) -> None:
        if self.has_reservation or self.released:
            raise RuntimeError("GR beam arena must be reserved once before retirement")
        if allocator.page_size != self.page_size:
            raise ValueError("GR beam arena and KV allocator page sizes differ")
        slots = None
        if self.arena_slots:
            slots = allocator.alloc(self.arena_slots)
            if slots is None:
                raise MemoryError(
                    f"Cannot reserve {self.arena_slots} slots for GR beam KV"
                )
        try:
            pages = (
                slots[:: self.page_size] // self.page_size
                if slots is not None
                else torch.empty(0, device=allocator.device, dtype=torch.int64)
            )
            beam_pages = pages.reshape(self.output_len - 1, self.pages_per_step)
            history = torch.empty(
                (self.beam_width, 0), device=pages.device, dtype=torch.int64
            )
        except Exception:
            # Metadata preparation can fail after the existing allocator has
            # handed us pages. Return that lease, then propagate the failure.
            if slots is not None:
                allocator.free(slots)
            raise
        self.beam_pages = beam_pages
        self.history = history
        self.has_reservation = True

    def prepare_decode(self, context_locs: torch.Tensor) -> GRPagedForwardMetadata:
        if not self.has_reservation or self.step >= self.output_len - 1:
            raise RuntimeError("GR beam arena has no remaining decode step")
        if context_locs.ndim != 1 or context_locs.numel() != self.context_len:
            raise ValueError(
                "GR decode requires exactly the leader's context locations"
            )
        # Radix checkpointing can replace the leader's context mapping. Capture
        # it after checkpointing, never an earlier member-row alias.
        context_locs = context_locs.to(dtype=torch.int64).clone()
        beam_columns = torch.arange(self.beam_width, device=self.beam_pages.device)
        topk_indices = torch.cat((self.history, beam_columns[:, None]), dim=1)
        self._prepared_topk_indices = topk_indices
        out_cache_loc = (
            self.beam_pages[self.step, beam_columns // self.page_size] * self.page_size
            + beam_columns % self.page_size
        )
        return GRPagedForwardMetadata(
            page_size=self.page_size,
            context_len=self.context_len,
            beam_width=self.beam_width,
            step=self.step,
            context_locs=context_locs,
            context_pages=context_locs[:: self.page_size] // self.page_size,
            beam_pages=self.beam_pages,
            topk_indices=topk_indices,
            out_cache_loc=out_cache_loc,
        )

    def advance(self, parent_idx: torch.Tensor) -> None:
        """Select the next beam ancestry without copying or freeing any K/V."""
        if self._prepared_topk_indices is None:
            raise RuntimeError(
                "Prepare the GR decode forward before advancing ancestry"
            )
        if parent_idx.ndim != 1 or parent_idx.numel() != self.beam_width:
            raise ValueError("GR paged KV requires a fixed number of beam parents")
        self.history = self._prepared_topk_indices.index_select(
            0, parent_idx.to(dtype=torch.int64)
        )
        self.step += 1
        self._prepared_topk_indices = None

    def release(self, free_tokens: Callable[[torch.Tensor], None]) -> None:
        """Retire once, including aborts before member rows were created.

        The eager POC has no overlapping forwards. Drain its current CUDA stream
        before returning arena pages; a fatal stream error propagates to worker
        teardown instead of returning potentially in-use storage to the pool.
        """
        if self.released:
            return
        if self.has_reservation and self.beam_pages.numel():
            if self.beam_pages.is_cuda:
                torch.cuda.current_stream(self.beam_pages.device).synchronize()
            # Free one representative token per whole owned page through the
            # caller's storage interface. At P=1 each token is its own page.
            free_tokens(self.beam_pages.reshape(-1) * self.page_size)
        self.has_reservation = False
        self.released = True
        self.beam_pages = None
        self.history = None
        self._prepared_topk_indices = None
