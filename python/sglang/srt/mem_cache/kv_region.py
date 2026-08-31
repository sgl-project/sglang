from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import msgspec
import torch

# (rows to touch, side info for the blob) / (rows to touch, index into saved rows)
SavePlan = Tuple[torch.Tensor, Optional[torch.Tensor]]
LoadPlan = Tuple[torch.Tensor, Optional[torch.Tensor]]


class RequestCtx(msgspec.Struct, frozen=True):
    """Addressing inputs for one request's save or load."""

    token_indices: torch.Tensor
    req_pool_idx: int

    @property
    def seq_len(self) -> int:
        return int(self.token_indices.numel())


class PageAligned(msgspec.Struct, frozen=True):
    """Paged buffer whose single row holds ``stride`` consecutive logical tokens.

    ``stride == compression_ratio * pool_page_size``: a compressed pool folds
    ``ratio`` tokens into one entry and packs ``pool_page_size`` entries per row.

    Sampling every ``stride``-th token instead of ``torch.unique`` keeps the row
    count a function of ``seq_len`` alone, which is what makes save and load
    positionally symmetric. Mirrors ``IndexKeyCache.cpu_copy``.
    """

    stride: int

    def save_plan(self, ctx: RequestCtx) -> SavePlan:
        return self._rows(ctx), None

    def load_plan(self, ctx: RequestCtx, side: Optional[torch.Tensor]) -> LoadPlan:
        del side
        return self._rows(ctx), None

    def _rows(self, ctx: RequestCtx) -> torch.Tensor:
        return ctx.token_indices[:: self.stride] // self.stride


def _swa_page_state(
    mapping: torch.Tensor, ctx: RequestCtx, page_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per *token page* of the request: its SWA page, and whether it is mapped.

    One representative token per page block, so the result is indexed by page
    ``k`` of the request on both the save and the load side regardless of how the
    SWA ring happens to be laid out (it can wrap).
    """
    representatives = ctx.token_indices[::page_size]
    swa_locs = mapping[representatives]
    return swa_locs // page_size, swa_locs > 0


class SwaMapped(msgspec.Struct, frozen=True):
    """SWA ring pages reached through ``full_to_swa_index_mapping``.

    Slot 0 is the reserved dummy slot: tail-only SWA allocation leaves
    out-of-window tokens unmapped. Which token pages are mapped depends on the
    allocator's ring state, so save and load can disagree; only pages mapped on
    both sides are restored (a page unmapped at save time had already left the
    window, and one unmapped at load time will not be read).
    """

    mapping: torch.Tensor
    page_size: int

    def save_plan(self, ctx: RequestCtx) -> SavePlan:
        rows, mapped = _swa_page_state(self.mapping, ctx, self.page_size)
        return rows[mapped], mapped.cpu()

    def load_plan(self, ctx: RequestCtx, side: Optional[torch.Tensor]) -> LoadPlan:
        assert side is not None, "SwaMapped needs the save-side mapped mask"
        rows, mapped = _swa_page_state(self.mapping, ctx, self.page_size)
        saved_mapped = side.to(rows.device)
        both = saved_mapped & mapped
        # Position of each token page inside the saved (save-side mapped) rows.
        saved_position = torch.cumsum(saved_mapped.to(torch.int64), 0) - 1
        return rows[both], saved_position[both]


class SwaPageRing(msgspec.Struct, frozen=True):
    """Compress-state rows: one whole ``ring_size`` block per SWA page.

    Matches ``CompressStatePool.translate_from_swa_loc_to_state_loc``, which maps
    a SWA location to ``(swa_loc // swa_page_size) * ring_size + ...``; the ring
    block moves as a unit because the live slot within it depends on positions
    that resume rewrites. Mapped-page bookkeeping mirrors :class:`SwaMapped`.
    """

    mapping: torch.Tensor
    swa_page_size: int
    ring_size: int

    def save_plan(self, ctx: RequestCtx) -> SavePlan:
        rows, mapped = _swa_page_state(self.mapping, ctx, self.swa_page_size)
        return self._expand(rows[mapped]), mapped.cpu()

    def load_plan(self, ctx: RequestCtx, side: Optional[torch.Tensor]) -> LoadPlan:
        assert side is not None, "SwaPageRing needs the save-side mapped mask"
        rows, mapped = _swa_page_state(self.mapping, ctx, self.swa_page_size)
        saved_mapped = side.to(rows.device)
        both = saved_mapped & mapped
        saved_position = torch.cumsum(saved_mapped.to(torch.int64), 0) - 1
        return self._expand(rows[both]), self._expand(saved_position[both])

    def _expand(self, pages: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.ring_size, device=pages.device)
        return (pages[:, None] * self.ring_size + offsets[None, :]).reshape(-1)


class ReqScoped(msgspec.Struct, frozen=True):
    """State addressed by ``req_pool_idx`` rather than by token location.

    Each request slot owns ``rows_per_req`` consecutive rows. Those rows are
    grouped into blocks of ``block_rows``; with ``block_tokens == 0`` the slot is
    a single block, otherwise the block holding the sequence's live remainder is
    selected. A sequence ending exactly on a ``block_tokens`` boundary owns no
    partial state and yields no rows -- the same rule as
    ``get_dsv4_c128_state_indices``, whose returned index counts whole blocks.
    """

    rows_per_req: int = 1
    block_rows: int = 1
    block_tokens: int = 0

    def save_plan(self, ctx: RequestCtx) -> SavePlan:
        return self._rows(ctx), None

    def load_plan(self, ctx: RequestCtx, side: Optional[torch.Tensor]) -> LoadPlan:
        del side
        return self._rows(ctx), None

    def _rows(self, ctx: RequestCtx) -> torch.Tensor:
        device = ctx.token_indices.device
        base = ctx.req_pool_idx * self.rows_per_req
        if self.block_tokens == 0:
            return torch.arange(
                base, base + self.block_rows, dtype=torch.int64, device=device
            )
        if ctx.seq_len == 0 or ctx.seq_len % self.block_tokens == 0:
            return torch.empty(0, dtype=torch.int64, device=device)
        num_blocks = self.rows_per_req // self.block_rows
        span = num_blocks * self.block_tokens
        block = ((ctx.seq_len - 1) % span) // self.block_tokens
        start = base + block * self.block_rows
        return torch.arange(
            start, start + self.block_rows, dtype=torch.int64, device=device
        )


Addressing = Union[PageAligned, SwaMapped, SwaPageRing, ReqScoped]


class KVRegion(msgspec.Struct, frozen=True):
    """One group of per-layer buffers sharing an addressing scheme."""

    name: str
    tensors: tuple
    addressing: Addressing
    # Called with req_pool_idx before loading, for state whose stale rows would
    # otherwise leak into the resumed request (see DecodePreallocQueue's
    # clear_c128_req_state call on the PD path).
    reset_before_load: Optional[Callable[[int], None]] = None


HostCopy = Dict[str, Optional[List[torch.Tensor]]]
SavedRegion = Tuple[List[torch.Tensor], Optional[torch.Tensor]]
HostBlob = Dict[str, Optional[SavedRegion]]


def save_regions(*, regions: Iterable[KVRegion], ctx: RequestCtx) -> HostBlob:
    """Copy every region's rows for this request to host memory."""
    host: HostBlob = {}
    for region in regions:
        rows, side = region.addressing.save_plan(ctx)
        if rows.numel() == 0:
            host[region.name] = None
            continue
        host[region.name] = (
            [tensor[rows].to("cpu") for tensor in region.tensors],
            side,
        )
    return host


def load_regions(
    *, regions: Iterable[KVRegion], host: HostBlob, ctx: RequestCtx
) -> None:
    """Write a :func:`save_regions` result back, using this request's *current*
    token indices and req_pool_idx (both differ from the save side)."""
    for region in regions:
        if region.reset_before_load is not None:
            region.reset_before_load(ctx.req_pool_idx)

        entry = host[region.name]
        if entry is None:
            continue
        saved, side = entry

        rows, source = region.addressing.load_plan(ctx, side)
        if rows.numel() == 0:
            continue
        if source is None:
            assert rows.numel() == saved[0].shape[0], (
                f"region {region.name!r} is not save/load symmetric: saved "
                f"{saved[0].shape[0]} rows, load side wants {rows.numel()}"
            )
        else:
            assert source.numel() == rows.numel(), (
                f"region {region.name!r}: {source.numel()} source rows for "
                f"{rows.numel()} destination rows"
            )
        for tensor, saved_layer in zip(region.tensors, saved):
            payload = saved_layer.to(tensor.device)
            if source is not None:
                payload = payload[source]
            tensor[rows] = payload
