from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import msgspec
import torch

# (rows to touch, side info for the blob) / (rows to touch, index into saved rows)
SavePlan = Tuple[torch.Tensor, Optional[torch.Tensor]]
LoadPlan = Tuple[torch.Tensor, Optional[torch.Tensor]]


class RequestCtx(msgspec.Struct, frozen=True):
    token_indices: torch.Tensor
    req_pool_idx: int

    @property
    def seq_len(self) -> int:
        return int(self.token_indices.numel())


class PageAligned(msgspec.Struct, frozen=True):
    """Paged buffer whose row holds ``stride == compression_ratio * pool_page_size``
    consecutive tokens. Sampling every ``stride``-th token keeps the row count a
    function of ``seq_len`` alone, which is what makes save and load symmetric.
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
    """Each token page's SWA page and whether it is mapped, keyed by page ``k`` of
    the request so save and load agree however the ring is laid out (it can wrap).
    """
    representatives = ctx.token_indices[::page_size]
    swa_locs = mapping[representatives]
    return swa_locs // page_size, swa_locs > 0


class SwaMapped(msgspec.Struct, frozen=True):
    """SWA ring pages via ``full_to_swa_index_mapping``; slot 0 is the reserved
    dummy for out-of-window tokens. Save and load can see different pages mapped,
    so only both-mapped pages are restored -- one unmapped at save had already
    left the window, one unmapped at load will not be read.
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
    """Compress-state rows: one whole ``ring_size`` block per SWA page, matching
    ``CompressStatePool.translate_from_swa_loc_to_state_loc``. The block moves as
    a unit: the live slot inside it depends on positions that resume rewrites.
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
    """State addressed by ``req_pool_idx``: each slot owns ``rows_per_req`` rows in
    blocks of ``block_rows``. With ``block_tokens == 0`` the slot is one block;
    otherwise the block holding the sequence's live remainder is selected, and a
    sequence ending exactly on a ``block_tokens`` boundary owns no partial state
    and yields no rows -- the same rule as ``get_dsv4_c128_state_indices``.
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
    # Called with req_pool_idx before loading; without it stale rows leak into the
    # resumed request (see clear_c128_req_state on the PD path).
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
    """Write a ``save_regions`` result back using this request's *current* token
    indices and req_pool_idx, which both differ from the save side."""
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
