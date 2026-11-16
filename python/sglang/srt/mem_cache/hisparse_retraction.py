"""Page-owned DSV4 state that the persistent HiSparse C4 host row does not hold."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.deepseek_v4_buffer_info import (
    DSV4CheckpointBufferInfo,
    DSV4PagedBufferInfo,
)
from sglang.srt.mem_cache.hicache_storage import PoolName

# Native page_slots count compressed tokens. FULL locations must be scaled
# before addressing those pages. Unknown regions fail before device state is freed.
_FULL_TOKEN_RATIOS = {
    PoolName.DEEPSEEK_V4_C1: 1,
    PoolName.DEEPSEEK_V4_C1_INDEXER: 1,
    PoolName.DEEPSEEK_V4_C1_INDEXER_SCALE: 1,
    PoolName.DEEPSEEK_V4_C2: 2,
    PoolName.DEEPSEEK_V4_C2_INDEXER: 2,
    PoolName.DEEPSEEK_V4_C2_INDEXER_SCALE: 2,
    PoolName.DEEPSEEK_V4_C4_INDEXER: 4,
    PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE: 4,
    PoolName.DEEPSEEK_V4_C4_ROPE: 4,
    PoolName.DEEPSEEK_V4_C128: 128,
    PoolName.DEEPSEEK_V4_C128_ROPE: 128,
}


@dataclass(frozen=True)
class _RegionLayout:
    name: PoolName
    indices_from: PoolName
    layer_ids: tuple[int, ...]
    tokens_per_page: int
    storage_format: tuple


def _regions(pool):
    """Refresh native views: pools may have been resized since capture."""
    regions = {}
    for info in pool.get_device_pool_infos():
        if info.pool_name == PoolName.DEEPSEEK_V4_C4:
            continue  # Persistent HiSparse host storage already covers this region.
        native = info.buffer_info
        if isinstance(native, DSV4PagedBufferInfo):
            buffers = native.buffers
            coverage = native.page_slots
            storage_format = (native.layout,)
        elif isinstance(native, DSV4CheckpointBufferInfo):
            buffers = tuple(state.pages for state in native.states)
            coverage = pool.swa_page_size
            storage_format = tuple(
                (state.ring_size, state.dtype) for state in native.states
            )
            if info.indices_from_pool != PoolName.SWA:
                raise ValueError(
                    f"{info.pool_name}: compressor rings must follow SWA pages"
                )
        else:
            raise ValueError(
                f"Unsupported HiSparse retraction region: {info.pool_name}"
            )
        if info.indices_from_pool == PoolName.KV:
            if info.pool_name not in _FULL_TOKEN_RATIOS:
                raise ValueError(f"Unknown HiSparse FULL region: {info.pool_name}")
            coverage *= _FULL_TOKEN_RATIOS[info.pool_name]
        elif info.indices_from_pool != PoolName.SWA:
            raise ValueError(f"Unknown HiSparse index space: {info.indices_from_pool}")
        if coverage <= 0 or len(buffers) != len(info.layer_ids):
            raise ValueError(f"Invalid HiSparse region geometry: {info.pool_name}")
        if info.pool_name in regions:
            raise ValueError(f"Duplicate HiSparse region: {info.pool_name}")
        if any(buffer.ndim < 2 or not buffer.is_contiguous() for buffer in buffers):
            raise ValueError(
                f"HiSparse region must have contiguous pages: {info.pool_name}"
            )
        # Copy raw page bytes, including padding and FP4 scale payloads. This
        # also avoids requiring advanced indexing support for packed FP4 dtypes.
        byte_buffers = tuple(
            buffer.view(torch.uint8).reshape(buffer.shape[0], -1) for buffer in buffers
        )
        layout = _RegionLayout(
            info.pool_name,
            info.indices_from_pool,
            info.layer_ids,
            coverage,
            (storage_format, tuple((tuple(b.shape[1:]), b.dtype) for b in buffers)),
        )
        regions[info.pool_name] = (layout, byte_buffers)
    return regions


def _page_rows(layout, full_locs, swa_locs):
    locs = full_locs if layout.indices_from == PoolName.KV else swa_locs
    # Native index/SWA pages can be smaller than an allocator page. Sampling
    # every logical page would lose rows; enumerate native pages in token order.
    return torch.unique_consecutive(locs // layout.tokens_per_page).long()


def _check_page_rows(name, rows, buffers):
    if rows.numel() and (
        int(rows.min()) < 0 or any(int(rows.max()) >= len(buffer) for buffer in buffers)
    ):
        raise ValueError(f"HiSparse retraction page out of range: {name}")


@dataclass(frozen=True)
class DSV4RetractionSnapshot:
    """CPU copies only; request-private rings remain on the retained request row.

    The native inventory includes C128 KV, all indexer payloads/scales, and
    page-owned attention/indexer compressor rings. Request-window SWA and
    request-private C128/pair state keep their original request-indexed storage.
    """

    committed_len: int
    window_start: int
    regions: dict[PoolName, tuple[_RegionLayout, tuple[torch.Tensor, ...]]]

    @classmethod
    def capture(cls, allocator, full_locs, window_size):
        if window_size is None or window_size <= 0:
            raise ValueError("HiSparse retraction requires a positive sliding window")
        regions = _regions(allocator.get_kvcache())
        start = max(0, full_locs.numel() - window_size)
        start = start // allocator.page_size * allocator.page_size
        swa_locs = allocator.translate_loc_from_full_to_swa(full_locs[start:])
        saved = {}
        for name, (layout, buffers) in regions.items():
            rows = _page_rows(layout, full_locs, swa_locs)
            _check_page_rows(name, rows, buffers)
            copies = []
            for buffer in buffers:
                copy = buffer[rows].to("cpu", copy=True)
                copies.append(copy.pin_memory() if buffer.is_cuda else copy)
            saved[name] = (layout, tuple(copies))
        return cls(full_locs.numel(), start, saved)

    def restore(self, allocator, full_locs):
        if full_locs.numel() != self.committed_len:
            raise ValueError("HiSparse retraction committed length changed")
        regions = _regions(allocator.get_kvcache())
        if regions.keys() != self.regions.keys():
            raise ValueError("HiSparse retraction region inventory changed")
        swa_locs = allocator.translate_loc_from_full_to_swa(
            full_locs[self.window_start :]
        )
        transfers = []
        # Validate EVERY region before starting writes. zip must never silently
        # truncate a changed layer group or a separate payload/scale region.
        for name, (layout, buffers) in regions.items():
            old_layout, copies = self.regions[name]
            rows = _page_rows(layout, full_locs, swa_locs)
            if layout != old_layout or len(buffers) != len(copies):
                raise ValueError(f"HiSparse retraction region layout changed: {name}")
            _check_page_rows(name, rows, buffers)
            for buffer, copy in zip(buffers, copies, strict=True):
                if copy.shape != (rows.numel(), buffer.shape[1]):
                    raise ValueError(f"HiSparse retraction page count changed: {name}")
                transfers.append((buffer, rows, copy))
        for buffer, rows, copy in transfers:
            buffer[rows] = copy.to(buffer.device, non_blocking=True)
