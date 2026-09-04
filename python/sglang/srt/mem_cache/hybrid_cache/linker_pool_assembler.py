"""Materialize hybrid device pools for external cache linkers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class DevicePoolEntry:
    """Zero-copy linker view over one physical device pool."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int | Sequence[int]],
        page_size: int,
        rows_are_pages: bool,
        packed: bool = True,
        index_mapper: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        self._page_offsets = torch.arange(page_size)
        self._row_span = 1 if rows_are_pages else page_size

        if not self.components or any(not component for component in self.components):
            raise ValueError(f"Device pool {name} has no storage buffers.")
        self.kv_buffer = [buffer for group in self.components for buffer in group]
        self._row_count = min(buffer.shape[0] for buffer in self.kv_buffer)

        self.buffer_meta = [
            [
                (
                    buffer.data_ptr(),
                    buffer.stride(0) * buffer.element_size(),
                    buffer.nbytes // buffer.shape[0] * self._row_span,
                )
                for buffer in component
            ]
            for component in self.components
        ]

        self._component_offsets = []
        offset = 0
        for component in self.buffer_meta:
            if not packed:
                offset = 0
            offsets = []
            for _, _, size in component:
                offsets.append(offset)
                offset += size
            self._component_offsets.append(offsets)

    def get_hybrid_pool_buffer(self) -> list[torch.Tensor]:
        return self.kv_buffer

    def translate_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return self._index_mapper(indices) if self._index_mapper else indices

    def _rows(self, indices: torch.Tensor) -> list[int]:
        slots = indices.detach().to(device="cpu", dtype=torch.int64).flatten()
        if slots.numel() % self.page_size:
            raise ValueError(
                f"Pool {self.name} got {slots.numel()} indices, expected a "
                f"multiple of page_size={self.page_size}."
            )
        if not slots.numel():
            return []

        pages = slots.reshape(-1, self.page_size)
        starts = pages[:, 0]
        if torch.any(starts.remainder(self.page_size)) or not torch.equal(
            pages, starts[:, None] + self._page_offsets
        ):
            raise ValueError(f"Pool {self.name} requires aligned contiguous pages.")
        rows = (
            starts.div(self.page_size, rounding_mode="floor")
            if self._row_span == 1
            else starts
        )
        first_row = int(rows.min())
        last_row = int(rows.max()) + self._row_span
        if first_row < 0 or last_row > self._row_count:
            raise ValueError(
                f"Pool {self.name} row range [{first_row}, {last_row}) exceeds "
                f"buffer shapes {[tuple(buffer.shape) for buffer in self.kv_buffer]}."
            )
        return rows.tolist()

    def get_page_buffer_meta(self, indices: torch.Tensor):
        rows = self._rows(indices)
        ptrs = [
            base_ptr + row * row_stride
            for row in rows
            for component in self.buffer_meta
            for base_ptr, row_stride, _ in component
        ]
        sizes = [
            size
            for _ in rows
            for component in self.buffer_meta
            for _, _, size in component
        ]
        return ptrs, sizes

    def prepare_locations(self, indices: torch.Tensor) -> list[int]:
        return self._rows(indices)

    def get_prepared_layer_range_meta(self, locations: list[int], layer: int):
        mapped = self.layer_mapping.get(layer)
        if mapped is None:
            return None
        buffer_indices = [mapped] if isinstance(mapped, int) else list(mapped)

        items_by_component = [
            [
                (*component[buffer_index], component_offsets[buffer_index])
                for buffer_index in buffer_indices
            ]
            for component, component_offsets in zip(
                self.buffer_meta, self._component_offsets
            )
        ]

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                [base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items]
                for items in items_by_component
            ]
            row_sizes = [
                [size for _, _, size, _ in items] for items in items_by_component
            ]
            row_offsets = [
                [offset for _, _, _, offset in items] for items in items_by_component
            ]
            if self.packed:
                ptrs.append([value for component in row_ptrs for value in component])
                sizes.append([value for component in row_sizes for value in component])
                offsets.append(
                    [value for component in row_offsets for value in component]
                )
            else:
                ptrs.extend(row_ptrs)
                sizes.extend(row_sizes)
                offsets.extend(row_offsets)
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """Physical device pools sharing one logical linker layer range."""

    def __init__(
        self,
        entries: Sequence[DevicePoolEntry],
        num_layers: int,
        page_size: int,
        *,
        rank_replicated: bool = False,
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.rank_replicated = rank_replicated
        self.kv_buffer = None

    def resolve_transfers(
        self,
        transfers: list[PoolTransfer],
        *,
        allow_partial: bool = False,
        allow_missing_kv: bool = False,
    ) -> list[PoolTransfer]:
        """Expand logical component transfers into physical device pools."""
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if not any(transfer.keys for transfer in transfers):
            return []
        if not allow_missing_kv and (kv is None or not kv.keys):
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None or not source.keys:
                continue
            indices = source.device_indices
            resolved.append(
                replace(
                    source,
                    name=name,
                    host_indices=(
                        self.entry_map[name].translate_indices(indices)
                        if indices is not None
                        else None
                    ),
                    keys=list(source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


def _deepseek_v4_state_views(state_pools: list[Any], global_layers: list[int]):
    views = []
    for layer in global_layers:
        pool = state_pools[layer]
        state = pool.kv_score_buffer.kv_score
        ring = int(pool.ring_size)
        usable = state.shape[0] // ring * ring
        views.append(
            state.view(torch.uint8)
            .reshape(state.shape[0], -1)[:usable]
            .reshape(usable // ring, -1)
        )
    return views


def _with_packed_draft_mapping(
    layer_mapping: dict[int, int],
    *,
    target_device_layer_num: int,
    draft_layer_num: int,
) -> dict[int, int | tuple[int, ...]]:
    """Attach draft depth N to the same transfer layer as target layer N."""
    if draft_layer_num > len(layer_mapping):
        raise ValueError(
            "Packed draft layers exceed the target transfer layer count: "
            f"{draft_layer_num} > {len(layer_mapping)}."
        )
    result: dict[int, int | tuple[int, ...]] = dict(layer_mapping)
    for depth in range(draft_layer_num):
        target = result[depth]
        target_indices = (target,) if isinstance(target, int) else target
        result[depth] = (*target_indices, target_device_layer_num + depth)
    return result


def _drop_empty_buffers_and_remap(
    buffers: list[torch.Tensor],
    layer_mapping: dict[int, int | Sequence[int]],
) -> tuple[list[torch.Tensor], dict[int, int | tuple[int, ...]]]:
    """Drop zero-row placeholders before DevicePoolEntry computes row metadata."""
    old_to_new = {
        old: new
        for new, old in enumerate(
            index for index, buffer in enumerate(buffers) if buffer.shape[0] > 0
        )
    }
    active_mapping = {}
    for layer, mapped in layer_mapping.items():
        indices = (mapped,) if isinstance(mapped, int) else mapped
        active = tuple(old_to_new[index] for index in indices if index in old_to_new)
        if active:
            active_mapping[layer] = active[0] if len(active) == 1 else active
    return [buffers[index] for index in old_to_new], active_mapping


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _dsv4_indexer_regions,
        _resolve_deepseek_v4_layer_mappings,
    )

    mappings = _resolve_deepseek_v4_layer_mappings(kvcache)
    if getattr(kvcache, "_unified_kv", False) or isinstance(
        kvcache.c4_kv_pool, HiSparseC4DevicePool
    ):
        raise ValueError(
            "The direct external linker does not support unified-KV or HiSparse."
        )
    if kvcache.swa_page_size != page_size:
        raise ValueError(
            "DeepSeek V4 SWA page size must match the tree page size: "
            f"{kvcache.swa_page_size} != {page_size}."
        )

    draft_swa_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.swa_kv_pool.kv_buffer
    ]
    swa_mapping = _with_packed_draft_mapping(
        mappings.swa,
        target_device_layer_num=len(kvcache.swa_kv_pool.kv_buffer),
        draft_layer_num=len(draft_swa_buffers),
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.SWA,
            indices_from_pool=PoolName.SWA,
            device_pool=kvcache.swa_kv_pool,
            components=[[*kvcache.swa_kv_pool.kv_buffer, *draft_swa_buffers]],
            layer_mapping=swa_mapping,
            page_size=page_size,
            rows_are_pages=True,
        )
    ]

    def add(name, source, pool, buffers, layer_mapping):
        if layer_mapping:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=pool,
                    components=[buffers],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )

    add(
        PoolName.DEEPSEEK_V4_C4,
        PoolName.KV,
        kvcache.c4_kv_pool,
        kvcache.c4_kv_pool.kv_buffer,
        mappings.c4,
    )
    for region in _dsv4_indexer_regions(kvcache, page_size):
        add(
            region.name,
            PoolName.KV,
            kvcache.c4_indexer_kv_pool,
            region.device_buffers,
            mappings.c4,
        )
    add(
        PoolName.DEEPSEEK_V4_C128,
        PoolName.KV,
        kvcache.c128_kv_pool,
        kvcache.c128_kv_pool.kv_buffer,
        mappings.c128,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.SWA,
        kvcache.compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    add(
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
        PoolName.SWA,
        kvcache.indexer_compress_state_pools,
        _deepseek_v4_state_views(
            kvcache.indexer_compress_state_pools,
            mappings.c4_state_global_layers,
        ),
        mappings.c4_state,
    )
    return DevicePoolGroup(
        entries,
        mappings.transfer_layer_num,
        page_size,
        rank_replicated=True,
    )


def _build_dsa_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )
    num_layers = kvcache.layer_num
    draft_pools = tuple(
        pool
        for pool in mtp_draft_device_pools
        if getattr(pool, "index_k_with_scale_buffer", None)
    )
    if any(pool.page_size != page_size for pool in draft_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [buffer for pool in draft_pools for buffer in pool.kv_buffer]
    draft_indexer_buffers = [
        buffer for pool in draft_pools for buffer in pool.index_k_with_scale_buffer
    ]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
    layer_mapping = _with_packed_draft_mapping(
        {layer: layer for layer in range(num_layers)},
        target_device_layer_num=num_layers,
        draft_layer_num=len(draft_kv_buffers),
    )
    indexer_buffers, indexer_mapping = _drop_empty_buffers_and_remap(
        [*kvcache.index_k_with_scale_buffer, *draft_indexer_buffers],
        layer_mapping,
    )
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.kv_buffer, *draft_kv_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=False,
        ),
    ]
    if indexer_buffers:
        entries.append(
            DevicePoolEntry(
                name=PoolName.INDEXER,
                indices_from_pool=PoolName.KV,
                device_pool=kvcache,
                components=[indexer_buffers],
                layer_mapping=indexer_mapping,
                page_size=page_size,
                rows_are_pages=True,
            )
        )
    return DevicePoolGroup(entries, num_layers, page_size, rank_replicated=True)


def _direct_linker_kv_components(pool: Any) -> list[list[torch.Tensor]]:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    if isinstance(pool, MHATokenToKVPool):
        if getattr(pool, "kv_cache_layout", "nhd") != "nhd":
            raise NotImplementedError(
                "The direct external linker only supports NHD MHA draft pools."
            )
        if getattr(pool, "k_scale_buffer", None) is not None:
            raise NotImplementedError(
                "The direct external linker does not support quantized MHA "
                "draft pools yet."
            )
        return [list(pool.k_buffer), list(pool.v_buffer)]
    return [list(pool.kv_buffer)]


def _build_direct_linker_draft_sidecars(
    draft_device_pools: tuple[Any, ...], page_size: int
) -> tuple[list[DevicePoolEntry], int]:
    """Build direct-linker counterparts of HiCache draft sidecars."""
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool, HybridLinearKVPool

    if len(draft_device_pools) != 1:
        raise ValueError(
            "Direct-linker draft sidecars require exactly one draft pool, got "
            f"{len(draft_device_pools)}."
        )

    draft_pool = draft_device_pools[0]
    if isinstance(draft_pool, BaseSWAKVPool):
        pool = draft_pool.swa_kv_pool
        name, source = PoolName.DRAFT_SWA, PoolName.SWA
        rows_are_pages = hasattr(pool, "bytes_per_page_padded")
    else:
        pool = (
            draft_pool.full_kv_pool
            if isinstance(draft_pool, HybridLinearKVPool)
            else draft_pool
        )
        name, source, rows_are_pages = PoolName.DRAFT, PoolName.KV, False

    if pool.page_size != page_size:
        raise ValueError("Draft pool page size must match the tree page size.")
    if pool.layer_num == 0:
        return [], 0

    layer_mapping = {layer: layer for layer in range(pool.layer_num)}
    components = _direct_linker_kv_components(pool)
    entries = [
        DevicePoolEntry(
            name=name,
            indices_from_pool=source,
            device_pool=pool,
            components=components,
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=rows_are_pages,
            packed=len(components) == 1,
        )
    ]
    if isinstance(pool, DSATokenToKVPool):
        indexer_buffers, indexer_mapping = _drop_empty_buffers_and_remap(
            list(pool.index_k_with_scale_buffer), layer_mapping
        )
        if indexer_buffers:
            entries.append(
                DevicePoolEntry(
                    name=PoolName.DRAFT_INDEXER,
                    indices_from_pool=PoolName.KV,
                    device_pool=pool,
                    components=[indexer_buffers],
                    layer_mapping=indexer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )
    return entries, pool.layer_num


def resolve_hybrid_device_pool_group(
    *,
    kvcache: Any,
    page_size: int,
    params: Any,
    components: set[ComponentType],
) -> DevicePoolGroup:
    """Materialize a direct-linker pool group through the hybrid registry."""
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _select_strategy,
    )

    group = _select_strategy(kvcache, components).build_direct_linker_pool_group(
        kvcache=kvcache,
        params=params,
        page_size=page_size,
    )
    draft_sidecars = getattr(params, "direct_linker_draft_device_pools", ())
    if not draft_sidecars:
        return group

    entries, draft_layer_num = _build_direct_linker_draft_sidecars(
        draft_sidecars, page_size
    )
    return DevicePoolGroup(
        [*group.entries, *entries],
        max(group.num_layers, draft_layer_num),
        page_size,
        rank_replicated=group.rank_replicated,
    )
