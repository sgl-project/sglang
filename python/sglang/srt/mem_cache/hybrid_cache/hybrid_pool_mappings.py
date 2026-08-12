from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, NamedTuple

import torch

from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)


class DeepSeekV4PoolMappings(NamedTuple):
    num_layers: int
    full: dict[int, int]
    swa: dict[int, int]
    c4: dict[int, int]
    c128: dict[int, int]
    c4_state: dict[int, int]
    c4_state_global_layers: list[int]


def resolve_deepseek_v4_pool_mappings(kvcache: Any) -> DeepSeekV4PoolMappings:
    num_layers = kvcache.end_layer - kvcache.start_layer
    full = {layer: layer for layer in range(num_layers)}
    swa = {} if getattr(kvcache, "_unified_kv", False) else full.copy()

    c4 = {}
    c128 = {}
    c4_state_global_layers = []
    for local_layer, item in enumerate(
        kvcache.layer_mapping[kvcache.start_layer : kvcache.end_layer]
    ):
        if item.compress_ratio == 4:
            c4[local_layer] = item.compress_layer_id
            c4_state_global_layers.append(kvcache.start_layer + local_layer)
        elif item.compress_ratio == 128:
            c128[local_layer] = item.compress_layer_id

    c4_state = {layer: index for index, layer in enumerate(c4)}
    return DeepSeekV4PoolMappings(
        num_layers, full, swa, c4, c128, c4_state, c4_state_global_layers
    )


class HybridLinearPoolMappings(NamedTuple):
    full: dict[int, int]
    mamba: dict[int, int]
    start_layer: int
    num_layers: int
    local_full: dict[int, int]
    local_mamba: dict[int, int]


def resolve_hybrid_linear_pool_mappings(
    kvcache: Any, req_to_token_pool: Any
) -> HybridLinearPoolMappings:
    full = dict(kvcache.full_attention_layer_id_mapping)
    mamba = dict(req_to_token_pool.mamba_map)
    layer_ids = set(full) | set(mamba)
    start_layer = min(layer_ids)
    num_layers = max(layer_ids) - start_layer + 1
    return HybridLinearPoolMappings(
        full=full,
        mamba=mamba,
        start_layer=start_layer,
        num_layers=num_layers,
        local_full={layer - start_layer: index for layer, index in full.items()},
        local_mamba={layer - start_layer: index for layer, index in mamba.items()},
    )


class DevicePoolEntry:
    """Storage view over one real device pool, analogous to a host PoolEntry."""

    def __init__(
        self,
        *,
        name: PoolName,
        indices_from_pool: PoolName,
        device_pool: Any,
        components: Sequence[Sequence[torch.Tensor]],
        layer_mapping: dict[int, int],
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
        buffer_index = self.layer_mapping.get(layer)
        if buffer_index is None:
            return None

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            base_ptr, row_stride, size = component[buffer_index]
            items.append((base_ptr, row_stride, size, offsets[buffer_index]))

        ptrs, sizes, offsets = [], [], []
        for row in locations:
            row_ptrs = [
                base_ptr + row * row_stride for base_ptr, row_stride, _, _ in items
            ]
            row_sizes = [size for _, _, size, _ in items]
            row_offsets = [offset for _, _, _, offset in items]
            if self.packed:
                ptrs.append(row_ptrs)
                sizes.append(row_sizes)
                offsets.append(row_offsets)
            else:
                ptrs.extend([[value] for value in row_ptrs])
                sizes.extend([[value] for value in row_sizes])
                offsets.extend([[value] for value in row_offsets])
        return ptrs, sizes, offsets


class DevicePoolGroup:
    """A group of physical device pools sharing one logical layer range."""

    def __init__(
        self, entries: Sequence[DevicePoolEntry], num_layers: int, page_size: int
    ):
        self.entries = list(entries)
        self.entry_map = {entry.name: entry for entry in entries}
        if len(self.entries) != len(self.entry_map):
            raise ValueError("DevicePoolGroup contains duplicate pool names.")
        self.sources = {entry.name: entry.indices_from_pool for entry in self.entries}
        self.num_layers = num_layers
        self.page_size = page_size
        self.kv_buffer = None

    def resolve_transfers(
        self, transfers: list[PoolTransfer], *, allow_partial: bool = False
    ) -> list[PoolTransfer]:
        """Map logical cache transfers to Mooncake's physical device pools.

        Cache components emit logical KV/SWA/MAMBA transfers, while Mooncake
        operates on every registered DevicePoolEntry. For example, DeepSeek V4
        C4/C128 pools reuse the KV transfer's page keys and indices, and its
        state pools reuse the SWA transfer. This method creates one transfer per
        physical pool and translates the source indices into that pool's rows.

        Lookup and load require every source pool. Offload passes
        ``allow_partial=True`` because a cache node may contain only some pools.
        """
        by_name = {transfer.name: transfer for transfer in transfers}
        kv = by_name.get(PoolName.KV)
        if kv is None or not kv.keys:
            return []
        if not allow_partial and not set(self.sources.values()) <= set(by_name):
            return []

        resolved = []
        for name, source_name in self.sources.items():
            source = by_name.get(source_name)
            if source is None:
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
                    keys=list(kv.keys if source_name == PoolName.KV else source.keys),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if source_name == PoolName.KV
                        else source.hit_policy
                    ),
                    indices_from_pool=None,
                )
            )
        return resolved


def _state_views(state_pools: Sequence[Any], global_layers: Sequence[int]):
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


def resolve_hybrid_device_pool_group(
    kvcache: Any, page_size: int, req_to_token_pool: Any
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4TokenToKVPool,
        HiSparseC4DevicePool,
    )
    from sglang.srt.mem_cache.memory_pool import (
        DSATokenToKVPool,
        HybridLinearKVPool,
    )

    if isinstance(kvcache, DeepSeekV4TokenToKVPool):
        if kvcache._unified_kv or isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
            raise ValueError(
                "Mooncake direct linker does not support unified-KV or HiSparse."
            )
        if kvcache.swa_page_size != page_size:
            raise ValueError(
                "DeepSeek V4 SWA page size must match the tree page size: "
                f"{kvcache.swa_page_size} != {page_size}."
            )

        mappings = resolve_deepseek_v4_pool_mappings(kvcache)
        entries = [
            DevicePoolEntry(
                name=PoolName.SWA,
                indices_from_pool=PoolName.SWA,
                device_pool=kvcache.swa_kv_pool,
                components=[kvcache.swa_kv_pool.kv_buffer],
                layer_mapping=mappings.swa,
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
        add(
            PoolName.DEEPSEEK_V4_C4_INDEXER,
            PoolName.KV,
            kvcache.c4_indexer_kv_pool,
            kvcache.c4_indexer_kv_pool.index_k_with_scale_buffer,
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
            _state_views(kvcache.compress_state_pools, mappings.c4_state_global_layers),
            mappings.c4_state,
        )
        add(
            PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
            PoolName.SWA,
            kvcache.indexer_compress_state_pools,
            _state_views(
                kvcache.indexer_compress_state_pools,
                mappings.c4_state_global_layers,
            ),
            mappings.c4_state,
        )
        return DevicePoolGroup(entries, mappings.num_layers, page_size)

    if isinstance(kvcache, HybridLinearKVPool):
        if getattr(req_to_token_pool, "mamba_ckpt_pool", None) is not None:
            raise ValueError(
                "Mooncake direct linker does not support int8 Mamba checkpoint storage."
            )
        mappings = resolve_hybrid_linear_pool_mappings(kvcache, req_to_token_pool)
        full_pool = kvcache.full_kv_pool
        if kvcache.use_mla or getattr(full_pool, "k_scale_buffer", None) is not None:
            raise ValueError(
                "Mooncake direct linker requires unquantized Qwen3.5 MHA KV."
            )
        kv = DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=full_pool,
            components=[full_pool.k_buffer, full_pool.v_buffer],
            layer_mapping=mappings.local_full,
            page_size=page_size,
            rows_are_pages=full_pool.k_buffer[0].shape[0] < full_pool.size + page_size,
        )

        state_pool = req_to_token_pool.mamba_pool
        state = state_pool.mamba_cache
        temporal_size = state.temporal[0, 0].numel() if state.temporal.numel() else 0
        state_tensors = ([state.temporal] if temporal_size else []) + list(state.conv)
        mamba = DevicePoolEntry(
            name=PoolName.MAMBA,
            indices_from_pool=PoolName.MAMBA,
            device_pool=state_pool,
            components=[list(tensor) for tensor in state_tensors],
            layer_mapping=mappings.local_mamba,
            page_size=1,
            rows_are_pages=True,
            packed=False,
            index_mapper=req_to_token_pool.translate_mamba_indices,
        )
        mamba.temporal_state_elem_size = temporal_size
        mamba.conv_buffer = list(state.conv)
        return DevicePoolGroup([kv, mamba], mappings.num_layers, page_size)

    if not isinstance(kvcache, DSATokenToKVPool):
        raise TypeError(
            "Mooncake direct linker supports DSA, DeepSeek V4, and hybrid linear "
            "KV pools."
        )
    if kvcache.page_size != page_size:
        raise ValueError(
            "DSA KV page size must match the tree page size: "
            f"{kvcache.page_size} != {page_size}."
        )

    num_layers = len(kvcache.kv_buffer)
    identity = {layer: layer for layer in range(num_layers)}
    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[kvcache.kv_buffer],
            layer_mapping=identity,
            page_size=page_size,
            rows_are_pages=False,
        ),
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[kvcache.index_k_with_scale_buffer],
            layer_mapping=identity,
            page_size=page_size,
            rows_are_pages=True,
        ),
    ]
    return DevicePoolGroup(entries, num_layers, page_size)
