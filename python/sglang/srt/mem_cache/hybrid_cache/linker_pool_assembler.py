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
from sglang.srt.runtime_context import get_memory, get_parallel


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
        shard: int | None = None,
    ):
        self.name = name
        self.indices_from_pool = indices_from_pool
        self.device_pool = device_pool
        self.components = [list(component) for component in components]
        self.layer_mapping = layer_mapping
        self.page_size = page_size
        self.packed = packed
        self._index_mapper = index_mapper
        # Ranks that declare the same shard hold identical objects, e.g. ranks
        # with the same DCP rank; None means every rank holds its own.
        self.shard = shard
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

        items = []
        for component, offsets in zip(self.buffer_meta, self._component_offsets):
            for buffer_index in buffer_indices:
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
        result[depth] = (layer_mapping[depth], target_device_layer_num + depth)
    return result


def _build_deepseek_v4_device_pool_group(
    kvcache: Any,
    page_size: int,
    mtp_draft_device_pools: tuple[Any, ...] = (),
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import HiSparseC4DevicePool
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _check_dsv4_indexer_page_size,
        _dsv4_low_ratio_regions,
        _dsv4_pool_infos,
        _dsv4_transfer_layer_mapping,
        _require_single_row_dsv4_swa_pages,
    )

    if isinstance(kvcache.c4_kv_pool, HiSparseC4DevicePool):
        raise ValueError("The direct external linker does not support HiSparse.")

    infos = _dsv4_pool_infos(kvcache)
    # Encoder SWA replay keeps the target window per request; the draft SWA is
    # dropped with it, a known draft accept-length cost.
    swa = infos.get(PoolName.SWA)
    entries = []
    if swa is not None:
        swa_info = swa.buffer_info
        _require_single_row_dsv4_swa_pages(
            logical_page_size=kvcache.swa_page_size,
            physical_page_size=swa_info.page_slots,
            consumer="DeepSeek-V4 direct external linker",
        )
        if kvcache.swa_page_size != page_size:
            raise ValueError(
                "DeepSeek V4 SWA page size must match the tree page size: "
                f"{kvcache.swa_page_size} != {page_size}."
            )
        draft_swa_buffers = []
        for pool in mtp_draft_device_pools:
            draft_swa = _dsv4_pool_infos(pool).get(PoolName.SWA)
            if draft_swa is None:
                raise ValueError("Packed DSV4 draft requires paged SWA storage")
            draft_swa_buffers.extend(draft_swa.buffer_info.buffers)
        swa_mapping = _with_packed_draft_mapping(
            _dsv4_transfer_layer_mapping(swa.layer_ids, kvcache.start_layer),
            target_device_layer_num=len(swa_info.buffers),
            draft_layer_num=len(draft_swa_buffers),
        )
        entries.append(
            DevicePoolEntry(
                name=PoolName.SWA,
                indices_from_pool=PoolName.SWA,
                device_pool=kvcache.swa_kv_pool,
                components=[[*swa_info.buffers, *draft_swa_buffers]],
                layer_mapping=swa_mapping,
                page_size=page_size,
                rows_are_pages=True,
            )
        )

    def add(name, source, pool, buffers, layer_mapping):
        if layer_mapping:
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=source,
                    device_pool=pool,
                    components=[list(buffers)],
                    layer_mapping=layer_mapping,
                    page_size=page_size,
                    rows_are_pages=True,
                )
            )

    def add_region(name, pool):
        info = infos.get(name)
        if info is not None:
            add(
                name,
                info.indices_from_pool,
                pool,
                info.buffer_info.buffers,
                _dsv4_transfer_layer_mapping(info.layer_ids, kvcache.start_layer),
            )

    add_region(PoolName.DEEPSEEK_V4_C4, kvcache.c4_kv_pool)
    if PoolName.DEEPSEEK_V4_C4_INDEXER in infos:
        _check_dsv4_indexer_page_size(
            infos[PoolName.DEEPSEEK_V4_C4_INDEXER].buffer_info, page_size
        )
    add_region(PoolName.DEEPSEEK_V4_C4_INDEXER, kvcache.c4_indexer_kv_pool)
    add_region(PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, kvcache.c4_indexer_kv_pool)
    add_region(PoolName.DEEPSEEK_V4_C128, kvcache.c128_kv_pool)
    for name, device_pool, buffers, layer_mapping in _dsv4_low_ratio_regions(
        kvcache, infos, page_size
    ):
        add(name, PoolName.KV, device_pool, buffers, layer_mapping)

    # fp8 two-pool unified_kv: one row index addresses both kv_buffer (fp8 nope) and
    # kv_buffer_rope (bf16 rope); shipping nope alone restores 512 of every 640
    # bytes and leaves stale rope. Non-fp8 layouts export no rope region.
    add_region(PoolName.DEEPSEEK_V4_C4_ROPE, kvcache.c4_kv_pool)
    add_region(PoolName.DEEPSEEK_V4_C128_ROPE, kvcache.c128_kv_pool)
    for name in (
        PoolName.DEEPSEEK_V4_C4_STATE,
        PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
    ):
        state = infos.get(name)
        if state is None:
            continue
        pool = (
            kvcache.compress_state_pools
            if name == PoolName.DEEPSEEK_V4_C4_STATE
            else kvcache.indexer_compress_state_pools
        )
        add(
            name,
            state.indices_from_pool,
            pool,
            [info.pages for info in state.buffer_info.states],
            _dsv4_transfer_layer_mapping(state.layer_ids, kvcache.start_layer),
        )
    if swa is None and any(
        entry.indices_from_pool == PoolName.SWA for entry in entries
    ):
        raise ValueError(
            "DeepSeek-V4 compress state pools take SWA page indices, which "
            "encoder SWA replay does not keep."
        )
    return DevicePoolGroup(
        entries,
        len(kvcache.stage_layer_ids),
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
    if any(pool.page_size != page_size for pool in mtp_draft_device_pools):
        raise ValueError("DSA MTP page size must match the tree page size.")
    draft_kv_buffers = [
        buffer for pool in mtp_draft_device_pools for buffer in pool.kv_buffer
    ]
    draft_indexer_buffers = [
        buffer
        for pool in mtp_draft_device_pools
        for buffer in pool.index_k_with_scale_buffer
    ]
    if len(draft_kv_buffers) != len(draft_indexer_buffers):
        raise ValueError("DSA MTP KV and indexer draft layer counts must match.")
    layer_mapping = _with_packed_draft_mapping(
        {layer: layer for layer in range(num_layers)},
        target_device_layer_num=num_layers,
        draft_layer_num=len(draft_kv_buffers),
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
        DevicePoolEntry(
            name=PoolName.INDEXER,
            indices_from_pool=PoolName.KV,
            device_pool=kvcache,
            components=[[*kvcache.index_k_with_scale_buffer, *draft_indexer_buffers]],
            layer_mapping=layer_mapping,
            page_size=page_size,
            rows_are_pages=True,
        ),
    ]
    return DevicePoolGroup(entries, num_layers, page_size, rank_replicated=True)


def _mha_device_pool_entry(
    name: PoolName,
    pool: Any,
    layer_mapping: dict[int, int],
    page_size: int,
    indices_from_pool: PoolName | None = None,
) -> DevicePoolEntry:
    # TODO(alphabetc1): support mxfp8.
    if pool.k_scale_buffer is not None or pool.native_k_scale_buffer is not None:
        raise ValueError(
            "The direct external linker does not support scaled KV caches yet."
        )
    num_layers = len(pool.k_buffer)
    return DevicePoolEntry(
        name=name,
        indices_from_pool=indices_from_pool or name,
        device_pool=pool,
        components=[[*pool.k_buffer, *pool.v_buffer]],
        layer_mapping={
            layer: (index, num_layers + index) for layer, index in layer_mapping.items()
        },
        page_size=page_size,
        rows_are_pages=False,
    )


def _mamba_device_pool_entry(state: Any, start_layer: int) -> DevicePoolEntry:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _stage_local_layer_mapping,
    )

    # TODO(alphabetc1): support unified memory for hybrid Mamba models.
    if get_memory().enable_unified_memory:
        raise ValueError(
            "The direct external linker does not support --enable-unified-memory "
            "on hybrid Mamba models yet."
        )
    # TODO(alphabetc1): support int8 Mamba checkpoints and Mamba side states.
    if (
        state.ckpt_pool is not None
        or state.short_conv_pool.enabled
        or state.ngram_pool.enabled
    ):
        raise ValueError(
            "The direct external linker does not support int8 Mamba checkpoints "
            "or Mamba side states yet."
        )

    mamba_pool = state.pool
    mamba_cache = mamba_pool.mamba_cache
    layers = range(mamba_pool.num_mamba_layers)
    # One row per state slot; conv-only models carry an empty temporal state.
    components = [
        [state[layer] for layer in layers]
        for state in (mamba_cache.temporal, *mamba_cache.conv)
        if state.numel() > 0
    ]
    return DevicePoolEntry(
        name=PoolName.MAMBA,
        indices_from_pool=PoolName.MAMBA,
        device_pool=mamba_pool,
        components=components,
        layer_mapping=_stage_local_layer_mapping(state.layer_map, start_layer),
        page_size=1,
        rows_are_pages=True,
    )


def _build_mamba_swa_device_pool_group(
    kvcache: Any, params: Any, page_size: int
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _swa_layer_mappings,
    )

    full_mapping, swa_mapping = _swa_layer_mappings(kvcache)
    entries = [
        _mha_device_pool_entry(
            PoolName.KV, kvcache.full_kv_pool, full_mapping, page_size
        ),
        _mha_device_pool_entry(
            PoolName.SWA, kvcache.swa_kv_pool, swa_mapping, page_size
        ),
        _mamba_device_pool_entry(
            params.token_to_kv_pool_allocator.state, kvcache.start_layer
        ),
    ]
    num_layers = max(layer for entry in entries for layer in entry.layer_mapping)
    return DevicePoolGroup(entries, num_layers + 1, page_size)


def _build_mamba_device_pool_group(
    kvcache: Any, params: Any, page_size: int
) -> DevicePoolGroup:
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        _stage_local_layer_mapping,
    )
    from sglang.srt.mem_cache.memory_pool import (
        HybridLinearKVPool,
        MLATokenToKVPool,
    )

    # TODO(alphabetc1): support other full-attention pools, DCP and packed draft KV.
    full_kv_pool = kvcache.full_kv_pool
    if (
        type(kvcache) is not HybridLinearKVPool
        or type(full_kv_pool) is not MLATokenToKVPool
        or get_parallel().attn_dcp_size > 1
        or params.mtp_draft_device_pools
    ):
        raise ValueError(
            "The direct external linker supports FULL + MAMBA models only with a "
            "plain MLA pool, without DCP or packed draft KV."
        )

    entries = [
        DevicePoolEntry(
            name=PoolName.KV,
            indices_from_pool=PoolName.KV,
            device_pool=full_kv_pool,
            components=[full_kv_pool.kv_buffer],
            layer_mapping=_stage_local_layer_mapping(
                kvcache.full_attention_layer_id_mapping, kvcache.start_layer
            ),
            page_size=page_size,
            rows_are_pages=False,
            # The MLA KV is identical on every attention TP rank.
            shard=0,
        ),
        _mamba_device_pool_entry(
            params.token_to_kv_pool_allocator.state, kvcache.start_layer
        ),
    ]
    for draft_pool in params.sidecar_draft_device_pools:
        if draft_pool.page_size != page_size:
            raise ValueError("Draft KV page size must match the tree page size.")
        # Draft layer i rides transfer layer i; the draft reads it only after
        # the batch's loads finish.
        entries.append(
            _mha_device_pool_entry(
                PoolName.DRAFT,
                draft_pool,
                {layer: layer for layer in range(draft_pool.layer_num)},
                page_size,
                indices_from_pool=PoolName.KV,
            )
        )
    num_layers = max(layer for entry in entries for layer in entry.layer_mapping)
    # Not rank_replicated: the Mamba state is TP-sharded.
    return DevicePoolGroup(entries, num_layers + 1, page_size)


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
    if getattr(params, "sidecar_draft_device_pools", ()) and (
        PoolName.DRAFT not in group.entry_map
    ):
        raise ValueError(
            "The direct external linker does not support this model's draft KV."
        )
    return group
