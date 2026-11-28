# SPDX-License-Identifier: Apache-2.0
"""Build unified L3 object identities for direct-linker device pools."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.srt.mem_cache.hicache_key_scheme import (
    normalize_dtype,
    plan_unified_kv,
    storage_model_id,
)
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolEntry
from sglang.srt.mem_cache.kv.memory.views import MHAKVView, MLAKVView
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.runtime_context import get_memory, get_parallel, get_spec

if TYPE_CHECKING:
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import DevicePoolGroup


@dataclass(frozen=True)
class LinkerUnifiedLayout:
    model_name: str
    suffixes: dict[PoolName, list[str]]
    entries: dict[PoolName, DevicePoolEntry]


def _canonical_entry(
    entry: DevicePoolEntry, num_layers: int
) -> tuple[DevicePoolEntry, int] | None:
    """Return the exact page-unified byte order for a dense primary KV pool."""
    pool = entry.device_pool
    if (
        entry.rows_are_pages
        or not entry.packed
        or getattr(pool, "layer_num", None) != num_layers
        or sorted(entry.layer_mapping) != list(range(num_layers))
    ):
        return None
    if isinstance(pool, MLAKVView):
        if (
            len(entry.kv_buffer) != num_layers
            or getattr(pool, "layer_shard_enabled", False)
            or getattr(pool, "kv_scale_buffer", None) is not None
            or any(entry.layer_mapping[i] != i for i in range(num_layers))
            or any(
                buffer.shape[1:] != (1, pool.kv_cache_dim)
                or buffer.dtype != pool.store_dtype
                or not buffer.is_contiguous()
                for buffer in entry.kv_buffer
            )
        ):
            return None
        return entry, 0
    if isinstance(pool, MHAKVView):
        if (
            len(entry.kv_buffer) != 2 * num_layers
            or pool.head_dim != pool.v_head_dim
            or any(
                buffer.shape[1:] != (pool.head_num, pool.head_dim)
                or buffer.dtype != pool.store_dtype
                or not buffer.is_contiguous()
                for buffer in entry.kv_buffer
            )
        ):
            return None
        pairs = list(zip(pool.k_buffer, pool.v_buffer, strict=True))
        if len(pairs) != num_layers:
            return None
        for layer, (key, value) in enumerate(pairs):
            mapped = entry.layer_mapping[layer]
            if (
                isinstance(mapped, int)
                or len(mapped) != 2
                or entry.kv_buffer[mapped[0]] is not key
                or entry.kv_buffer[mapped[1]] is not value
            ):
                return None
        return (
            DevicePoolEntry(
                name=entry.name,
                indices_from_pool=entry.indices_from_pool,
                device_pool=pool,
                components=[[buffer for pair in pairs for buffer in pair]],
                layer_mapping={i: (2 * i, 2 * i + 1) for i in range(num_layers)},
                page_size=entry.page_size,
                rows_are_pages=False,
                tp_replicated=entry.tp_replicated,
            ),
            pool.head_num,
        )
    return None


def _raw_layout_identity(entry: DevicePoolEntry, draft_identity: tuple | None) -> str:
    """Identify raw bytes by buffer order, geometry, and layer assignments."""
    geometry = (
        str(entry.name),
        str(entry.indices_from_pool),
        type(entry.device_pool).__qualname__,
        str(getattr(entry.device_pool, "kv_layout", "")),
        str(getattr(entry.device_pool, "kv_cache_layout", "")),
        entry.packed,
        entry.rows_are_pages,
        [
            [
                (str(buffer.dtype), buffer.shape[1:], buffer.stride()[1:])
                for buffer in group
            ]
            for group in entry.components
        ],
        sorted(entry.layer_mapping.items()),
        draft_identity,
    )
    digest = hashlib.sha256(msgspec.msgpack.encode(geometry)).hexdigest()
    return f"linker-device-v1-{digest}"


def _head_group_entry(entry: DevicePoolEntry, head_group_num: int) -> DevicePoolEntry:
    """Expose each token's contiguous head slice as a page-strided buffer."""
    page_size = entry.page_size
    buffers = entry.components[0]
    num_heads = buffers[0].size(1)
    heads_per_group = num_heads // head_group_num
    # Each object is ordered by layer, K/V, then token. Selecting a token's
    # position within every page keeps these views affine in the page index.
    components = [
        [
            buffer[token::page_size, head : head + heads_per_group, :]
            for buffer in buffers
            for token in range(page_size)
        ]
        for head in range(0, num_heads, heads_per_group)
    ]
    ranges_per_layer = 2 * page_size
    return DevicePoolEntry(
        name=entry.name,
        indices_from_pool=entry.indices_from_pool,
        device_pool=entry.device_pool,
        components=components,
        layer_mapping={
            layer: tuple(
                range(layer * ranges_per_layer, (layer + 1) * ranges_per_layer)
            )
            for layer in entry.layer_mapping
        },
        page_size=page_size,
        rows_are_pages=True,
        packed=False,
        tp_replicated=entry.tp_replicated,
    )


def build_linker_unified_layout(
    kvcache,
    pool_group: DevicePoolGroup,
    *,
    model_name: str,
    params: CacheInitParams,
) -> LinkerUnifiedLayout:
    """Match HiCache keys for canonical KV and isolate all other byte formats."""
    memory = get_memory()
    parallel = get_parallel()
    if params.attn_cp_size > 1 or parallel.attn_dcp_size > 1:
        raise ValueError("Unified linker keys require attention CP and DCP sizes of 1.")
    if memory.hicache_storage_layer_partition is not None:
        raise ValueError("Unified linker keys do not support layer partitioning.")
    if not model_name:
        raise ValueError("Unified linker keys require a model name.")

    entries = dict(pool_group.entry_map)
    primary = entries.get(PoolName.KV)
    pool_dtype = (
        getattr(primary.device_pool, "dtype", kvcache.dtype)
        if primary is not None
        else kvcache.dtype
    )
    if not isinstance(pool_dtype, torch.dtype):
        raise ValueError(
            f"Unified linker keys require a resolved logical KV dtype, got {pool_dtype!r}."
        )
    dtype = normalize_dtype(pool_dtype)
    model_name = storage_model_id(model_name, dtype)
    start_layer = getattr(kvcache, "start_layer", 0)
    parallel_geometry = dict(
        attn_tp_rank=parallel.attn_tp_rank,
        attn_tp_size=parallel.attn_tp_size,
        attn_cp_size=params.attn_cp_size,
        start_layer=start_layer,
        end_layer=start_layer + pool_group.num_layers,
        is_final_stage=params.pp_rank == params.pp_size - 1,
    )
    canonical = (
        _canonical_entry(primary, pool_group.num_layers)
        if primary is not None
        else None
    )
    suffixes = {}
    if canonical is not None:
        entry, local_heads = canonical
        plan = plan_unified_kv(
            model_id=model_name,
            dtype=dtype,
            page_size=entry.page_size,
            rank_replicated=local_heads == 0,
            local_kv_heads=local_heads,
            head_group=memory.hicache_storage_head_partition,
            **parallel_geometry,
        )
        # Use the same physical constraints as the matching host layout.
        itemsize = entry.kv_buffer[0].element_size()
        PageUnifiedLayout(
            page_size=entry.page_size,
            layer_num=pool_group.num_layers,
            head_num=local_heads or 1,
            head_group_num=plan.head_group_num,
            head_dim=entry.kv_buffer[0].shape[-1],
            itemsize=itemsize,
            is_mla=local_heads == 0,
        )
        entries[PoolName.KV] = (
            _head_group_entry(entry, plan.head_group_num)
            if plan.head_group_num > 1
            else entry
        )
        suffixes[PoolName.KV] = plan.suffixes
    elif (
        memory.hicache_storage_head_partition is not None
        and not pool_group.rank_replicated
        and (primary is None or not primary.tp_replicated)
    ):
        raise ValueError(
            "--hicache-storage-head-partition requires a dense page-unified KV "
            "pool in the direct linker."
        )

    draft_identity = None
    if (
        params.mtp_draft_device_pools
        or params.sidecar_draft_device_pools
        or params.mtp_draft_mamba_pools
    ):
        spec = get_spec()
        draft_identity = (
            spec.speculative_algorithm,
            spec.speculative_draft_model_path,
            spec.speculative_draft_model_revision,
            spec.speculative_draft_model_quantization,
            spec.speculative_draft_kv_cache_dtype,
            spec.speculative_draft_window_size,
            spec.speculative_draft_sink_size,
        )
    for name, entry in entries.items():
        if name in suffixes:
            continue
        if not entry.packed:
            raise ValueError(f"Unified linker keys require a packed entry for {name}.")
        if any(
            not buffer[0].is_contiguous()
            or (
                not entry.rows_are_pages
                and buffer.stride(0) * buffer.element_size()
                != buffer.nbytes // buffer.shape[0]
            )
            for buffer in entry.kv_buffer
        ):
            raise ValueError(f"Unified linker keys require dense page rows for {name}.")
        entry_dtype = getattr(entry.device_pool, "dtype", None)
        if not isinstance(entry_dtype, torch.dtype):
            entry_dtype = entry.kv_buffer[0].dtype
        plan = plan_unified_kv(
            model_id=model_name,
            dtype=normalize_dtype(entry_dtype),
            page_size=entry.page_size,
            rank_replicated=pool_group.rank_replicated or entry.tp_replicated,
            # Raw layouts keep the entire local shard in one object. One dummy
            # head per rank makes the namespace specific to the attention TP size.
            local_kv_heads=1,
            head_group=1,
            object_layout=_raw_layout_identity(entry, draft_identity),
            **parallel_geometry,
        )
        suffixes[name] = plan.suffixes
    return LinkerUnifiedLayout(model_name, suffixes, entries)
