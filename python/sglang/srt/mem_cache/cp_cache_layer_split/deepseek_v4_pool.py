"""DeepSeek V4 token-to-KV pool with CP layer-sharded KV buffers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.mem_cache.cp_cache_layer_split.broadcast import (
    BroadcastSlots,
    get_pynccl_broadcast_comm,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_layout import (
    build_cp_cache_layer_split_deepseek_v4_pool_layout,
)
from sglang.srt.mem_cache.cp_cache_layer_split.pool_base import (
    CpCacheLayerSplitPoolBase,
)
from sglang.srt.mem_cache.cp_cache_layer_split.staging import (
    StagingBufferManager,
    active_pages_for_indices,
    remap_indices_to_staging,
    remap_page_table_to_staging,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_ATTENTION_STATE,
    DSV4_TRANSFER_C4_INDEXER_KV,
    DSV4_TRANSFER_C4_KV,
    DSV4_TRANSFER_C128_KV,
    DSV4_TRANSFER_C128_STATE,
    DSV4_TRANSFER_INDEXER_STATE,
    DSV4_TRANSFER_SWA_KV,
    DeepSeekV4LayerItem,
    DeepSeekV4TokenToKVPool,
    NopeFp8RopeBf16Pack,
)
from sglang.srt.utils import ceil_div

logger = logging.getLogger(__name__)


@dataclass
class _BatchActivePages:
    """Per-forward active pages plus remapped indices/page table."""

    selected_pages: Optional[torch.Tensor] = None
    remapped: Optional[torch.Tensor] = None


class CpCacheLayerSplitDeepSeekV4TokenToKVPool(
    CpCacheLayerSplitPoolBase, DeepSeekV4TokenToKVPool
):
    """DeepSeek V4 pool that reads non-owned layer KV through staging buffers."""

    requires_descriptor_matched_transfer = True

    def __init__(
        self,
        *,
        cp_rank: int,
        cp_size: int,
        **kwargs,
    ):
        staging_context_len = kwargs.pop(
            "cp_cache_layer_split_staging_context_len", None
        )
        staging_chunked_prefill_size = kwargs.pop(
            "cp_cache_layer_split_staging_chunked_prefill_size", None
        )
        staging_max_prefill_tokens = kwargs.pop(
            "cp_cache_layer_split_staging_max_prefill_tokens", None
        )

        self._clear_swa_read_state()
        self._clear_extra_read_state()
        self._clear_indexer_read_state()

        # SWA, C128, and page-table-driven C4/indexer selections are constant
        # within one forward. Sparse C4 indices remain layer-specific.
        self._batch_active_pages: dict[str, _BatchActivePages] = {
            "swa": _BatchActivePages(),
            "indexer": _BatchActivePages(),
            "extra_c4_page_table": _BatchActivePages(),
            "extra_c128": _BatchActivePages(),
        }

        pp_start = kwargs.get("start_layer")
        if pp_start is None:
            pp_start = 0
        pp_end = kwargs.get("end_layer")
        layer_num = kwargs["layer_num"]
        if pp_end is None:
            pp_end = pp_start + layer_num

        layout = build_cp_cache_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            pp_start,
            pp_end,
            kwargs["compression_ratios"],
        )
        if layout.swa_layer_num == 0:
            logger.warning(
                "CpCacheLayerSplitDeepSeekV4TokenToKVPool: cp_rank=%s owns no SWA layers "
                "in PP slice [%s, %s)",
                cp_rank,
                pp_start,
                pp_end,
            )

        self._staging_context_len = staging_context_len
        self._staging_chunked_prefill_size = staging_chunked_prefill_size
        self._staging_max_prefill_tokens = staging_max_prefill_tokens
        self._staging = StagingBufferManager()
        self._broadcast_slots = BroadcastSlots(("swa", "extra", "indexer"))

        self._init_cp_cache_layer_split(
            cp_rank=cp_rank,
            cp_size=cp_size,
            layer_shard_start_layer=pp_start,
            layer_shard_layer_num=pp_end - pp_start,
        )
        super().__init__(
            **kwargs,
            cp_cache_layer_split_layout=layout,
        )
        self._swa_global_to_local = self._build_owned_layer_local_index_map()
        self._log_layer_shard_plan()
        logger.info(
            "DSV4 Cache LayerSplit buffers: SWA=%s, C4=%s, C128=%s, "
            "C4_INDEXER=%s, C4_STATE=%s, C128_STATE=%s, C4_INDEXER_STATE=%s",
            layout.swa_layer_num,
            layout.c4_layer_num,
            layout.c128_layer_num,
            layout.c4_indexer_layer_num,
            layout.c4_state_layer_num,
            layout.c128_state_layer_num,
            layout.c4_indexer_state_layer_num,
        )
        self._rebuild_compressed_layer_mapping_for_cp()

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        total_L = len(self.compression_ratios)
        self.compress_state_pools: list[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: list[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            if self._owns_attention_state_layer_id(idx):
                self.compress_state_pools[idx] = self._make_attn_state_pool(
                    ratio, enable_memory_saver
                )

            if ratio == 4 and self._owns_indexer_state_layer_id(idx):
                self.indexer_compress_state_pools[idx] = self._make_indexer_state_pool(
                    ratio, enable_memory_saver
                )

    def get_kv_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's sharded V4 KV buffer list."""
        layout: list = []
        stage_layers = range(self._stage_start, self._stage_end)

        layout.extend(
            (DSV4_TRANSFER_C4_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_c4_kv_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_C4_INDEXER_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_kv_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_C128_KV, layer_id)
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 128
            and self._owns_c128_kv_layer_id(layer_id)
        )
        return layout

    def get_state_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's sharded V4 state buffer list."""
        layout: list = []
        swa_layers = sorted(
            self._swa_global_to_local,
            key=lambda layer_id: self._swa_global_to_local[layer_id],
        )
        swa_layers = swa_layers[: len(self.swa_kv_pool.kv_buffer)]
        layout.extend((DSV4_TRANSFER_SWA_KV, layer_id) for layer_id in swa_layers)

        layout.extend(
            (DSV4_TRANSFER_ATTENTION_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 4
            and self._owns_attention_state_layer_id(layer_id)
        )
        layout.extend(
            (DSV4_TRANSFER_INDEXER_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_state_layer_id(layer_id)
        )
        return layout

    def get_c128_state_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's C128 state buffer list."""
        return [
            (DSV4_TRANSFER_C128_STATE, layer_id)
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 128
            and self._owns_attention_state_layer_id(layer_id)
        ]

    def get_hicache_host_layer_mapping(self) -> dict[str, dict[int, int]]:
        """Per-family ``{pp_local_layer_id -> local_buffer_index}`` for HiCache."""
        pp_start = self._stage_start

        swa_layers_sorted = sorted(
            self._swa_global_to_local.items(), key=lambda kv: kv[1]
        )
        swa_layers_kept = swa_layers_sorted[: len(self.swa_kv_pool.kv_buffer)]
        swa_map: dict[int, int] = {
            gid - pp_start: local_id for gid, local_id in swa_layers_kept
        }

        c4_kv_map: dict[int, int] = {}
        c128_kv_map: dict[int, int] = {}
        for gidx in range(pp_start, self._stage_end):
            item = self.layer_mapping[gidx]
            if item is None or item.compress_layer_id is None:
                continue
            pp_local = gidx - pp_start
            if item.compress_ratio == 4 and self._owns_c4_kv_layer_id(gidx):
                c4_kv_map[pp_local] = item.compress_layer_id
            elif item.compress_ratio == 128 and self._owns_c128_kv_layer_id(gidx):
                c128_kv_map[pp_local] = item.compress_layer_id

        c4_indexer_map: dict[int, int] = {}
        c4_indexer_local = 0
        for gidx in range(pp_start, self._stage_end):
            if self.compression_ratios[gidx] != 4:
                continue
            if not self._owns_indexer_kv_layer_id(gidx):
                continue
            c4_indexer_map[gidx - pp_start] = c4_indexer_local
            c4_indexer_local += 1

        c4_state_map: dict[int, int] = {}
        c128_state_map: dict[int, int] = {}
        c4_indexer_state_map: dict[int, int] = {}
        c4s_local = c128s_local = c4is_local = 0
        for gidx in range(pp_start, self._stage_end):
            ratio = self.compression_ratios[gidx]
            pp_local = gidx - pp_start
            if ratio == 4 and self._owns_attention_state_layer_id(gidx):
                c4_state_map[pp_local] = c4s_local
                c4s_local += 1
            elif ratio == 128 and self._owns_attention_state_layer_id(gidx):
                c128_state_map[pp_local] = c128s_local
                c128s_local += 1
            if ratio == 4 and self._owns_indexer_state_layer_id(gidx):
                c4_indexer_state_map[pp_local] = c4is_local
                c4is_local += 1

        return {
            "swa": swa_map,
            "c4_kv": c4_kv_map,
            "c128_kv": c128_kv_map,
            "c4_indexer": c4_indexer_map,
            "c4_state": c4_state_map,
            "c128_state": c128_state_map,
            "c4_indexer_state": c4_indexer_state_map,
        }

    def _owns_swa_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def should_skip_swa_write(self, layer_id: int) -> bool:
        return not self._owns_swa_layer_id(layer_id)

    def _owns_c4_kv_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def _owns_c128_kv_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def _owns_core_layer_id(self, layer_id: int) -> bool:
        if self.compression_ratios[layer_id] not in (4, 128):
            return False
        return self._is_layer_owned(layer_id)

    def _owns_extra_key_layer_id(self, layer_id: int) -> bool:
        return self._owns_core_layer_id(layer_id)

    def _owns_indexer_kv_layer_id(self, layer_id: int) -> bool:
        return self._is_layer_owned(layer_id)

    def _owns_attention_state_layer_id(self, layer_id: int) -> bool:
        return self._owns_core_layer_id(layer_id)

    def _owns_indexer_state_layer_id(self, layer_id: int) -> bool:
        return self._owns_indexer_kv_layer_id(layer_id)

    def should_skip_core_compressor_write(self, layer_id: int) -> bool:
        return not self._is_layer_owned(layer_id)

    def should_skip_indexer_compressor_write(self, layer_id: int) -> bool:
        return not self._is_layer_owned(layer_id)

    def should_use_c4_extra_broadcast_overlap(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        return item is not None and item.compress_ratio == 4

    @staticmethod
    def _pool_num_pages(pool) -> int:
        # Paged KV allocators reserve page 0 for dummy/padded tokens and allocate
        # real cache pages from 1..size//page_size.
        return pool.size // pool.page_size + 1

    def _active_pages_for_indices(
        self,
        indices: torch.Tensor,
        page_size: int,
        max_pages: int,
    ) -> torch.Tensor:
        return active_pages_for_indices(
            indices, page_size, max_pages, get_pynccl_broadcast_comm()
        )

    def _compact_broadcast_for_read(
        self,
        layer_id: int,
        indices: torch.Tensor,
        source: Optional[torch.Tensor],
        staging: torch.Tensor,
        selected_pages: torch.Tensor,
        page_size: int,
        max_pages: int,
        broadcast_kind: str,
        remap_indices: bool = True,
    ) -> torch.Tensor:
        """Broadcast owner's compact active pages and optionally remap indices."""
        if selected_pages.numel() > staging.shape[0]:
            raise RuntimeError(
                "CP Cache LayerSplit staging buffer is smaller than active page set: "
                f"active_pages={selected_pages.numel()}, capacity={staging.shape[0]}"
            )

        active_pages = selected_pages.numel()
        broadcast_pages = max(1, active_pages)
        owner_cp = self._get_layer_owner_rank(layer_id)
        if self.cp_rank == owner_cp:
            assert source is not None
            if active_pages > 0:
                staging[:active_pages].copy_(source[selected_pages.to(torch.long)])
            else:
                staging[:1].copy_(source[:1])

        pynccl_comm = get_pynccl_broadcast_comm()
        self._broadcast_slots.start(
            broadcast_kind,
            layer_id,
            staging[:broadcast_pages],
            owner_cp,
            pynccl_comm,
        )
        if not remap_indices:
            return indices
        return remap_indices_to_staging(indices, selected_pages, page_size, max_pages)

    def _bounded_full_token_staging_size(self) -> int:
        context_len = self._staging_context_len or self.swa_kv_pool.size
        if (
            self._staging_chunked_prefill_size is not None
            and self._staging_chunked_prefill_size > 0
        ):
            return min(context_len, self._staging_chunked_prefill_size)
        if (
            self._staging_max_prefill_tokens is not None
            and self._staging_max_prefill_tokens > 0
        ):
            return min(context_len, self._staging_max_prefill_tokens)
        return context_len

    def _initial_swa_staging_pages(self) -> int:
        token_bound = self._bounded_full_token_staging_size() + self.swa_page_size
        token_bound = min(self.swa_kv_pool.size, token_bound)
        return max(1, ceil_div(token_bound, self.swa_kv_pool.page_size))

    def _initial_extra_staging_pages(self, compress_ratio: int) -> int:
        context_len = self._staging_context_len
        if context_len is None:
            context_len = (
                self.c4_size * 4 if compress_ratio == 4 else self.c128_size * 128
            )
        compressed_tokens = ceil_div(context_len, compress_ratio)
        pool = self.c4_kv_pool if compress_ratio == 4 else self.c128_kv_pool
        compressed_tokens = min(pool.size, compressed_tokens)
        return max(1, ceil_div(compressed_tokens, pool.page_size))

    def _get_swa_staging_buffer(self, required_pages: int = 0) -> torch.Tensor:
        num_pages = max(self._initial_swa_staging_pages(), required_pages)
        return self._staging.get_or_grow(
            "swa",
            num_pages,
            lambda n: self.swa_kv_pool.create_buffer(num_pages=n),
        )

    def _extra_family_name(self, compress_ratio: int) -> str:
        if compress_ratio == 4:
            return "extra_c4"
        if compress_ratio == 128:
            return "extra_c128"
        raise ValueError(f"extra KV staging only for C4/C128, got {compress_ratio}")

    def _get_extra_staging_buffer(
        self, compress_ratio: int, required_pages: int = 0
    ) -> torch.Tensor:
        family = self._extra_family_name(compress_ratio)
        pool = self.c4_kv_pool if compress_ratio == 4 else self.c128_kv_pool
        num_pages = max(
            self._initial_extra_staging_pages(compress_ratio), required_pages
        )
        return self._staging.get_or_grow(
            family,
            num_pages,
            lambda n: pool.create_buffer(num_pages=n),
        )

    def _indexer_page_bytes(self) -> int:
        pool = self.c4_indexer_kv_pool
        page_bytes = pool.page_size * pool.index_head_dim
        page_bytes += (
            pool.page_size * (pool.index_head_dim // pool.quant_block_size) * 4
        )
        return page_bytes

    def _initial_indexer_staging_pages(self) -> int:
        context_len = self._staging_context_len
        if context_len is None:
            context_len = self.c4_indexer_kv_pool.size * 4
        compressed_tokens = min(self.c4_indexer_kv_pool.size, ceil_div(context_len, 4))
        return max(1, ceil_div(compressed_tokens, self.c4_indexer_kv_pool.page_size))

    def _get_indexer_staging_buffer(self, required_pages: int = 0) -> torch.Tensor:
        pool = self.c4_indexer_kv_pool
        num_pages = max(self._initial_indexer_staging_pages(), required_pages)

        def _allocate(n: int) -> torch.Tensor:
            return torch.empty(
                (n, self._indexer_page_bytes()),
                dtype=pool.index_k_with_scale_buffer_dtype,
                device=pool.device,
            )

        return self._staging.get_or_grow("indexer", num_pages, _allocate)

    def reset_batch_active_pages(self) -> None:
        """Invalidate batch-constant active-page caches."""
        for cache in self._batch_active_pages.values():
            cache.selected_pages = None
            cache.remapped = None

    def _clear_swa_read_state(self) -> None:
        self._swa_remapped_indices: Optional[torch.Tensor] = None
        self._swa_remapped_layer_id: Optional[int] = None

    def _clear_extra_read_state(self) -> None:
        self._extra_staging_layer_id: Optional[int] = None
        self._extra_page_table_selected_pages: Optional[torch.Tensor] = None
        self._extra_page_table_remap_layer_id: Optional[int] = None
        self._extra_page_table_page_size: Optional[int] = None
        self._extra_page_table_max_pages: Optional[int] = None
        self._extra_remapped_indices: Optional[torch.Tensor] = None
        self._extra_remapped_layer_id: Optional[int] = None

    def _clear_indexer_read_state(self) -> None:
        self._indexer_staging_layer_id: Optional[int] = None
        self._indexer_remapped_page_table: Optional[torch.Tensor] = None
        self._indexer_remapped_layer_id: Optional[int] = None

    def _build_owner_local_layer_map(self, compress_ratio: int) -> dict[int, int]:
        """Map each family layer to its local index in the owning rank's pool."""
        owner_offsets = [0] * self.cp_size
        result = {}
        for layer_id in range(self._stage_start, self._stage_end):
            if self.compression_ratios[layer_id] != compress_ratio:
                continue
            owner_cp = self._get_layer_owner_rank(layer_id)
            result[layer_id] = owner_offsets[owner_cp]
            owner_offsets[owner_cp] += 1
        return result

    def _owner_compress_layer_item(self, layer_id: int) -> DeepSeekV4LayerItem:
        item = self.layer_mapping[layer_id]
        assert item is not None
        assert item.compress_ratio in (
            4,
            128,
        ), f"layer {layer_id} has no compressed KV (ratio={item.compress_ratio})"
        if not (self._stage_start <= layer_id < self._stage_end):
            raise RuntimeError(f"layer {layer_id} is outside the local PP stage")
        owner_local = (
            self._c4_owner_local_by_layer[layer_id]
            if item.compress_ratio == 4
            else self._c128_owner_local_by_layer[layer_id]
        )
        compress_kv_pool = (
            self.c4_kv_pool if item.compress_ratio == 4 else self.c128_kv_pool
        )
        return DeepSeekV4LayerItem(
            compress_ratio=item.compress_ratio,
            compress_layer_id=owner_local,
            compress_kv_pool=compress_kv_pool,
        )

    def _c4_indexer_local_layer_id(self, layer_id: int) -> int:
        try:
            return self._c4_indexer_owner_local_by_layer[layer_id]
        except KeyError:
            raise RuntimeError(f"layer {layer_id} not in C4 indexer KV set") from None

    def prefetch_swa_layer(self, layer_id: int, indices: torch.Tensor) -> None:
        """Start current-layer SWA page broadcast; caller waits before attention."""
        self.wait_layer_transfer(layer_id)
        self._clear_extra_read_state()
        max_pages = self._pool_num_pages(self.swa_kv_pool)
        page_size = self.swa_kv_pool.page_size
        swa_cache = self._batch_active_pages["swa"]
        if swa_cache.selected_pages is None:
            swa_cache.selected_pages = self._active_pages_for_indices(
                indices, page_size, max_pages
            )
            swa_cache.remapped = remap_indices_to_staging(
                indices, swa_cache.selected_pages, page_size, max_pages
            )
        selected_pages = swa_cache.selected_pages
        staging = self._get_swa_staging_buffer(selected_pages.numel())
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]
        self._compact_broadcast_for_read(
            layer_id,
            indices,
            owner_buf,
            staging,
            selected_pages,
            page_size,
            max_pages,
            broadcast_kind="swa",
            remap_indices=False,
        )
        self._swa_remapped_indices = swa_cache.remapped
        self._swa_remapped_layer_id = layer_id

    def wait_swa_prefetch(self, layer_id: int) -> None:
        """Wait for the pending SWA prefetch before attention consumes it."""
        self._broadcast_slots.finish("swa", layer_id)

    def _prefetch_extra_key_pages(self, layer_id: int, indices: torch.Tensor) -> None:
        """Broadcast compact owner C4/C128 KV pages and remap indices."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio not in (4, 128):
            return
        owner_item = self._owner_compress_layer_item(layer_id)
        assert owner_item.compress_kv_pool is not None
        pool = owner_item.compress_kv_pool
        max_pages = self._pool_num_pages(pool)
        page_size = pool.page_size
        if item.compress_ratio == 128:
            c128_cache = self._batch_active_pages["extra_c128"]
            if c128_cache.selected_pages is None:
                c128_cache.selected_pages = self._active_pages_for_indices(
                    indices, page_size, max_pages
                )
                c128_cache.remapped = remap_indices_to_staging(
                    indices, c128_cache.selected_pages, page_size, max_pages
                )
            selected_pages = c128_cache.selected_pages
            cached_remapped_indices = c128_cache.remapped
        else:
            selected_pages = self._active_pages_for_indices(
                indices, page_size, max_pages
            )
            cached_remapped_indices = None
        staging = self._get_extra_staging_buffer(
            item.compress_ratio, selected_pages.numel()
        )
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = pool.kv_buffer[owner_item.compress_layer_id]
        remapped_indices = self._compact_broadcast_for_read(
            layer_id,
            indices,
            owner_buf,
            staging,
            selected_pages,
            page_size,
            max_pages,
            broadcast_kind="extra",
            remap_indices=cached_remapped_indices is None,
        )
        self._extra_remapped_indices = (
            cached_remapped_indices
            if cached_remapped_indices is not None
            else remapped_indices
        )
        self._extra_remapped_layer_id = layer_id
        self._extra_staging_layer_id = layer_id

    def prefetch_extra_key_layer_from_page_table(
        self, layer_id: int, page_table: torch.Tensor
    ) -> None:
        """Prefetch C4 extra KV pages selected by the batch page table."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio != 4:
            return
        self._clear_extra_read_state()
        owner_item = self._owner_compress_layer_item(layer_id)
        assert owner_item.compress_kv_pool is not None
        pool = owner_item.compress_kv_pool
        max_pages = self._pool_num_pages(pool)
        page_size = pool.page_size
        c4_pt_cache = self._batch_active_pages["extra_c4_page_table"]
        if c4_pt_cache.selected_pages is None:
            c4_pt_cache.selected_pages = self._active_pages_for_indices(
                page_table, 1, max_pages
            )
        selected_pages = c4_pt_cache.selected_pages
        staging = self._get_extra_staging_buffer(4, selected_pages.numel())
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_buf = None
        if self.cp_rank == owner_cp:
            owner_buf = pool.kv_buffer[owner_item.compress_layer_id]
        self._compact_broadcast_for_read(
            layer_id,
            page_table,
            owner_buf,
            staging,
            selected_pages,
            1,
            max_pages,
            broadcast_kind="extra",
            remap_indices=False,
        )
        self._extra_staging_layer_id = layer_id
        self._extra_page_table_selected_pages = selected_pages
        self._extra_page_table_remap_layer_id = layer_id
        self._extra_page_table_page_size = page_size
        self._extra_page_table_max_pages = max_pages

    def _prefetch_index_k_pages(self, layer_id: int, page_table: torch.Tensor) -> None:
        """Broadcast compact owner C4 indexer pages for ``layer_id``."""
        self.wait_layer_transfer(layer_id)
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self._clear_indexer_read_state()
        owner_cp = self._get_layer_owner_rank(layer_id)
        owner_local = self._c4_indexer_local_layer_id(layer_id)
        pool = self.c4_indexer_kv_pool
        max_pages = self._pool_num_pages(pool)
        indexer_cache = self._batch_active_pages["indexer"]
        if indexer_cache.selected_pages is None:
            indexer_cache.selected_pages = self._active_pages_for_indices(
                page_table, 1, max_pages
            )
            indexer_cache.remapped = remap_page_table_to_staging(
                page_table, indexer_cache.selected_pages, max_pages
            )
        selected_pages = indexer_cache.selected_pages
        staging = self._get_indexer_staging_buffer(selected_pages.numel())
        if self.cp_rank == owner_cp:
            owner_buf = pool.get_index_k_with_scale_buffer(owner_local)
        else:
            owner_buf = None
        self._compact_broadcast_for_read(
            layer_id,
            page_table,
            owner_buf,
            staging,
            selected_pages,
            1,
            max_pages,
            broadcast_kind="indexer",
            remap_indices=False,
        )
        self._indexer_remapped_page_table = indexer_cache.remapped
        self._indexer_remapped_layer_id = layer_id
        self._indexer_staging_layer_id = layer_id

    def prefetch_extra_key_layer(self, layer_id: int, core_metadata) -> None:
        """Prefetch current-layer C4/C128 KV after the owner-side write."""
        item = self.layer_mapping[layer_id]
        if item is None:
            return
        if item.compress_ratio in (4, 128):
            indices = (
                core_metadata.c4_sparse_page_indices
                if item.compress_ratio == 4
                else core_metadata.c128_page_indices
            )
            if indices is None:
                raise RuntimeError(
                    f"CP Cache LayerSplit missing compressed indices for layer {layer_id}"
                )
            self._prefetch_extra_key_pages(layer_id, indices)

    def prefetch_index_k_layer(self, layer_id: int, page_table: torch.Tensor) -> None:
        """Prefetch current-layer C4 indexer KV after the owner-side write."""
        item = self.layer_mapping[layer_id]
        if item is None:
            return
        if item.compress_ratio == 4:
            self._prefetch_index_k_pages(layer_id, page_table)

    def _rebuild_compressed_layer_mapping_for_cp(self) -> None:
        """``compress_layer_id`` counts only layers owned by this CP rank."""
        self._c4_owner_local_by_layer = self._build_owner_local_layer_map(4)
        self._c128_owner_local_by_layer = self._build_owner_local_layer_map(128)
        self._c4_indexer_owner_local_by_layer = self._build_owner_local_layer_map(4)

        c1_cnt = 0
        self.layer_mapping = [None] * len(self.compression_ratios)

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 4:
                owns_extra = self._owns_c4_kv_layer_id(idx)
            elif ratio == 128:
                owns_extra = self._owns_c128_kv_layer_id(idx)
            else:
                owns_extra = self._owns_swa_layer_id(idx)

            if not owns_extra:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=ratio,
                    compress_layer_id=None,
                    compress_kv_pool=None,
                )
                continue

            if ratio == 0:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=0,
                    compress_layer_id=c1_cnt,
                )
                c1_cnt += 1
            elif ratio == 4:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=4,
                    compress_layer_id=self._c4_owner_local_by_layer[idx],
                    compress_kv_pool=self.c4_kv_pool,
                )
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=self._c128_owner_local_by_layer[idx],
                    compress_kv_pool=self.c128_kv_pool,
                )
            else:
                raise ValueError(f"Unsupported compression ratio: {ratio}")

    def _swa_local_layer_id(self, layer_id: int) -> int:
        if layer_id not in self._swa_global_to_local:
            raise RuntimeError(
                f"cp_rank={self.cp_rank} does not own SWA for global layer {layer_id}"
            )
        return self._swa_global_to_local[layer_id]

    def clear_staging_remap_for_read(self) -> None:
        for kind in self._broadcast_slots.kinds():
            self._broadcast_slots.clear(kind)
        self._clear_swa_read_state()
        self._clear_extra_read_state()
        self._clear_indexer_read_state()
        self.reset_batch_active_pages()

    def get_swa_key_buffer(self, layer_id: int) -> torch.Tensor:
        swa_staging = self._staging.get_existing("swa")
        if self._swa_remapped_layer_id == layer_id and swa_staging is not None:
            return swa_staging
        if self._owns_swa_layer_id(layer_id):
            return super().get_swa_key_buffer(layer_id)
        return self._get_swa_staging_buffer()

    def get_swa_key_buffer_radix(self, layer_id: int) -> torch.Tensor:
        swa_staging = self._staging.get_existing("swa")
        if self._swa_remapped_layer_id == layer_id and swa_staging is not None:
            return swa_staging
        if self._owns_swa_layer_id(layer_id):
            return super().get_swa_key_buffer_radix(layer_id)
        return self._get_swa_staging_buffer()

    def remap_swa_indices_for_read(
        self, layer_id: int, indices: torch.Tensor
    ) -> torch.Tensor:
        if self._swa_remapped_layer_id == layer_id:
            assert self._swa_remapped_indices is not None
            return self._swa_remapped_indices
        return indices

    def set_swa_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        super().set_swa_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def set_swa_key_buffer_radix(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        super().set_swa_key_buffer_radix(
            layer_id, swa_loc, cache_nope_fp8_rope_bf16_pack
        )

    def set_swa_key_buffer_radix_fused(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        return super().set_swa_key_buffer_radix_fused(layer_id, swa_loc, cache_k)

    def set_swa_key_buffer_radix_fused_norm_rope(
        self,
        layer_id: int,
        swa_loc: torch.Tensor,
        kv: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if not self._owns_swa_layer_id(layer_id):
            return
        return super().set_swa_key_buffer_radix_fused_norm_rope(
            layer_id,
            swa_loc,
            kv,
            kv_weight,
            eps,
            freqs_cis,
            positions,
        )

    def get_extra_key_buffer(self, layer_id: int) -> torch.Tensor | None:
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio not in (4, 128):
            return None
        if self._extra_staging_layer_id == layer_id:
            family = self._extra_family_name(item.compress_ratio)
            extra_staging = self._staging.get_existing(family)
            if extra_staging is not None:
                self._broadcast_slots.finish("extra", layer_id)
                return extra_staging
        if self._owns_extra_key_layer_id(layer_id):
            return super().get_extra_key_buffer(layer_id)
        return self._get_extra_staging_buffer(item.compress_ratio)

    def remap_extra_indices_for_read(
        self, layer_id: int, indices: torch.Tensor
    ) -> torch.Tensor:
        if self._extra_remapped_layer_id == layer_id:
            assert self._extra_remapped_indices is not None
            return self._extra_remapped_indices
        if self._extra_page_table_remap_layer_id == layer_id:
            assert self._extra_page_table_selected_pages is not None
            assert self._extra_page_table_page_size is not None
            assert self._extra_page_table_max_pages is not None
            return remap_indices_to_staging(
                indices,
                self._extra_page_table_selected_pages,
                self._extra_page_table_page_size,
                self._extra_page_table_max_pages,
            )
        return indices

    def set_extra_key_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_nope_fp8_rope_bf16_pack: NopeFp8RopeBf16Pack,
    ) -> None:
        if not self._owns_extra_key_layer_id(layer_id):
            return
        super().set_extra_key_buffer(layer_id, loc, cache_nope_fp8_rope_bf16_pack)

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self._indexer_staging_layer_id == layer_id:
            indexer_staging = self._staging.get_existing("indexer")
            if indexer_staging is not None:
                self._broadcast_slots.finish("indexer", layer_id)
                return indexer_staging
        if self._owns_indexer_kv_layer_id(layer_id):
            return self.c4_indexer_kv_pool.get_index_k_with_scale_buffer(
                self._c4_indexer_local_layer_id(layer_id)
            )
        return self._get_indexer_staging_buffer()

    def remap_indexer_page_table_for_read(
        self, layer_id: int, page_table: torch.Tensor
    ) -> torch.Tensor:
        if self._indexer_remapped_layer_id == layer_id:
            assert self._indexer_remapped_page_table is not None
            return self._indexer_remapped_page_table
        return page_table

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        if not self._owns_indexer_kv_layer_id(layer_id):
            return
        self.c4_indexer_kv_pool.set_index_k_scale_buffer(
            self._c4_indexer_local_layer_id(layer_id), loc, index_k, index_k_scale
        )

    def set_extra_key_buffer_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
    ) -> None:
        if not self._owns_extra_key_layer_id(layer_id):
            return
        super().set_extra_key_buffer_fused(layer_id, loc, cache_k)

    def set_index_k_fused(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        if not self._owns_indexer_kv_layer_id(layer_id):
            return
        self.c4_indexer_kv_pool.set_key_buffer_fused(
            self._c4_indexer_local_layer_id(layer_id), loc, index_k
        )
