"""DeepSeek V4 token-to-KV pool with CP layer-sharded KV buffers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4 import index_buf_accessor
from sglang.srt.mem_cache.cp_kv_layer_split.deepseek_v4_layout import (
    build_cp_kv_layer_split_deepseek_v4_pool_layout,
    shard_cp_kv_layer_split_c4,
    shard_cp_kv_layer_split_c4_indexer,
    shard_cp_kv_layer_split_c128,
    shard_cp_kv_layer_split_swa,
)
from sglang.srt.mem_cache.cp_kv_layer_split.ownership import (
    build_owned_layer_local_index_map,
    kv_layer_owner,
    owned_kv_layer_range,
    owns_kv_layer,
)
from sglang.srt.mem_cache.cp_kv_layer_split.pool_base import CpKvLayerSplitPoolBase
from sglang.srt.mem_cache.cp_kv_layer_split.staging import (
    remap_indices_to_staging,
    remap_page_table_to_staging,
)
from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DSV4_TRANSFER_ATTENTION_STATE,
    DSV4_TRANSFER_C4_INDEXER_KV,
    DSV4_TRANSFER_C4_KV,
    DSV4_TRANSFER_C128_KV,
    DSV4_TRANSFER_INDEXER_STATE,
    DSV4_TRANSFER_SWA_KV,
    ONLINE_C128,
    DeepSeekV4LayerItem,
    DeepSeekV4TokenToKVPool,
    NopeFp8RopeBf16Pack,
)
from sglang.srt.utils import ceil_div

logger = logging.getLogger(__name__)


@dataclass
class _BatchActivePages:
    """Per-forward active pages plus remapped indices/page table."""

    selected_pages: Optional["torch.Tensor"] = None
    remapped: Optional["torch.Tensor"] = None


class CpKvLayerSplitDeepSeekV4TokenToKVPool(
    CpKvLayerSplitPoolBase, DeepSeekV4TokenToKVPool
):
    """DeepSeek V4 pool that reads non-owned layer KV through staging buffers."""

    def __init__(
        self,
        *,
        cp_rank: int,
        cp_size: int,
        model_num_hidden_layers: int,
        **kwargs,
    ):
        staging_context_len = kwargs.pop("cp_kv_layer_split_staging_context_len", None)
        staging_chunked_prefill_size = kwargs.pop(
            "cp_kv_layer_split_staging_chunked_prefill_size", None
        )
        staging_max_prefill_tokens = kwargs.pop(
            "cp_kv_layer_split_staging_max_prefill_tokens", None
        )

        self._shard_swa = shard_cp_kv_layer_split_swa()
        self._shard_c4 = shard_cp_kv_layer_split_c4()
        self._shard_c128 = shard_cp_kv_layer_split_c128()
        self._shard_c4_indexer = shard_cp_kv_layer_split_c4_indexer()

        self._indexer_staging_layer_id: Optional[int] = None
        self._indexer_remapped_page_table: Optional[torch.Tensor] = None
        self._indexer_remapped_layer_id: Optional[int] = None
        self._swa_remapped_indices: Optional[torch.Tensor] = None
        self._swa_remapped_layer_id: Optional[int] = None
        self._extra_staging_layer_id: Optional[int] = None
        self._extra_page_table_selected_pages: Optional[torch.Tensor] = None
        self._extra_page_table_remap_layer_id: Optional[int] = None
        self._extra_page_table_page_size: Optional[int] = None
        self._extra_page_table_max_pages: Optional[int] = None
        self._extra_remapped_indices: Optional[torch.Tensor] = None
        self._extra_remapped_layer_id: Optional[int] = None

        # Batch-constant metadata can reuse active-page selection across layers.
        # The hit/miss pattern depends only on layer order after the per-forward
        # reset, so all CP ranks still enter the same collectives.
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

        layout = build_cp_kv_layer_split_deepseek_v4_pool_layout(
            cp_rank,
            cp_size,
            model_num_hidden_layers,
            pp_start,
            pp_end,
            kwargs["compression_ratios"],
        )
        self.cp_kv_layer_split_layout = layout

        owned_start, owned_end = owned_kv_layer_range(
            cp_rank,
            cp_size,
            model_num_hidden_layers,
            pp_start,
            pp_end,
        )
        if self._shard_swa:
            self._swa_global_to_local = build_owned_layer_local_index_map(
                cp_rank,
                cp_size,
                model_num_hidden_layers,
                pp_start,
                pp_end,
            )
        else:
            self._swa_global_to_local = {
                layer_id: layer_id - pp_start for layer_id in range(pp_start, pp_end)
            }

        if layout.swa_layer_num == 0:
            logger.warning(
                "CpKvLayerSplitDeepSeekV4TokenToKVPool: cp_rank=%s owns no SWA layers "
                "in PP slice [%s, %s)",
                cp_rank,
                pp_start,
                pp_end,
            )

        logger.info(
            "CpKvLayerSplitDeepSeekV4TokenToKVPool: cp_rank=%s cp_size=%s owns global "
            "SWA [%s, %s) -> %s buffers, C4=%s, C128=%s, C4_INDEXER=%s, "
            "C4_STATE=%s, C128_STATE=%s, C4_INDEXER_STATE=%s (PP [%s, %s))",
            cp_rank,
            cp_size,
            owned_start,
            owned_end,
            layout.swa_layer_num,
            layout.c4_layer_num,
            layout.c128_layer_num,
            layout.c4_indexer_layer_num,
            layout.c4_state_layer_num,
            layout.c128_state_layer_num,
            layout.c4_indexer_state_layer_num,
            pp_start,
            pp_end,
        )

        super().__init__(
            cp_rank=cp_rank,
            cp_size=cp_size,
            model_num_hidden_layers=model_num_hidden_layers,
            broadcast_slot_kinds=("swa", "extra", "indexer"),
            staging_context_len=staging_context_len,
            staging_chunked_prefill_size=staging_chunked_prefill_size,
            staging_max_prefill_tokens=staging_max_prefill_tokens,
            **kwargs,
            cp_kv_layer_split_layout=layout,
        )
        self._rebuild_compressed_layer_mapping_for_cp()
        if logger.isEnabledFor(logging.DEBUG):
            host_mapping = self.get_hicache_host_layer_mapping()
            logger.debug(
                "CpKvLayerSplitDeepSeekV4TokenToKVPool: cp_rank=%s HiCache host-pool "
                "owned counts: swa=%s c4_kv=%s c128_kv=%s c4_indexer=%s c4_state=%s "
                "c128_state=%s c4_indexer_state=%s",
                cp_rank,
                len(host_mapping["swa"]),
                len(host_mapping["c4_kv"]),
                len(host_mapping["c128_kv"]),
                len(host_mapping["c4_indexer"]),
                len(host_mapping["c4_state"]),
                len(host_mapping["c128_state"]),
                len(host_mapping["c4_indexer_state"]),
            )

    def _init_paged_compress_states(self, enable_memory_saver: bool):
        c4_state_pool_size = self.c4_state_pool_size
        c128_state_pool_size = self.c128_state_pool_size
        total_L = len(self.compression_ratios)
        self.compress_state_pools: list[Optional[CompressStatePool]] = [None] * total_L
        self.indexer_compress_state_pools: list[Optional[CompressStatePool]] = [
            None
        ] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio == 0:
                continue

            overlap = ratio == 4
            size = c4_state_pool_size if ratio == 4 else c128_state_pool_size
            ring_size = self.get_ring_size(ratio)
            state_dtype = (
                self.c4_state_dtype if ratio == 4 else self.c128_state_dtype
            )

            if self._owns_attention_state_layer_id(idx):
                self.compress_state_pools[idx] = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
                    dtype=state_dtype,
                    device=self.device,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    online=(ratio == 128 and ONLINE_C128),
                    swa_page_size=self.swa_page_size,
                    online_mtp_max_draft_tokens=(
                        self.online_mtp_max_draft_tokens if ratio == 128 else 0
                    ),
                )

            if ratio == 4 and self._owns_indexer_state_layer_id(idx):
                self.indexer_compress_state_pools[idx] = CompressStatePool(
                    size=size,
                    ring_size=ring_size,
                    overlap=overlap,
                    head_dim=self.indexer_head_dim,
                    device=self.device,
                    dtype=self.c4_state_dtype,
                    enable_memory_saver=enable_memory_saver,
                    ratio=ratio,
                    swa_page_size=self.swa_page_size,
                )

    def _transfer_swa_layer_id(self, layer_id: int) -> bool:
        if self._shard_swa:
            return self.owns_kv_layer_id(layer_id)
        return self.cp_rank == 0

    def _transfer_core_layer_id(self, layer_id: int) -> bool:
        ratio = self.compression_ratios[layer_id]
        if ratio == 4:
            sharded = self._shard_c4
        elif ratio == 128:
            sharded = self._shard_c128
        else:
            sharded = False
        if sharded:
            return self.owns_kv_layer_id(layer_id)
        return self.cp_rank == 0

    def _transfer_indexer_layer_id(self, layer_id: int) -> bool:
        sharded = self._shard_c4_indexer
        if sharded:
            return self.owns_kv_layer_id(layer_id)
        return self.cp_rank == 0

    def get_kv_transfer_layout(self) -> list:
        """Descriptors parallel to this rank's sharded V4 KV buffer list."""
        layout: list = []
        stage_layers = range(self._stage_start, self._stage_end)

        layout.extend(
            (
                (DSV4_TRANSFER_C4_KV, layer_id)
                if self._transfer_core_layer_id(layer_id)
                else None
            )
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_c4_kv_layer_id(layer_id)
        )
        layout.extend(
            (
                (DSV4_TRANSFER_C4_INDEXER_KV, layer_id)
                if self._transfer_indexer_layer_id(layer_id)
                else None
            )
            for layer_id in stage_layers
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_kv_layer_id(layer_id)
        )
        layout.extend(
            (
                (DSV4_TRANSFER_C128_KV, layer_id)
                if self._transfer_core_layer_id(layer_id)
                else None
            )
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
        layout.extend(
            (
                (DSV4_TRANSFER_SWA_KV, layer_id)
                if self._transfer_swa_layer_id(layer_id)
                else None
            )
            for layer_id in swa_layers
        )

        layout.extend(
            (
                (DSV4_TRANSFER_ATTENTION_STATE, layer_id)
                if self._transfer_core_layer_id(layer_id)
                else None
            )
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] != 0
            and self._owns_attention_state_layer_id(layer_id)
        )
        layout.extend(
            (
                (DSV4_TRANSFER_INDEXER_STATE, layer_id)
                if self._transfer_indexer_layer_id(layer_id)
                else None
            )
            for layer_id in range(self._stage_start, self._stage_end)
            if self.compression_ratios[layer_id] == 4
            and self._owns_indexer_state_layer_id(layer_id)
        )
        return layout

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

    def owns_kv_layer_id(self, layer_id: int) -> bool:
        return owns_kv_layer(
            layer_id, self.cp_rank, self.cp_size, self.model_num_hidden_layers
        )

    def _owns_swa_layer_id(self, layer_id: int) -> bool:
        return not self._shard_swa or self.owns_kv_layer_id(layer_id)

    def should_skip_swa_write(self, layer_id: int) -> bool:
        return not self._owns_swa_layer_id(layer_id)

    def _owns_c4_kv_layer_id(self, layer_id: int) -> bool:
        return not self._shard_c4 or self.owns_kv_layer_id(layer_id)

    def _owns_c128_kv_layer_id(self, layer_id: int) -> bool:
        return not self._shard_c128 or self.owns_kv_layer_id(layer_id)

    def _owns_extra_key_layer_id(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio == 4:
            return self._owns_c4_kv_layer_id(layer_id)
        if item.compress_ratio == 128:
            return self._owns_c128_kv_layer_id(layer_id)
        return False

    def _owns_indexer_kv_layer_id(self, layer_id: int) -> bool:
        return not self._shard_c4_indexer or self.owns_kv_layer_id(layer_id)

    def _owns_attention_state_layer_id(self, layer_id: int) -> bool:
        ratio = self.compression_ratios[layer_id]
        if ratio == 4:
            return not self._shard_c4 or self.owns_kv_layer_id(layer_id)
        if ratio == 128:
            return not self._shard_c128 or self.owns_kv_layer_id(layer_id)
        return False

    def _owns_indexer_state_layer_id(self, layer_id: int) -> bool:
        return not self._shard_c4_indexer or self.owns_kv_layer_id(layer_id)

    def _broadcast_swa_layer_id(self, layer_id: int) -> bool:
        return self._shard_swa and self.cp_size > 1

    def _broadcast_extra_key_layer_id(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio == 4:
            return (self._shard_c4) and self.cp_size > 1
        if item.compress_ratio == 128:
            return (self._shard_c128) and self.cp_size > 1
        return False

    def _broadcast_indexer_layer_id(self, layer_id: int) -> bool:
        return (self._shard_c4_indexer) and self.cp_size > 1

    def should_skip_core_compressor_write(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio == 4:
            sharded = self._shard_c4
        elif item.compress_ratio == 128:
            sharded = self._shard_c128
        else:
            sharded = False
        return sharded and not self.owns_kv_layer_id(layer_id)

    def should_skip_indexer_compressor_write(self, layer_id: int) -> bool:
        sharded = self._shard_c4_indexer
        return sharded and not self.owns_kv_layer_id(layer_id)

    def should_use_c4_extra_broadcast_overlap(self, layer_id: int) -> bool:
        item = self.layer_mapping[layer_id]
        return (
            item is not None
            and item.compress_ratio == 4
            and self._broadcast_extra_key_layer_id(layer_id)
        )

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

    def is_any_family_sharded(self) -> bool:
        return any(
            (
                self._shard_swa,
                self._shard_c4,
                self._shard_c128,
                self._shard_c4_indexer,
            )
        )

    def _owner_compress_layer_item(self, layer_id: int) -> DeepSeekV4LayerItem:
        item = self.layer_mapping[layer_id]
        assert item is not None
        assert item.compress_ratio in (
            4,
            128,
        ), f"layer {layer_id} has no compressed KV (ratio={item.compress_ratio})"
        if not (self._stage_start <= layer_id < self._stage_end):
            owner_cp = self._kv_owner_cp_rank(layer_id)
            raise RuntimeError(
                f"layer {layer_id} not in owner cp_rank={owner_cp} compress set"
            )
        owner_cp = self._kv_owner_cp_rank(layer_id)
        family_sharded = (
            self._shard_c4 if item.compress_ratio == 4 else self._shard_c128
        )
        owner_local = sum(
            1
            for idx in range(self._stage_start, layer_id)
            if self.compression_ratios[idx] == item.compress_ratio
            and (
                not family_sharded
                or kv_layer_owner(idx, self.cp_size, self.model_num_hidden_layers)
                == owner_cp
            )
        )
        compress_kv_pool = (
            self.c4_kv_pool if item.compress_ratio == 4 else self.c128_kv_pool
        )
        return DeepSeekV4LayerItem(
            compress_ratio=item.compress_ratio,
            compress_layer_id=owner_local,
            compress_kv_pool=compress_kv_pool,
        )

    def _async_kind_for(self, kind: str) -> Optional[str]:
        """Resolve a broadcast slot kind to its async/inline status."""
        if kind == "swa":
            return (
                "swa"
                if envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_ENABLE_SWA_BROADCAST_OVERLAP.get()
                else None
            )
        if kind in ("extra", "indexer"):
            if envs.SGLANG_CP_KV_LAYER_SPLIT_DSV4_DISABLE_ASYNC_NON_SWA_BROADCAST.get():
                return None
            return kind
        raise ValueError(f"unknown broadcast kind: {kind}")

    def _c4_indexer_local_layer_id(
        self, layer_id: int, owner_cp: int | None = None
    ) -> int:
        if owner_cp is None:
            owner_cp = self.cp_rank
        local_id = 0
        for idx in range(self._stage_start, self._stage_end):
            if idx == layer_id:
                return local_id
            if self.compression_ratios[idx] != 4:
                continue
            if (
                not self._shard_c4_indexer
                or kv_layer_owner(idx, self.cp_size, self.model_num_hidden_layers)
                == owner_cp
            ):
                local_id += 1
        raise RuntimeError(f"layer {layer_id} not in C4 indexer KV set")

    def start_swa_layer_for_read(self, layer_id: int, indices: torch.Tensor) -> None:
        """Start owner SWA page broadcast; caller must finish before attention read."""
        self.wait_layer_transfer(layer_id)
        self._extra_staging_layer_id = None
        self._extra_page_table_selected_pages = None
        self._extra_page_table_remap_layer_id = None
        self._extra_page_table_page_size = None
        self._extra_page_table_max_pages = None
        self._extra_remapped_indices = None
        self._extra_remapped_layer_id = None
        self._broadcast_slots.clear("swa", next_layer_id=layer_id)
        if not self._broadcast_swa_layer_id(layer_id):
            self._swa_remapped_indices = indices
            self._swa_remapped_layer_id = layer_id
            return

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
        owner_cp = self._kv_owner_cp_rank(layer_id)
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
            async_kind=self._async_kind_for("swa"),
            remap_indices=False,
        )
        self._swa_remapped_indices = swa_cache.remapped
        self._swa_remapped_layer_id = layer_id

    def finish_swa_layer_for_read(self, layer_id: int) -> None:
        """Wait for a pending SWA broadcast before the layer reads SWA KV."""
        self._broadcast_slots.finish("swa", layer_id)

    def prepare_extra_key_layer_for_read(
        self, layer_id: int, indices: torch.Tensor
    ) -> None:
        """Broadcast compact owner C4/C128 KV pages and remap read indices."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio not in (4, 128):
            return
        if not self._broadcast_extra_key_layer_id(layer_id):
            self._extra_remapped_indices = indices
            self._extra_remapped_layer_id = layer_id
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
        owner_cp = self._kv_owner_cp_rank(layer_id)
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
            async_kind=self._async_kind_for("extra"),
            remap_indices=cached_remapped_indices is None,
        )
        self._extra_remapped_indices = (
            cached_remapped_indices
            if cached_remapped_indices is not None
            else remapped_indices
        )
        self._extra_remapped_layer_id = layer_id
        self._extra_staging_layer_id = layer_id

    def start_page_table_extra_key_layer_for_read(
        self, layer_id: int, page_table: torch.Tensor
    ) -> None:
        """Broadcast C4 extra KV pages selected by the batch page table."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio != 4:
            return
        self._extra_staging_layer_id = None
        self._extra_page_table_selected_pages = None
        self._extra_page_table_remap_layer_id = None
        self._extra_page_table_page_size = None
        self._extra_page_table_max_pages = None
        self._extra_remapped_indices = None
        self._extra_remapped_layer_id = None
        if not self._broadcast_extra_key_layer_id(layer_id):
            return

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
        owner_cp = self._kv_owner_cp_rank(layer_id)
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
            async_kind=self._async_kind_for("extra"),
            remap_indices=False,
        )
        self._extra_staging_layer_id = layer_id
        self._extra_page_table_selected_pages = selected_pages
        self._extra_page_table_remap_layer_id = layer_id
        self._extra_page_table_page_size = page_size
        self._extra_page_table_max_pages = max_pages

    def prepare_index_k_layer_for_read(
        self, layer_id: int, page_table: torch.Tensor
    ) -> None:
        """Broadcast compact owner C4 indexer pages for ``layer_id``."""
        self.wait_layer_transfer(layer_id)
        compress_ratio, _, _ = self.layer_mapping[layer_id]
        assert compress_ratio == 4, f"only c4 has indexer, got {compress_ratio = }"
        self._indexer_staging_layer_id = None
        self._indexer_remapped_page_table = None
        self._indexer_remapped_layer_id = None
        if not self._broadcast_indexer_layer_id(layer_id):
            return
        owner_cp = self._kv_owner_cp_rank(layer_id)
        owner_local = self._c4_indexer_local_layer_id(layer_id, owner_cp)
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
            async_kind=self._async_kind_for("indexer"),
            remap_indices=False,
        )
        self._indexer_remapped_page_table = indexer_cache.remapped
        self._indexer_remapped_layer_id = layer_id
        self._indexer_staging_layer_id = layer_id

    def sync_extra_key_layer_for_read(self, layer_id: int, core_metadata) -> None:
        """Collective after owner C4/C128 KV write and before extra-KV reads."""
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
                    f"CP KV LayerSplit missing compressed indices for layer {layer_id}"
                )
            self.prepare_extra_key_layer_for_read(layer_id, indices)

    def sync_index_k_layer_for_read(
        self, layer_id: int, page_table: torch.Tensor
    ) -> None:
        """Collective after owner C4 indexer KV write and before indexer reads."""
        item = self.layer_mapping[layer_id]
        if item is None:
            return
        if item.compress_ratio == 4:
            self.prepare_index_k_layer_for_read(layer_id, page_table)

    def _rebuild_compressed_layer_mapping_for_cp(self) -> None:
        """``compress_layer_id`` counts only layers owned by this CP rank."""
        c1_cnt = c4_cnt = c128_cnt = 0
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
                    compress_layer_id=c4_cnt,
                    compress_kv_pool=self.c4_kv_pool,
                )
                c4_cnt += 1
            elif ratio == 128:
                self.layer_mapping[idx] = DeepSeekV4LayerItem(
                    compress_ratio=128,
                    compress_layer_id=c128_cnt,
                    compress_kv_pool=self.c128_kv_pool,
                )
                c128_cnt += 1
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
        self._swa_remapped_indices = None
        self._swa_remapped_layer_id = None
        self._extra_staging_layer_id = None
        self._extra_page_table_selected_pages = None
        self._extra_page_table_remap_layer_id = None
        self._extra_page_table_page_size = None
        self._extra_page_table_max_pages = None
        self._extra_remapped_indices = None
        self._extra_remapped_layer_id = None
        self._indexer_staging_layer_id = None
        self._indexer_remapped_page_table = None
        self._indexer_remapped_layer_id = None
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

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len: int,
        page_indices: torch.Tensor,
    ):
        if self._indexer_staging_layer_id == layer_id:
            indexer_staging = self._staging.get_existing("indexer")
            if indexer_staging is not None:
                self._broadcast_slots.finish("indexer", layer_id)
                return index_buf_accessor.GetKAndS.execute(
                    self.c4_indexer_kv_pool,
                    indexer_staging,
                    seq_len=seq_len,
                    page_indices=page_indices,
                )
        if self._owns_indexer_kv_layer_id(layer_id):
            return self.c4_indexer_kv_pool.get_index_k_scale_buffer(
                self._c4_indexer_local_layer_id(layer_id), seq_len, page_indices
            )
        buf = self._get_indexer_staging_buffer()
        return index_buf_accessor.GetKAndS.execute(
            self.c4_indexer_kv_pool, buf, seq_len=seq_len, page_indices=page_indices
        )

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
