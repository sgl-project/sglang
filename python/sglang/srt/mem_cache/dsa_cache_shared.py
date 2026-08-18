# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    quantize_k_cache_separate,
)
from sglang.kernels.ops.kvcache.dsa_shared import (
    materialize_indexer_pages_triton,
    set_mla_kv_buffer_owner_and_current_triton,
    set_mla_kv_buffer_owner_triton,
)
from sglang.srt.layers.attention.dsa.utils import (
    INDEXER_K_CACHE_PRESHUFFLE_TILE,
    aiter_can_use_preshuffle_paged_mqa,
)
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    DSATokenToKVPool,
    RadixAttention,
    maybe_detect_oob,
)
from sglang.srt.mem_cache.shared_kv.demand_cache import PoolDemandCache
from sglang.srt.mem_cache.shared_kv.family import (
    OwnerShardedFamily,
    OwnerShardedFamilySpec,
)
from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.dsa_shared_demand import SharedMLACurrentRows


@dataclass(frozen=True)
class SharedDSACapacityAccounting:
    logical_tokens: int
    logical_pages: int
    cp_size: int
    main_physical_token_slots_per_layer_per_rank: int
    indexer_mode: str
    indexer_physical_token_slots_per_layer_per_rank: int
    authoritative_bytes_per_rank: int
    indexer_demand_bytes_per_rank: int
    main_demand_bytes_per_rank: int
    tracked_total_bytes_per_rank: int


def log_shared_dsa_capacity_accounting(
    pool: SharedDSATokenToKVPool, *, main_demand_workspace_bytes: int
) -> SharedDSACapacityAccounting:
    """Log logical capacity beside the physical HBM charged to each CP rank."""
    main = pool.main_family.accounting()
    publication_bytes = pool.shared_write_publisher.mapped_bytes_per_rank
    if pool.share_indexer:
        indexer = pool.index_key_cache.family.accounting()
        indexer_mode = "shared"
        indexer_physical_token_slots = indexer.minimum_blocks_per_rank * pool.page_size
        indexer_authoritative_bytes = indexer.mapped_bytes_per_rank
        indexer_demand_bytes = pool.index_key_cache.pool_cache.allocated_bytes
    else:
        indexer_mode = "replicated"
        indexer_physical_token_slots = (
            pool.index_key_cache.buffer[0].shape[0] * pool.page_size
        )
        indexer_authoritative_bytes = sum(
            buffer.nbytes for buffer in pool.index_key_cache.buffer
        )
        indexer_demand_bytes = 0

    authoritative_bytes = (
        main.mapped_bytes_per_rank + indexer_authoritative_bytes + publication_bytes
    )
    tracked_total_bytes = (
        authoritative_bytes + indexer_demand_bytes + main_demand_workspace_bytes
    )
    accounting = SharedDSACapacityAccounting(
        logical_tokens=pool.size,
        logical_pages=(pool.size + pool.page_size - 1) // pool.page_size,
        cp_size=pool.shared_size,
        main_physical_token_slots_per_layer_per_rank=(
            main.minimum_blocks_per_rank * pool.page_size
        ),
        indexer_mode=indexer_mode,
        indexer_physical_token_slots_per_layer_per_rank=(indexer_physical_token_slots),
        authoritative_bytes_per_rank=authoritative_bytes,
        indexer_demand_bytes_per_rank=indexer_demand_bytes,
        main_demand_bytes_per_rank=main_demand_workspace_bytes,
        tracked_total_bytes_per_rank=tracked_total_bytes,
    )
    logger.info(
        "Shared DSA capacity ledger: logical_tokens=%s logical_pages=%s "
        "cp_size=%s main_physical_token_slots_per_layer_per_rank=%s "
        "indexer_mode=%s indexer_physical_token_slots_per_layer_per_rank=%s "
        "authoritative_bytes_per_rank=%s "
        "indexer_demand_bytes_per_rank=%s main_demand_bytes_per_rank=%s "
        "tracked_total_bytes_per_rank=%s",
        accounting.logical_tokens,
        accounting.logical_pages,
        accounting.cp_size,
        accounting.main_physical_token_slots_per_layer_per_rank,
        accounting.indexer_mode,
        accounting.indexer_physical_token_slots_per_layer_per_rank,
        accounting.authoritative_bytes_per_rank,
        accounting.indexer_demand_bytes_per_rank,
        accounting.main_demand_bytes_per_rank,
        accounting.tracked_total_bytes_per_rank,
    )
    return accounting


def gather_shared_index_rows(
    pool: SharedDSATokenToKVPool,
    buffer: torch.Tensor,
    loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed Indexer rows from Shared-KV's page-major storage."""
    loc = loc.view(-1).long()
    page_size = pool.page_size
    head_dim = pool.index_head_dim
    scale_bytes = head_dim // pool.quant_block_size * 4
    bytes_per_page = page_size * (head_dim + scale_bytes)
    pages = torch.div(loc, page_size, rounding_mode="floor")
    tokens = loc.remainder(page_size)
    columns = torch.arange(head_dim, device=loc.device, dtype=torch.int64)
    if aiter_can_use_preshuffle_paged_mqa():
        tile = INDEXER_K_CACHE_PRESHUFFLE_TILE
        k_offsets = (
            pages[:, None] * bytes_per_page
            + torch.div(tokens, tile, rounding_mode="floor")[:, None]
            * (tile * head_dim)
            + torch.div(columns, tile, rounding_mode="floor")[None, :] * (tile * tile)
            + tokens.remainder(tile)[:, None] * tile
            + columns.remainder(tile)[None, :]
        )
    else:
        k_offsets = (
            pages[:, None] * bytes_per_page
            + tokens[:, None] * head_dim
            + columns[None, :]
        )
    scale_columns = torch.arange(scale_bytes, device=loc.device, dtype=torch.int64)
    scale_offsets = (
        pages[:, None] * bytes_per_page
        + page_size * head_dim
        + tokens[:, None] * scale_bytes
        + scale_columns[None, :]
    )
    flat = buffer.view(-1)
    return flat[k_offsets], flat[scale_offsets]


def indexer_pool_cache_shape(
    index_buf_size: int,
    page_size: int,
    page_bytes: int,
) -> tuple[int, int]:
    if index_buf_size < 0 or min(page_size, page_bytes) <= 0:
        raise ValueError("Indexer Pool Demand Cache dimensions must be positive")
    logical_pages = (index_buf_size + page_size + 1) // page_size
    return logical_pages, page_bytes


@dataclass(frozen=True)
class SharedDSAPageLayout:
    owner_layout: OwnerShardedLayout
    local_pages_per_layer: Optional[int] = None

    @property
    def cp_size(self) -> int:
        return self.owner_layout.cp_size

    @property
    def page_size(self) -> int:
        return self.owner_layout.ownership_granule

    @property
    def pages_per_rank(self) -> int:
        return self.owner_layout.blocks_per_rank

    def translate_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        return self.owner_layout.physical_rows(slot_indices)

    def translate_local_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        return self.owner_layout.owner_local_rows(slot_indices)

    def owned_slot_mask(
        self, slot_indices: torch.Tensor, *, owner_rank: int
    ) -> torch.Tensor:
        return self.owner_layout.owned_row_mask(slot_indices, rank=owner_rank)


class SharedIndexKeyCache:
    """Page-owner-sharded Indexer K/scale storage exposed as one global VMM."""

    def __init__(self, pool: SharedDSATokenToKVPool, index_buf_size: int):
        self.pool = pool
        logical_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        requested_pages = (logical_pages + pool.shared_size - 1) // pool.shared_size
        page_bytes = pool.page_size * (
            pool.index_head_dim + pool.index_head_dim // pool.quant_block_size * 4
        )
        self.family = OwnerShardedFamily.create(
            spec=OwnerShardedFamilySpec(
                name="dsa_indexer_k",
                num_layers=pool.layer_num,
                logical_rows_per_layer=requested_pages * pool.shared_size,
                ownership_granule=1,
                storage_rows_per_granule=1,
                row_shape=(page_bytes,),
                dtype=pool.index_k_with_scale_buffer_dtype,
                # DeepGEMM validates that the base pointer belongs to the
                # process's current CUDA device. Put this rank's physical
                # segment first and translate page IDs into rank-relative order.
                map_rank_local=True,
            ),
            cp_size=pool.shared_size,
            cpu_group=pool._get_cp_group().cpu_group,
            zero_initialize=False,
        )
        self.layout = self.family.layout
        self.buffer = self.family.slab.rank_local_views
        self.local_buffer = self.family.slab.local_views
        cache_shape = indexer_pool_cache_shape(
            index_buf_size,
            pool.page_size,
            page_bytes,
        )
        with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.pool_cache = PoolDemandCache.create(
                keys=pool.indexer_cache_layer_ids,
                entries_per_key=cache_shape[0],
                entry_bytes=cache_shape[1],
                dtype=pool.index_k_with_scale_buffer_dtype,
                device=pool.device,
            )

    def clear(self) -> None:
        self.buffer = []
        self.local_buffer = []
        self.pool_cache.clear()
        self.family.close()

    def get_global_buffer(self, layer_id: int) -> torch.Tensor:
        return self.buffer[layer_id - self.pool.start_layer]

    def get_local_buffer(self, layer_id: int) -> torch.Tensor:
        return self.local_buffer[layer_id - self.pool.start_layer]

    def materialize_pages(
        self,
        layer_id: int,
        source_pages: torch.Tensor,
        target_pages: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.get_global_buffer(layer_id)
        cache_buffer, cache_tags = self.pool_cache.storage_for(layer_id)
        materialize_indexer_pages_triton(
            cache_buffer,
            source,
            source_pages,
            target_pages,
            seq_len,
            page_size=self.pool.page_size,
            tags=cache_tags,
            epoch=self.pool_cache.epoch_tensor,
        )
        return cache_buffer, target_pages

    def invalidate_cache(self) -> None:
        self.pool_cache.invalidate()

    def snapshot_move(
        self, tgt_loc: torch.Tensor, src_loc: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        owned = self.pool.index_layout.owned_row_mask(
            torch.div(tgt_loc, self.pool.page_size, rounding_mode="floor"),
            rank=self.pool.shared_rank,
        )
        owned_targets = tgt_loc[owned]
        global_sources = self.pool.translate_index_slots(src_loc[owned])
        snapshots = [
            gather_shared_index_rows(self.pool, buffer, global_sources)
            for buffer in self.buffer
        ]
        return owned_targets, snapshots

    def restore_move(
        self,
        owned_targets: torch.Tensor,
        snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for buffer, (k_bytes, scale_bytes) in zip(self.local_buffer, snapshots):
            index_buf_accessor.SetKAndS.execute(
                pool=self.pool,
                buf=buffer,
                loc=owned_targets,
                index_k=k_bytes.view(torch.float8_e4m3fn),
                index_k_scale=scale_bytes.view(torch.float32),
                owner_rank=self.pool.shared_rank,
                owner_size=self.pool.shared_size,
            )

    def cpu_copy(self, indices: torch.Tensor):
        """Copy logical Indexer pages through the rank-relative VMM view."""
        page_indices = torch.unique(
            torch.div(indices, self.pool.page_size, rounding_mode="floor"),
            sorted=True,
        )
        translated_pages = self.pool.translate_index_pages(page_indices)
        page_chunk_size = max(
            1, self.pool.cpu_offloading_chunk_size // self.pool.page_size
        )
        torch.cuda.synchronize()
        index_k_cpu = []
        for buffer in self.buffer:
            layer_copy = []
            for i in range(0, len(translated_pages), page_chunk_size):
                layer_copy.append(
                    buffer[translated_pages[i : i + page_chunk_size]].to(
                        "cpu", non_blocking=True
                    )
                )
            index_k_cpu.append(layer_copy)
        torch.cuda.synchronize()
        return index_k_cpu

    def load_cpu_copy(self, index_k_cpu, indices: torch.Tensor) -> None:
        """Restore only the Indexer pages physically owned by this rank."""
        page_indices = torch.unique(
            torch.div(indices, self.pool.page_size, rounding_mode="floor"),
            sorted=True,
        )
        page_chunk_size = max(
            1, self.pool.cpu_offloading_chunk_size // self.pool.page_size
        )
        torch.cuda.synchronize()
        for layer_id, local_buffer in enumerate(self.local_buffer):
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_pages = page_indices[i : i + page_chunk_size]
                owned = self.layout.owned_row_mask(
                    chunk_pages, rank=self.pool.shared_rank
                )
                if not bool(owned.any().item()):
                    continue
                local_pages = self.layout.owner_local_rows(chunk_pages[owned])
                cpu_chunk = index_k_cpu[layer_id][i // page_chunk_size]
                local_buffer[local_pages] = cpu_chunk[owned.cpu()].to(
                    local_buffer.device, non_blocking=True
                )
        torch.cuda.synchronize()

    def state_buf_infos(self):
        data_ptrs = [buf.data_ptr() for buf in self.local_buffer]
        data_lens = [buf.nbytes for buf in self.local_buffer]
        item_lens = [buf[0].nbytes for buf in self.local_buffer]
        return data_ptrs, data_lens, item_lens


class ReplicatedIndexKeyCache(IndexKeyCache):
    """Full local Indexer storage used with owner-sharded Main KV."""

    def snapshot_move(
        self, tgt_loc: torch.Tensor, src_loc: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        snapshots = [
            gather_shared_index_rows(self.pool, buffer, src_loc)
            for buffer in self.buffer
        ]
        return tgt_loc, snapshots

    def restore_move(
        self,
        targets: torch.Tensor,
        snapshots: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        for buffer, (k_bytes, scale_bytes) in zip(self.buffer, snapshots):
            index_buf_accessor.SetKAndS.execute(
                pool=self.pool,
                buf=buffer,
                loc=targets,
                index_k=k_bytes.view(torch.float8_e4m3fn),
                index_k_scale=scale_bytes.view(torch.float32),
            )


class SharedDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        indexer_cache_layer_ids: tuple[int, ...],
        share_indexer: bool = True,
        **kwargs,
    ):
        assert shared_size > 1
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.share_indexer = share_indexer
        self.indexer_cache_layer_ids = indexer_cache_layer_ids
        self.shared_cp_group = None
        self.main_layout: Optional[SharedDSAPageLayout] = None
        self.main_family: Optional[OwnerShardedFamily] = None
        self.index_layout: Optional[OwnerShardedLayout] = None
        self.local_kv_buffer: list[torch.Tensor] = []
        self.shared_write_publisher: Optional[SharedWritePublisher] = None
        self.shared_cache_access = None
        super().__init__(*args, **kwargs)
        from sglang.srt.layers.attention.dsa.shared_cache_access import (
            DSASharedCacheAccess,
        )

        self.shared_cache_access = DSASharedCacheAccess(self)

    def _get_cp_group(self):
        if self.shared_cp_group is None:
            from sglang.srt.runtime_context import get_parallel

            self.shared_cp_group = get_parallel().attn_cp_group
        return self.shared_cp_group

    def _create_buffers(self) -> None:
        logical_pages = (self.size + 2 * self.page_size - 1) // self.page_size
        requested_pages = (logical_pages + self.shared_size - 1) // self.shared_size + 1
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.main_family = OwnerShardedFamily.create(
                spec=OwnerShardedFamilySpec(
                    name="dsa_main_kv",
                    num_layers=self.layer_num,
                    logical_rows_per_layer=(
                        requested_pages * self.shared_size * self.page_size
                    ),
                    ownership_granule=self.page_size,
                    storage_rows_per_granule=self.page_size,
                    row_shape=(1, self.kv_cache_dim),
                    dtype=self.store_dtype,
                    map_rank_local=False,
                ),
                cp_size=self.shared_size,
                cpu_group=self._get_cp_group().cpu_group,
                zero_initialize=False,
            )
        self.kv_buffer = self.main_family.slab.global_views
        self.local_kv_buffer = self.main_family.slab.local_views
        self.main_layout = SharedDSAPageLayout(
            self.main_family.layout,
            local_pages_per_layer=requested_pages,
        )
        self.shared_write_publisher = SharedWritePublisher(self._get_cp_group())
        logger.info(
            "DSA shared Main KV: rank=%s size=%s pages_per_layer=%s "
            "rank_stride_pages=%s",
            self.shared_rank,
            self.shared_size,
            requested_pages,
            self.main_layout.pages_per_rank,
        )

    def _create_index_key_cache(self) -> SharedIndexKeyCache | ReplicatedIndexKeyCache:
        if not getattr(self, "share_indexer", True):
            logger.info("DSA shared Main KV with replicated Indexer")
            return ReplicatedIndexKeyCache(self, self.index_buf_size)
        cache = SharedIndexKeyCache(self, self.index_buf_size)
        self.index_layout = cache.layout
        logger.info(
            "DSA shared Indexer: rank=%s size=%s pages_per_layer=%s "
            "rank_stride_pages=%s",
            self.shared_rank,
            self.shared_size,
            cache.layout.minimum_blocks_per_rank,
            cache.layout.blocks_per_rank,
        )
        return cache

    def _clear_buffers(self) -> None:
        publisher = self.shared_write_publisher
        main_family = self.main_family
        index_key_cache = getattr(self, "index_key_cache", None)
        self.kv_buffer = []
        self.local_kv_buffer = []
        self.shared_write_publisher = None
        self.main_family = None
        self.index_layout = None
        self.index_key_cache = None
        self.shared_cache_access = None
        if publisher is not None:
            publisher.close()
        if main_family is not None:
            main_family.close()
        if index_key_cache is not None:
            index_key_cache.clear()

    def translate_index_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "share_indexer", True):
            return page_indices
        assert self.index_layout is not None
        return self.index_layout.rank_relative_rows(page_indices, rank=self.shared_rank)

    def translate_index_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "share_indexer", True):
            return slot_indices
        assert self.index_layout is not None
        pages = torch.div(slot_indices, self.page_size, rounding_mode="floor")
        offsets = slot_indices.remainder(self.page_size)
        return torch.where(
            slot_indices >= 0,
            self.index_layout.rank_relative_rows(pages, rank=self.shared_rank)
            * self.page_size
            + offsets,
            slot_indices,
        )

    def prepare_paged_index_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "share_indexer", True):
            return page_table
        assert self.index_layout is not None
        return self.index_layout.rank_relative_rows(
            page_table, rank=self.shared_rank
        ).to(torch.int32)

    def translate_main_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        assert self.main_layout is not None
        return self.main_layout.translate_slots(slot_indices)

    def synchronize_shared_writes(self) -> None:
        assert self.shared_write_publisher is not None
        self.shared_write_publisher.publish()

    def synchronize_shared_status(self, local_success: bool) -> bool:
        assert self.shared_write_publisher is not None
        return self.shared_write_publisher.publish_status(local_success)

    def get_cpu_copy(self, indices, mamba_indices=None):
        """Copy logical Main and Indexer rows through their shared views."""
        del mamba_indices
        self.synchronize_shared_writes()
        translated = self.translate_main_slots(indices)
        chunk_size = self.cpu_offloading_chunk_size
        torch.cuda.synchronize()
        kv_cache_cpu = []
        for buffer in self.kv_buffer:
            layer_copy = []
            for i in range(0, len(translated), chunk_size):
                layer_copy.append(
                    buffer[translated[i : i + chunk_size]].to("cpu", non_blocking=True)
                )
            kv_cache_cpu.append(layer_copy)
        torch.cuda.synchronize()
        return {
            "kv": kv_cache_cpu,
            "index_k": self.index_key_cache.cpu_copy(indices),
        }

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        """Restore owner-local rows, then publish them to peer VMM readers."""
        del mamba_indices
        assert self.main_layout is not None
        chunk_size = self.cpu_offloading_chunk_size
        torch.cuda.synchronize()
        for layer_id, local_buffer in enumerate(self.local_kv_buffer):
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                owned = self.main_layout.owned_slot_mask(
                    chunk_indices, owner_rank=self.shared_rank
                )
                if not bool(owned.any().item()):
                    continue
                local_rows = self.main_layout.translate_local_slots(
                    chunk_indices[owned]
                )
                cpu_chunk = kv_cache_cpu_dict["kv"][layer_id][i // chunk_size]
                local_buffer[local_rows] = cpu_chunk[owned.cpu()].to(
                    local_buffer.device, non_blocking=True
                )
        self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)
        self.synchronize_shared_writes()

    def get_kv_size_bytes(self) -> int:
        assert self.main_family is not None
        assert self.shared_write_publisher is not None
        if not getattr(self, "share_indexer", True):
            return (
                self.main_family.accounting().mapped_bytes_per_rank
                + sum(buffer.nbytes for buffer in self.index_key_cache.buffer)
                + self.shared_write_publisher.mapped_bytes_per_rank
            )
        return (
            self.main_family.accounting().mapped_bytes_per_rank
            + self.index_key_cache.family.accounting().mapped_bytes_per_rank
            + self.index_key_cache.pool_cache.allocated_bytes
            + self.shared_write_publisher.mapped_bytes_per_rank
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)[..., : self.kv_lora_rank]

    def _write_owned_mla_kv_buffer(
        self,
        kv_buffer: torch.Tensor,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        set_mla_kv_buffer_owner_triton(
            kv_buffer,
            loc,
            cache_k_nope,
            cache_k_rope,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
            page_size=self.page_size,
        )

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        maybe_detect_oob(
            loc, 0, self.size + self.page_size, "set_mla_kv_buffer (DSA shared)"
        )
        self._write_mla_kv_buffer(
            self.local_kv_buffer[layer.layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
            write_fn=self._write_owned_mla_kv_buffer,
        )

    def set_mla_kv_buffer_with_current_rows(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
        current_rows: SharedMLACurrentRows,
        *,
        query_rows: int,
        rows_per_request: int,
    ) -> None:
        if not self.dsa_kv_cache_store_fp8:
            raise ValueError("Shared MLA current rows require scaled-FP8 Main KV")
        if query_rows != loc.numel():
            raise ValueError(
                "Shared MLA current-row query count must match locations: "
                f"query_rows={query_rows} locs={loc.numel()}"
            )
        expected_nope_shape = (query_rows, 1, 512)
        expected_rope_shape = (query_rows, 1, 64)
        if tuple(cache_k_nope.shape) != expected_nope_shape:
            raise ValueError(
                "Shared MLA current-row NoPE shape must be "
                f"{expected_nope_shape}, got {tuple(cache_k_nope.shape)}"
            )
        if tuple(cache_k_rope.shape) != expected_rope_shape:
            raise ValueError(
                "Shared MLA current-row RoPE shape must be "
                f"{expected_rope_shape}, got {tuple(cache_k_rope.shape)}"
            )
        if rows_per_request <= 0 or query_rows % rows_per_request:
            raise ValueError(
                "Shared MLA current rows require complete request groups: "
                f"query_rows={query_rows} rows_per_request={rows_per_request}"
            )
        if query_rows > current_rows.encoded_rows.shape[0]:
            raise ValueError("Shared MLA current-row query capacity exceeded")
        if rows_per_request > current_rows.encoded_rows.shape[1]:
            raise ValueError("Shared MLA per-request row capacity exceeded")
        assert self.main_layout is not None

        maybe_detect_oob(
            loc,
            0,
            self.size + self.page_size,
            "set_mla_kv_buffer_with_current_rows (DSA shared)",
        )
        cache_k_nope_fp8, cache_k_rope_fp8 = quantize_k_cache_separate(
            cache_k_nope, cache_k_rope
        )
        set_mla_kv_buffer_owner_and_current_triton(
            self.local_kv_buffer[layer.layer_id - self.start_layer],
            current_rows.encoded_rows,
            current_rows.physical_rows,
            current_rows.counts,
            loc,
            cache_k_nope_fp8,
            cache_k_rope_fp8,
            rows_per_request=rows_per_request,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
            page_size=self.page_size,
            pages_per_rank=self.main_layout.pages_per_rank,
        )

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")
        if tgt_loc.numel() == 0:
            return
        assert self.main_layout is not None
        owned = self.main_layout.owned_slot_mask(tgt_loc, owner_rank=self.shared_rank)
        local_targets = self.main_layout.translate_local_slots(tgt_loc[owned])
        shared_sources = self.main_layout.translate_slots(src_loc[owned])
        main_snapshots = [
            shared_kv[shared_sources].clone() for shared_kv in self.kv_buffer
        ]
        index_targets, index_snapshots = self.index_key_cache.snapshot_move(
            tgt_loc, src_loc
        )

        # EAGLE compaction can form a cross-owner swap or chain. Every rank must
        # finish reading its remote sources before any owner overwrites a target
        # that is another rank's source.
        self.synchronize_shared_writes()

        for local_kv, snapshot in zip(self.local_kv_buffer, main_snapshots):
            local_kv[local_targets] = snapshot
        self.index_key_cache.restore_move(index_targets, index_snapshots)

        # Accepted-token relocation writes each destination only on its owner.
        # Publish the completed Main-K and Indexer copies before a peer reads them.
        self.synchronize_shared_writes()

    def get_index_k_write_targets(
        self, layer_id: int
    ) -> tuple[tuple[torch.Tensor, int, int], ...]:
        if not getattr(self, "share_indexer", True):
            return ((self.index_key_cache.get_local_buffer(layer_id), 0, 1),)
        return (
            (
                self.index_key_cache.get_local_buffer(layer_id),
                self.shared_rank,
                self.shared_size,
            ),
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_key_cache.get_local_buffer(layer_id)

    def get_paged_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if not getattr(self, "share_indexer", True):
            return self.index_key_cache.get_local_buffer(layer_id)
        return self.index_key_cache.get_global_buffer(layer_id)

    def materialize_index_pages(
        self,
        layer_id: int,
        source_pages: torch.Tensor,
        target_pages: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not getattr(self, "share_indexer", True):
            raise RuntimeError("Indexer Page Demand requires Shared Indexer storage")
        return self.index_key_cache.materialize_pages(
            layer_id, source_pages, target_pages, seq_len
        )

    def invalidate_indexer_cache(self) -> None:
        if getattr(self, "share_indexer", True):
            self.index_key_cache.invalidate_cache()

    def get_index_k_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_paged_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self,
            buffer,
            seq_len=seq_len,
            page_indices=self.translate_index_pages(page_indices),
        )

    def get_index_k_scale_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_paged_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetS.execute(
            self,
            buffer,
            seq_len=seq_len,
            page_indices=self.translate_index_pages(page_indices),
        )

    def get_index_k_scale_buffer(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buffer = self.get_paged_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self,
            buffer,
            page_indices=self.translate_index_pages(page_indices),
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        owner_rank = self.shared_rank if self.share_indexer else 0
        owner_size = self.shared_size if self.share_indexer else 1
        index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.index_key_cache.get_local_buffer(layer_id),
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            owner_rank=owner_rank,
            owner_size=owner_size,
        )

    def get_state_buf_infos(self):
        return self.index_key_cache.state_buf_infos()
