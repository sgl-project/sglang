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
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.kernels.ops.kvcache.dsa_shared import set_mla_kv_buffer_owner_triton
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    DSATokenToKVPool,
    RadixAttention,
    maybe_detect_oob,
)
from sglang.srt.mem_cache.shared_kv.family import (
    OwnerShardedFamily,
    OwnerShardedFamilySpec,
)
from sglang.srt.mem_cache.shared_kv.layout import OwnerShardedLayout
from sglang.srt.mem_cache.shared_kv.synchronization import SharedWritePublisher

logger = logging.getLogger(__name__)


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

    @property
    def padding_value(self) -> int:
        return -1

    def translate_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        valid = page_indices >= 0
        translated = self.owner_layout.physical_rows(page_indices * self.page_size)
        return torch.where(valid, translated // self.page_size, page_indices)

    def translate_pages_for_rank(
        self, page_indices: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        valid = page_indices >= 0
        translated = self.owner_layout.rank_relative_rows(
            page_indices * self.page_size, rank=rank
        )
        return torch.where(valid, translated // self.page_size, page_indices)

    def translate_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        return self.owner_layout.physical_rows(slot_indices)

    def translate_slots_for_rank(
        self, slot_indices: torch.Tensor, *, rank: int
    ) -> torch.Tensor:
        return self.owner_layout.rank_relative_rows(slot_indices, rank=rank)

    def translate_local_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        return self.owner_layout.owner_local_rows(slot_indices)

    def owned_slot_mask(
        self, slot_indices: torch.Tensor, *, owner_rank: int
    ) -> torch.Tensor:
        return self.owner_layout.owned_row_mask(slot_indices, rank=owner_rank)


class SharedDSATokenToKVPool(DSATokenToKVPool):
    def __init__(
        self,
        *args,
        shared_rank: int,
        shared_size: int,
        **kwargs,
    ):
        assert shared_size > 1
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.shared_cp_group = None
        self.main_layout: Optional[SharedDSAPageLayout] = None
        self.index_layout: Optional[SharedDSAPageLayout] = None
        self.main_family: Optional[OwnerShardedFamily] = None
        self.index_family: Optional[OwnerShardedFamily] = None
        self.local_kv_buffer: list[torch.Tensor] = []
        self.local_index_k_with_scale_buffer: list[torch.Tensor] = []
        self.rank_local_index_k_with_scale_buffer: list[torch.Tensor] = []
        self.shared_write_publisher: Optional[SharedWritePublisher] = None
        super().__init__(*args, **kwargs)

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

    def _create_index_buffers(self) -> None:
        logical_pages = (self.index_buf_size + self.page_size + 1) // self.page_size
        requested_pages = (logical_pages + self.shared_size - 1) // self.shared_size + 1
        with (
            torch.cuda.use_mem_pool(self.custom_mem_pool)
            if self.custom_mem_pool
            else nullcontext()
        ):
            # DeepGEMM resolves the cache device from its base pointer, so the
            # paged read alias maps this rank's segment first over the same HBM.
            self.index_family = OwnerShardedFamily.create(
                spec=OwnerShardedFamilySpec(
                    name="dsa_indexer",
                    num_layers=self.layer_num,
                    logical_rows_per_layer=(
                        requested_pages * self.shared_size * self.page_size
                    ),
                    ownership_granule=self.page_size,
                    storage_rows_per_granule=1,
                    row_shape=self._index_buffer_shape(1)[1:],
                    dtype=self.index_k_with_scale_buffer_dtype,
                    map_rank_local=True,
                ),
                cp_size=self.shared_size,
                cpu_group=self._get_cp_group().cpu_group,
                zero_initialize=False,
            )
        self.index_k_with_scale_buffer = self.index_family.slab.global_views
        self.local_index_k_with_scale_buffer = self.index_family.slab.local_views
        self.rank_local_index_k_with_scale_buffer = (
            self.index_family.slab.rank_local_views
        )
        self.index_layout = SharedDSAPageLayout(
            self.index_family.layout,
            local_pages_per_layer=requested_pages,
        )
        logger.info(
            "DSA shared Indexer: rank=%s size=%s pages_per_layer=%s "
            "rank_stride_pages=%s",
            self.shared_rank,
            self.shared_size,
            requested_pages,
            self.index_layout.pages_per_rank,
        )

    def _clear_buffers(self) -> None:
        main_family = self.main_family
        index_family = self.index_family
        self.kv_buffer = []
        self.local_kv_buffer = []
        self.index_k_with_scale_buffer = []
        self.local_index_k_with_scale_buffer = []
        self.rank_local_index_k_with_scale_buffer = []
        self.main_family = None
        self.index_family = None
        if main_family is not None:
            main_family.close()
        if index_family is not None:
            index_family.close()
        self.shared_write_publisher = None

    def translate_index_pages(self, page_indices: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_pages(page_indices)

    def translate_index_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_slots(slot_indices)

    def prepare_paged_index_page_table(self, page_table: torch.Tensor) -> torch.Tensor:
        assert self.index_layout is not None
        return self.index_layout.translate_pages_for_rank(
            page_table, rank=self.shared_rank
        ).to(torch.int32)

    def translate_main_slots(self, slot_indices: torch.Tensor) -> torch.Tensor:
        assert self.main_layout is not None
        return self.main_layout.translate_slots(slot_indices)

    def synchronize_shared_writes(self) -> None:
        assert self.shared_write_publisher is not None
        self.shared_write_publisher.publish()

    def get_kv_size_bytes(self) -> int:
        assert self.main_family is not None
        assert self.index_family is not None
        return (
            self.main_family.accounting().mapped_bytes_per_rank
            + self.index_family.accounting().mapped_bytes_per_rank
        )

    def get_contiguous_buf_infos(self):
        buffers = self.local_kv_buffer
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes * self.page_size for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self):
        buffers = self.local_index_k_with_scale_buffer
        data_ptrs = [buf.data_ptr() for buf in buffers]
        data_lens = [buf.nbytes for buf in buffers]
        item_lens = [buf[0].nbytes for buf in buffers]
        return data_ptrs, data_lens, item_lens

    def get_pd_transfer_tensors(self):
        return self.local_kv_buffer, [self.local_index_k_with_scale_buffer]

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
        for local_kv, shared_kv in zip(self.local_kv_buffer, self.kv_buffer):
            local_kv[local_targets] = shared_kv[shared_sources]

        local_pages = local_targets // self.page_size
        shared_pages = shared_sources // self.page_size
        for local_index, shared_index in zip(
            self.local_index_k_with_scale_buffer,
            self.index_k_with_scale_buffer,
        ):
            local_index[local_pages] = shared_index[shared_pages]

    def get_index_k_write_owner(self) -> tuple[int, int]:
        return self.shared_rank, self.shared_size

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.local_index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_broadcastable_index_k_with_scale_buffer(
        self, layer_id: int
    ) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_paged_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.rank_local_index_k_with_scale_buffer[layer_id - self.start_layer]

    def get_index_k_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
        return index_buf_accessor.GetK.execute(
            self,
            buffer,
            seq_len=seq_len,
            page_indices=self.translate_index_pages(page_indices),
        )

    def get_index_k_scale_continuous(
        self, layer_id: int, seq_len: int, page_indices: torch.Tensor
    ):
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
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
        buffer = self.get_broadcastable_index_k_with_scale_buffer(layer_id)
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
        index_buf_accessor.SetKAndS.execute(
            pool=self,
            buf=self.local_index_k_with_scale_buffer[layer_id - self.start_layer],
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
            owner_rank=self.shared_rank,
            owner_size=self.shared_size,
        )
