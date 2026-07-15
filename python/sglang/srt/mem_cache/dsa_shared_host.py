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
# ==============================================================================

from __future__ import annotations

import logging
import threading

import torch
import torch.distributed as dist

from sglang.srt.distributed.device_communicators.vmm_utils import all_ranks_ok
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DSAIndexerPoolHost,
    HostPoolGroup,
    PoolEntry,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

logger = logging.getLogger(__name__)


class _LocalDSADevicePool:
    def __init__(self, pool):
        self._pool = pool

    @property
    def kv_buffer(self):
        return self._pool.local_kv_buffer

    @property
    def index_k_with_scale_buffer(self):
        return self._pool.local_index_k_with_scale_buffer

    @property
    def size(self):
        return self.kv_buffer[0].shape[0]

    def __getattr__(self, name):
        return getattr(self._pool, name)


class _SharedDSAHostPageAllocator:
    def __init__(self, local_page_num: int, page_size: int, cp_size: int):
        self.local_page_num = local_page_num
        self.page_size = page_size
        self.cp_size = cp_size
        self.global_page_num = local_page_num * cp_size
        self._lock = threading.RLock()
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self.free_pages = torch.arange(self.global_page_num, dtype=torch.int64)
            self.page_used = torch.zeros(self.global_page_num, dtype=torch.bool)

    def available_size(self) -> int:
        with self._lock:
            return self.free_pages.numel() * self.page_size

    def alloc(self, need_size: int) -> torch.Tensor | None:
        assert need_size % self.page_size == 0
        with self._lock:
            page_num = need_size // self.page_size
            if page_num > self.free_pages.numel():
                return None
            pages = self.free_pages[:page_num]
            self.free_pages = self.free_pages[page_num:]
            self.page_used[pages] = True
            return self._expand_pages(pages)

    def allocate_for_device(self, device_indices: torch.Tensor) -> torch.Tensor | None:
        assert device_indices.numel() % self.page_size == 0
        device_pages = (
            device_indices.reshape(-1, self.page_size)[:, 0].cpu() // self.page_size
        )
        owners = device_pages % self.cp_size
        with self._lock:
            free_owners = self.free_pages % self.cp_size
            selected = torch.empty_like(owners)
            selected_mask = torch.zeros_like(self.free_pages, dtype=torch.bool)
            for owner in range(self.cp_size):
                request_pos = torch.nonzero(owners == owner, as_tuple=True)[0]
                free_pos = torch.nonzero(free_owners == owner, as_tuple=True)[0]
                if request_pos.numel() > free_pos.numel():
                    return None
                chosen_pos = free_pos[: request_pos.numel()]
                selected[request_pos] = self.free_pages[chosen_pos]
                selected_mask[chosen_pos] = True
            self.free_pages = self.free_pages[~selected_mask]
            self.page_used[selected] = True
            return self._expand_pages(selected)

    def free(self, indices: torch.Tensor) -> int:
        pages = torch.unique(indices.cpu() // self.page_size)
        with self._lock:
            assert self.page_used[pages].all()
            self.page_used[pages] = False
            self.free_pages = torch.cat((self.free_pages, pages))
        return pages.numel() * self.page_size

    def _expand_pages(self, pages: torch.Tensor) -> torch.Tensor:
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        return (pages[:, None] * self.page_size + offsets).reshape(-1)


class SharedDSAHostPoolGroup(HostPoolGroup):
    """Transfer DSA Main and Indexer between shared L1 and shared L2."""

    def __init__(
        self,
        entries,
        *,
        shared_rank: int,
        shared_size: int,
        cpu_sync_group,
        device_sync_group,
        sync_device: str,
    ):
        super().__init__(entries)
        self.shared_rank = shared_rank
        self.shared_size = shared_size
        self.cpu_sync_group = cpu_sync_group
        self.device_sync_group = device_sync_group
        self.logical_allocator = _SharedDSAHostPageAllocator(
            local_page_num=self.anchor_entry.host_pool.page_num,
            page_size=self.page_size,
            cp_size=shared_size,
        )
        self.size = self.logical_allocator.global_page_num * self.page_size
        self._sync_flag = torch.ones(1, dtype=torch.int32, device=sync_device)

    def _synchronize(self, error):
        success = all_ranks_ok(self.cpu_sync_group, error is None)
        if not success:
            return False, error
        self._sync_flag.fill_(1)
        dist.all_reduce(self._sync_flag, group=self.device_sync_group)
        return True, error

    def clear(self) -> None:
        self.logical_allocator.clear()

    def available_size(self) -> int:
        return self.logical_allocator.available_size()

    def alloc(self, need_size: int) -> torch.Tensor | None:
        return self.logical_allocator.alloc(need_size)

    def allocate_host_for_device(
        self, device_indices: torch.Tensor
    ) -> torch.Tensor | None:
        return self.logical_allocator.allocate_for_device(device_indices)

    def allocate_device_for_host(
        self, device_allocator, host_indices: torch.Tensor
    ) -> torch.Tensor | None:
        host_pages = host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        return device_allocator.alloc_for_page_owners(
            host_pages % self.shared_size, self.shared_size
        )

    def free(self, indices: torch.Tensor) -> int:
        return self.logical_allocator.free(indices)

    def destroy(self) -> None:
        for entry in self.entries:
            entry.host_pool.destroy()
        dist.destroy_process_group(self.device_sync_group)

    def _local_owner_pairs(self, host_indices, device_indices):
        if host_indices.numel() != device_indices.numel():
            raise ValueError("DSA shared L2 host and device indices must match.")
        host_pages = host_indices.cpu() // self.page_size
        device_pages = device_indices.cpu() // self.page_size
        if not torch.equal(
            host_pages % self.shared_size, device_pages % self.shared_size
        ):
            raise ValueError("DSA shared L2 host and device page owners must match.")
        owned = (device_pages % self.shared_size) == self.shared_rank
        positions = torch.nonzero(owned, as_tuple=True)[0]
        return (
            self._to_local_slots(
                host_indices.index_select(0, positions.to(host_indices.device))
            ),
            self._to_local_slots(
                device_indices.index_select(0, positions.to(device_indices.device))
            ),
        )

    def _to_local_slots(self, indices: torch.Tensor) -> torch.Tensor:
        pages = indices // self.page_size
        offsets = indices % self.page_size
        return (pages // self.shared_size) * self.page_size + offsets

    def backup_from_device_all_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        io_backend,
        pool_transfers=None,
    ) -> None:
        local_pool = _LocalDSADevicePool(device_pool)
        error = None
        try:
            local_host, local_device = self._local_owner_pairs(
                host_indices, device_indices
            )
            if local_host.numel() > 0:
                anchor = self.anchor_entry
                anchor.host_pool.backup_from_device_all_layer(
                    local_pool,
                    local_host,
                    local_device,
                    io_backend,
                )
            for transfer in pool_transfers or []:
                entry = self.entry_map.get(transfer.name)
                if entry is None or transfer.host_indices is None:
                    continue
                local_host, local_device = self._local_owner_pairs(
                    transfer.host_indices, transfer.device_indices
                )
                if local_host.numel() > 0:
                    entry.host_pool.backup_from_device_all_layer(
                        local_pool,
                        local_host,
                        local_device,
                        io_backend,
                    )
        except Exception as exc:
            error = exc

        success, error = self._synchronize(error)
        if not success:
            if error is not None:
                raise error
            raise RuntimeError("A DSA shared L2 rank failed to back up KV cache.")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        pool_transfers=None,
    ) -> None:
        local_pool = _LocalDSADevicePool(device_pool)
        error = None
        try:
            anchor = self.anchor_entry
            local_layer_id = anchor.layer_mapper(layer_id)
            local_host, local_device = self._local_owner_pairs(
                host_indices, device_indices
            )
            if local_layer_id is not None and local_host.numel() > 0:
                anchor.host_pool.load_to_device_per_layer(
                    local_pool,
                    local_host,
                    local_device,
                    local_layer_id,
                    io_backend,
                )

            for transfer in pool_transfers or []:
                entry = self.entry_map.get(transfer.name)
                if entry is None or transfer.host_indices is None:
                    continue
                local_layer_id = entry.layer_mapper(layer_id)
                if local_layer_id is None:
                    continue
                local_host, local_device = self._local_owner_pairs(
                    transfer.host_indices, transfer.device_indices
                )
                if local_host.numel() > 0:
                    entry.host_pool.load_to_device_per_layer(
                        local_pool,
                        local_host,
                        local_device,
                        local_layer_id,
                        io_backend,
                    )
        except Exception as exc:
            error = exc

        success, error = self._synchronize(error)
        if not success:
            if error is not None:
                raise error
            raise RuntimeError("A DSA shared L2 rank failed to restore KV cache.")


def build_shared_dsa_hicache_stack(
    *,
    params,
    server_args,
    kv_pool,
    full_layer_mapping,
    page_size,
    tp_group,
    load_cache_event,
    attn_cp_group,
    attn_tp_group=None,
    pp_group=None,
    storage_backend=None,
    prefetch_threshold=256,
    model_name=None,
    storage_backend_extra_config=None,
    enable_storage_metrics=False,
):
    if attn_cp_group is None:
        raise ValueError("DSA shared HiCache requires an attention CP group.")
    if storage_backend is not None:
        raise ValueError("DSA shared HiCache does not support L3 storage yet.")

    shared_rank = dist.get_rank(attn_cp_group)
    shared_size = dist.get_world_size(attn_cp_group)
    local_pool = _LocalDSADevicePool(kv_pool)
    local_host_size = (
        server_args.hicache_size / shared_size if server_args.hicache_size > 0 else 0
    )
    main_host = MLATokenToKVPoolHost(
        local_pool,
        server_args.hicache_ratio,
        local_host_size,
        page_size,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        override_kv_cache_dim=kv_pool.kv_cache_dim,
    )
    index_host = DSAIndexerPoolHost(
        local_pool,
        main_host,
        server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
    )
    transfer_layer_num = len(full_layer_mapping)

    def layer_mapper(layer_id):
        if not 0 <= layer_id < transfer_layer_num:
            return None
        return full_layer_mapping.get(layer_id)

    entries = [
        PoolEntry(
            name=PoolName.KV,
            host_pool=main_host,
            device_pool=kv_pool,
            layer_mapper=layer_mapper,
            is_primary_index_anchor=True,
        ),
        PoolEntry(
            name=PoolName.INDEXER,
            host_pool=index_host,
            device_pool=kv_pool,
            layer_mapper=layer_mapper,
        ),
    ]
    device_sync_group = dist.new_group(
        ranks=dist.get_process_group_ranks(attn_cp_group), backend="nccl"
    )
    host_group = SharedDSAHostPoolGroup(
        entries,
        shared_rank=shared_rank,
        shared_size=shared_size,
        cpu_sync_group=attn_cp_group,
        device_sync_group=device_sync_group,
        sync_device=kv_pool.device,
    )
    main_bytes = main_host.kv_buffer.nbytes
    index_bytes = index_host.index_k_with_scale_buffer.nbytes
    logger.info(
        "DSA shared L2: rank=%d local_pages=%d global_pages=%d "
        "page_size=%d local_bytes=%.2f GB aggregate_bytes=%.2f GB.",
        shared_rank,
        host_group.logical_allocator.local_page_num,
        host_group.logical_allocator.global_page_num,
        page_size,
        (main_bytes + index_bytes) / 1e9,
        (main_bytes + index_bytes) * shared_size / 1e9,
    )
    controller = HybridCacheController(
        params.token_to_kv_pool_allocator,
        host_group,
        page_size,
        tp_group,
        load_cache_event=load_cache_event,
        attn_cp_group=attn_cp_group,
        attn_tp_group=attn_tp_group,
        pp_group=pp_group,
        write_policy=server_args.hicache_write_policy,
        io_backend=server_args.hicache_io_backend,
        storage_backend=storage_backend,
        prefetch_threshold=prefetch_threshold,
        model_name=model_name,
        storage_backend_extra_config=storage_backend_extra_config,
        transfer_layer_num=transfer_layer_num,
        enable_storage_metrics=enable_storage_metrics,
    )
    return host_group, controller
