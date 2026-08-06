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

"""Layer-sharded DSA KV cache pool for context-parallel prefill.

``LayerSplitDSATokenToKVPool`` splits the DSA (DeepSeek Sparse Attention) GPU
KV/indexer cache layers across context-parallel (CP) ranks so that each rank
only materializes the layers it owns, reducing per-rank KV memory. When a rank
needs to read a layer it does not own, the owning rank broadcasts that layer's
buffer into a small per-rank remote scratch buffer.

This subclass keeps the core ``KVCache`` / ``MLATokenToKVPool`` /
``DSATokenToKVPool`` pools untouched: all sharding, broadcast, and remote-scratch
bookkeeping lives here. Layer split is only ever enabled for DSA MLA models on
PD prefill workers under prefill-CP (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.attention.dsa import index_buf_accessor
from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.mem_cache.memory_pool import (
    GPU_MEMORY_TYPE_KV_CACHE,
    DSATokenToKVPool,
    RadixAttention,
    get_tensor_size_bytes,
    maybe_detect_oob,
    unwrap_write_loc,
)
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


class LayerSplitIndexKeyCache(IndexKeyCache):
    def __init__(self, pool: LayerSplitDSATokenToKVPool, index_buf_size: int):
        super().__init__(pool, index_buf_size)
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.custom_mem_pool
            else nullcontext()
        ):
            self.remote_buffer = torch.empty(
                self._buffer_shape(num_pages),
                dtype=pool.index_k_with_scale_buffer_dtype,
                device=pool.device,
            )
        self.remote_layer_id: Optional[int] = None

    def _layer_num_pages(self, layer_idx: int, num_pages: int) -> int:
        layer_id = self.pool.start_layer + layer_idx
        return num_pages if self.pool._is_layer_owned(layer_id) else 0

    def clear(self) -> None:
        super().clear()
        del self.remote_buffer

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for index_k in self.buffer:
            if index_k.shape[0] != 0:
                index_k[tgt_loc_flat] = index_k[src_loc_flat]

    def get_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pool.layer_transfer_counter is not None:
            self.pool.layer_transfer_counter.wait_until(
                layer_id - self.pool.start_layer
            )
        return self.get_broadcastable_buffer(layer_id)

    def get_k_and_scale(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buf = self.get_buffer(layer_id)
        self.pool.prefetch_kv_buffer(layer_id)
        return index_buf_accessor.GetKAndS.execute(
            self.pool,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )

    def store_quantized(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        self.invalidate(layer_id)
        if self.pool._is_layer_owned(layer_id):
            super().store_quantized(layer_id, loc, index_k, index_k_scale)

    def invalidate(self, layer_id: int) -> None:
        if self.remote_layer_id == layer_id:
            self.remote_layer_id = None

    def get_broadcastable_buffer(self, layer_id: int) -> torch.Tensor:
        if self.remote_layer_id != layer_id:
            local_idx = layer_id - self.pool.start_layer
            src_tensor = (
                self.buffer[local_idx] if self.pool._is_layer_owned(layer_id) else None
            )
            self.pool._broadcast_tensor_from_owner(
                self.remote_buffer,
                layer_id,
                src_tensor=src_tensor,
            )
            self.remote_layer_id = layer_id
        return self.remote_buffer

    def state_buf_infos(self):
        owned_layer_ids = [
            i
            for i in range(self.pool.layer_num)
            if self.pool._is_layer_owned(self.pool.start_layer + i)
        ]
        data_ptrs = [self.buffer[i].data_ptr() for i in owned_layer_ids]
        data_lens = [self.buffer[i].nbytes for i in owned_layer_ids]
        item_lens = [self.buffer[i][0].nbytes for i in owned_layer_ids]
        return data_ptrs, data_lens, item_lens

    def cpu_copy(self, indices):
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.layer_num):
            index_k_cpu.append([])
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.buffer[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()
        return index_k_cpu

    def load_cpu_copy(self, index_k_cpu, indices) -> None:
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.layer_num):
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = index_k_cpu[layer_id][i // page_chunk_size]
                assert idx_cpu.shape[0] == len(chunk_page_indices)
                idx_chunk = idx_cpu.to(self.buffer[layer_id].device, non_blocking=True)
                self.buffer[layer_id][chunk_page_indices] = idx_chunk
        torch.cuda.synchronize()


class LayerSplitDSATokenToKVPool(DSATokenToKVPool):
    """DSA KV pool that shards layers across CP ranks with owner-broadcast reads."""

    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        **kwargs,
    ):
        assert (
            layer_shard_rank is not None and layer_shard_size > 1
        ), "LayerSplitDSATokenToKVPool requires layer_shard_size > 1"
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        self.layer_broadcast_comm = None
        super().__init__(*args, **kwargs)
        # First global layer index owned by this rank (used by PD transfer to
        # label the contiguous owned-buffer range).
        my_start, _ = self._owned_local_layer_range()
        self.layer_shard_start = self.start_layer + my_start

    # ---- layer ownership helpers ------------------------------------------

    def _local_layer_idx(self, layer_id: int) -> int:
        return layer_id - self.start_layer

    def _owned_local_layer_range(self) -> tuple[int, int]:
        return get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        local_idx = self._local_layer_idx(layer_id)
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_idx < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            self._local_layer_idx(layer_id), self.layer_shard_size, self.layer_num
        )

    def _log_layer_shard_plan(self) -> None:
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)
            partitions.append(f"r{rank}:[{st},{ed})")
        my_start, my_end = self._owned_local_layer_range()
        logger.info(
            "Layer shard plan (continuous): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, local=[{my_start},{my_end}), "
            f"global=[{self.start_layer + my_start},{self.start_layer + my_end}), "
            f"partitions={'; '.join(partitions)}"
        )

    # ---- broadcast plumbing -----------------------------------------------

    def _init_layer_broadcast_comm(self) -> None:
        cp_group = get_parallel().attn_cp_group
        if cp_group.world_size <= 1 or cp_group.pynccl_comm is None:
            return

        from sglang.srt.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        self.layer_broadcast_comm = PyNcclCommunicator(
            group=cp_group.cpu_group,
            device=cp_group.device,
        )
        logger.info(
            "Initialized dedicated layer-shard broadcast NCCL communicator: "
            f"rank={cp_group.rank_in_group}, world_size={cp_group.world_size}"
        )

    def _broadcast_tensor_from_owner(
        self,
        tensor: torch.Tensor,
        layer_id: int,
        src_tensor: Optional[torch.Tensor] = None,
        use_layer_broadcast_comm: bool = False,
    ) -> torch.Tensor:
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_tensor is not None
            if tensor.data_ptr() != src_tensor.data_ptr():
                tensor.copy_(src_tensor)

        cp_group = get_parallel().attn_cp_group
        comm = (
            self.layer_broadcast_comm
            if use_layer_broadcast_comm and self.layer_broadcast_comm is not None
            else cp_group.pynccl_comm
        )
        if comm is not None:
            # PyNcclCommunicator defaults to disabled=True (it is only enabled
            # inside CUDA-graph capture via change_state). Without re-enabling it
            # here, comm.broadcast() is a silent no-op and non-owner CP ranks read
            # stale remote buffers, corrupting layer-split attention. Mirror the
            # standard usage in parallel_state.py.
            with comm.change_state(enable=True):
                comm.broadcast(tensor, src=owner_rank)
        else:
            torch.distributed.broadcast(
                tensor, src=owner_rank, group=cp_group.cpu_group
            )
        return tensor

    # ---- buffer allocation (owned-only + remote scratch) ------------------

    def _create_buffers(self):
        self._log_layer_shard_plan()
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                # Owned layers get the full buffer; non-owned layers allocate a
                # 0-row placeholder so ``kv_buffer`` stays index-aligned by layer.
                self.kv_buffer = [
                    torch.zeros(
                        (
                            (
                                (self.size + self.page_size)
                                if self._is_layer_owned(self.start_layer + i)
                                else 0
                            ),
                            1,
                            self.kv_cache_dim,
                        ),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    for i in range(self.layer_num)
                ]
                self.remote_kv_buffer = torch.empty(
                    (self.size + self.page_size, 1, self.kv_cache_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                self.remote_kv_layer_id: Optional[int] = None
                self.device_module = torch.get_device_module(self.device)
                self.kv_broadcast_stream = self.device_module.Stream()
                self.pending_remote_kv_layer_id: Optional[int] = None
                self.pending_remote_kv_broadcast = False
        self._init_layer_broadcast_comm()

    def _create_index_key_cache(self) -> IndexKeyCache:
        return LayerSplitIndexKeyCache(self, self.index_buf_size)

    def _clear_buffers(self):
        del self.kv_buffer
        del self.remote_kv_buffer
        self.index_key_cache.clear()

    # ---- MLA latent KV: owned-only writes, owner-broadcast reads ----------

    def get_kv_size_bytes(self):
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        for index_k_cache in self.index_k_with_scale_buffer:
            kv_size_bytes += get_tensor_size_bytes(index_k_cache)
        return kv_size_bytes

    def get_contiguous_buf_infos(self):
        # Only report buffers owned by the current CP rank; non-owned layers
        # are empty and are pulled from their owner via PD transfer.
        owned_layer_ids = [
            i
            for i in range(self.layer_num)
            if self._is_layer_owned(self.start_layer + i)
        ]
        kv_data_ptrs = [self.kv_buffer[i].data_ptr() for i in owned_layer_ids]
        kv_data_lens = [self.kv_buffer[i].nbytes for i in owned_layer_ids]
        kv_item_lens = [
            self.kv_buffer[i][0].nbytes * self.page_size for i in owned_layer_ids
        ]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)
        return kv_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.store_dtype != self.dtype:
            return kv_buffer[..., : self.kv_lora_rank].view(self.dtype)
        return kv_buffer[..., : self.kv_lora_rank]

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_kv_buffer (MLA)")
        layer_id = layer.layer_id
        assert not self.dsa_kv_cache_store_fp8
        # A write invalidates any cached remote copy for this layer.
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None
        if not self._is_layer_owned(layer_id):
            return
        if cache_k.dtype != self.dtype:
            cache_k = cache_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k.view(
                self.store_dtype
            )
        else:
            self.kv_buffer[layer_id - self.start_layer][loc] = cache_k

    def set_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc: torch.Tensor,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=True)
        remote_kv_updatable = self.remote_kv_layer_id == layer_id
        if remote_kv_updatable:
            self._write_mla_kv_buffer(
                self.remote_kv_buffer, loc, cache_k_nope, cache_k_rope
            )
        if not self._is_layer_owned(layer_id):
            return
        self._write_mla_kv_buffer(
            self.kv_buffer[layer_id - self.start_layer],
            loc,
            cache_k_nope,
            cache_k_rope,
        )
        if not remote_kv_updatable and self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None

    def _finalize_pending_kv_broadcast(
        self, *, set_remote_layer_id: bool = True
    ) -> None:
        if not self.pending_remote_kv_broadcast:
            return
        self.device_module.current_stream().wait_stream(self.kv_broadcast_stream)
        self.pending_remote_kv_broadcast = False
        if set_remote_layer_id and self.pending_remote_kv_layer_id is not None:
            self.remote_kv_layer_id = self.pending_remote_kv_layer_id
        self.pending_remote_kv_layer_id = None

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
    ) -> None:
        """Kick off an async owner-broadcast of ``layer_id``'s latent KV.

        Called ahead of the layer's attention so the remote scratch buffer is
        ready by the time a non-owner rank reads it (see the prefetch wiring in
        ``DeepseekV2DecoderLayer``).
        """
        if self.remote_kv_layer_id == layer_id:
            return
        if self.pending_remote_kv_broadcast:
            if self.pending_remote_kv_layer_id == layer_id:
                return
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)

        local_idx = self._local_layer_idx(layer_id)
        src_tensor = (
            self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
        )
        if self.layer_broadcast_comm is None:
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_kv_layer_id = layer_id
            return

        self.kv_broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_broadcast_stream):
            if layer_transfer_counter is not None and layer_transfer_idx is not None:
                layer_transfer_counter.wait_until(layer_transfer_idx)
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
        self.pending_remote_kv_layer_id = layer_id
        self.pending_remote_kv_broadcast = True

    def _get_broadcastable_kv_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pending_remote_kv_broadcast:
            self._finalize_pending_kv_broadcast(
                set_remote_layer_id=self.pending_remote_kv_layer_id == layer_id
            )
        if self.remote_kv_layer_id != layer_id:
            local_idx = self._local_layer_idx(layer_id)
            src_tensor = (
                self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
            )
            self._broadcast_tensor_from_owner(
                self.remote_kv_buffer,
                layer_id,
                src_tensor=src_tensor,
                use_layer_broadcast_comm=True,
            )
            self.remote_kv_layer_id = layer_id
        return self.remote_kv_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")
        if tgt_loc.numel() == 0:
            return
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        for kv_cache in self.kv_buffer:
            if kv_cache.shape[0] == 0:
                continue
            kv_cache[tgt_loc_flat] = kv_cache[src_loc_flat]
        self.index_key_cache.move(tgt_loc, src_loc)

    # ---- DSA indexer buffer: owned-only writes, owner-broadcast reads -----

    def get_broadcastable_index_k_with_scale_buffer(
        self, layer_id: int
    ) -> torch.Tensor:
        return self.index_key_cache.get_buffer(layer_id)

    def invalidate_index_buffer_for_layer(self, layer_id: int) -> None:
        self.index_key_cache.invalidate(layer_id)

    def _get_broadcastable_index_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_key_cache.get_broadcastable_buffer(layer_id)

    # ---- HiCache CPU offload: skip empty (non-owned) layers ---------------

    def get_cpu_copy(self, indices, mamba_indices=None):
        from sglang.srt.utils import current_platform

        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()

        return {"kv": kv_cache_cpu, "index_k": self.index_key_cache.cpu_copy(indices)}

    def load_cpu_copy(self, kv_cache_cpu_dict, indices, mamba_indices=None):
        from sglang.srt.utils import current_platform

        kv_cache_cpu = kv_cache_cpu_dict["kv"]
        current_platform.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            if self.kv_buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = kv_cache_cpu[layer_id][i // chunk_size]
                assert kv_cpu.shape[0] == len(chunk_indices)
                kv_chunk = kv_cpu.to(self.kv_buffer[layer_id].device, non_blocking=True)
                self.kv_buffer[layer_id][chunk_indices] = kv_chunk
        current_platform.synchronize()

        self.index_key_cache.load_cpu_copy(kv_cache_cpu_dict["index_k"], indices)
