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
bookkeeping lives here. Unified NVFP4 serving also computes decode on each
layer's owner and transfers only the required pages during eager prefill (see
``sglang.srt.layers.cp.utils.is_glm_dsa_cache_layer_split_enabled``).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.kernels.ops.kvcache import dsa_index_k_cache
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

PREFILL_TRANSFER_MAX_PAGES = 1024


def get_dsa_prefill_transfer_workspace_bytes(size: int, page_size: int = 64) -> int:
    """Exact target-pool workspace for unified NVFP4, without index pooling."""
    assert page_size == 64 and size >= 0 and size % page_size == 0
    num_pages = size // page_size + 1
    chunk_pages = min(PREFILL_TRANSFER_MAX_PAGES, num_pages)
    return chunk_pages * page_size * (416 + 132) + num_pages * 8


def _collect_prefill_page_ids(
    full_page_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    extend_seq_lens_cpu: list[int],
    out_cache_loc: torch.Tensor,
    staging: torch.Tensor,
    page_size: int,
    num_pages: int,
) -> torch.Tensor:
    """Collect valid whole-batch pages on CPU, using bounded device scratch.

    Only CP rank zero calls this. The source table precedes CP request filtering.
    CPU lengths determine every copy shape; neither unique nor nonzero runs on
    the GPU. Reuse the idle main-transfer stage instead of allocating a second
    device page table or a data-dependent GPU sorting workspace.
    """
    lengths = seq_lens_cpu.tolist()
    assert len(lengths) == len(extend_seq_lens_cpu) == full_page_table.shape[0]
    assert full_page_table.dtype in (torch.int32, torch.int64)
    if any(
        extend < 0 or (extend > 0 and not extend <= length <= full_page_table.shape[1])
        for length, extend in zip(lengths, extend_seq_lens_cpu)
    ):
        raise ValueError("Invalid sequence length in prefill page plan")
    seen = torch.zeros(num_pages, dtype=torch.bool)

    def add_slots(slots: torch.Tensor) -> None:
        pages = slots.to(torch.int64) // page_size
        if pages.numel() and bool(((pages <= 0) | (pages >= num_pages)).any()):
            raise ValueError("Prefill page plan contains an invalid physical page")
        seen[pages] = True

    columns = (full_page_table.shape[1] + page_size - 1) // page_size
    if columns:
        storage = staging.view(full_page_table.dtype).flatten()
        # Split columns too: even a tiny pool must not allocate more staging
        # just because the request table has a large padded context width.
        for col_start in range(0, columns, storage.numel()):
            width = min(columns - col_start, storage.numel())
            rows_per_chunk = max(1, storage.numel() // width)
            for row_start in range(0, len(lengths), rows_per_chunk):
                row_end = min(row_start + rows_per_chunk, len(lengths))
                packed = storage[: (row_end - row_start) * width].view(-1, width)
                packed.copy_(
                    full_page_table[
                        row_start:row_end,
                        col_start * page_size : (col_start + width)
                        * page_size : page_size,
                    ]
                )
                host = packed.cpu()
                for row in range(row_start, row_end):
                    if extend_seq_lens_cpu[row] == 0:
                        continue  # padded request, with no real query
                    length = int(lengths[row])
                    valid = max(
                        0, min(width, (length + page_size - 1) // page_size - col_start)
                    )
                    add_slots(host[row - row_start, :valid])

    num_new_tokens = sum(extend_seq_lens_cpu)
    if num_new_tokens > out_cache_loc.numel():
        raise ValueError("Prefill write locations do not cover the whole batch")
    add_slots(out_cache_loc[:num_new_tokens].cpu())
    return seen.nonzero().flatten()


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
            self.prefill_staging = (
                torch.empty(
                    (pool.prefill_chunk_pages, self.remote_buffer.shape[1]),
                    dtype=pool.index_k_with_scale_buffer_dtype,
                    device=pool.device,
                )
                if pool.unified_decode
                else None
            )
        self.remote_layer_id: Optional[int] = None
        self.remote_scope: Optional[int] = None

    def _layer_num_pages(self, layer_idx: int, num_pages: int) -> int:
        layer_id = self.pool.start_layer + layer_idx
        if not self.pool._is_layer_owned(layer_id):
            return 0
        return super()._layer_num_pages(layer_idx, num_pages)

    def clear(self) -> None:
        super().clear()
        del self.remote_buffer
        del self.prefill_staging

    def move(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
        if tgt_loc.numel() == 0:
            return
        super().move(tgt_loc, src_loc)
        self.pool.clear_prefill_page_plan()

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
        if self.pool.prefill_page_plan_active:
            buf = self.get_prefill_buffer(layer_id)
        else:
            buf = self.get_buffer(layer_id)
            self.pool.prefetch_kv_buffer(layer_id)
        return dsa_index_k_cache.GetKAndS.execute(
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
            self.remote_scope = None

    def prepare_prefill_buffer(self, layer_id: int) -> None:
        """Called collectively after the owner stores this layer's index keys."""
        self.get_prefill_buffer(layer_id)
        self.pool.prefetch_prefill_kv_buffer(layer_id)

    def get_prefill_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.pool.prefill_page_plan_active
        scope = self.pool.prefill_plan_epoch
        if self.remote_layer_id != layer_id or self.remote_scope not in (None, scope):
            local_idx = layer_id - self.pool.start_layer
            src = (
                self.buffer[local_idx] if self.pool._is_layer_owned(layer_id) else None
            )
            self.pool._broadcast_prefill_pages(
                self.remote_buffer, src, layer_id, self.prefill_staging
            )
            self.remote_layer_id = layer_id
            self.remote_scope = scope
        return self.remote_buffer

    def get_broadcastable_buffer(self, layer_id: int) -> torch.Tensor:
        if self.remote_layer_id != layer_id or self.remote_scope is not None:
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
            self.remote_scope = None
        return self.remote_buffer

    def state_buf_infos(self):
        owned_layer_ids = [
            i
            for i in range(self.pool.layer_num)
            if self.pool._is_layer_owned(self.pool.start_layer + i)
        ]
        data_ptrs = [self.buffer[i].data_ptr() for i in owned_layer_ids]
        data_lens = [self.buffer[i].nbytes for i in owned_layer_ids]
        item_lens = [self._item_len(i) for i in owned_layer_ids]
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
        self.pool.clear_prefill_page_plan()
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
        unified_decode: bool = False,
        **kwargs,
    ):
        assert layer_shard_rank is not None and layer_shard_size > 1, (
            "LayerSplitDSATokenToKVPool requires layer_shard_size > 1"
        )
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        self.layer_broadcast_comm = None
        self.unified_decode = unified_decode
        self.prefill_plan_epoch = 0
        self.prefill_num_pages: Optional[int] = None
        super().__init__(*args, **kwargs)
        if unified_decode and not self.dsa_kv_cache_store_nvfp4:
            raise ValueError("Non-PD layer split requires the NVFP4 DSA cache")
        if unified_decode and self.index_kpool != 1:
            raise ValueError("Non-PD layer split requires unpooled index keys")
        # First global layer index owned by this rank (used by PD transfer to
        # label the contiguous owned-buffer range).
        my_start, _ = self._owned_local_layer_range()
        self.layer_shard_start = self.start_layer + my_start

    def use_owner_compute(self, forward_mode) -> bool:
        """Run replicated non-PD decode/verify queries where the KV resides."""
        return self.unified_decode and (
            forward_mode.is_decode() or forward_mode.is_target_verify()
        )

    def prepare_owner_compute(self) -> None:
        """Invalidate scratch before eager execution and before graph replay.

        Graph replay updates owned KV without executing Python bookkeeping.
        The attention metadata boundary calls this on every forward, including
        a prefill that follows a replay, so a previous scratch label can never
        hide writes performed by a captured owner-compute path.
        """
        self.clear_prefill_page_plan()

    @property
    def prefill_page_plan_active(self) -> bool:
        return self.prefill_num_pages is not None

    def clear_prefill_page_plan(self) -> None:
        # The stage and page-id storage may still be read on the prefetch stream.
        self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        self.prefill_num_pages = None
        self.remote_kv_layer_id = None
        self.remote_kv_scope = None
        self.index_key_cache.remote_layer_id = None
        self.index_key_cache.remote_scope = None

    def begin_prefill_page_plan(
        self,
        full_page_table: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        extend_seq_lens_cpu: list[int],
        out_cache_loc: torch.Tensor,
    ) -> None:
        """Publish one whole-batch plan before CP filters requests per rank."""
        assert self.unified_decode and self.dsa_kv_cache_store_nvfp4
        assert self.index_kpool == 1 and self.page_size == 64
        assert not (
            full_page_table.is_cuda and torch.cuda.is_current_stream_capturing()
        ), "Prefill page plans must be built outside CUDA graph capture"
        self.clear_prefill_page_plan()
        page_ids, error = None, None
        if self.layer_shard_rank == 0:
            try:
                page_ids = _collect_prefill_page_ids(
                    full_page_table,
                    seq_lens_cpu,
                    extend_seq_lens_cpu,
                    out_cache_loc,
                    self.prefill_main_staging,
                    self.page_size,
                    self.prefill_page_ids.numel(),
                )
            except Exception as exc:
                # Publish a root-side planning failure before peers enter the
                # device collective, rather than leaving them waiting for it.
                error = f"{type(exc).__name__}: {exc}"
        self.prefill_num_pages = self._broadcast_prefill_plan(page_ids, error)
        self.prefill_plan_epoch += 1

    def _broadcast_prefill_plan(
        self, page_ids: Optional[torch.Tensor], error: Optional[str]
    ) -> int:
        group = get_parallel().attn_cp_group
        error, count = group.broadcast_object(
            (error, page_ids.numel() if page_ids is not None else 0), src=0
        )
        if error is not None:
            raise ValueError(error)
        if count:
            ids = self.prefill_page_ids[:count]
            if self.layer_shard_rank == 0:
                ids.copy_(page_ids)
            group.broadcast(ids, src=0)
        return count

    def _broadcast_prefill_pages(
        self,
        remote: torch.Tensor,
        source: Optional[torch.Tensor],
        layer_id: int,
        staging: torch.Tensor,
        *,
        use_layer_broadcast_comm: bool = False,
    ) -> None:
        assert self.prefill_page_plan_active
        remote_pages = remote.view(self.size // self.page_size + 1, -1)
        source_pages = source.view_as(remote_pages) if source is not None else None
        for start in range(0, self.prefill_num_pages, self.prefill_chunk_pages):
            ids = self.prefill_page_ids[
                start : min(start + self.prefill_chunk_pages, self.prefill_num_pages)
            ]
            packed = staging[: ids.numel()]
            if source_pages is not None:
                torch.index_select(source_pages, 0, ids, out=packed)
            self._broadcast_tensor_from_owner(
                packed,
                layer_id,
                src_tensor=packed if source_pages is not None else None,
                use_layer_broadcast_comm=use_layer_broadcast_comm,
            )
            remote_pages.index_copy_(0, ids, packed)

    def run_on_layer_owner(
        self,
        layer_id: int,
        compute: Callable[[], Optional[torch.Tensor]],
        *,
        shape: Optional[tuple[int, ...]],
        dtype: torch.dtype,
        device,
    ) -> Optional[torch.Tensor]:
        """Compute on the owner and broadcast a fixed-shape result to CP ranks.

        The caller guarantees identical queries and physical cache locations
        across this CP group. The callback must use owned-buffer accessors;
        ordinary layer-split getters would issue unmatched collectives. No
        allocation here depends on tensor contents, including under capture.
        ``shape=None`` executes an owner-only cache write without a result.
        """
        assert self.unified_decode
        self.prepare_owner_compute()
        if self._is_layer_owned(layer_id):
            result = compute()
            if shape is None:
                return None
            assert result is not None and tuple(result.shape) == shape
            assert result.dtype == dtype
            result = result.contiguous()
        else:
            if shape is None:
                return None
            result = torch.empty(shape, dtype=dtype, device=device)
        return self._broadcast_tensor_from_owner(
            result, layer_id, src_tensor=result, use_layer_broadcast_comm=True
        )

    def get_owned_key_buffer(self, layer_id: int) -> torch.Tensor:
        assert self._is_layer_owned(layer_id)
        return self.kv_buffer[self._local_layer_idx(layer_id)]

    def get_owned_index_buffer(self, layer_id: int) -> torch.Tensor:
        assert self._is_layer_owned(layer_id)
        return self.index_key_cache.get_local_buffer(layer_id)

    def set_owned_mla_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ) -> None:
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "owner MLA KV write")
        self._write_mla_kv_buffer(
            layer.layer_id,
            self.get_owned_key_buffer(layer.layer_id),
            loc,
            cache_k_nope,
            cache_k_rope,
        )

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
        self.prefill_num_pages = None
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
                self.remote_kv_scope: Optional[int] = None
                num_pages = (self.size + self.page_size) // self.page_size
                self.prefill_chunk_pages = min(PREFILL_TRANSFER_MAX_PAGES, num_pages)
                self.prefill_main_staging = (
                    torch.empty(
                        (self.prefill_chunk_pages, self.page_size * self.kv_cache_dim),
                        dtype=self.store_dtype,
                        device=self.device,
                    )
                    if self.unified_decode
                    else None
                )
                self.prefill_page_ids = (
                    torch.empty(num_pages, dtype=torch.int64, device=self.device)
                    if self.unified_decode
                    else None
                )
                self.device_module = torch.get_device_module(self.device)
                self.kv_broadcast_stream = self.device_module.Stream()
                self.pending_remote_kv_layer_id: Optional[int] = None
                self.pending_remote_kv_scope: Optional[int] = None
                self.pending_remote_kv_broadcast = False
        self._init_layer_broadcast_comm()

    def _create_index_key_cache(self) -> IndexKeyCache:
        return LayerSplitIndexKeyCache(self, self.index_buf_size)

    def _clear_buffers(self):
        self.clear_prefill_page_plan()
        del self.kv_buffer
        del self.remote_kv_buffer
        del self.prefill_main_staging
        del self.prefill_page_ids
        self.index_key_cache.clear()

    # ---- MLA latent KV: owned-only writes, owner-broadcast reads ----------

    def get_kv_size_bytes(self):
        # The base counts owned KV/indexer layers and persistent global scales.
        # Both remote scratch buffers are real allocations on every CP rank.
        return (
            super().get_kv_size_bytes()
            + get_tensor_size_bytes(self.remote_kv_buffer)
            + get_tensor_size_bytes(self.index_key_cache.remote_buffer)
            + sum(self.get_prefill_transfer_allocation_bytes().values())
        )

    def get_prefill_transfer_allocation_bytes(self) -> dict[str, int]:
        return {
            "prefill_main_staging_bytes": (
                self.prefill_main_staging.nbytes
                if self.prefill_main_staging is not None
                else 0
            ),
            "prefill_index_staging_bytes": (
                self.index_key_cache.prefill_staging.nbytes
                if self.index_key_cache.prefill_staging is not None
                else 0
            ),
            "prefill_page_ids_bytes": (
                self.prefill_page_ids.nbytes if self.prefill_page_ids is not None else 0
            ),
        }

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
        if self.dsa_kv_cache_store_packed_fp4:
            return kv_buffer
        if self.store_dtype != self.dtype:
            return kv_buffer.view(self.dtype)
        return kv_buffer

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)

        kv_buffer = self._get_broadcastable_kv_buffer(layer_id)
        if self.dsa_kv_cache_store_packed_fp4:
            # The NVFP4 consumer reads the same packed row for K and V.
            return kv_buffer
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
        assert not self.dsa_kv_cache_store_packed_fp4
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
        loc_info,
        cache_k_nope: torch.Tensor,
        cache_k_rope: torch.Tensor,
    ):
        loc, _, _ = unwrap_write_loc(loc_info)
        maybe_detect_oob(loc, 0, self.size + self.page_size, "set_mla_kv_buffer (MLA)")
        layer_id = layer.layer_id
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=True)
        remote_kv_updatable = self.remote_kv_layer_id == layer_id
        if remote_kv_updatable:
            self._write_mla_kv_buffer(
                layer_id,
                self.remote_kv_buffer,
                loc,
                cache_k_nope,
                cache_k_rope,
            )
        if not self._is_layer_owned(layer_id):
            return
        self._write_mla_kv_buffer(
            layer_id,
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
            self.remote_kv_scope = self.pending_remote_kv_scope
        self.pending_remote_kv_layer_id = None
        self.pending_remote_kv_scope = None

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
        self._prefetch_kv_buffer(
            layer_id, None, layer_transfer_counter, layer_transfer_idx
        )

    def prefetch_prefill_kv_buffer(self, layer_id: int) -> None:
        assert self.prefill_page_plan_active
        self._prefetch_kv_buffer(layer_id, self.prefill_plan_epoch)

    def _prefetch_kv_buffer(
        self,
        layer_id: int,
        scope: Optional[int],
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
    ) -> None:
        if self.pending_remote_kv_broadcast:
            if (
                self.pending_remote_kv_layer_id == layer_id
                and self.pending_remote_kv_scope in (None, scope)
            ):
                return
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id and self.remote_kv_scope in (
            None,
            scope,
        ):
            return
        self.remote_kv_layer_id = None
        self.remote_kv_scope = None

        local_idx = self._local_layer_idx(layer_id)
        src_tensor = (
            self.kv_buffer[local_idx] if self._is_layer_owned(layer_id) else None
        )

        def transfer():
            if scope is None:
                self._broadcast_tensor_from_owner(
                    self.remote_kv_buffer,
                    layer_id,
                    src_tensor=src_tensor,
                    use_layer_broadcast_comm=True,
                )
            else:
                self._broadcast_prefill_pages(
                    self.remote_kv_buffer,
                    src_tensor,
                    layer_id,
                    self.prefill_main_staging,
                    use_layer_broadcast_comm=True,
                )

        if self.layer_broadcast_comm is None:
            transfer()
            self.remote_kv_layer_id = layer_id
            self.remote_kv_scope = scope
            return

        self.kv_broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_broadcast_stream):
            if layer_transfer_counter is not None and layer_transfer_idx is not None:
                layer_transfer_counter.wait_until(layer_transfer_idx)
            transfer()
        self.pending_remote_kv_layer_id = layer_id
        self.pending_remote_kv_scope = scope
        self.pending_remote_kv_broadcast = True

    def get_prefill_key_buffer(self, layer_id: int) -> torch.Tensor:
        assert self.prefill_page_plan_active
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self.prefetch_prefill_kv_buffer(layer_id)
        self._finalize_pending_kv_broadcast()
        return self.remote_kv_buffer

    def _get_broadcastable_kv_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pending_remote_kv_broadcast:
            self._finalize_pending_kv_broadcast(
                set_remote_layer_id=self.pending_remote_kv_layer_id == layer_id
            )
        if self.remote_kv_layer_id != layer_id or self.remote_kv_scope is not None:
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
            self.remote_kv_scope = None
        return self.remote_kv_buffer

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        size_limit = self.size + self.page_size
        maybe_detect_oob(tgt_loc, 0, size_limit, "move_kv_cache tgt_loc")
        maybe_detect_oob(src_loc, 0, size_limit, "move_kv_cache src_loc")
        if tgt_loc.numel() == 0:
            return
        self.clear_prefill_page_plan()
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

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
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

        return {
            "kv": kv_cache_cpu,
            "index_k": self.index_key_cache.cpu_copy(indices),
        }

    def load_cpu_copy(
        self,
        kv_cache_cpu_dict,
        indices,
        mamba_indices=None,
        req_pool_index=None,
    ):
        from sglang.srt.utils import current_platform

        self.clear_prefill_page_plan()
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
