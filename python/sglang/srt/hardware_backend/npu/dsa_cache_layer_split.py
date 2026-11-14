# SPDX-License-Identifier: Apache-2.0
"""Layer-sharded NPU MLA KV cache pool for context-parallel prefill.

NPU counterpart of ``sglang.srt.mem_cache.dsa_cache_layer_split.LayerSplitDSATokenToKVPool``:
splits the DSA MLA KV/indexer cache layers across context-parallel (CP) ranks so
each rank only materializes the layers it owns, reducing per-rank KV memory.
When a rank needs to read a layer it does not own, the owning rank broadcasts
that layer's buffer into a full-size per-rank remote scratch tensor.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.layers.cp.utils import get_layer_owner, get_layer_shard_range
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


class LayerSplitNPUMLATokenToKVPool(NPUMLATokenToKVPool):
    """NPU MLA KV pool that shards layers across CP ranks with owner-broadcast reads.

    Layout: ``k_buffer``/``v_buffer`` (and the indexer buffers) keep their stacked
    layer-first shape, but dim 0 holds only the *owned* layers; ``_kv_local_offset``
    rebases a global layer id onto the owned rows. Reads of non-owned layers
    broadcast the owner's layer buffer into the dedicated ``remote_*`` scratch
    tensors.
    """

    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        **kwargs,
    ):
        assert layer_shard_rank is not None and layer_shard_size > 1, (
            "LayerSplitNPUMLATokenToKVPool requires layer_shard_size > 1"
        )
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        super().__init__(*args, **kwargs)
        # First global layer index owned by this rank (used by PD transfer to
        # label the contiguous owned-buffer range).
        self.layer_shard_start = self.start_layer + self._owned_local_layer_range()[0]

    def _owned_local_layer_range(self) -> tuple[int, int]:
        # host_pool_config.py addresses the split-pool root through the
        # mem_cache backend's method name, so the NPU class must expose it
        # too when hicache is combined with split.
        return get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _is_local_layer_owned(self, local_layer_id: int) -> bool:
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_layer_id < owned_end

    def _is_layer_owned(self, layer_id: int) -> bool:
        return self._is_local_layer_owned(layer_id - self.start_layer)

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        return get_layer_owner(
            layer_id - self.start_layer, self.layer_shard_size, self.layer_num
        )

    def _log_layer_shard_plan(self) -> None:
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)
            partitions.append(f"r{rank}:[{st},{ed})")
        my_start, my_end = self._owned_local_layer_range()
        logger.info(
            "Layer shard plan (continuous, NPU): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, local=[{my_start},{my_end}), "
            f"global=[{self.start_layer + my_start},{self.start_layer + my_end}), "
            f"partitions={'; '.join(partitions)}"
        )

    def _create_buffers(self):
        self._log_layer_shard_plan()
        if self.enable_sparsity_driven_kv_offload:
            raise ValueError(
                "--enable-dsa-cache-layer-split is incompatible with "
                "sparsity-driven KV offload"
            )
        owned_start, owned_end = self._owned_local_layer_range()
        num_owned = owned_end - owned_start
        # Rebuild the indexer bookkeeping over owned layers only: the base
        # read/write paths resolve index buffers via _get_indexer_slot and
        # iterate self.indexer_layer_ids for transfer/state metadata.
        self.indexer_layer_ids = tuple(
            layer_id
            for layer_id in self.indexer_layer_ids
            if owned_start <= layer_id - self.start_layer < owned_end
        )
        self.num_indexer_layers = len(self.indexer_layer_ids)
        self.indexer_layer_id_to_slot = {
            layer_id: slot for slot, layer_id in enumerate(self.indexer_layer_ids)
        }
        kv_pages = self.size // self.page_size + self.kv_page_padding
        index_pages = self.index_size // self.index_page_size + self.index_page_padding

        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            self.k_buffer = torch.zeros(
                (
                    num_owned,
                    kv_pages,
                    self.page_size,
                    1,
                    self.kv_cache_dim,
                ),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (
                    num_owned,
                    kv_pages,
                    self.page_size,
                    1,
                    self.kr_cache_dim,
                ),
                dtype=(
                    torch.bfloat16 if self.dsa_kv_cache_store_fp8 else self.store_dtype
                ),
                device=self.device,
            )
            self.index_k_buffer = None
            if self.index_head_dim is not None:
                self.index_k_buffer = torch.zeros(
                    (
                        self.num_indexer_layers,
                        index_pages,
                        self.index_page_size,
                        1,
                        self.index_head_dim,
                    ),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                if self.dsa_kv_cache_store_fp8 and self.num_indexer_layers > 0:
                    from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
                        create_npu_hadamard_128,
                    )

                    self.index_k_scale_buffer = torch.zeros(
                        (*self.index_k_buffer.shape[:-2], 1),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    self.indexer_hadamard_128 = create_npu_hadamard_128(
                        self.index_head_dim, self.device
                    )
            # Full-size remote scratch per buffer type, refreshed by the
            # owner-broadcast whenever a non-owned layer is read.
            self.remote_k_buffer = torch.empty(
                (kv_pages, self.page_size, 1, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            # v_buffer is always materialized (bfloat16 in fp8 mode), so the
            # remote v scratch mirrors that dtype instead of store_dtype.
            self.remote_v_buffer = torch.empty(
                (kv_pages, self.page_size, 1, self.kr_cache_dim),
                dtype=(
                    torch.bfloat16 if self.dsa_kv_cache_store_fp8 else self.store_dtype
                ),
                device=self.device,
            )
            self.remote_index_k_buffer = None
            self.remote_index_k_scale_buffer = None
            if self.index_head_dim is not None:
                self.remote_index_k_buffer = torch.empty(
                    (index_pages, self.index_page_size, 1, self.index_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                if self.index_k_scale_buffer is not None:
                    self.remote_index_k_scale_buffer = torch.empty(
                        (index_pages, self.index_page_size, 1),
                        dtype=torch.float32,
                        device=self.device,
                    )

        self.remote_kv_layer_id: Optional[int] = None
        self.remote_index_layer_id: Optional[int] = None
        self.pending_remote_kv_layer_id: Optional[int] = None
        self.pending_remote_kv_broadcast = False
        self.device_module = torch.get_device_module(self.device)
        self.kv_broadcast_stream = self.device_module.Stream()

    def _kv_local_offset(self, layer_id: int) -> int:
        owned_start, _ = self._owned_local_layer_range()
        return layer_id - self.start_layer - owned_start

    def _broadcast_buffer_from_owner(
        self,
        buf: torch.Tensor,
        layer_id: int,
        src_buf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Broadcast one layer's buffer from its owner into ``buf`` (scratch)."""
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_buf is not None
            if buf.data_ptr() != src_buf.data_ptr():
                buf.copy_(src_buf)
        cp_group = get_parallel().attn_cp_group
        # Use the CP group's HCCL device group.
        # torch.distributed expects the *global* rank of the source.
        torch.distributed.broadcast(
            buf, src=cp_group.ranks[owner_rank], group=cp_group.device_group
        )
        return buf

    def _maybe_view_dtype(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

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

        Called ahead of the layer's attention so the remote scratch is
        ready by the time a non-owner rank reads it.
        """
        if self.remote_kv_layer_id == layer_id:
            return
        if self.pending_remote_kv_broadcast:
            if self.pending_remote_kv_layer_id == layer_id:
                return
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)

        src_k = src_v = None
        if self._is_layer_owned(layer_id):
            offset = self._kv_local_offset(layer_id)
            src_k = self.k_buffer[offset]
            src_v = self.v_buffer[offset]
        self.kv_broadcast_stream.wait_stream(self.device_module.current_stream())
        with self.device_module.stream(self.kv_broadcast_stream):
            if layer_transfer_counter is not None and layer_transfer_idx is not None:
                layer_transfer_counter.wait_until(layer_transfer_idx)
            self._broadcast_buffer_from_owner(self.remote_k_buffer, layer_id, src_k)
            if not self.dsa_kv_cache_store_fp8:
                self._broadcast_buffer_from_owner(self.remote_v_buffer, layer_id, src_v)
        self.pending_remote_kv_layer_id = layer_id
        self.pending_remote_kv_broadcast = True

    def _ensure_remote_layer(self, layer_id: int) -> None:
        """Synchronously materialize ``layer_id`` in the remote scratch if remote."""
        if self.pending_remote_kv_broadcast:
            self._finalize_pending_kv_broadcast(
                set_remote_layer_id=self.pending_remote_kv_layer_id == layer_id
            )
        if self.remote_kv_layer_id == layer_id:
            return
        src_k = src_v = None
        if self._is_layer_owned(layer_id):
            offset = self._kv_local_offset(layer_id)
            src_k = self.k_buffer[offset]
            src_v = self.v_buffer[offset]
        self._broadcast_buffer_from_owner(self.remote_k_buffer, layer_id, src_k)
        if not self.dsa_kv_cache_store_fp8:
            self._broadcast_buffer_from_owner(self.remote_v_buffer, layer_id, src_v)
        self.remote_kv_layer_id = layer_id

    def _ensure_remote_index_layer(self, layer_id: int) -> None:
        """Materialize ``layer_id``'s indexer rows on every rank (paired broadcast)."""
        if self.remote_index_layer_id == layer_id:
            return
        if self.remote_index_k_buffer is None:
            return
        index_src = scale_src = None
        if self._is_layer_owned(layer_id):
            slot = self.indexer_layer_id_to_slot.get(layer_id)
            if slot is not None:
                index_src = self.index_k_buffer[slot]
                if self.index_k_scale_buffer is not None:
                    scale_src = self.index_k_scale_buffer[slot]
        self._broadcast_buffer_from_owner(
            self.remote_index_k_buffer, layer_id, index_src
        )
        if self.remote_index_k_scale_buffer is not None:
            self._broadcast_buffer_from_owner(
                self.remote_index_k_scale_buffer, layer_id, scale_src
            )
        self.remote_index_layer_id = layer_id

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        self._ensure_remote_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_kv_buffer(layer_id)
        return (
            self._maybe_view_dtype(self.remote_k_buffer),
            self._maybe_view_dtype(self.remote_v_buffer),
        )

    def get_key_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        self._ensure_remote_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_key_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_k_buffer)

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        self._ensure_remote_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_value_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_v_buffer)

    def set_kv_buffer(
        self,
        layer: RadixAttention,
        loc_info: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = layer.layer_id
        # Consume/invalidate the pending remote copy on EVERY rank
        # (before the owned-only early return), so the follow-up get-time
        # broadcast decision stays paired across the CP group. A write
        # this layer must re-broadcast on both ranks.
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None
        if not self._is_layer_owned(layer_id):
            return
        super().set_kv_buffer(layer, loc_info, cache_k, cache_v)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        # A write invalidates any cached remote copy before the owned-only
        # early return, so the next read re-broadcasts on every rank.
        self.invalidate_index_layer(layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_index_k_buffer(layer_id, loc, index_k)

    def set_index_k_scale_buffer(self, layer_id: int, loc, scale):
        self.invalidate_index_layer(layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_index_k_scale_buffer(layer_id, loc, scale)

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.index_k_buffer is None:
            raise RuntimeError("NPU MLA index KV cache is not allocated.")
        # Every rank (owner included) goes through the broadcastable path so the
        # HCCL broadcast is paired across the CP group.
        self._ensure_remote_index_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_index_k_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_index_k_buffer)

    def get_index_k_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._ensure_remote_index_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_index_k_scale_buffer(layer_id)
        return self.remote_index_k_scale_buffer

    def invalidate_index_layer(self, layer_id: int) -> None:
        if self.remote_index_layer_id == layer_id:
            self.remote_index_layer_id = None

    def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor):
        if tgt_loc.numel() == 0:
            return
        tgt_flat = tgt_loc.view(-1).long()
        src_flat = src_loc.view(-1).long()
        for offset in range(self.k_buffer.shape[0]):
            self.k_buffer[offset][tgt_flat] = self.k_buffer[offset][src_flat]
            self.v_buffer[offset][tgt_flat] = self.v_buffer[offset][src_flat]
        if self.index_k_buffer is not None:
            for slot in range(self.index_k_buffer.shape[0]):
                self.index_k_buffer[slot][tgt_flat] = self.index_k_buffer[slot][
                    src_flat
                ]
                if self.index_k_scale_buffer is not None:
                    self.index_k_scale_buffer[slot][tgt_flat] = (
                        self.index_k_scale_buffer[slot][src_flat]
                    )

    def get_kv_layer_ids(self):
        owned_start, owned_end = self._owned_local_layer_range()
        owned = list(
            range(self.start_layer + owned_start, self.start_layer + owned_end)
        )
        return owned * 2 + self.get_state_layer_ids()

    def get_kv_transfer_layer_span(self, num_entries: int) -> int:
        """Layer count covered by ``num_entries`` contiguous buf entries.

        ``get_contiguous_buf_infos`` reports one entry per layer per buffer
        group (the groups of ``_get_disagg_buffer_entries``: latent k, non-fp8
        v, indexer k, fp8 index scale), so the owned layer count is the entry
        count divided by the group count — not the raw entry count.
        """
        groups = 1  # latent k (v stays packed into k in fp8 mode)
        if not self.dsa_kv_cache_store_fp8:
            groups += 1  # v
        if self.index_head_dim is not None:
            groups += 1  # indexer k
            if self.index_k_scale_buffer is not None:
                groups += 1  # fp8 index scale
        assert groups > 0 and num_entries % groups == 0, (
            f"layer-split transfer reported {num_entries} entries, "
            f"not a multiple of {groups} buffer groups"
        )
        return num_entries // groups

    def _mla_native_buffers(self):
        from sglang.srt.mem_cache.device_pool_info import MLASplitBuffers

        # The buffers hold only owned layers (dim 0 == owned layers), so the
        # host pool consumes them positionally without slicing: row i holds
        # owned layer i, addressed in the owned-row space.
        return MLASplitBuffers(
            k=self.k_buffer,
            v=self.v_buffer,
            index_k=self.index_k_buffer,
            index_scales=self.index_k_scale_buffer,
            index_layer_indices=(
                tuple(range(self.index_k_buffer.shape[0]))
                if self.index_k_buffer is not None
                else ()
            ),
            k_width=self.kv_cache_dim,
            v_width=self.kr_cache_dim,
            index_width=self.index_head_dim or 0,
        )

    def _mla_buffer_info(self):
        from sglang.srt.mem_cache.device_pool_info import MLABufferInfo

        info = super()._mla_buffer_info()
        owned = self.k_buffer.shape[0]
        return MLABufferInfo(
            buffers=self._mla_native_buffers(),
            native_buffers=self._mla_native_buffers,
            data_ptrs=None,
            layer_indices=tuple(range(owned)),
            device_layer_num=owned,
            page_size=info.page_size,
            kv_lora_rank=info.kv_lora_rank,
            qk_rope_head_dim=info.qk_rope_head_dim,
            kv_cache_dim=info.kv_cache_dim,
            dtype=info.dtype,
            device=info.device,
            packed_fp8=info.packed_fp8,
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        torch.npu.synchronize()
        buf_of_layers = [
            self._get_cpu_offload_layer_buffers(i)
            if self._is_local_layer_owned(i)
            else []
            for i in range(self.layer_num)
        ]
        uses_global_slots_per_layer = []
        for buffers in buf_of_layers:
            # MLA K/V is rank-local under DCP; replicated indexer buffers keep
            # allocator-global slot identities (same convention as the base pool).
            uses_global_slots_per_layer.append(
                [self.is_draft_worker, self.is_draft_worker]
                + [True] * (len(buffers) - 2)
                if buffers
                else []
            )

        kv_cache_cpu = self._chunk_copy_npu_to_cpu(
            buf_of_layers, indices, uses_global_slots_per_layer
        )
        torch.npu.synchronize()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        torch.npu.synchronize()
        chunk_size = self.cpu_offloading_chunk_size
        for local_layer_id in range(self.layer_num):
            if not self._is_local_layer_owned(local_layer_id):
                continue
            buffers = self._get_cpu_offload_layer_buffers(local_layer_id)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                chunk = kv_cache_cpu[local_layer_id][i // chunk_size]
                cpu_index = 0
                for buffer, uses_global_slots in zip(
                    buffers,
                    [self.is_draft_worker, self.is_draft_worker]
                    + [True] * (len(buffers) - 2),
                    strict=True,
                ):
                    cpu = chunk[cpu_index]
                    cpu_index += 1
                    target_indices = self._copy_indices_for_buffer(
                        chunk_indices, uses_global_slots
                    )
                    assert cpu.shape[0] == len(target_indices)
                    buffer[target_indices] = cpu.to(buffer.device, non_blocking=True)
        torch.npu.synchronize()
