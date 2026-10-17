"""Layer-sharded NPU MLA KV cache pool for DSA context-parallel prefill.

NPU companion of the CUDA ``LayerSplitDSATokenToKVPool``
(``sglang.srt.mem_cache.dsa_cache_layer_split``). The NPU MLA pool uses a
completely different layout (batched 5D k/v/index tensors, scatter writes,
NPU indexer pool methods), so the CUDA pool cannot be reused and a dedicated
subclass shards the layers instead.

Each prefill-CP rank only materializes the k/v/index layers it owns plus one
full-size remote scratch per buffer type; reads of non-owned layers broadcast
the owning rank's layer buffer over the CP group (HCCL via
``torch.distributed`` — NPU groups disable pynccl). PD transfer reports
owned-only buffers and the decode peer pulls from every CP rank to reassemble
the full cache.
"""

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.cache_controller import LayerDoneCounter

logger = logging.getLogger(__name__)


class NPULayerSplitMLATokenToKVPool(NPUMLATokenToKVPool):
    def __init__(
        self,
        *args,
        layer_shard_rank: int,
        layer_shard_size: int,
        **kwargs,
    ):
        assert layer_shard_rank is not None and layer_shard_size > 1, (
            "NPULayerSplitMLATokenToKVPool requires layer_shard_size > 1"
        )
        self.layer_shard_rank = layer_shard_rank
        self.layer_shard_size = layer_shard_size
        self.layer_shard_enabled = True
        # Layers whose buffers currently populate the remote scratch, mirroring
        # the CUDA pool's remote_kv_layer_id bookkeeping (a cache hit skips the
        # broadcast; any write to the layer invalidates it). KV (k+v) and index
        # (index_k + scale) scratch are cached independently so an attention
        # read does not re-broadcast indexer buffers and vice versa.
        self.remote_kv_layer_id: Optional[int] = None
        self.remote_index_layer_id: Optional[int] = None
        super().__init__(*args, **kwargs)
        owned_start, _ = self._owned_local_layer_range()
        # First global layer owned by this rank; consumed by PD transfer
        # (KVArgs.prefill_start_layer) to place owned-only entries on decode.
        self.layer_shard_start = self.start_layer + owned_start

    # ---- layer ownership helpers ------------------------------------------

    def _owned_local_layer_range(self) -> tuple[int, int]:
        from sglang.srt.layers.cp.utils import get_layer_shard_range

        return get_layer_shard_range(
            self.layer_shard_rank, self.layer_shard_size, self.layer_num
        )

    def _is_layer_owned(self, layer_id: int) -> bool:
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= layer_id - self.start_layer < owned_end

    def _is_local_layer_owned(self, local_layer_id: int) -> bool:
        owned_start, owned_end = self._owned_local_layer_range()
        return owned_start <= local_layer_id < owned_end

    def _get_layer_owner_rank(self, layer_id: int) -> int:
        from sglang.srt.layers.cp.utils import get_layer_owner

        return get_layer_owner(
            layer_id - self.start_layer, self.layer_shard_size, self.layer_num
        )

    def _kv_local_offset(self, layer_id: int) -> int:
        owned_start, _ = self._owned_local_layer_range()
        return layer_id - self.start_layer - owned_start

    # ---- buffer allocation (owned-only + remote scratch) -------------------

    def _create_buffers(self) -> None:
        if self.enable_sparsity_driven_kv_offload:
            raise ValueError(
                "--enable-dsa-cache-layer-split is incompatible with "
                "sparsity-driven KV offload"
            )
        owned_start, owned_end = self._owned_local_layer_range()
        owned_kv_layers = owned_end - owned_start
        # Rebuild the indexer bookkeeping over owned layers only: base
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
                (owned_kv_layers, kv_pages, self.page_size, 1, self.kv_cache_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            self.v_buffer = torch.zeros(
                (owned_kv_layers, kv_pages, self.page_size, 1, self.kr_cache_dim),
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
        partitions = []
        for rank in range(self.layer_shard_size):
            st, ed = self._owned_local_layer_range_for(rank)
            partitions.append(f"r{rank}:[{st},{ed})")
        logger.info(
            "NPU layer shard plan (continuous): "
            f"layer_num={self.layer_num}, shard_size={self.layer_shard_size}, "
            f"rank={self.layer_shard_rank}, owned={partitions[self.layer_shard_rank]}, "
            f"partitions={'; '.join(partitions)}"
        )

    def _owned_local_layer_range_for(self, rank: int) -> tuple[int, int]:
        from sglang.srt.layers.cp.utils import get_layer_shard_range

        return get_layer_shard_range(rank, self.layer_shard_size, self.layer_num)

    # ---- owner-broadcast plumbing ------------------------------------------

    def _broadcast_to_owner_scratch(
        self,
        scratch: torch.Tensor,
        layer_id: int,
        src: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fill ``scratch`` with ``layer_id``'s buffer broadcast by its owner.

        ``src`` must be the local owned buffer when this rank owns the layer;
        the owner copies it into the scratch before the collective so the
        broadcast publishes the freshly written data. The HCCL process group
        picks up the current NPU stream, keeping the copy/broadcast ordered.
        """
        cp_group = get_parallel().attn_cp_group
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src is not None
            scratch.copy_(src)
        torch.distributed.broadcast(
            scratch, src=cp_group.ranks[owner_rank], group=cp_group.device_group
        )
        return scratch

    # ---- reads: owned buffers, owner-broadcast for the rest ----------------

    def _maybe_view_dtype(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

    def get_kv_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        # Every rank joins the owner-broadcast for every layer read (the owner
        # sources it), otherwise the HCCL collective would hang on mismatched
        # participation. Owned layers still read their local buffer.
        self._broadcast_kv_layer(layer_id)
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
        self._broadcast_kv_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_key_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_k_buffer)

    def get_value_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._raise_if_native_kv_cache_disabled()
        self._broadcast_kv_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_value_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_v_buffer)

    def get_index_k_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        if self.index_k_buffer is None:
            raise RuntimeError("NPU MLA index KV cache is not allocated.")
        self._broadcast_index_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_index_k_buffer(layer_id)
        return self._maybe_view_dtype(self.remote_index_k_buffer)

    def get_index_k_scale_buffer(self, layer_id: int):
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        self._broadcast_index_layer(layer_id)
        if self._is_layer_owned(layer_id):
            return super().get_index_k_scale_buffer(layer_id)
        return self.remote_index_k_scale_buffer

    def _broadcast_kv_layer(self, layer_id: int) -> None:
        """Refresh the k/v remote scratch with ``layer_id``'s owner data."""
        if self.remote_kv_layer_id == layer_id:
            # Scratch already holds this layer's latest owner broadcast.
            return
        k_src = v_src = None
        if self._is_layer_owned(layer_id):
            offset = self._kv_local_offset(layer_id)
            k_src = self.k_buffer[offset]
            if self.v_buffer is not None:
                v_src = self.v_buffer[offset]
        self._broadcast_to_owner_scratch(self.remote_k_buffer, layer_id, k_src)
        if self.remote_v_buffer is not None:
            self._broadcast_to_owner_scratch(self.remote_v_buffer, layer_id, v_src)
        self.remote_kv_layer_id = layer_id

    def _broadcast_index_layer(self, layer_id: int) -> None:
        """Refresh the indexer remote scratch with ``layer_id``'s owner data."""
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
        self._broadcast_to_owner_scratch(
            self.remote_index_k_buffer, layer_id, index_src
        )
        if self.remote_index_k_scale_buffer is not None:
            self._broadcast_to_owner_scratch(
                self.remote_index_k_scale_buffer, layer_id, scale_src
            )
        self.remote_index_layer_id = layer_id

    def _invalidate_remote_layer(self, layer_id: int) -> None:
        # A write to ``layer_id`` (local or remote via the owner) makes any
        # cached remote copy stale; drop it so the next read re-broadcasts.
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None
        if self.remote_index_layer_id == layer_id:
            self.remote_index_layer_id = None

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional["LayerDoneCounter"] = None,
        layer_transfer_idx: Optional[int] = None,
    ) -> None:
        """Eagerly fill the k/v remote scratch with ``layer_id``'s owner data.

        Parity with the CUDA pool's prefetch hook
        (``DeepseekV2DecoderLayer.maybe_prefetch_next_full_attention_kv``):
        the following kv read of ``layer_id`` then hits the
        ``remote_kv_layer_id`` cache instead of issuing a fresh broadcast.
        NPU broadcasts run on the current stream, so unlike the CUDA side
        stream this is a synchronous prefetch.
        """
        if layer_transfer_counter is not None and layer_transfer_idx is not None:
            layer_transfer_counter.wait_until(layer_transfer_idx)
        self._broadcast_kv_layer(layer_id)

    # ---- PD transfer layer span --------------------------------------------

    def get_kv_transfer_layer_span(self, num_entries: int) -> int:
        """Number of layers covered by ``num_entries`` contiguous buf entries.

        ``get_contiguous_buf_infos`` on the NPU MLA pool reports one entry per
        layer per buffer group (k, non-fp8 v, index-k, fp8 index scale —
        mirroring ``_get_disagg_buffer_entries``), so the owned layer count is
        the entry count divided by the group count — not the entry count
        itself, which is what the CUDA packed-kv layout assumes.
        """
        groups = 1  # k
        if not self.dsa_kv_cache_store_fp8:
            groups += 1  # v
        if self.index_head_dim is not None:
            groups += 1  # index_k
            if self.index_k_scale_buffer is not None:
                groups += 1  # index scale
        assert groups > 0 and num_entries % groups == 0, (
            f"layer-split transfer reported {num_entries} entries, "
            f"not a multiple of {groups} buffer groups"
        )
        return num_entries // groups

    # ---- writes: owned layers only -----------------------------------------

    def set_kv_buffer(
        self,
        layer: "RadixAttention",
        loc_info,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        self._invalidate_remote_layer(layer.layer_id)
        if not self._is_layer_owned(layer.layer_id):
            return
        super().set_kv_buffer(layer, loc_info, cache_k, cache_v)

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ):
        self._invalidate_remote_layer(layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_index_k_buffer(layer_id, loc, index_k)

    def set_index_k_scale_buffer(self, layer_id: int, loc, scale):
        self._invalidate_remote_layer(layer_id)
        if not self._is_layer_owned(layer_id):
            return
        super().set_index_k_scale_buffer(layer_id, loc, scale)

    # ---- CPU offload: skip non-owned layers (empty per-layer lists) ---------

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
                    if buffer is None:
                        continue
                    cpu = chunk[cpu_index]
                    cpu_index += 1
                    target_indices = self._copy_indices_for_buffer(
                        chunk_indices, uses_global_slots
                    )
                    assert cpu.shape[0] == len(target_indices)
                    buffer[target_indices] = cpu.to(buffer.device, non_blocking=True)
        torch.npu.synchronize()
