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
from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.device import is_npu

if is_npu():
    import torch_npu

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
                    torch.bfloat16 if self.dsa_kv_cache_store_c8 else self.store_dtype
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
                if self.dsa_kv_cache_store_c8 and self.num_indexer_layers > 0:
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
            # owner-broadcast whenever a non-owned layer is read. K and V
            # live in ONE flat byte allocation (one allocation, two dtype
            # views).
            _v_dtype = (
                torch.bfloat16 if self.dsa_kv_cache_store_c8 else self.store_dtype
            )
            _k_bytes = (
                kv_pages
                * self.page_size
                * self.kv_cache_dim
                * torch.empty((), dtype=self.store_dtype).element_size()
            )
            _v_bytes = (
                kv_pages
                * self.page_size
                * self.kr_cache_dim
                * torch.empty((), dtype=_v_dtype).element_size()
            )
            self.remote_kv_all = torch.empty(
                _k_bytes + _v_bytes, dtype=torch.uint8, device=self.device
            )
            self.remote_k_buffer = (
                self.remote_kv_all[:_k_bytes]
                .view(self.store_dtype)
                .view(kv_pages, self.page_size, 1, self.kv_cache_dim)
            )
            # v_buffer is always materialized (bfloat16 in c8 mode), so the
            # remote v scratch mirrors that dtype instead of store_dtype.
            self.remote_v_buffer = (
                self.remote_kv_all[_k_bytes:]
                .view(_v_dtype)
                .view(kv_pages, self.page_size, 1, self.kr_cache_dim)
            )
            self._v_dtype = _v_dtype
            self._kv_capacity_tokens = kv_pages * self.page_size
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
        self.pending_remote_index_layer_id: Optional[int] = None
        self.pending_remote_index_broadcast = False
        # Async kick handles (torch.distributed Work) for the in-flight
        # prefix broadcasts.
        self.pending_kv_works: list = []
        self.pending_index_works: list = []
        # Row-space bound of the current kick's broadcast: rows [0, bound)
        # below the batch's patch-write region.
        self._kv_used_bound = 0
        self._idx_used_bound = 0

    def _kv_local_offset(self, layer_id: int) -> int:
        owned_start, _ = self._owned_local_layer_range()
        return layer_id - self.start_layer - owned_start

    def _kv_buffer_layer_ids(self) -> range:
        owned_start, owned_end = self._owned_local_layer_range()
        return range(self.start_layer + owned_start, self.start_layer + owned_end)

    def _kv_slice_tokens(self, used_bound: Optional[int]) -> int:
        """Row-space slice of the remote scratches for a bound.
        The bound is POOL ROWS (token-major flat rows).
        """
        if not used_bound or used_bound <= 0:
            return self._kv_capacity_tokens
        return min(used_bound, self._kv_capacity_tokens)

    def _broadcast_kv_from_owner(self, layer_id: int, used_bound: Optional[int] = None):
        """Broadcast K and V token rows [0, slice_tokens) from the owner."""
        slice_tokens = self._kv_slice_tokens(used_bound)
        owner_rank = self._get_layer_owner_rank(layer_id)
        offset = self._kv_local_offset(layer_id)
        cp_group = get_parallel().attn_cp_group
        # torch.distributed expects the *global* rank of the source.
        src = cp_group.ranks[owner_rank]
        if self.layer_shard_rank == owner_rank:
            k_buf = self.k_buffer[offset].view(-1, self.kv_cache_dim)[:slice_tokens]
            v_buf = self.v_buffer[offset].view(-1, self.kr_cache_dim)[:slice_tokens]
        else:
            k_buf = self.remote_k_buffer.view(-1, self.kv_cache_dim)[:slice_tokens]
            v_buf = self.remote_v_buffer.view(-1, self.kr_cache_dim)[:slice_tokens]
        return [
            torch.distributed.broadcast(
                k_buf, src=src, group=cp_group.device_group, async_op=True
            ),
            torch.distributed.broadcast(
                v_buf, src=src, group=cp_group.device_group, async_op=True
            ),
        ]

    def _broadcast_buffer_from_owner(
        self,
        buf: torch.Tensor,
        layer_id: int,
        src_buf: Optional[torch.Tensor] = None,
    ):
        """Broadcast one layer's buffer from its owner into ``buf`` (scratch)."""
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            assert src_buf is not None
            if buf.data_ptr() != src_buf.data_ptr():
                buf.copy_(src_buf)
        cp_group = get_parallel().attn_cp_group
        # Use the CP group's HCCL device group.
        # torch.distributed expects the *global* rank of the source.
        work = torch.distributed.broadcast(
            buf,
            src=cp_group.ranks[owner_rank],
            group=cp_group.device_group,
            async_op=True,
        )
        return work

    def _maybe_view_dtype(self, buffer: torch.Tensor) -> torch.Tensor:
        if self.store_dtype != self.dtype:
            return buffer.view(self.dtype)
        return buffer

    def _finalize_pending_kv_broadcast(
        self, *, set_remote_layer_id: bool = True, wait_works: bool = True
    ) -> None:
        if not self.pending_remote_kv_broadcast:
            return
        # Wait the async kick handles before any main-stream use of the
        # scratch (set-side scatter / get-side read) or of the owner's src
        # buffer (the layer set overwrites the broadcast's source).
        if wait_works:
            for w in self.pending_kv_works:
                if w is not None:
                    w.wait()
        self.pending_kv_works = []
        self.pending_remote_kv_broadcast = False
        if set_remote_layer_id and self.pending_remote_kv_layer_id is not None:
            self.remote_kv_layer_id = self.pending_remote_kv_layer_id
        self.pending_remote_kv_layer_id = None

    def _finalize_pending_index_broadcast(
        self, *, set_remote_layer_id: bool = True, wait_works: bool = True
    ) -> None:
        if not self.pending_remote_index_broadcast:
            return
        if wait_works:
            for w in self.pending_index_works:
                if w is not None:
                    w.wait()
        self.pending_index_works = []
        self.pending_remote_index_broadcast = False
        if set_remote_layer_id and self.pending_remote_index_layer_id is not None:
            self.remote_index_layer_id = self.pending_remote_index_layer_id
        self.pending_remote_index_layer_id = None

    def prefetch_index_k_buffer(
        self, layer_id: int, used_bound: Optional[int] = None
    ) -> None:
        """Kick off an owner-broadcast of ``layer_id``'s index-K scratch.

        Skipped for quant indexers: the fp32 scale rows are not covered
        by the patch-write path, so those configs keep the get-time broadcast.
        """
        if (
            self.remote_index_k_buffer is None
            or self.remote_index_k_scale_buffer is not None
        ):
            return
        if self.remote_index_layer_id == layer_id:
            return
        if self.pending_remote_index_broadcast:
            if self.pending_remote_index_layer_id == layer_id:
                return
            self._finalize_pending_index_broadcast(
                set_remote_layer_id=False, wait_works=False
            )
        # Same assign-not-max as the KV bound: row-space, per-chunk correct.
        self._idx_used_bound = used_bound or 0
        work = self._broadcast_index_from_owner(layer_id)
        self.pending_index_works = [work]
        self.pending_remote_index_layer_id = layer_id
        self.pending_remote_index_broadcast = True

    def _broadcast_index_from_owner(self, layer_id: int, full: bool = False):
        """Owner-broadcast of the index-K scratch, sliced to the live rows."""
        head_dim = self.index_head_dim
        cap_elems = self.remote_index_k_buffer.numel()
        cap_tokens = cap_elems // head_dim
        if not full and self._idx_used_bound > 0:
            n_tok = min(self._idx_used_bound, cap_tokens)
        else:
            n_tok = cap_tokens
        n_elems = n_tok * head_dim
        owner_rank = self._get_layer_owner_rank(layer_id)
        if self.layer_shard_rank == owner_rank:
            slot = self.indexer_layer_id_to_slot.get(layer_id)
            buf = self.index_k_buffer[slot].view(-1)[:n_elems]
        else:
            buf = self.remote_index_k_buffer.view(-1)[:n_elems]
        cp_group = get_parallel().attn_cp_group
        return torch.distributed.broadcast(
            buf,
            src=cp_group.ranks[owner_rank],
            group=cp_group.device_group,
            async_op=True,
        )

    def prefetch_kv_buffer(
        self,
        layer_id: int,
        layer_transfer_counter: Optional[LayerDoneCounter] = None,
        layer_transfer_idx: Optional[int] = None,
        used_bound: Optional[int] = None,
    ) -> None:
        """Kick off an owner-broadcast of ``layer_id``'s latent KV.

        One layer ahead of the read. This is the prefix-row delivery of the
        patch-write scheme: the broadcast carries the owner's pre-write bytes
        (prefix rows correct, this forward's new-token rows stale), and the
        layer's ``set_kv_buffer`` (``cp_gathered_chunk=True``) scatters the
        CP-gathered new rows on top and registers the layer so the get skips
        the broadcast entirely. Without the set-side patch-write the pending
        copy is discarded and the get falls back to the synchronous
        broadcast.
        """
        if self.remote_kv_layer_id == layer_id:
            return
        if self.pending_remote_kv_broadcast:
            if self.pending_remote_kv_layer_id == layer_id:
                return
            self._finalize_pending_kv_broadcast(
                set_remote_layer_id=False, wait_works=False
            )

        if layer_transfer_counter is not None and layer_transfer_idx is not None:
            layer_transfer_counter.wait_until(layer_transfer_idx)
        self._kv_used_bound = used_bound or 0
        works = self._broadcast_kv_from_owner(layer_id, self._kv_used_bound)
        self.pending_kv_works = works
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
        # Fallback materialization: wait immediately, the caller reads the
        # scratch right after this returns.
        for _w in self._broadcast_kv_from_owner(layer_id, None):
            _w.wait()
        self.remote_kv_layer_id = layer_id

    def _ensure_remote_index_layer(self, layer_id: int) -> None:
        """Materialize ``layer_id``'s indexer rows on every rank (paired broadcast)."""
        if self.pending_remote_index_broadcast:
            self._finalize_pending_index_broadcast(
                set_remote_layer_id=False, wait_works=False
            )
        if self.remote_index_layer_id == layer_id:
            return
        if self.remote_index_k_buffer is None:
            return
        scale_src = None
        if self._is_layer_owned(layer_id):
            slot = self.indexer_layer_id_to_slot.get(layer_id)
            if slot is not None and self.index_k_scale_buffer is not None:
                scale_src = self.index_k_scale_buffer[slot]
        self._broadcast_index_from_owner(layer_id, full=True).wait()
        if self.remote_index_k_scale_buffer is not None:
            self._broadcast_buffer_from_owner(
                self.remote_index_k_scale_buffer, layer_id, scale_src
            ).wait()
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
        cp_gathered_full: bool = False,
        cp_gathered_chunk: bool = False,
    ):
        layer_id = layer.layer_id
        prefix_broadcast_pending = (
            self.pending_remote_kv_broadcast
            and self.pending_remote_kv_layer_id == layer_id
        )
        had_remote_copy = self.remote_kv_layer_id == layer_id
        # Consume/invalidate the pending remote copy on EVERY rank
        # (before the owned-only early return), so the follow-up get-time
        # broadcast decision stays paired across the CP group. A write
        # this layer must re-broadcast on both ranks.
        if self.pending_remote_kv_layer_id == layer_id:
            self._finalize_pending_kv_broadcast(set_remote_layer_id=False)
        if self.remote_kv_layer_id == layer_id:
            self.remote_kv_layer_id = None
        if (
            cp_gathered_full
            or (cp_gathered_chunk and (prefix_broadcast_pending or had_remote_copy))
        ) and self._remote_store_supported():
            if self._is_layer_owned(layer_id):
                super().set_kv_buffer(layer, loc_info, cache_k, cache_v)
                self.remote_kv_layer_id = layer_id
            else:
                self._store_remote_copy(layer, loc_info, cache_k, cache_v)
            return
        if not self._is_layer_owned(layer_id):
            return
        super().set_kv_buffer(layer, loc_info, cache_k, cache_v)

    def _remote_store_supported(self) -> bool:
        """Whether the remote scratch can be seeded by the CP gather path.

        Must be config-only (identical on every rank) so the get-time
        broadcast decision stays paired across the CP group.
        """
        return not (self.use_combined_kv_buffer or self.use_fia_nz)

    def _store_remote_copy(
        self,
        layer: RadixAttention,
        loc_info: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ) -> None:
        """Cache a CP-gathered full-sequence copy into the remote scratch."""
        if self.use_combined_kv_buffer or self.use_fia_nz:
            return
        loc, _, _ = unwrap_write_loc(loc_info)
        if self.dsa_kv_cache_store_c8:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            packed = self._pack_dsa_quant_kv_cache(cache_k, cache_v)
            torch_npu.npu_scatter_nd_update_(
                self.remote_k_buffer.view(-1, 1, self.kv_cache_dim),
                loc.view(-1, 1),
                packed.view(-1, 1, self.kv_cache_dim),
            )
        else:
            if cache_v is None:
                cache_k, cache_v = cache_k.split(
                    [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            modelslim_kv_scheme = getattr(layer, "scheme", None)
            if modelslim_kv_scheme is not None:
                cache_k, cache_v, _ = modelslim_kv_scheme.quantize_cache(
                    cache_k, cache_v
                )
            elif cache_k.dtype != self.dtype:
                cache_k = cache_k.to(self.dtype)
                cache_v = cache_v.to(self.dtype)
            if self.store_dtype != self.dtype:
                cache_k = cache_k.view(self.store_dtype)
                cache_v = cache_v.view(self.store_dtype)
            torch_npu.npu_scatter_nd_update_(
                self.remote_k_buffer.view(-1, 1, self.kv_lora_rank),
                loc.view(-1, 1),
                cache_k.view(-1, 1, self.kv_lora_rank),
            )
            torch_npu.npu_scatter_nd_update_(
                self.remote_v_buffer.view(-1, 1, self.qk_rope_head_dim),
                loc.view(-1, 1),
                cache_v.view(-1, 1, self.qk_rope_head_dim),
            )
        self.remote_kv_layer_id = layer.layer_id

    def set_index_k_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        cp_gathered_full: bool = False,
        cp_gathered_chunk: bool = False,
    ):
        prefix_broadcast_pending = (
            self.pending_remote_index_broadcast
            and self.pending_remote_index_layer_id == layer_id
        )
        had_remote_copy = self.remote_index_layer_id == layer_id
        if self.pending_remote_index_layer_id == layer_id:
            self._finalize_pending_index_broadcast(set_remote_layer_id=False)
        # A write invalidates any cached remote copy before the owned-only
        # early return, so the next read re-broadcasts on every rank.
        self.invalidate_index_layer(layer_id)
        if (
            cp_gathered_full
            or (cp_gathered_chunk and (prefix_broadcast_pending or had_remote_copy))
        ) and self._remote_index_store_supported():
            if self._is_layer_owned(layer_id):
                super().set_index_k_buffer(layer_id, loc, index_k)
                self.remote_index_layer_id = layer_id
            else:
                self._store_remote_index_copy(layer_id, loc, index_k)
            return
        if not self._is_layer_owned(layer_id):
            return
        super().set_index_k_buffer(layer_id, loc, index_k)

    def _remote_index_store_supported(self) -> bool:
        # The quant indexer keeps a separate fp32 scale row per token that
        # the scratch store below does not handle; quant configs keep the
        # broadcast path. Config-only so the get-time broadcast decision
        # stays paired across the CP group.
        return self.remote_index_k_scale_buffer is None

    def _store_remote_index_copy(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
    ) -> None:
        """Cache a CP-gathered full-sequence index row copy into the remote
        scratch, so the follow-up get_index_k_buffer read hits
        ``remote_index_layer_id`` instead of issuing a synchronous owner
        broadcast."""
        if index_k.dtype != self.dtype:
            index_k = index_k.to(self.dtype)
        if self.store_dtype != self.dtype:
            index_k = index_k.view(self.store_dtype)
        torch_npu.npu_scatter_nd_update_(
            self.remote_index_k_buffer.view(-1, 1, self.index_head_dim),
            loc.view(-1, 1),
            index_k.view(-1, 1, self.index_head_dim),
        )
        self.remote_index_layer_id = layer_id

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

    def get_kv_transfer_layer_span(self, num_entries: int) -> int:
        """Layer count covered by ``num_entries`` contiguous buf entries.

        ``get_contiguous_buf_infos`` reports one entry per layer per buffer
        group (the groups of ``_get_disagg_buffer_entries``: latent k, non-c8
        v, indexer k, index scale), so the owned layer count is the entry
        count divided by the group count — not the raw entry count.
        """
        groups = 1  # latent k (v stays packed into k in c8 mode)
        if not self.dsa_kv_cache_store_c8:
            groups += 1  # v
        if self.index_head_dim is not None:
            groups += 1  # indexer k
            if self.index_k_scale_buffer is not None:
                groups += 1  # index scale
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
