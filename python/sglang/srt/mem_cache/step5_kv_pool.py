"""Native FULL/SWA pool owning Step5's indexer cache and compression state."""

import logging
from contextlib import nullcontext

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    _SwaStrategy,
    build_pool_entry,
    register_stack_strategy,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
from sglang.srt.mem_cache.swa_memory_pool import GB, SWAKVPool

logger = logging.getLogger(__name__)


def get_step5_ring_capacity(max_verify_width: int, compression_ratio: int = 8) -> int:
    """Retain R-1 history inputs and every candidate, rounded to complete regions.

    max_verify_width includes the initial target input; use 1 without speculation.
    Writing p aliases p-C < floor(base/R)*R for all candidates in this capacity.
    """
    if max_verify_width < 1:
        raise ValueError("maximum verify input width must be positive")
    required = compression_ratio - 1 + max_verify_width
    return ((required + compression_ratio - 1) // compression_ratio) * compression_ratio


class Step5Fp8FullKVPool(MHATokenToKVPool):
    def _create_buffers(self):
        if self.post_capture_active or self.use_hnd or self.head_num != 1:
            raise ValueError(
                "Step5 FP8 FULL KV requires eager NHD storage with one KV head"
            )
        with (
            self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            slots = self.size + self.page_size
            self.store_dtype = torch.float8_e4m3fn
            self.k_buffer = [
                torch.zeros(
                    (slots, self.head_num, self.head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.v_buffer = [
                torch.zeros(
                    (slots, self.head_num, self.v_head_dim),
                    dtype=self.store_dtype,
                    device=self.device,
                )
                for _ in range(self.layer_num)
            ]
            self.k_scale_buffer = [
                torch.ones(
                    (slots, self.head_num), dtype=torch.float32, device=self.device
                )
                for _ in range(self.layer_num)
            ]
            self.v_scale_buffer = [
                torch.ones(
                    (slots, self.head_num), dtype=torch.float32, device=self.device
                )
                for _ in range(self.layer_num)
            ]
        self.dq_k_buffer = None
        self.dq_v_buffer = None
        self._kv_buffer_descs = self._build_kv_buffer_descs()
        self._init_data_ptrs_and_strides()

    def _get_key_buffer(self, layer_id: int):
        return self.k_buffer[layer_id - self.start_layer]

    def _get_value_buffer(self, layer_id: int):
        return self.v_buffer[layer_id - self.start_layer]

    def get_kv_scale_buffer(self, layer_id: int):
        index = layer_id - self.start_layer
        return self.k_scale_buffer[index], self.v_scale_buffer[index]

    def set_kv_buffer(
        self,
        layer,
        loc_info,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
        dcp_kv_mask=None,
    ):
        from sglang.kernels.ops.attention.step5.fp8_kv import step5_fp8_kv_store

        if dcp_kv_mask is not None:
            raise NotImplementedError("Step5 FP8 FULL KV does not support DCP")
        loc, _, _ = unwrap_write_loc(loc_info)
        layer_id = (
            layer_id_override if layer_id_override is not None else layer.layer_id
        )
        index = layer_id - self.start_layer
        step5_fp8_kv_store(
            cache_k,
            cache_v,
            loc,
            self.k_buffer[index],
            self.v_buffer[index],
            self.k_scale_buffer[index],
            self.v_scale_buffer[index],
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return [
            [
                tensor[indices].to("cpu", copy=True)
                for tensor in (
                    self.k_buffer[i],
                    self.v_buffer[i],
                    self.k_scale_buffer[i],
                    self.v_scale_buffer[i],
                )
            ]
            for i in range(self.layer_num)
        ]

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        for i, saved in enumerate(kv_cache_cpu):
            for tensor, value in zip(
                (
                    self.k_buffer[i],
                    self.v_buffer[i],
                    self.k_scale_buffer[i],
                    self.v_scale_buffer[i],
                ),
                saved,
            ):
                tensor[indices] = value.to(tensor.device)
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def get_kv_scale_buf_infos(self):
        tensors = self.k_scale_buffer + self.v_scale_buffer
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes * self.page_size for tensor in tensors],
        )


def _require_outside_capture(device: torch.device) -> None:
    if device.type == "cuda":
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Step5 DSA storage construction/reset is outside capture only"
                )


class Step5TokenToKVPool(SWAKVPool):
    """Target-only indexer cache with explicit page and request ownership.

    Compressed index keys follow FULL physical pages. Raw K/Z rings are
    request-owned, with layer-major backing tensors and per-layer lists that alias the same storage.

    The caller supplies the actual request-pool capacity, including padding
    slot zero and any PD preallocation slots. Existing backend, HiCache and PD
    consumers can use the same tensor views and buffer ordering. Draft SWA and
    MTP input history remain separately owned.

    Memory-saver requires a drained reset before KV pause and after resume,
    before request admission: KV-tag storage has no CPU backup. Reset must not
    write unmapped storage, and callers own cross-stream ordering.
    """

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        if req_pool_index is None or not 0 < req_pool_index < self.num_req_slots:
            raise ValueError("Step5 retraction requires a live request slot")
        cpu_copy = super().get_cpu_copy(indices, mamba_indices, req_pool_index)
        # Physical region indices preserve logical token order across page remapping.
        regions = torch.unique_consecutive(indices // self.compression_ratio)
        keys = self.compressed_index_key_buffer.flatten(1, 2)
        cpu_copy["compressed_index_keys"] = keys[:, regions].to("cpu", copy=True)
        cpu_copy["index_key_ring"] = self.index_key_ring[:, req_pool_index].to(
            "cpu", copy=True
        )
        cpu_copy["compression_logits_ring"] = self.compression_logits_ring[
            :, req_pool_index
        ].to("cpu", copy=True)
        cpu_copy["index_positions"] = self.index_position_buffer[:, req_pool_index].to(
            "cpu", copy=True
        )
        return cpu_copy

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        if req_pool_index is None or not 0 < req_pool_index < self.num_req_slots:
            raise ValueError("Step5 retraction requires a live request slot")
        regions = torch.unique_consecutive(indices // self.compression_ratio)
        if regions.numel() != kv_cache_cpu["compressed_index_keys"].shape[1]:
            raise ValueError("Step5 retraction changed the compressed region layout")
        super().load_cpu_copy(kv_cache_cpu, indices, mamba_indices, req_pool_index)
        keys = self.compressed_index_key_buffer.flatten(1, 2)
        keys[:, regions] = kv_cache_cpu["compressed_index_keys"].to(keys.device)
        self.index_key_ring[:, req_pool_index].copy_(kv_cache_cpu["index_key_ring"])
        self.compression_logits_ring[:, req_pool_index].copy_(
            kv_cache_cpu["compression_logits_ring"]
        )
        # Publish validity only after restoring the corresponding raw inputs.
        self.index_position_buffer[:, req_pool_index].copy_(
            kv_cache_cpu["index_positions"]
        )
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def __init__(
        self,
        *,
        size: int,
        size_swa: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        swa_attention_layer_ids: list[int],
        full_attention_layer_ids: list[int],
        device: str,
        num_req_slots: int,
        index_head_dim: int = 256,
        compression_ratio: int = 8,
        index_ring_size: int = 8,
        enable_memory_saver: bool = False,
        enable_kv_cache_copy: bool = False,
        enable_alt_stream: bool = True,
        kv_cache_layout: str | None = None,
        token_to_kv_pool_class: type[MHATokenToKVPool] = MHATokenToKVPool,
        post_capture_active: bool = False,
        full_kv_fp8: bool = False,
    ):
        # Kernel layout constraints; engine configuration owns device, layer
        # selection and int32 address-capacity validation.
        if (page_size, compression_ratio, index_head_dim) != (64, 8, 256):
            raise ValueError("Step5 indexer kernels require P64/R8/D256")
        if size <= 0 or size % page_size:
            raise ValueError("Step5 requires a positive, page-aligned capacity")
        if num_req_slots < 2:
            raise ValueError("num_req_slots must include padding and a usable slot")
        if index_ring_size < compression_ratio or index_ring_size % compression_ratio:
            raise ValueError("Step5 ring capacity must be a positive multiple of R8")
        if dtype != torch.bfloat16 or post_capture_active:
            raise ValueError(
                "Step5 requires BF16 KV storage without post-capture allocation"
            )
        _require_outside_capture(torch.device(device))
        # The parent constructor calls get_kv_size_bytes before allocating
        # the indexer buffers. Recompute mem_usage after all allocations.
        self.compressed_index_key_buffer: torch.Tensor | None = None
        self.index_key_ring: torch.Tensor | None = None
        self.compression_logits_ring: torch.Tensor | None = None
        self.index_position_buffer: torch.Tensor | None = None
        super().__init__(
            size=size,
            size_swa=size_swa,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            swa_attention_layer_ids=swa_attention_layer_ids,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            enable_memory_saver=enable_memory_saver,
            enable_kv_cache_copy=enable_kv_cache_copy,
            enable_alt_stream=enable_alt_stream,
            kv_cache_layout=kv_cache_layout,
            token_to_kv_pool_class=token_to_kv_pool_class,
            full_kv_pool_class=Step5Fp8FullKVPool if full_kv_fp8 else None,
            post_capture_active=post_capture_active,
        )
        self.full_kv_fp8 = full_kv_fp8
        self.dsa_layer_ids = tuple(full_attention_layer_ids)
        self.full_num_pages = size // page_size + 1
        self.num_req_slots = num_req_slots
        self.index_head_dim = index_head_dim
        self.compression_ratio = compression_ratio
        self.index_ring_size = index_ring_size
        request_shape = (self.full_layer_nums, self.num_req_slots)
        ring_shape = (*request_shape, self.index_ring_size, 1, self.index_head_dim)

        # Indexer buffers share the FULL KV pool's memory-saver and PD allocator.
        pool = self.full_kv_pool
        with (
            pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE),
            (
                torch.cuda.use_mem_pool(pool.custom_mem_pool)
                if pool.enable_custom_mem_pool
                else nullcontext()
            ),
        ):
            self.compressed_index_key_buffer = torch.empty(
                (
                    self.full_layer_nums,
                    self.full_num_pages,
                    self.page_size // self.compression_ratio,
                    1,
                    self.index_head_dim,
                ),
                dtype=torch.uint8,
                device=pool.device,
            )
            # Pending index keys and compression logits for non-online R8,
            # indexed by layer, request slot and absolute-position residue.
            self.index_key_ring = torch.empty(
                ring_shape, dtype=torch.bfloat16, device=pool.device
            )
            # Z contains per-channel compression logits, not region scores.
            self.compression_logits_ring = torch.empty(
                ring_shape, dtype=torch.bfloat16, device=pool.device
            )
            self.index_position_buffer = torch.empty(
                (*request_shape, self.index_ring_size),
                dtype=torch.int64,
                device=pool.device,
            )
            self.reset_dsa_state()

        # Cache layer aliases once; getters do not create views in the hot path.
        self.compressed_index_key_buffer_pool = list(
            self.compressed_index_key_buffer.unbind(0)
        )
        self.index_key_ring_pool = list(self.index_key_ring.unbind(0))
        self.compression_logits_ring_pool = list(self.compression_logits_ring.unbind(0))
        self.index_position_buffer_pool = list(self.index_position_buffer.unbind(0))
        k_size, v_size = self.get_kv_size_bytes()
        self.mem_usage = (k_size + v_size) / GB

    def _transfer_full_attention_id(self, layer_id: int) -> int:
        index, is_swa = self.layers_mapping[layer_id]
        if is_swa:
            raise ValueError("Step5 indexer state belongs to FULL attention layers")
        self._wait_for_layer(layer_id)
        return index

    def get_compressed_index_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.compressed_index_key_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def get_index_key_ring(self, layer_id: int) -> torch.Tensor:
        return self.index_key_ring_pool[self._transfer_full_attention_id(layer_id)]

    def get_compression_logits_ring(self, layer_id: int) -> torch.Tensor:
        return self.compression_logits_ring_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def get_index_position_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_position_buffer_pool[
            self._transfer_full_attention_id(layer_id)
        ]

    def reset_dsa_state(self) -> None:
        """Invalidate in place after a drained flush; never call during capture.

        Only construction or an owner-proven drained flush may invoke this.
        All readers' WAR dependencies must have completed before these writes,
        and later readers must observe their stream ordering. CompressedIndexKeys/raw K/Z
        bytes remain untouched; invalid position tags prohibit
        their use. This is not page eviction or request-generation advancement.
        """
        _require_outside_capture(torch.device(self.device))
        self.index_position_buffer.fill_(-1)

    @staticmethod
    def _get_state_buf_infos(tensors: list[torch.Tensor]):
        """PD geometry: each layer tensor's first axis is a page or request slot."""
        if not tensors or any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("Step5 PD state tensors must be nonempty and contiguous")
        return (
            [tensor.data_ptr() for tensor in tensors],
            [tensor.nbytes for tensor in tensors],
            [tensor[0].nbytes for tensor in tensors],
        )

    def get_compressed_index_key_buf_infos(self):
        """Physical-page-indexed completed R8 compressed_index_keys for PD transfer."""
        compressed_index_keys = self.compressed_index_key_buffer
        return self._get_state_buf_infos(
            [
                compressed_index_keys[index]
                for index in range(compressed_index_keys.shape[0])
            ]
        )

    def get_committed_tail_buf_infos(self):
        """Request-slot-indexed input ring and absolute-position validity tags."""
        tensors = [
            tensor[layer]
            for layer in range(len(self.dsa_layer_ids))
            for tensor in (
                self.index_key_ring,
                self.compression_logits_ring,
                self.index_position_buffer,
            )
        ]
        return self._get_state_buf_infos(tensors)

    @property
    def aux_state(self):
        """Expose pool-owned DSA state through the existing PD interface."""
        return self

    def attach_aux_state(self, state) -> None:
        raise RuntimeError("Step5 target state is owned by the pool constructor")

    def get_pd_state_components(self):
        """Page-owned compressed_index_keys first, then the request-slot-owned tail."""
        components = (
            ("step5_compressed_index_key", self.get_compressed_index_key_buf_infos()),
        )
        if self.full_kv_fp8:
            components += (("block_scale", self.get_kv_scale_buf_infos()),)
        return components + (("step5_dsa_tail", self.get_committed_tail_buf_infos()),)

    def invalidate_pd_receive(self, request_slot: int) -> None:
        """Invalidate a decode destination before publishing its remote address."""
        if not 0 < request_slot < self.num_req_slots:
            raise ValueError("Step5 PD destination request slot is out of range")
        self.index_position_buffer[:, request_slot].fill_(-1)

    def initialize_synthetic_pd_receive(
        self, *, request_slot: int, frontier: int
    ) -> None:
        """Install a canonical synthetic tail for the fake PD transport only.

        Decode startup warmup deliberately bypasses the prefill worker and does
        not transfer state.  Step5 nevertheless executes a real decode graph,
        so give that synthetic request valid local tail geometry.  Missing raw
        K/Z values are zeroed; fake-transfer output is discarded and must never
        be exposed as a real handoff.
        """
        if not 0 < request_slot < self.num_req_slots or not 0 <= frontier <= 1048576:
            raise ValueError("Step5 synthetic PD destination is out of range")
        self.invalidate_pd_receive(request_slot)
        self.index_key_ring[:, request_slot].zero_()
        self.compression_logits_ring[:, request_slot].zero_()
        for position in range(frontier // 8 * 8, frontier):
            self.index_position_buffer[
                :, request_slot, position % self.index_ring_size
            ] = position

    def bind_pd_receive(
        self, *, request_slot: int, generation: int, frontier: int
    ) -> bool:
        """Validate the common PD handoff coordinates; ring tags travel with K/Z."""
        if (
            not 0 < request_slot < self.num_req_slots
            or not 0 < generation < 2**63
            or not 0 <= frontier <= 1048576
        ):
            return False
        return True

    def get_kv_size_bytes(self):
        k_bytes, v_bytes = super().get_kv_size_bytes()
        k_bytes += sum(self.get_dsa_size_bytes().values())
        return k_bytes, v_bytes

    def get_dsa_size_bytes(self) -> dict[str, int]:
        """Count backing buffers once; exclude layer aliases and workspaces."""
        return {
            "compressed_index_key_buffer": (
                self.compressed_index_key_buffer.nbytes
                if self.compressed_index_key_buffer is not None
                else 0
            ),
            "index_ring": sum(
                tensor.nbytes
                for tensor in (
                    self.index_key_ring,
                    self.compression_logits_ring,
                    self.index_position_buffer,
                )
                if tensor is not None
            ),
        }


class _Step5SwaStrategy(_SwaStrategy):
    """Native FULL/SWA stack plus the page-owned compressed index key sidecar."""

    def matches(self, kvcache, components):
        return isinstance(kvcache, Step5TokenToKVPool) and super().matches(
            kvcache, components
        )

    def build(self, **kwargs):
        from sglang.srt.mem_cache.hicache_storage import PoolName, SidecarPoolSpec
        from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
        from sglang.srt.mem_cache.pool_host.common import get_allocator_type
        from sglang.srt.runtime_context import get_memory

        memory = get_memory()
        # The sidecar reuses a generic page-row host pool whose other
        # layout/backend branches are not validated for Step5.
        if (memory.hicache_mem_layout, memory.hicache_io_backend) != (
            "page_first",
            "kernel",
        ):
            raise NotImplementedError(
                "Step5 DSA HiCache supports kernel/page_first only"
            )
        kv_pool = kwargs["kvcache"]
        compressed_index_keys = kv_pool.compressed_index_key_buffer
        # flatten() of a non-contiguous buffer returns a copy, which would
        # detach the host pool from the live device pages.
        if not compressed_index_keys.is_contiguous():
            raise ValueError("Step5 compressed_index_keys must be contiguous")

        result = super().build(**kwargs)
        kv_host = result.host_pool_group.get_pool(PoolName.KV)
        if kv_pool.page_size != kv_host.page_size:
            raise ValueError(
                "Step5 compressed_index_keys and FULL host page sizes must match"
            )

        def add_page_sidecar(name, page_rows):
            host_pool = DeepSeekV4PagedHostPool(
                pool_name=str(name),
                device_buffers=page_rows,
                item_bytes=page_rows[0].shape[-1],
                num_host_pages=kv_host.page_num,
                slot_page_size=kv_host.page_size,
                layout=kv_host.layout,
                device=kv_host.device,
                pin_memory=kv_host.pin_memory,
                allocator_type=get_allocator_type(),
                page_aligned_only=True,
            )
            result.cache_controller.register_host_pool_entry(
                build_pool_entry(
                    name=name,
                    host_pool=host_pool,
                    device_pool=kv_pool,
                    layer_mapping={
                        global_id - kv_pool.start_layer: local_id
                        for local_id, global_id in enumerate(kv_pool.dsa_layer_ids)
                    },
                    transfer_layer_id_max=result.cache_controller.transfer_layer_id_max,
                )
            )
            result.sidecars.append(
                SidecarPoolSpec(pool_name=name, indices_from_pool=PoolName.KV)
            )

        # One page row holds a FULL page's completed R8 keys for one DSA layer,
        # so the sidecar moves whole rows at the FULL page number.
        add_page_sidecar(
            PoolName.STEP5_COMPRESSED_INDEX_KEY,
            list(compressed_index_keys.flatten(2).unbind(0)),
        )
        result.pools_desc += " + Step5 compressed index keys"

        full_kv_pool = kv_pool.full_kv_pool
        if isinstance(full_kv_pool, Step5Fp8FullKVPool):
            if memory.hicache_storage_backend is not None:
                raise NotImplementedError(
                    "Step5 FP8 FULL KV does not support HiCache storage backends"
                )
            for name, buffers in (
                (PoolName.STEP5_FP8_K_SCALE, full_kv_pool.k_scale_buffer),
                (PoolName.STEP5_FP8_V_SCALE, full_kv_pool.v_scale_buffer),
            ):
                if any(not buffer.is_contiguous() for buffer in buffers):
                    raise ValueError("Step5 FP8 KV scales must be contiguous")
                add_page_sidecar(
                    name,
                    [
                        buffer.view(torch.uint8).view(-1, kv_host.page_size * 4)
                        for buffer in buffers
                    ],
                )
            result.pools_desc += " + Step5 FP8 KV scales"
        return result


# Loading the device pool installs its HiCache strategy before cache construction.
register_stack_strategy(_Step5SwaStrategy())
