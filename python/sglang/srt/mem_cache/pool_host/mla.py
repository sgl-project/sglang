from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_hybm,
    ascendc_io_enabled,
    ensure_hybm_capacity,
    hybm_host_memory_enabled,
    to_device_no_sync,
    track_pinned_staging,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.utils import is_cuda, is_hip, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)

_ASCENDC_LAYER_GROUP_DEFAULT = 8


def _ascendc_layer_group_size() -> int:
    """Layer-group size for the AscendC sparse-copy pipeline (env-tunable)."""
    raw = os.environ.get("SGLANG_HICACHE_LAYER_GROUP_SIZE", "")
    try:
        size = int(raw) if raw else _ASCENDC_LAYER_GROUP_DEFAULT
    except ValueError:
        logger.warning(
            "Invalid SGLANG_HICACHE_LAYER_GROUP_SIZE=%r; using %d",
            raw,
            _ASCENDC_LAYER_GROUP_DEFAULT,
        )
        size = _ASCENDC_LAYER_GROUP_DEFAULT
    return size


class MLATokenToKVPoolHost(HiSparseHostPoolMixin, HostKVCache):
    device_pool: MLATokenToKVPool
    mtp_draft_device_pools: tuple[MLATokenToKVPool, ...] = ()

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
        mtp_draft_device_pools: Sequence[MLATokenToKVPool] = (),
        dcp_size: int = 1,
        dcp_rank: int = 0,
        *,
        pool_label: str = "kv",
    ):
        self.override_kv_cache_dim = override_kv_cache_dim
        self.mtp_draft_device_pools = tuple(mtp_draft_device_pools)
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            pool_label=pool_label,
        )
        # The JIT HiCache kernels also build with hipcc (ROCm): the PTX-only
        # helpers in hicache.cuh are guarded by USE_ROCM and the staged
        # write-back kernel has a ROCm path, so enable them on HIP too. This
        # keeps the ROCm write-back path consistent with CUDA.
        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize
        )

        if self.layout == "page_first":
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data
            transposed = self.kv_buffer.transpose(0, 1)
            self.data_refs = [transposed[i] for i in range(self.layer_num)]
        else:
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.mtp_draft_device_pools:
            device_pools = (self.device_pool, *self.mtp_draft_device_pools)
            self.packed_device_data_ptrs = torch.cat(
                [pool.data_ptrs for pool in device_pools]
            )
            self.packed_device_kv_buffers = [
                buffer for pool in device_pools for buffer in pool.kv_buffer
            ]
        self._init_write_back_staging_buffers()

    def get_contiguous_buf_infos(self):
        """Return (data_ptrs, data_lens, item_lens) in the same format as device pool,
        for registering host memory with the disaggregation transfer engine."""
        data_ptrs = [int(self.data_ptrs[i].item()) for i in range(self.layer_num)]
        data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size * self.page_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        # FP8 DSA packs K/V into the single device k_buffer (device v_buffer is
        # empty and never transferred). Exposed for the L3 store to skip the
        # dead v component when generating per-page keys/pointers.
        self.dsa_kv_cache_store_fp8 = getattr(
            self.device_pool, "dsa_kv_cache_store_fp8", False
        )
        self.target_layer_num = self._effective_host_layer_num()
        self.layer_num = self.target_layer_num + len(self.mtp_draft_device_pools)
        self.kv_cache_dim = self.override_kv_cache_dim or (
            self.kv_lora_rank + self.qk_rope_head_dim
        )
        size_per_token = self.kv_cache_dim * self.dtype.itemsize * self.layer_num
        if (
            self.layout == "page_first_kv_split"
            and self.device_pool.index_head_dim is not None
        ):
            # Ascend allocates Indexer K as a third per-layer host buffer.  It
            # must participate in fixed-byte host capacity sizing as well.
            # Only physical Indexer layers carry one, which can be a subset of
            # all layers (e.g. GLM 5.2: 21 of 78).
            num_indexer_layers = getattr(
                self.device_pool, "num_indexer_layers", None
            )
            if num_indexer_layers is None:
                num_indexer_layers = self.layer_num
            size_per_token += (
                self.device_pool.index_head_dim
                * self.dtype.itemsize
                * num_indexer_layers
            )
            if getattr(self.device_pool, "index_k_scale_buffer", None) is not None:
                # FP32 quantization scale per token per indexer layer.
                size_per_token += 4 * num_indexer_layers
        return size_per_token

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            # Indexer buffers only exist for physical Indexer layers, which can
            # be a subset of all layers (e.g. GLM 5.2: 21 of 78).  The device
            # pool packs them as (num_indexer_layers, page, ...); mirror that
            # layer count here so transfer_kv_dim_exchange's layer check
            # (device dim0 == host dim1) holds.
            num_indexer_layers = getattr(
                self.device_pool, "num_indexer_layers", None
            )
            if num_indexer_layers is None:
                num_indexer_layers = self.layer_num
            indexer_dims = (
                self.page_num,
                num_indexer_layers,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            if getattr(self.device_pool, "dsa_kv_cache_store_fp8", False):
                # FP8 DSA packs latent+RoPE+scale into the device k_buffer;
                # mirror the packed width so the 2D memcpy row width matches.
                k_width = self.device_pool.kv_cache_dim
            else:
                k_width = self.kv_lora_rank
            if _is_npu and hybm_host_memory_enabled():
                # hybm-mapped host DRAM: required by the AIV sparse-copy IO
                # path (SGLANG_HICACHE_IO_ASCENDC) and usable by the legacy
                # memcpy2d path unchanged.  Size the GB-aligned reserve with
                # the combined bytes of all buffers allocated below.
                total_bytes = (
                    self.page_num
                    * self.page_size
                    * self.layer_num
                    * (k_width + self.qk_rope_head_dim)
                    * self.dtype.itemsize
                )
                if self.device_pool.index_head_dim is not None:
                    total_bytes += (
                        self.page_num
                        * self.page_size
                        * num_indexer_layers
                        * self.device_pool.index_head_dim
                        * self.dtype.itemsize
                    )
                if getattr(self.device_pool, "index_k_scale_buffer", None) is not None:
                    # FP32 scale mirror
                    total_bytes += (
                        self.page_num * self.page_size * num_indexer_layers * 4
                    )
                ensure_hybm_capacity(total_bytes, torch.npu.current_device())
                alloc_func = alloc_with_hybm
            self.k_buffer = alloc_func(
                (*base_dims, k_width),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*indexer_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Host-side mirror of the NPU quantized-Indexer FP32 scale cache
            # (see NPUMLATokenToKVPool.index_k_scale_buffer). Only present when
            # the device pool carries one (FP8 DSA + npu_quant_lightning_indexer).
            self.index_k_scale_buffer = None
            if getattr(self.device_pool, "index_k_scale_buffer", None) is not None:
                self.index_k_scale_buffer = alloc_func(
                    (*indexer_dims, 1),
                    dtype=torch.float32,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Return k_buffer to preserve original kv_buffer and data_refs init logic,
            # though Ascend doesn't use these parameters.
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def _init_write_back_staging_buffers(self):
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_buffer = None
        self.can_use_write_back_jit = False
        if self.layout != "page_first" or (_is_npu or _is_xpu or _is_mps):
            return

        # The staged write-back JIT kernel builds with hipcc and has a ROCm
        # path, so enable it on HIP too (consistent with the CUDA path).
        self.can_use_write_back_jit = (
            _is_cuda or _is_hip
        ) and can_use_write_back_jit_kernel(
            element_size=self.kv_cache_dim * self.dtype.itemsize,
        )
        if not self.can_use_write_back_jit:
            return

        self.staging_page_capacity = min(self.page_num, _WRITE_BACK_STAGING_PAGE_CHUNK)
        self.staging_token_capacity = self.staging_page_capacity * self.page_size
        self.staging_buffer = torch.empty(
            (
                self.staging_token_capacity,
                self.layer_num,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device=self.device_pool.device,
        )

    def _ascendc_layer_group(self, device_pool, device_layer_id):
        """Layer-group descriptor for the AscendC sparse-copy pipeline.

        Returns ``(layer_start, layer_num, index_k_layer_start,
        index_k_layer_num)`` when ``device_layer_id`` is the first layer of a
        group in the device pool's owned-layer space, otherwise None.  The
        group size comes from SGLANG_HICACHE_LAYER_GROUP_SIZE: 1 means one
        layer per group (a transfer launch per layer), <1 disables grouping
        (one-shot), default 8; a size covering all owned layers also
        degenerates to the one-shot form.

        The indexer range maps the group's absolute layer ids onto the
        indexer slot space (device_pool.indexer_layer_ids is sorted, so the
        covered slots form a contiguous range).  A group containing no
        indexer layers gets index_k_layer_num=0, which skips those
        components.
        """
        start, end = self._device_owned_layer_range(device_pool)
        span = end - start
        if span <= 0:
            return None
        group = _ascendc_layer_group_size()
        if group < 1 or group >= span:
            # group < 1 explicitly disables grouping (one-shot transfer of
            # the whole owned range); group >= span has no split point
            # either.  group == 1 is VALID and means one layer per group
            # (finest-grained pipelining: a kv_exchange launch per layer).
            group = span
        if (device_layer_id - start) % group != 0:
            return None
        lo = device_layer_id
        hi = min(lo + group, end)

        index_k_layer_start, index_k_layer_num = 0, -1
        indexer_layer_ids = getattr(device_pool, "indexer_layer_ids", None)
        if indexer_layer_ids and self.index_k_buffer is not None:
            # indexer_layer_ids holds absolute layer ids (PP global space);
            # the owned range above is in the pool-local layer space, so
            # convert before matching.  The ids are sorted, so the covered
            # slots form a contiguous range.
            start_layer = getattr(device_pool, "start_layer", 0)
            slot_lo = None
            slot_hi = None
            for slot, layer in enumerate(indexer_layer_ids):
                local = layer - start_layer
                if lo <= local < hi:
                    if slot_lo is None:
                        slot_lo = slot
                    slot_hi = slot + 1
            if slot_lo is None:
                index_k_layer_start, index_k_layer_num = 0, 0
            else:
                index_k_layer_start = slot_lo
                index_k_layer_num = slot_hi - slot_lo
        return lo, hi - lo, index_k_layer_start, index_k_layer_num

    def _transfer_ascendc_sparse_copy(
        self,
        device_pool,
        host_indices,
        device_indices,
        direction: TransferDirection,
        layer_start: int = 0,
        layer_num: int = -1,
        index_k_layer_start: int = 0,
        index_k_layer_num: int = -1,
    ) -> None:
        """One-shot KV transfer via the Memfabric acc_offload fused AIV kernel.

        Sends a compact metadata array (per-component layout pitches and
        layer ranges) plus the device-resident token indices to the acc_offload
        ``kv_exchange_copy`` kernel, which derives every (page, layer, split)
        block address on the device: no (src, dst, len) entry table is built
        on the host and the indices never round-trip through the CPU, so the
        transfer launch does not synchronize the load stream.

        The layer range arguments restrict the transfer to one layer group
        (layer-group pipelining); the defaults transfer everything.

        Requires the host pool to be hybm-backed (SGLANG_HICACHE_IO_ASCENDC
        implies SGLANG_HICACHE_HOST_MEM=hybm) since the kernel de-references
        host pointers directly.
        """
        from memfabric_hybrid import offload

        device = device_pool.k_buffer.device
        # The kernel reads the token indices directly from device memory.
        # Upload without a stream sync: a plain .to(device) from pageable
        # memory synchronizes the stream and would serialize the pipeline.
        if host_indices.device.type != "npu":
            host_indices = to_device_no_sync(host_indices, device)
        if device_indices.device.type != "npu":
            device_indices = to_device_no_sync(device_indices, device)
        # The kernel runs on the current (load) stream while the indices were
        # allocated on another stream; keep them alive until the copy retires.
        stream = torch.npu.current_stream()
        host_indices.record_stream(stream)
        device_indices.record_stream(stream)

        def comp_meta(dev_t, host_t, lo, hi):
            # dev_t: (layer, page, page_size, [1,] width) layer-first;
            # host_t: (page, layer, page_size, [1,] width) page-first.
            itemsize = dev_t.dtype.itemsize
            width = 1
            for dim in dev_t.shape[2:]:
                width *= dim
            return (
                dev_t.data_ptr(),
                host_t.data_ptr(),
                dev_t.stride(0) * itemsize,
                dev_t.stride(1) * itemsize,
                host_t.stride(0) * itemsize,
                host_t.stride(1) * itemsize,
                width * itemsize,
                lo,
                hi,
            )

        k_lo = layer_start
        if layer_num < 0:
            k_hi = device_pool.k_buffer.shape[0]
        else:
            k_hi = k_lo + layer_num
        # Both pools must share the layer index space (same limitation as the
        # legacy memcpy2d exchange op); catches e.g. MTP draft pools, whose
        # host rows live past the main pool's layers.
        if k_hi > self.k_buffer.shape[1] or k_hi > device_pool.k_buffer.shape[0]:
            raise RuntimeError(
                f"AscendC kv_exchange layer range [{k_lo}, {k_hi}) exceeds the "
                f"pool layer space (device={device_pool.k_buffer.shape[0]}, "
                f"host={self.k_buffer.shape[1]})"
            )

        comps = [
            comp_meta(device_pool.k_buffer, self.k_buffer, k_lo, k_hi),
        ]
        # FP8 DSA packs V into the device k_buffer; the device v_buffer is
        # empty and must be skipped.
        if device_pool.v_buffer.numel() > 0 and self.v_buffer.numel() > 0:
            comps.append(comp_meta(device_pool.v_buffer, self.v_buffer, k_lo, k_hi))

        device_index_k = getattr(device_pool, "index_k_buffer", None)
        if self.index_k_buffer is not None and device_index_k is not None:
            if index_k_layer_num < 0:
                ik_lo, ik_hi = 0, self.index_k_buffer.shape[1]
            else:
                ik_lo, ik_hi = index_k_layer_start, index_k_layer_start + index_k_layer_num
            if ik_hi > ik_lo:
                comps.append(
                    comp_meta(device_index_k, self.index_k_buffer, ik_lo, ik_hi)
                )
                device_scale = getattr(device_pool, "index_k_scale_buffer", None)
                if self.index_k_scale_buffer is not None and device_scale is not None:
                    comps.append(
                        comp_meta(
                            device_scale,
                            self.index_k_scale_buffer,
                            ik_lo,
                            ik_hi,
                        )
                    )
        if len(comps) > 4:
            raise RuntimeError(
                f"AscendC kv_exchange supports at most 4 components, got {len(comps)}"
            )

        num_pages = host_indices.numel() // self.page_size
        direction_value = direction.value if isinstance(direction, TransferDirection) else int(direction)
        vals = [
            len(comps),
            num_pages,
            self.page_size,
            direction_value,
            device_indices.data_ptr(),
            host_indices.data_ptr(),
        ]
        for comp in comps:
            vals.extend(comp)
        
        pinned_meta = torch.tensor(vals, dtype=torch.int64, pin_memory=True)
        meta = torch.empty(pinned_meta.shape, dtype=torch.int64, device=device)
        meta.copy_(pinned_meta, non_blocking=True)
        track_pinned_staging(pinned_meta)
        ret = offload.kv_exchange_copy(meta, device)
        if ret != 0:
            raise RuntimeError(f"offload.kv_exchange_copy failed with code {ret}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if layer_id == 0 and torch.distributed.get_rank() == 0:
            logger.debug(
                "[%s][L2-LOAD] slots=%d host_idx_dev=%s dev_idx_dev=%s "
                "layout=%s io_backend=%s num_layers=%d",
                self.__class__.__name__,
                host_indices.numel(),
                host_indices.device,
                device_indices.device,
                self.layout,
                io_backend,
                self.device_pool.layer_num,
            )
        if not is_draft and not self._is_device_layer_owned(device_pool, layer_id):
            return
        host_indices = self.dcp_kernel_indices(host_indices)
        device_indices = self.dcp_kernel_indices(device_indices)
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        device_layer_id = 0 if is_draft else layer_id

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[device_layer_id],
                        cache_src=self.kv_buffer[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[host_layer_id],
                        dst=device_pool.kv_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=device_pool.kv_buffer[device_layer_id],
                        cache_src=self.data_refs[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_pool.kv_buffer[device_layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=host_layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[host_layer_id]],
                    dst_layers=[device_pool.kv_buffer[device_layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[device_layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=host_layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                if _is_npu and ascendc_io_enabled():
                    # AscendC layer-group pipelining: at each group boundary
                    # layer, launch one fused kv_exchange kernel covering the
                    # group.  The per-layer complete(i) events recorded by the
                    # caller then gate compute at group granularity, letting
                    # later groups' DMA overlap the current group's compute.
                    group = self._ascendc_layer_group(device_pool, device_layer_id)
                    if group is not None:
                        layer_start, layer_num, ik_start, ik_num = group
                        self._transfer_ascendc_sparse_copy(
                            device_pool,
                            host_indices,
                            device_indices,
                            TransferDirection.H2D,
                            layer_start=layer_start,
                            layer_num=layer_num,
                            index_k_layer_start=ik_start,
                            index_k_layer_num=ik_num,
                        )
                    return
                # memcpy2d layer-group pipelining: at each group boundary
                # layer, enqueue that group's 2D copies through the op's
                # layer_start/layer_num range (the op's run-merge loop
                # restricts both its merged and per-page forms to the range).
                # The per-layer complete(i) events recorded by the caller then
                # gate compute at group granularity, letting later groups' DMA
                # overlap the current group's compute — the same pipelining as
                # the AscendC path above.
                group = self._ascendc_layer_group(device_pool, device_layer_id)
                if group is not None:
                    layer_start, layer_num, ik_start, ik_num = group
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        device_index_k_scale=getattr(
                            device_pool, "index_k_scale_buffer", None
                        ),
                        host_index_k_scale=self.index_k_scale_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                        layer_start=layer_start,
                        layer_num=layer_num,
                        index_k_layer_start=ik_start,
                        index_k_layer_num=ik_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        # Indices arrive already translated by backup_from_device_all_layer.
        # MTP draft layers do not participate in CP layer sharding.
        host_layer_id = layer_id if is_draft else self._host_layer_index(layer_id)
        device_layer_id = 0 if is_draft else layer_id

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.kv_buffer[host_layer_id],
                        cache_src=device_pool.kv_buffer[device_layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    transfer_kv_per_layer_mla(
                        src=device_pool.kv_buffer[device_layer_id],
                        dst=self.kv_buffer[host_layer_id],
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        cache_dst=self.data_refs[host_layer_id],
                        cache_src=device_pool.kv_buffer[device_layer_id],
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    raise ValueError(
                        "Layer-sharded MLA HiCache backup with page_first layout "
                        "requires the JIT one-layer kernel."
                    )
            else:
                raise ValueError(
                    f"Layer-sharded HiCache backup does not support layout: {self.layout}"
                )
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[device_pool.kv_buffer[device_layer_id]],
                    dst_layers=[self.kv_buffer[host_layer_id]],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(
                    "Layer-sharded direct HiCache backup only supports "
                    f"layer_first layout, got {self.layout}"
                )
        else:
            raise ValueError(
                f"Layer-sharded HiCache backup does not support IO backend: {io_backend}"
            )

    def _resolve_device_transfer_buffers(self, device_pool):
        if self.mtp_draft_device_pools:
            return self.packed_device_data_ptrs, self.packed_device_kv_buffers
        return device_pool.data_ptrs, device_pool.kv_buffer

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if torch.distributed.get_rank() == 0:
            logger.debug(
                "[%s][L2-BACKUP] slots=%d host_idx_dev=%s dev_idx_dev=%s "
                "layout=%s io_backend=%s num_layers=%d",
                self.__class__.__name__,
                device_indices.numel(),
                host_indices.device,
                device_indices.device,
                self.layout,
                io_backend,
                self.device_pool.layer_num,
            )
        host_indices = self.dcp_kernel_indices(host_indices)
        device_indices = self.dcp_kernel_indices(device_indices)
        if self._is_device_layer_sharded(device_pool):
            for layer_id in self._owned_device_layer_ids(device_pool):
                self._backup_from_device_per_layer(
                    device_pool, host_indices, device_indices, layer_id, io_backend
                )
            for draft_layer_id, draft_device_pool in enumerate(
                self.mtp_draft_device_pools
            ):
                self._backup_from_device_per_layer(
                    draft_device_pool,
                    host_indices,
                    device_indices,
                    self.device_pool.layer_num + draft_layer_id,
                    io_backend,
                    is_draft=True,
                )
            return

        # data_ptrs/kv_buffers are only needed by the kernel/direct backends;
        # kernel_ascend (NPU) uses k_buffer/v_buffer/index_k_buffer instead and
        # NPU device pools do not expose data_ptrs.
        if io_backend != "kernel_ascend":
            device_data_ptrs, device_kv_buffers = self._resolve_device_transfer_buffers(
                device_pool
            )

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer_mla(
                        src_layers=device_data_ptrs,
                        dst_layers=self.data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                if self.can_use_write_back_jit:
                    jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                        ptr_src=device_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        staging=self.staging_buffer,
                        dst=self.kv_buffer,
                        page_size=self.page_size,
                    )
                else:
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_kv_buffers,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_kv_buffers,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                if _is_npu and ascendc_io_enabled():
                    self._transfer_ascendc_sparse_copy(
                        device_pool,
                        host_indices,
                        device_indices,
                        TransferDirection.D2H,
                    )
                    return
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    device_index_k=device_pool.index_k_buffer,
                    host_index_k=self.index_k_buffer,
                    device_index_k_scale=getattr(
                        device_pool, "index_k_scale_buffer", None
                    ),
                    host_index_k_scale=self.index_k_scale_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        assert self.dcp_size == 1, (
            "HiCache L3 storage paths are not yet DCP-aware (per-rank shards "
            "need dcp_rank-scoped keys); --hicache-storage-backend with "
            "--dcp-size > 1 should have been rejected at server start."
        )
        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "page_first_kv_split":
            # TODO (iforgetmyname): merge mla kv
            k_buffer_data_ptr = self.k_buffer.data_ptr()
            v_buffer_data_ptr = self.v_buffer.data_ptr()
            index_k_buffer = getattr(self, "index_k_buffer", None)
            index_k_buffer_data_ptr = (
                index_k_buffer.data_ptr() if index_k_buffer is not None else None
            )
            scale_buffer = getattr(self, "index_k_scale_buffer", None)
            scale_buffer_data_ptr = (
                scale_buffer.data_ptr() if scale_buffer is not None else None
            )
            # k row width mirrors the device pool (packed dim for FP8 DSA).
            k_width = self.k_buffer.shape[-1]
            k_item_size = self.k_buffer.element_size()
            # Indexer buffers cover only physical Indexer layers, which can be
            # a subset of all layers (e.g. GLM 5.2: 21 of 78).
            num_indexer_layers = (
                index_k_buffer.shape[1] if index_k_buffer is not None else 0
            )
            index_k_width = (
                index_k_buffer.shape[-1] if index_k_buffer is not None else 0
            )
            index_k_item_size = (
                index_k_buffer.element_size() if index_k_buffer is not None else 0
            )
            # FP8 DSA packs V into k_buffer; the device v_buffer is empty and
            # never transferred, so the host v mirror holds no valid data and
            # must not be persisted to storage.
            skip_v = getattr(self, "dsa_kv_cache_store_fp8", False)
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    k_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * k_width
                    * k_item_size
                )
                ptr_list.append(k_ptr)
                if not skip_v:
                    v_ptr = (
                        v_buffer_data_ptr
                        + indices[index]
                        * self.layer_num
                        * self.qk_rope_head_dim
                        * self.dtype.itemsize
                    )
                    ptr_list.append(v_ptr)
                if index_k_buffer_data_ptr is not None:
                    # Host index_k layout is (page_num, num_indexer_layers,
                    # page_size, 1, index_head_dim).
                    ptr_list.append(
                        index_k_buffer_data_ptr
                        + indices[index]
                        * num_indexer_layers
                        * index_k_width
                        * index_k_item_size
                    )
                if scale_buffer_data_ptr is not None:
                    # Host scale layout is (page_num, num_indexer_layers,
                    # page_size, 1, 1) FP32: one scale value per token per
                    # indexer layer.
                    ptr_list.append(
                        scale_buffer_data_ptr
                        + indices[index] * num_indexer_layers * 4
                    )
            k_element_size = self.layer_num * k_item_size * self.page_size * k_width
            v_element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.qk_rope_head_dim
            )
            index_k_element_size = (
                num_indexer_layers * index_k_item_size * self.page_size * index_k_width
            )
            scale_element_size = num_indexer_layers * 4 * self.page_size
            element_size_list = []
            for _ in range(0, len(indices), self.page_size):
                element_size_list.append(k_element_size)
                if not skip_v:
                    element_size_list.append(v_element_size)
                if index_k_buffer_data_ptr is not None:
                    element_size_list.append(index_k_element_size)
                if scale_buffer_data_ptr is not None:
                    element_size_list.append(scale_element_size)
            return ptr_list, element_size_list
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.kv_cache_dim
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.kv_cache_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        """Return True if per-page strides are multiples of *page_size_bytes*.

        When O_DIRECT is used with any file-based NIXL backend, every data pointer
        passed to the kernel must be page-aligned.  In zero-copy mode the
        pointer for KV page ``p`` is:

            base_ptr + p * page_size * layer_num * kv_cache_dim * itemsize

        For this to be page-aligned (given a page-aligned ``base_ptr``) the per-page
        stride must itself be a multiple of the OS page size.
        """
        if self.layout not in ("page_first", "page_first_direct"):
            return False
        stride = (
            self.page_size * self.layer_num * self.kv_cache_dim * self.dtype.itemsize
        )
        base_aligned = self.kv_buffer.data_ptr() % page_size_bytes == 0
        return base_aligned and stride % page_size_bytes == 0
