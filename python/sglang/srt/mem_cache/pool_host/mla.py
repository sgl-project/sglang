from __future__ import annotations

import logging
from typing import Optional, Sequence

import msgspec
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_hicache_jit_kernel,
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla as jit_transfer_hicache_all_layer_mla,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_page_unified as jit_transfer_hicache_all_layer_mla_staged_lf_page_unified,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla_page_unified_lf as jit_transfer_hicache_one_layer_mla_page_unified_lf,
)
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.device_pool_info import MLABufferInfo, MLASplitBuffers
from sglang.srt.mem_cache.pool_host._kvcacheio import (
    MLA_KVCACHEIO_OPS,
    require_kvcacheio_ops,
)
from sglang.srt.mem_cache.pool_host.base import (
    _WRITE_BACK_STAGING_PAGE_CHUNK,
    HostKVCache,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    make_kernel_ptr_table,
)
from sglang.srt.mem_cache.pool_host.hisparse import HiSparseHostPoolMixin
from sglang.srt.mem_cache.pool_host.npu_host_memory import reserve_mla_split_host_memory
from sglang.srt.mem_cache.pool_host.page_unified import PageUnifiedLayout
from sglang.srt.platforms import current_platform
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
elif _is_xpu:
    from sglang.srt.mem_cache.pool_host._kvcacheio import (
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
    )
if _is_npu:
    from sgl_kernel_npu.operators.mem_cache import (
        transfer_kv_all_layer_kv_split,
        transfer_kv_all_layer_kv_split_lf_pf,
        transfer_kv_per_layer_kv_split,
        transfer_kv_per_layer_kv_split_pf_lf,
    )

logger = logging.getLogger(__name__)


class MLAHostAllocation(msgspec.Struct, frozen=True, kw_only=True):
    """Allocator-only geometry; it carries no device buffers or transfer promise."""

    dtype: torch.dtype
    kv_lora_rank: int
    qk_rope_head_dim: int
    kv_cache_dim: int
    size_per_token: int
    draft_layer_num: int = 0
    packed_fp8: bool = False


class MLATokenToKVPoolHost(HiSparseHostPoolMixin, HostKVCache):
    """Host mirror built by the assembler or HiSparse from captured MLA buffers.

    Capacity and reserved CP rows remain caller policy. Transfer methods retain
    the shared HostKVCache signature but use the captured tensors and slot map.
    """

    def __init__(
        self,
        buffer_info: MLABufferInfo | MLAHostAllocation,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
        draft_buffer_infos: Sequence[MLABufferInfo] = (),
        dcp_size: int = 1,
        dcp_rank: int = 0,
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        target_layer_num: int,
        layer_shard_enabled: bool = False,
        pool_label: str = "kv",
        write_back_staging: torch.Tensor | None = None,
        page_unified_layout: PageUnifiedLayout | None = None,
        host_token_capacity: int | None = None,
    ):
        self._is_dummy = isinstance(buffer_info, MLAHostAllocation)
        self.override_kv_cache_dim = override_kv_cache_dim
        if _is_xpu and not self._is_dummy:
            require_kvcacheio_ops(
                pool=type(self).__name__, layout=layout, ops=self._kvcacheio_ops
            )
        self.dtype = buffer_info.dtype
        self.layout = layout
        self.page_unified_layout = page_unified_layout
        self.kv_lora_rank = buffer_info.kv_lora_rank
        self.qk_rope_head_dim = buffer_info.qk_rope_head_dim
        # Existing builders explicitly resolve packed row width before construction.
        self.kv_cache_dim = override_kv_cache_dim or (
            buffer_info.kv_cache_dim
            if self._is_dummy
            else self.kv_lora_rank + self.qk_rope_head_dim
        )
        self.dsa_kv_cache_store_fp8 = buffer_info.packed_fp8
        self.target_layer_num = target_layer_num
        self.layer_num = target_layer_num + (
            buffer_info.draft_layer_num if self._is_dummy else len(draft_buffer_infos)
        )
        self._layer_shard_enabled = layer_shard_enabled
        self._native_buffers = None
        if self._is_dummy:
            if draft_buffer_infos:
                raise ValueError(
                    "Allocator-only MLA does not accept physical draft buffers"
                )
            self._size_per_token = buffer_info.size_per_token
        else:
            infos = (buffer_info, *draft_buffer_infos)
            for index, info in enumerate(infos):
                info.validate()
                if info.scales is not None:
                    role = "target" if index == 0 else f"draft[{index - 1}]"
                    raise ValueError(
                        f"MLA host cache does not support {role} FP4 KV: "
                        "packed payload and separate block scales cannot be restored"
                    )
                if index and (
                    (info.kv_cache_dim, info.dtype)
                    != (buffer_info.kv_cache_dim, buffer_info.dtype)
                    or type(info.buffers) is not type(buffer_info.buffers)
                ):
                    raise ValueError(
                        f"packed draft KV row {(info.kv_cache_dim, info.dtype)} differs "
                        f"from target KV row {(buffer_info.kv_cache_dim, buffer_info.dtype)}"
                    )
                if index and len(info.layer_indices) != 1:
                    raise ValueError(
                        f"MLA packed draft[{index - 1}] must contain one layer, got {info.layer_indices}"
                    )
            self._transfer_device = buffer_info.device
            self._device_layer_num = buffer_info.device_layer_num
            self._target_layer_indices = buffer_info.layer_indices
            self._draft_layer_num = len(draft_buffer_infos)
            slots = (
                *buffer_info.layer_indices,
                *range(
                    self._device_layer_num,
                    self._device_layer_num + self._draft_layer_num,
                ),
            )
            self._buffer_index_by_layer = {layer: i for i, layer in enumerate(slots)}
            self._host_layer_by_controller = dict(
                zip(buffer_info.layer_indices, range(len(buffer_info.layer_indices)))
            )
            self._host_layer_by_controller.update(
                {
                    self._device_layer_num + i: target_layer_num + i
                    for i in range(self._draft_layer_num)
                }
            )
            self._size_per_token = (
                self.kv_cache_dim * self.dtype.itemsize * self.layer_num
            )
            if isinstance(buffer_info.buffers, MLASplitBuffers):
                self._native_buffers = buffer_info.buffers
                self._native_infos = infos
                native = self._native_buffers
                index_layers = len(native.index_layer_indices)
                if (
                    layout in ("page_first_kv_split", "layer_first_kv_split")
                    and native.index_k is not None
                ):
                    self._size_per_token += (
                        native.index_width * self.dtype.itemsize * index_layers
                    )
                    if native.index_scales is not None:
                        self._size_per_token += 4 * index_layers
                self.device_kv_buffers = ()
                self.device_data_ptrs = None
            else:
                self.device_kv_buffers = tuple(
                    tensor for info in infos for tensor in info.buffers
                )
                self.device_data_ptrs = (
                    torch.cat([info.data_ptrs for info in infos])
                    if draft_buffer_infos
                    else buffer_info.data_ptrs
                )
        super().__init__(
            dtype=self.dtype,
            device_capacity_tokens=device_capacity_tokens,
            size_per_token=self.get_size_per_token(),
            start_layer=start_layer,
            end_layer=end_layer,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            page_size=page_size,
            layout=layout,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
            dcp_size=dcp_size,
            dcp_rank=dcp_rank,
            pool_label=pool_label,
            host_token_capacity=host_token_capacity,
            is_dummy=self._is_dummy,
        )
        if self._is_dummy:
            self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
            self.layout_dim = self.token_stride_size * self.layer_num
            self.can_use_jit = False
            self.staging_page_capacity = 0
            self.staging_token_capacity = 0
            self.staging_buffer = None
            self.kv_buffer = None
            self.data_refs = None
            self.data_ptrs = None
            logger.info(
                "MLATokenToKVPoolHost dummy mode: allocator-only, size=%d tokens, "
                "saving %.2f GB host memory",
                self.size,
                self.size * self.size_per_token / 1e9,
            )
            return

        self._initialize_transfer_views(write_back_staging)
        if (
            self._native_buffers is not None
            and self._transfer_device.type == "npu"
            and self.layout in ("page_first_kv_split", "layer_first_kv_split")
        ):
            self._init_npu_transfer_bindings()

    def _initialize_transfer_views(self, write_back_staging=None) -> None:
        # The JIT HiCache kernels also build with hipcc (ROCm): the PTX-only
        # helpers in hicache.cuh are guarded by USE_ROCM and the staged
        # write-back kernel has a ROCm path, so enable them on HIP too. This
        # keeps the ROCm write-back path consistent with CUDA.
        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
            page_size=self.page_size,
            element_size=self.kv_cache_dim * self.dtype.itemsize,
        )

        if self.layout == "page_unified":
            # Transfers address the fused page buffer directly.
            self.data_refs = []
        elif self.layout in ("page_first", "page_first_kv_split"):
            # Transpose [page, layer, ...] -> [layer, page, ...] to get per-layer views
            # This swaps strides without copying data.  For page_first_kv_split,
            # kv_buffer is the k_buffer with the same page-major dims.
            transposed = self.kv_buffer.transpose(0, 1)
            self.data_refs = [transposed[i] for i in range(self.layer_num)]
        else:
            self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = (
            make_kernel_ptr_table(
                self.data_refs,
                self._transfer_device,
                host_memory_registered=self.pin_memory,
            )
            if self.data_refs
            else None
        )
        if self._native_buffers is None:
            self._init_device_row_stride(self.device_kv_buffers)
        self._init_write_back_staging_buffers(write_back_staging)

    def get_contiguous_buf_infos(self):
        """Return (data_ptrs, data_lens, item_lens) in the same format as device pool,
        for registering host memory with the disaggregation transfer engine."""
        if self._is_dummy:
            return [], [], []
        if self.layout == "page_unified":
            return (
                [self.kv_buffer.data_ptr()],
                [self.kv_buffer.nbytes],
                [self.page_unified_layout.bytes_per_page],
            )
        data_ptrs = [tensor.data_ptr() for tensor in self.data_refs]
        if self.layout == "page_first_kv_split":
            # data_refs are per-layer views of the k_buffer (page-major), so
            # take the per-layer slab size instead of kv_buffer[i] (a page slab).
            data_lens = [x.nbytes for x in self.data_refs]
        else:
            data_lens = [self.kv_buffer[i].nbytes for i in range(self.layer_num)]
        item_lens = [self.token_stride_size * self.page_size] * self.layer_num
        return data_ptrs, data_lens, item_lens

    def get_size_per_token(self):
        return self._size_per_token

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def get_mapping_lengths(self) -> list[int]:
        if self.layout != "page_first_kv_split" or self._native_buffers is None:
            return super().get_mapping_lengths()
        native = self._native_buffers
        k_width = native.k_width if self.dsa_kv_cache_store_fp8 else self.kv_lora_rank
        per_row_widths = [
            self.layer_num * k_width * self.dtype.itemsize,
            self.layer_num * self.qk_rope_head_dim * self.dtype.itemsize,
        ]
        index_layers = len(native.index_layer_indices)
        if native.index_k is not None:
            per_row_widths.append(
                index_layers * native.index_width * self.dtype.itemsize
            )
        if native.index_scales is not None:
            # FP32 scale mirror
            per_row_widths.append(index_layers * 4)
        rows = self.page_num * self.page_size
        return [rows * width for width in per_row_widths]

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
        elif self.layout == "page_unified":
            # Same bytes as page_first_direct -- MLA is rank-replicated, so the
            # grid never cuts a head axis and there is nothing to permute. It
            # is still its own layout so the L3 namespace, the transfer arms
            # and the MHA side all name one thing.
            self.page_unified_layout = self.page_unified_layout or PageUnifiedLayout(
                page_size=self.page_size,
                layer_num=self.layer_num,
                head_num=1,
                head_group_num=1,
                head_dim=self.kv_cache_dim,
                itemsize=self.dtype.itemsize,
                is_mla=True,
            )
            dims = self.page_unified_layout.page_dims(self.page_num)
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "layer_first_kv_split":
            # NPU AscendC path: allocate layer-first split K/V via the shared
            # helper. FP8 DSA packs V into K, so v_width is 0 and v_buffer
            # becomes a 5-D zero-element empty tensor.
            native = self._native_buffers
            base_dims = (self.layer_num, self.page_num, self.page_size, 1)
            indexer_dims = (
                len(native.index_layer_indices),
                self.page_num,
                self.page_size,
                1,
            )
            return self._init_kv_split_buffer(
                base_dims,
                indexer_dims,
            )
        elif self.layout == "page_first_kv_split":
            # Indexer buffers only exist for physical Indexer layers, which can
            # be a subset of all layers (e.g. GLM 5.2: 21 of 78).  The device
            # pool packs them as (num_indexer_layers, page, ...); the Host
            # mirror uses (page, num_indexer_layers, ...).
            native = self._native_buffers
            base_dims = (self.page_num, self.layer_num, self.page_size, 1)
            indexer_dims = (
                self.page_num,
                len(native.index_layer_indices),
                self.page_size,
                1,
            )
            return self._init_kv_split_buffer(
                base_dims,
                indexer_dims,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
            registration_granularity_bytes=(
                self.page_size * self.layout_dim
                if self.layout in ("page_first", "page_first_direct", "page_unified")
                else None
            ),
        )
        return buffer

    def _init_kv_split_buffer(self, base_dims, indexer_dims):
        """Allocate split K/V (+ indexer) host buffers for NPU.

        ``base_dims``/``indexer_dims`` carry the axis order, so both
        ``page_first_kv_split`` (page-major) and ``layer_first_kv_split``
        (layer-major) share this allocation tail. ``v_width == 0`` yields a 5-D
        zero-element empty tensor because FP8 DSA packs V into K.
        """
        native = self._native_buffers
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        k_width = native.k_width if self.dsa_kv_cache_store_fp8 else self.kv_lora_rank
        v_width = 0 if self.dsa_kv_cache_store_fp8 else native.v_width

        reserve_mla_split_host_memory(
            num_pages=self.page_num,
            page_size=self.page_size,
            num_layers=self.layer_num,
            k_width=k_width,
            v_width=v_width,
            dtype=self.dtype,
            buffers=native,
        )
        self.k_buffer = alloc_func(
            (*base_dims, k_width),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        if v_width == 0:
            # FP8 DSA: V is packed into K, so the host v mirror holds no valid
            # data. Keep a 5-D zero-element tensor (not None) so component
            # builders can safely check ``numel() > 0`` without extra guards.
            self.v_buffer = torch.empty(
                (*base_dims, 0), dtype=self.dtype, device=self.device
            )
        else:
            self.v_buffer = alloc_func(
                (*base_dims, v_width),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        self.index_k_buffer = None
        if native.index_k is not None:
            self.index_k_buffer = alloc_func(
                (*indexer_dims, native.index_width),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        # Host-side mirror of the NPU quantized-Indexer FP32 scale cache.
        self.index_k_scale_buffer = None
        if native.index_scales is not None:
            self.index_k_scale_buffer = alloc_func(
                (*indexer_dims, 1),
                dtype=torch.float32,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        # Return k_buffer to preserve kv_buffer/data_refs init logic.
        return self.k_buffer

    def _init_write_back_staging_buffers(self, write_back_staging=None):
        self.staging_page_capacity = 0
        self.staging_token_capacity = 0
        self.staging_buffer = None
        self.can_use_write_back_jit = False
        if not current_platform.capabilities.hicache_device_kernels:
            return
        if self.layout == "page_unified":
            self._require_packed_device_rows("page_unified staged write-back")
            self._init_page_unified_staging(write_back_staging)
            return
        # The staged kernel reads whole device pages, so it needs packed rows.
        if (
            self.layout != "page_first"
            or self._layer_shard_enabled
            or not self.device_rows_packed
        ):
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
            device=self._transfer_device,
        )

    if _is_xpu:
        # {io_backend: {layout: kvcacheio ops}} the transfer methods below call,
        # checked at construction; keep in step with their branches.
        _kvcacheio_ops = MLA_KVCACHEIO_OPS

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
        assert not self._is_dummy, "load on a dummy (non-src MLA) host pool"
        if not is_draft and layer_id not in self._target_layer_indices:
            return
        host_indices = maybe_dcp_kernel_indices(
            host_indices, self.dcp_size, self.dcp_rank
        )
        device_indices = maybe_dcp_kernel_indices(
            device_indices, self.dcp_size, self.dcp_rank
        )
        # Draft transfer IDs follow all target device layers, while host rows
        # follow only this rank's target layers.
        host_layer_id = self._host_layer_by_controller[layer_id]
        if io_backend != "kernel_ascend":
            device_buffer = self.device_kv_buffers[
                self._buffer_index_by_layer[layer_id]
            ]

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        page_size=self.page_size,
                        cache_dst=device_buffer,
                        cache_src=self.kv_buffer[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    self._require_packed_device_rows("transfer_kv_per_layer_mla")
                    transfer_kv_per_layer_mla(
                        src=self.kv_buffer[host_layer_id],
                        dst=device_buffer,
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        page_size=self.page_size,
                        cache_dst=device_buffer,
                        cache_src=self.data_refs[host_layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    self._require_packed_device_rows("transfer_kv_per_layer_mla_pf_lf")
                    transfer_kv_per_layer_mla_pf_lf(
                        src=self.kv_buffer,
                        dst=device_buffer,
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        layer_id=host_layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )
            elif self.layout == "page_unified":
                jit_transfer_hicache_one_layer_mla_page_unified_lf(
                    device_buffer,
                    self.kv_buffer,
                    host_indices,
                    device_indices,
                    host_layer_id,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[host_layer_id]],
                    dst_layers=[device_buffer],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                self._require_packed_device_rows("transfer_kv_per_layer_direct_pf_lf")
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_buffer],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=host_layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                transfer = transfer_kv_per_layer_kv_split_pf_lf
            elif self.layout == "layer_first_kv_split":
                transfer = transfer_kv_per_layer_kv_split
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
            binding = self._npu_layer_bindings[layer_id]
            if binding is None:
                return
            transfer(
                device_indices=device_indices,
                host_indices=host_indices,
                **binding,
                impl="auto",
            )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_from_device_per_layer(
        self,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        assert not self._is_dummy, "backup on a dummy (non-src MLA) host pool"
        if io_backend not in ("kernel", "direct"):
            raise ValueError(
                f"Layer-sharded HiCache backup does not support IO backend: {io_backend}"
            )
        # Indices arrive already translated by backup_from_device_all_layer.
        # Draft transfer IDs follow all target device layers, while host rows
        # follow only this rank's target layers.
        host_layer_id = self._host_layer_by_controller[layer_id]
        device_buffer = self.device_kv_buffers[self._buffer_index_by_layer[layer_id]]

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        page_size=self.page_size,
                        cache_dst=self.kv_buffer[host_layer_id],
                        cache_src=device_buffer,
                        indices_dst=host_indices,
                        indices_src=device_indices,
                        element_dim=self.kv_cache_dim,
                    )
                else:
                    self._require_packed_device_rows("transfer_kv_per_layer_mla")
                    transfer_kv_per_layer_mla(
                        src=device_buffer,
                        dst=self.kv_buffer[host_layer_id],
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer_mla(
                        page_size=self.page_size,
                        cache_dst=self.data_refs[host_layer_id],
                        cache_src=device_buffer,
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
                    src_layers=[device_buffer],
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

    def _init_npu_transfer_bindings(self):
        self._npu_layer_bindings = {}
        pools = []
        layer_start = 0
        host_layer_axis = 0 if self.layout == "layer_first_kv_split" else 1

        def layer_view(buffer, axis, layer, count=1):
            if buffer is None or not buffer.numel():
                return buffer
            if layer == 0 and buffer.shape[axis] == count:
                return buffer
            return buffer.narrow(axis, layer, count)

        for i, info in enumerate(self._native_infos):
            native = info.native_buffers()
            num_layers = native.k.shape[0]
            layer_end = layer_start + num_layers
            is_target = i == 0
            if not is_target:
                if native.index_k is not None:
                    self._warn_draft_indexer_skipped(native)
                if not self._npu_draft_pool_compatible(native):
                    self._npu_layer_bindings[self._device_layer_num + i - 1] = None
                    layer_start = layer_end
                    continue
            host_k = layer_view(self.k_buffer, host_layer_axis, layer_start, num_layers)
            host_v = layer_view(self.v_buffer, host_layer_axis, layer_start, num_layers)
            pools.append((native, host_k, host_v, is_target))
            index_slots = (
                {layer: slot for slot, layer in enumerate(native.index_layer_indices)}
                if is_target and self.index_k_buffer is not None
                else {}
            )
            layers = (
                self._target_layer_indices
                if is_target
                else (self._device_layer_num + i - 1,)
            )
            for layer_id in layers:
                device_layer = layer_id if is_target else 0
                host_layer = self._host_layer_by_controller[layer_id]
                slot = index_slots.get(layer_id)
                index_k = (
                    layer_view(native.index_k, 0, slot) if slot is not None else None
                )
                index_scale = (
                    layer_view(native.index_scales, 0, slot)
                    if slot is not None
                    else None
                )
                if index_scale is not None and index_scale.ndim == 4:
                    index_scale = index_scale.unsqueeze(-1)
                self._npu_layer_bindings[layer_id] = dict(
                    device_k=layer_view(native.k, 0, device_layer),
                    host_k=layer_view(self.k_buffer, host_layer_axis, host_layer),
                    device_v=layer_view(native.v, 0, device_layer),
                    host_v=layer_view(self.v_buffer, host_layer_axis, host_layer),
                    device_index_k=index_k,
                    host_index_k=(
                        layer_view(self.index_k_buffer, host_layer_axis, slot)
                        if slot is not None
                        else None
                    ),
                    device_index_k_scale=index_scale,
                    host_index_k_scale=(
                        layer_view(self.index_k_scale_buffer, host_layer_axis, slot)
                        if slot is not None
                        else None
                    ),
                    page_size=self.page_size,
                    device_layer_id=0,
                    host_layer_id=0,
                    device_index_layer_id=0 if slot is not None else None,
                    host_index_layer_id=0 if slot is not None else None,
                )
            layer_start = layer_end
        self._npu_pool_bindings = tuple(pools)

    def _warn_draft_indexer_skipped(self, native):
        if getattr(self, "_draft_indexer_skip_warned", False):
            return
        self._draft_indexer_skip_warned = True
        logger.warning(
            "NPU HiCache kernel_ascend IO: MTP draft pool %s carries an "
            "index_k_buffer but the host MLA pool allocates indexer rows "
            "only for the target pool; draft indexer KV is not backed up.",
            type(native).__name__,
        )

    def _npu_draft_pool_compatible(self, native) -> bool:
        device_k = native.k
        device_v = native.v
        mismatch = None
        if (
            device_k.shape[-1] != self.k_buffer.shape[-1]
            or device_k.dtype != self.k_buffer.dtype
        ):
            mismatch = ("k_buffer", device_k, self.k_buffer)
        elif (
            device_v is not None
            and device_v.numel() > 0
            and self.v_buffer.numel() > 0
            and (
                device_v.shape[-1] != self.v_buffer.shape[-1]
                or device_v.dtype != self.v_buffer.dtype
            )
        ):
            mismatch = ("v_buffer", device_v, self.v_buffer)
        if mismatch is None:
            return True
        name, dev_t, host_t = mismatch
        if not getattr(self, "_draft_kv_skip_warned", False):
            self._draft_kv_skip_warned = True
            logger.warning(
                "NPU HiCache kernel_ascend IO: skipping MTP draft pool whose "
                "%s (%s, %s) does not match the host %s (%s, %s); "
                "its KV is not transferred.",
                name,
                tuple(dev_t.shape),
                dev_t.dtype,
                name,
                tuple(host_t.shape),
                host_t.dtype,
            )
        return False

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        assert not self._is_dummy, "backup on a dummy (non-src MLA) host pool"
        host_indices = maybe_dcp_kernel_indices(
            host_indices, self.dcp_size, self.dcp_rank
        )
        device_indices = maybe_dcp_kernel_indices(
            device_indices, self.dcp_size, self.dcp_rank
        )
        if self._layer_shard_enabled:
            for layer_id in self._target_layer_indices:
                self._backup_from_device_per_layer(
                    host_indices, device_indices, layer_id, io_backend
                )
            for draft_layer_id in range(self._draft_layer_num):
                self._backup_from_device_per_layer(
                    host_indices,
                    device_indices,
                    self._device_layer_num + draft_layer_id,
                    io_backend,
                    is_draft=True,
                )
            return

        if io_backend == "kernel_ascend":
            # NPU pools use contiguous multi-layer tensors and intentionally do
            # not build the CUDA-style data_ptrs array.
            device_data_ptrs, device_kv_buffers = None, None
        else:
            device_data_ptrs = self.device_data_ptrs
            device_kv_buffers = self.device_kv_buffers

        if io_backend == "kernel":
            if self.layout == "layer_first":
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer_mla(
                        page_size=self.page_size,
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.token_stride_size,
                        cache_src_stride_bytes=self.device_row_stride_bytes,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    self._require_packed_device_rows("transfer_kv_all_layer_mla")
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
                elif self.can_use_jit and not self.device_rows_packed:
                    # The per-layer host views of page_first rows sit
                    # `layout_dim` apart, which the all-layer kernel can step.
                    jit_transfer_hicache_all_layer_mla(
                        page_size=self.page_size,
                        ptr_dst=self.data_ptrs,
                        indices_dst=host_indices,
                        ptr_src=device_data_ptrs,
                        indices_src=device_indices,
                        cache_dst_stride_bytes=self.layout_dim,
                        cache_src_stride_bytes=self.device_row_stride_bytes,
                        element_size=self.kv_cache_dim * self.dtype.itemsize,
                    )
                else:
                    self._require_packed_device_rows("transfer_kv_all_layer_mla_lf_pf")
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_data_ptrs,
                        dst=self.kv_buffer,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_unified":
                jit_transfer_hicache_all_layer_mla_staged_lf_page_unified(
                    device_data_ptrs,
                    self._page_ids(device_indices),
                    self._page_ids(host_indices),
                    self.staging_buffer,
                    self.kv_buffer,
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
                self._require_packed_device_rows("transfer_kv_all_layer_direct_lf_pf")
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
                transfer = transfer_kv_all_layer_kv_split_lf_pf
            elif self.layout == "layer_first_kv_split":
                transfer = transfer_kv_all_layer_kv_split
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
            for native, host_k, host_v, is_target in self._npu_pool_bindings:
                transfer(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=native.k,
                    host_k=host_k,
                    device_v=native.v,
                    host_v=host_v,
                    device_index_k=native.index_k,
                    host_index_k=self.index_k_buffer if is_target else None,
                    device_index_k_scale=native.index_scales,
                    host_index_k_scale=self.index_k_scale_buffer if is_target else None,
                    page_size=self.page_size,
                    impl="auto",
                )
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        index = self._storage_page_index(index)
        if self.layout == "page_unified":
            page_id = index // self.page_size
            data_page = self.kv_buffer[page_id : page_id + 1]
        elif self.layout == "layer_first":
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
        index = self._storage_page_index(index)
        if self.layout == "page_unified":
            page_id = index // self.page_size
            self.kv_buffer[page_id : page_id + 1] = data_page.reshape(
                self.page_unified_layout.page_dims(1)
            )
        elif self.layout == "layer_first":
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

    def get_hybrid_pool_buffer(self):
        """Every physical host buffer one logical storage page spans.

        ``get_page_buffer_meta`` emits one pointer per physical buffer and
        Mooncake packs them into a single object per page, so each of them has
        to be registered with the transfer engine.  Registering only
        ``kv_buffer`` leaves the RoPE mirror and the DSA indexer buffers
        unregistered, which forces those segments onto a copy fallback (and
        fails outright on URAN/memfabric transfers).
        """
        buffers = [self.kv_buffer]
        if self.layout == "page_first_kv_split":
            # FP8 DSA packs V into K; the host V mirror holds no valid data.
            if not getattr(self, "dsa_kv_cache_store_fp8", False):
                buffers.append(getattr(self, "v_buffer", None))
            # Indexer buffers exist only for physical Indexer layers.
            buffers.append(getattr(self, "index_k_buffer", None))
            buffers.append(getattr(self, "index_k_scale_buffer", None))
        return [buf for buf in buffers if buf is not None and buf.numel() > 0]

    def get_page_buffer_meta(self, indices):
        """Return local buffer pointers and byte sizes for logical storage pages."""
        if self.dcp_size > 1 and self.layout == "page_first_kv_split":
            raise NotImplementedError(
                "DCP L3 zero-copy storage does not support split KV buffers."
            )
        assert len(indices) % self.logical_page_size == 0
        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "page_first_kv_split":
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
                    + indices[index] * self.layer_num * k_width * k_item_size
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
                        scale_buffer_data_ptr + indices[index] * num_indexer_layers * 4
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
        # Each logical page maps to one physical page in this rank's buffer.
        page_starts = [
            self._storage_page_index(index)
            for index in indices[:: self.logical_page_size]
        ]
        if self.layout == "layer_first":
            for index in page_starts:
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + index * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in page_starts:
                k_ptr = (
                    kv_buffer_data_ptr
                    + index * self.layer_num * self.kv_cache_dim * self.dtype.itemsize
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
