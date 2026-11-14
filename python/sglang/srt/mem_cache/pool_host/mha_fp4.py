"""HiCache L2 for NVFP4 payload bytes and both persistent scale layouts.

The device description carries token-linear scales and, when allocated, native
page/head/token scales. Payload copies reuse the ordinary MHA host pool with
byte-width geometry; scales travel through page-granular side buffers. Target
and draft buffers share the same packed local layer order as their payloads.
"""

from __future__ import annotations

from typing import Sequence

import msgspec
import torch

from sglang.kernels.ops.kvcache.hicache import (
    can_use_write_back_jit_kernel,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_all_layer_mla_staged_lf_pf as jit_transfer_hicache_all_layer_mla_staged_lf_pf,
)
from sglang.kernels.ops.kvcache.hicache import (
    transfer_hicache_one_layer_mla as jit_transfer_hicache_one_layer_mla,
)
from sglang.srt.mem_cache.device_pool_info import (
    KVBufferLayout,
    KVScaleLayout,
    MHABufferInfo,
)
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)


def nvfp4_payload_info(info: MHABufferInfo) -> MHABufferInfo:
    """Describe existing packed bytes without copying or allocating tensors."""
    info.validate()
    if info.scales is None or info.scales.layout != KVScaleLayout.NVFP4:
        raise ValueError("NVFP4 host transfers require an explicit NVFP4 scale layout")
    if info.head_dim != info.v_head_dim or info.head_dim % 16:
        raise NotImplementedError(
            "NVFP4 host transfers require equal K/V head dimensions divisible by 16"
        )
    if info.dtype != torch.uint8 or info.buffers.layout != KVBufferLayout.NHD:
        raise NotImplementedError(
            "NVFP4 host transfers require packed uint8 NHD buffers"
        )
    if info.transfer_buffers is not None:
        raise NotImplementedError(
            "NVFP4 host transfers do not support relocatable buffers"
        )
    token_capacity = info.buffers.k[0].shape[0] if info.buffers.k else None
    for side in (info.buffers.k, info.buffers.v):
        for buffer in side:
            if (
                buffer.ndim != 3
                or buffer.shape[0] != token_capacity
                or tuple(buffer.shape[1:]) != (info.head_num, info.head_dim // 2)
                or buffer.shape[0] % info.page_size
                or not buffer.is_contiguous()
            ):
                raise ValueError(
                    "NVFP4 payload must contain contiguous, page-aligned "
                    "[token, head, packed_dim] byte rows"
                )
    # The underlying tensors and pointer tables stay exactly the same. Removing
    # scales here is deliberate: this descriptor is only for the payload copy;
    # the subclass separately retains and transfers every scale allocation.
    return msgspec.structs.replace(
        info, head_dim=info.head_dim // 2, v_head_dim=info.v_head_dim // 2, scales=None
    )


def _scale_buffers(info: MHABufferInfo):
    scales = info.scales
    result = [
        ("k_scale_buffer", False, scales.k),
        ("v_scale_buffer", False, scales.v),
    ]
    if scales.native_k or scales.native_v:
        result.extend(
            [
                ("native_k_scale_buffer", True, scales.native_k),
                ("native_v_scale_buffer", True, scales.native_v),
            ]
        )
    return result


class MHATokenToKVPoolFP4Host(MHATokenToKVPoolHost):
    def __init__(
        self,
        buffer_info: MHABufferInfo,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        device_capacity_tokens: int,
        start_layer: int,
        end_layer: int,
        draft_buffer_infos: Sequence[MHABufferInfo] = (),
        write_back_staging: tuple[torch.Tensor, torch.Tensor] | None = None,
        pool_label: str = "kv",
        head_group_num: int = 1,
        host_token_capacity: int | None = None,
    ):
        if layout != "page_first":
            raise NotImplementedError("NVFP4 host transfers support only page_first")
        infos = (buffer_info, *draft_buffer_infos)
        payload_infos = tuple(nvfp4_payload_info(info) for info in infos)
        if not buffer_info.buffers.k:
            raise NotImplementedError(
                "NVFP4 host pool requires at least one target layer"
            )
        if any(info.page_size != page_size for info in infos):
            raise ValueError(
                "NVFP4 target, drafts and host must use the same page size"
            )
        if any(len(info.buffers.k) != 1 for info in draft_buffer_infos):
            raise ValueError("NVFP4 packed draft descriptions must contain one layer")

        self._scale_names = []
        self._scale_page_bytes = {}
        kinds = _scale_buffers(buffer_info)
        expected_names = {name for name, _, _ in kinds}
        for info in infos:
            if {name for name, _, _ in _scale_buffers(info)} != expected_names:
                raise ValueError(
                    "NVFP4 target and drafts must carry the same scale layouts"
                )
        for name, rows_are_pages, _ in kinds:
            buffers = []
            page_bytes = page_size * buffer_info.head_num * buffer_info.head_dim // 16
            for info in infos:
                selected = dict((n, b) for n, _, b in _scale_buffers(info))[name]
                for payload, scale in zip(info.buffers.k, selected):
                    expected_shape = (
                        (
                            payload.shape[0] // page_size,
                            info.head_num,
                            page_size,
                            info.head_dim // 16,
                        )
                        if rows_are_pages
                        else (payload.shape[0], info.head_num, info.head_dim // 16)
                    )
                    if (
                        tuple(scale.shape) != expected_shape
                        or scale.element_size() != 1
                        or scale.device != info.device
                        or not scale.is_contiguous()
                    ):
                        raise ValueError(
                            f"NVFP4 {name} has invalid shape, dtype, device or strides"
                        )
                    buffers.append(scale)
            self._scale_names.append(name)
            self._scale_page_bytes[name] = page_bytes
            setattr(self, f"_scale_dev_{name}", buffers)
            setattr(self, f"_scale_host_{name}", None)

        if not all(
            can_use_write_back_jit_kernel(element_size=n)
            for n in self._scale_page_bytes.values()
        ):
            raise NotImplementedError(
                "NVFP4 scale pages require the staged JIT transfer kernel"
            )
        super().__init__(
            payload_infos[0],
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            device_capacity_tokens=device_capacity_tokens,
            start_layer=start_layer,
            end_layer=end_layer,
            draft_buffer_infos=payload_infos[1:],
            write_back_staging=write_back_staging,
            pool_label=pool_label,
            head_group_num=head_group_num,
            host_token_capacity=host_token_capacity,
        )
        if not self.can_use_jit or not self.can_use_write_back_jit:
            raise NotImplementedError(
                "NVFP4 host pool requires staged JIT kernels (CUDA/HIP, io_backend='kernel')"
            )
        self._init_scale_buffers()

    def get_size_per_token(self):
        payload = super().get_size_per_token()
        scales = sum(self._scale_page_bytes.values()) // self.page_size
        return payload + scales * self.layer_num

    def _init_scale_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        for name in self._scale_names:
            page_bytes = self._scale_page_bytes[name]
            host = alloc_func(
                (self.page_num, self.layer_num, page_bytes),
                dtype=torch.uint8,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            setattr(self, f"_scale_host_{name}", host)
            setattr(self, f"_scale_host_layers_{name}", list(host.transpose(0, 1)))
            rows = [
                buffer.view(torch.uint8).reshape(-1, page_bytes)
                for buffer in getattr(self, f"_scale_dev_{name}")
            ]
            setattr(self, f"_scale_rows_{name}", rows)
            setattr(
                self,
                f"_scale_ptrs_{name}",
                torch.tensor(
                    [row.data_ptr() for row in rows],
                    dtype=torch.uint64,
                    device=self._transfer_device,
                ),
            )
            setattr(
                self,
                f"_scale_staging_{name}",
                torch.empty(
                    (self.staging_page_capacity, self.layer_num, page_bytes),
                    dtype=torch.uint8,
                    device=self._transfer_device,
                ),
            )

    def _validate_transfer(self, host_indices, device_indices, io_backend):
        if io_backend != "kernel":
            raise NotImplementedError(
                "NVFP4 host transfers support only io_backend='kernel'"
            )
        if (
            host_indices.ndim != 1
            or device_indices.ndim != 1
            or host_indices.numel() != device_indices.numel()
            or device_indices.numel() % self.page_size
        ):
            raise ValueError(
                "NVFP4 host transfers require matching whole-page index lists"
            )

    def _page_ids(self, indices):
        # The allocator supplies canonical aligned pages in token order. Checking
        # shape/count here catches partial transfers without a GPU-to-CPU sync;
        # arbitrary within-page permutations are not a supported input contract.
        return indices[:: self.page_size] // self.page_size

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        self._validate_transfer(host_indices, device_indices, io_backend)
        if device_indices.numel() == 0:
            return
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        device_pages, host_pages = (
            self._page_ids(device_indices),
            self._page_ids(host_indices),
        )
        if host_pages.is_cuda:
            host_pages = host_pages.cpu()
        for name in self._scale_names:
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=getattr(self, f"_scale_ptrs_{name}"),
                src_indices=device_pages,
                dst_indices=host_pages,
                staging=getattr(self, f"_scale_staging_{name}"),
                dst=getattr(self, f"_scale_host_{name}"),
                page_size=1,
            )

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft=False,
    ):
        self._validate_transfer(host_indices, device_indices, io_backend)
        if device_indices.numel() == 0:
            return
        if not is_draft and not 0 <= layer_id < self.target_layer_num:
            return
        if is_draft and not self.target_layer_num <= layer_id < self.layer_num:
            raise ValueError("NVFP4 draft layer must be a packed local draft index")
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        device_pages, host_pages = (
            self._page_ids(device_indices),
            self._page_ids(host_indices),
        )
        for name in self._scale_names:
            jit_transfer_hicache_one_layer_mla(
                cache_dst=getattr(self, f"_scale_rows_{name}")[layer_id],
                indices_dst=device_pages,
                cache_src=getattr(self, f"_scale_host_layers_{name}")[layer_id],
                indices_src=host_pages,
                element_dim=self._scale_page_bytes[name],
            )

    def destroy(self):
        for name in getattr(self, "_scale_names", ()):
            host = getattr(self, f"_scale_host_{name}", None)
            if host is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(host)
            for attr in ("host", "host_layers", "staging", "rows", "ptrs", "dev"):
                setattr(self, f"_scale_{attr}_{name}", None)
        super().destroy()

    def _storage_pages_unsupported(self):
        return NotImplementedError(
            "NVFP4 host pool does not expose flat L3 pages: scales live separately"
        )

    def get_data_page(self, index, flat=True):
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self):
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index, data_page):
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(self, indices, split_factor):
        raise self._storage_pages_unsupported()
