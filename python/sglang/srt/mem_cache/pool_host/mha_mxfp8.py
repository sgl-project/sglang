"""Host (L2) pool for the MXFP8 block-scaled MHA KV cache.

The fp8 payload rides the regular MHA host pool; this subclass adds the
per-page UE8M0 scale blocks, which the device pool keeps in the FA4
interleaved layout (num_pages, head, 32, page_size // 32, sf_dim). A page's
scales are one contiguous block, so they move as whole-page rows of
`n_heads * page_size * sf_dim` bytes with the same staged JIT kernels the
payload uses, just at page granularity (page_size=1 in kernel terms).
"""

from __future__ import annotations

from typing import Sequence

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
from sglang.srt.mem_cache.device_pool_info import KVScaleLayout, MHABufferInfo
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
)
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    _is_cuda,
    _is_hip,
)


class MHATokenToKVPoolMXFP8Host(MHATokenToKVPoolHost):
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
    ):
        if layout != "page_first":
            raise NotImplementedError(
                f"MXFP8 KV host pool supports only the page_first layout, got {layout!r}."
            )
        if draft_buffer_infos:
            raise NotImplementedError(
                "MXFP8 KV host pool does not pack MTP draft KV layers."
            )
        interleaved = (
            buffer_info.scales is not None
            and buffer_info.scales.layout == KVScaleLayout.MXFP8_INTERLEAVED
        )
        if not interleaved:
            raise NotImplementedError(
                "MXFP8 KV host pool requires the interleaved (page_size=128) scale layout."
            )
        k_scales = buffer_info.scales.k
        v_scales = buffer_info.scales.v
        # HostKVCache calls get_size_per_token during base initialization.
        self.k_sf_page_bytes = k_scales[0][0].numel()
        self.v_sf_page_bytes = v_scales[0][0].numel()
        self.k_scale_host: torch.Tensor | None = None
        self.v_scale_host: torch.Tensor | None = None
        super().__init__(
            buffer_info,
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
            write_back_staging=write_back_staging,
            pool_label=pool_label,
        )
        if self.page_size != buffer_info.page_size:
            raise ValueError(
                "MXFP8 KV host pool moves scales per page, so the host page size "
                f"({self.page_size}) must equal the device page size "
                f"({buffer_info.page_size})."
            )
        if not self.can_use_write_back_jit or not all(
            can_use_write_back_jit_kernel(element_size=size)
            for size in (self.k_sf_page_bytes, self.v_sf_page_bytes)
        ):
            raise NotImplementedError(
                "MXFP8 KV host pool needs the staged JIT write-back kernel "
                "(io_backend='kernel', page_first layout, CUDA or HIP)."
            )
        self._init_scale_buffers(k_scales, v_scales)

    def get_size_per_token(self):
        payload = super().get_size_per_token()
        scales = (self.k_sf_page_bytes + self.v_sf_page_bytes) // self.page_size
        return payload + scales * self.layer_num

    def _init_scale_buffers(self, k_scales, v_scales):
        alloc_func = ALLOC_MEMORY_FUNCS[self._transfer_device.type]
        # Host: page-first like the payload, one contiguous scale block per
        # (page, layer) so a page's layers are a single memcpy span.
        self.k_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        self.v_scale_host = alloc_func(
            (self.page_num, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        # [page, layer, bytes] -> per-layer strided [page, bytes] views for H2D.
        self.k_scale_host_layers = list(self.k_scale_host.transpose(0, 1))
        self.v_scale_host_layers = list(self.v_scale_host.transpose(0, 1))

        # Device: each layer's interleaved scale tensor as [page, bytes] rows.
        self.k_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1) for buf in k_scales
        ]
        self.v_scale_device_rows = [
            buf.view(torch.uint8).reshape(buf.shape[0], -1) for buf in v_scales
        ]
        self.k_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.k_scale_device_rows],
            dtype=torch.uint64,
            device=self._transfer_device,
        )
        self.v_scale_device_ptrs = torch.tensor(
            [rows.data_ptr() for rows in self.v_scale_device_rows],
            dtype=torch.uint64,
            device=self._transfer_device,
        )
        self.k_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.k_sf_page_bytes),
            dtype=torch.uint8,
            device=self._transfer_device,
        )
        self.v_scale_staging = torch.empty(
            (self.staging_page_capacity, self.layer_num, self.v_sf_page_bytes),
            dtype=torch.uint8,
            device=self._transfer_device,
        )

    def _page_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Page-aligned token indices -> one page id per page, same device."""
        return indices[:: self.page_size] // self.page_size

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        if io_backend != "kernel":
            raise NotImplementedError(
                f"MXFP8 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        if host_pages.is_cuda:
            host_pages = host_pages.cpu()
        for ptr_src, staging, dst in (
            (self.k_scale_device_ptrs, self.k_scale_staging, self.k_scale_host),
            (self.v_scale_device_ptrs, self.v_scale_staging, self.v_scale_host),
        ):
            jit_transfer_hicache_all_layer_mla_staged_lf_pf(
                ptr_src=ptr_src,
                src_indices=device_pages,
                dst_indices=host_pages,
                staging=staging,
                dst=dst,
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
        is_draft: bool = False,
    ):
        super().load_to_device_per_layer(
            device_pool,
            host_indices,
            device_indices,
            layer_id,
            io_backend,
            is_draft=is_draft,
        )
        if is_draft:
            raise NotImplementedError("MXFP8 KV host pool has no draft layers.")
        if io_backend != "kernel":
            raise NotImplementedError(
                f"MXFP8 KV host pool supports only io_backend='kernel', got {io_backend!r}."
            )
        if not 0 <= layer_id < self.target_layer_num:
            return
        host_layer_id = layer_id
        device_pages = self._page_ids(device_indices)
        host_pages = self._page_ids(host_indices)
        for dst_rows, src_rows, sf_page_bytes in (
            (
                self.k_scale_device_rows[layer_id],
                self.k_scale_host_layers[host_layer_id],
                self.k_sf_page_bytes,
            ),
            (
                self.v_scale_device_rows[layer_id],
                self.v_scale_host_layers[host_layer_id],
                self.v_sf_page_bytes,
            ),
        ):
            jit_transfer_hicache_one_layer_mla(
                cache_dst=dst_rows,
                indices_dst=device_pages,
                cache_src=src_rows,
                indices_src=host_pages,
                element_dim=sf_page_bytes,
            )

    def destroy(self):
        for buf in (self.k_scale_host, self.v_scale_host):
            if buf is not None and self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(buf)
        self.k_scale_host = None
        self.v_scale_host = None
        super().destroy()

    def _storage_pages_unsupported(self) -> NotImplementedError:
        return NotImplementedError(
            "MXFP8 KV host pool does not expose flat storage (L3) pages yet: "
            "a page's UE8M0 scales live outside kv_buffer."
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise self._storage_pages_unsupported()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise self._storage_pages_unsupported()

    def get_page_buffer_meta(self, indices):
        raise self._storage_pages_unsupported()

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        raise self._storage_pages_unsupported()
