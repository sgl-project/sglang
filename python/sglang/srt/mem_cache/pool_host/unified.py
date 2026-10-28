from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import torch

from sglang.srt.mem_cache.kv.memory.host_backing import _SharedPageEnvelopeHostBacking
from sglang.srt.mem_cache.pool_host.base import HostKVCache
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_direct, transfer_kv_per_layer_mla


class UnifiedPageEnvelopeHostPool(HostKVCache):
    """Host mirror that transfers complete unified-memory page envelopes."""

    stores_page_envelope = True
    storage_format_tag = "unified-token-major"

    def __init__(
        self,
        device_pool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        *,
        mtp_draft_device_pools: Sequence = (),
        pool_label: str = "kv",
        _shared_backing: _SharedPageEnvelopeHostBacking | None = None,
    ):
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Unified page-envelope host pools do not support packed MTP draft pools."
            )

        device_page_buffer = device_pool.get_page_envelope_buffer()
        if device_page_buffer.ndim != 2:
            raise ValueError(
                "Unified page-envelope buffers must have shape [pages, bytes]; "
                f"got {tuple(device_page_buffer.shape)}."
            )
        page_bytes = int(device_page_buffer.shape[1])
        if page_bytes % page_size != 0:
            raise ValueError(
                "Unified page-envelope bytes must be divisible by page_size; "
                f"got page_bytes={page_bytes}, page_size={page_size}."
            )
        if device_pool.page_size != page_size:
            raise ValueError(
                "Unified device and host page sizes must match; "
                f"got device={device_pool.page_size}, host={page_size}."
            )
        if layout not in {"layer_first", "page_first", "page_first_direct"}:
            raise ValueError(f"Unsupported layout: {layout}")

        if _shared_backing is None:
            _shared_backing = _SharedPageEnvelopeHostBacking(
                device_pools=(device_pool,),
                pool_names=(pool_label,),
                host_to_device_ratio=host_to_device_ratio,
                host_size=host_size,
                pin_memory=pin_memory,
                device=device,
                allocator_type=allocator_type,
            )

        self.device_pool = device_pool
        self.pool_label = pool_label
        self._shared_backing = _shared_backing
        self._device_page_buffer = device_page_buffer
        self._host_page_buffer_view = _shared_backing.page_buffer(pool_label)
        self.size_per_token = page_bytes // page_size
        self.item_bytes = page_bytes
        self.page_size = page_size
        self.page_num = _shared_backing.sides[pool_label].page_num
        self.device_capacity_tokens = (
            device_pool.host_capacity_tokens
            if device_pool.host_capacity_tokens is not None
            else device_pool.size
        )
        self.size = self.page_num * page_size
        self.dcp_size = 1
        self.dcp_rank = 0
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.dtype = torch.uint8
        self.allocator = _shared_backing.allocator
        self.layer_num = 1
        self.can_use_write_back_jit = False
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer
        self.kv_buffer = (
            _shared_backing.raw
            if pool_label == _shared_backing.registration_owner
            else None
        )
        self.fd = _shared_backing.fd
        self.lock = _shared_backing.lock
        self._destroyed = False
        _shared_backing.retain(pool_label)

    @classmethod
    def build_hybrid_swa_pool_pair(
        cls,
        *,
        device_pools: Sequence,
        host_to_device_ratio: float,
        host_size: float,
        page_size: int,
        layout: str,
        allocator_type: str,
        mtp_draft_device_pools: Sequence = (),
        pool_names: Sequence[str] = ("full", "swa"),
        pin_memory: bool = True,
        device: str = "cpu",
    ) -> tuple[HostKVCache, HostKVCache]:
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Unified page-envelope host pools do not support packed MTP draft pools."
            )
        if len(device_pools) != 2:
            raise ValueError("A hybrid-SWA host pool pair requires exactly two pools.")

        pool_names = tuple(pool_names)
        backing = _SharedPageEnvelopeHostBacking(
            device_pools=device_pools,
            pool_names=pool_names,
            host_to_device_ratio=host_to_device_ratio,
            host_size=host_size,
            pin_memory=pin_memory,
            device=device,
            allocator_type=allocator_type,
        )
        host_pools = []
        try:
            for device_pool, pool_name in zip(device_pools, pool_names, strict=True):
                host_pools.append(
                    cls(
                        device_pool,
                        host_to_device_ratio,
                        0,
                        page_size,
                        layout,
                        pin_memory=pin_memory,
                        device=device,
                        allocator_type=allocator_type,
                        pool_label=pool_name,
                        _shared_backing=backing,
                    )
                )
        except Exception:
            for host_pool in host_pools:
                host_pool.destroy()
            backing.destroy_if_unused()
            raise
        return tuple(host_pools)

    def get_size_per_token(self):
        return self.size_per_token

    def get_ksize_per_token(self):
        return self.size_per_token

    @property
    def shared_allocation_domain(self):
        return self._shared_backing

    def init_kv_buffer(self):
        return self._host_page_buffer_view

    def get_hybrid_pool_buffer(self):
        if self.pool_label == self._shared_backing.registration_owner:
            return [self._shared_backing.raw]
        return []

    def clear(self) -> None:
        self._shared_backing.clear(self.pool_label)

    def available_size(self):
        return self._shared_backing.available_size(self.pool_label)

    def alloc(self, need_size: int) -> torch.Tensor | None:
        return self._shared_backing.alloc(self.pool_label, need_size)

    def free(self, indices: torch.Tensor) -> int:
        return self._shared_backing.free(self.pool_label, indices)

    def prepare_transfer_indices(
        self, host_indices, device_indices, io_backend
    ) -> tuple[torch.Tensor, torch.Tensor]:
        host_indices = self._shared_backing.translate_indices(
            self.pool_label, host_indices
        )
        if io_backend == "kernel":
            if not host_indices.is_cuda:
                host_indices = host_indices.to(
                    self._device_page_buffer.device, non_blocking=True
                )
            return host_indices, device_indices
        if io_backend == "direct":
            if self.layout == "layer_first":
                device_indices = device_indices.cpu()
                host_indices, order = host_indices.sort()
                return host_indices, device_indices.index_select(0, order)
            if self.layout == "page_first_direct":
                return host_indices, device_indices.cpu()
            raise ValueError(
                f"Unsupported layout {self.layout!r} for io backend 'direct'"
            )
        if io_backend == "kernel_ascend":
            return host_indices, device_indices.cpu()
        raise ValueError(f"Unsupported io backend: {io_backend}")

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        self.kv_buffer = None
        self._host_page_buffer_view = None
        self._shared_backing.release(self.pool_label)

    def _to_page_indices(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.to(dtype=torch.int64)
        if indices.numel() == 0:
            return indices
        if indices.numel() % self.page_size != 0:
            raise ValueError(
                "Unified page-envelope transfer index count must be page-aligned; "
                f"got {indices.numel()} indices for page_size={self.page_size}."
            )
        grouped = indices.reshape(-1, self.page_size)
        # Device indices are produced by the page allocator. Validating every
        # value here would turn both reductions into host-device synchronizations
        # on every offload. Keep the full check for CPU/debug inputs while the
        # hot CUDA/ROCm path relies on that allocator invariant.
        if indices.device.type == "cpu":
            offsets = torch.arange(self.page_size, dtype=torch.int64)
            if bool(torch.any(grouped[:, 0] % self.page_size != 0)) or not torch.equal(
                grouped, grouped[:, :1] + offsets
            ):
                raise ValueError("Unified page-envelope transfers require whole pages.")
        return grouped[:, 0] // self.page_size

    def _has_transfer_indices(
        self, host_indices: torch.Tensor | None, device_indices: torch.Tensor | None
    ) -> bool:
        if host_indices is None or device_indices is None:
            return False
        if host_indices.numel() != device_indices.numel():
            raise ValueError(
                "Unified page-envelope transfer index size mismatch: "
                f"host={host_indices.numel()}, device={device_indices.numel()}."
            )
        return host_indices.numel() > 0

    def _transfer_pages(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        src_pages: torch.Tensor,
        dst_pages: torch.Tensor,
        io_backend: str,
    ) -> None:
        if io_backend not in {"kernel", "direct"}:
            raise ValueError(f"Unsupported io_backend: {io_backend}")
        if self._device_page_buffer.device.type == "cpu":
            dst.index_copy_(0, dst_pages, src.index_select(0, src_pages))
            return

        if not (_is_cuda or _is_hip):
            raise RuntimeError(
                "Unified page-envelope transfers require a CUDA or ROCm device."
            )
        if io_backend == "kernel":
            transfer_kv_per_layer_mla(
                src=src,
                dst=dst,
                src_indices=src_pages,
                dst_indices=dst_pages,
                item_size=self.item_bytes,
            )
        else:
            transfer_kv_direct(
                src_layers=[src],
                dst_layers=[dst],
                src_indices=src_pages,
                dst_indices=dst_pages,
                page_size=1,
            )

    @contextmanager
    def _direct_transfer(self, host_indices, device_indices, io_backend):
        self._shared_backing.acquire_layout()
        finish_event = None
        transfer_key = None
        try:
            yield self.prepare_transfer_indices(
                host_indices, device_indices, io_backend
            )
            if self._device_page_buffer.device.type != "cpu":
                finish_event = torch.cuda.Event()
                finish_event.record()
                transfer_key = (
                    "direct",
                    int(torch.cuda.current_stream().cuda_stream),
                )
        except Exception:
            if self._device_page_buffer.device.type != "cpu":
                torch.cuda.current_stream().synchronize()
            raise
        finally:
            self._shared_backing.release_layout(finish_event, transfer_key)

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        with self._direct_transfer(host_indices, device_indices, io_backend) as (
            host_indices,
            device_indices,
        ):
            self.backup_from_device_all_layer_physical(
                device_pool, host_indices, device_indices, io_backend
            )

    def backup_from_device_all_layer_physical(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        if not self._has_transfer_indices(host_indices, device_indices):
            return
        self._transfer_pages(
            self._device_page_buffer,
            self._host_page_buffer_view,
            self._to_page_indices(device_indices),
            self._to_page_indices(host_indices),
            io_backend,
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
        if is_draft:
            raise NotImplementedError(
                "Unified page-envelope host pools do not contain draft KV."
            )
        if layer_id != 0 or not self._has_transfer_indices(
            host_indices, device_indices
        ):
            return
        with self._direct_transfer(host_indices, device_indices, io_backend) as (
            host_indices,
            device_indices,
        ):
            self.load_to_device_per_layer_physical(
                device_pool,
                host_indices,
                device_indices,
                layer_id,
                io_backend,
                is_draft=is_draft,
            )

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        *,
        is_draft: bool = False,
    ):
        if is_draft:
            raise NotImplementedError(
                "Unified page-envelope host pools do not contain draft KV."
            )
        if layer_id != 0 or not self._has_transfer_indices(
            host_indices, device_indices
        ):
            return
        self._transfer_pages(
            self._host_page_buffer_view,
            self._device_page_buffer,
            self._to_page_indices(host_indices),
            self._to_page_indices(device_indices),
            io_backend,
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        with self._shared_backing.layout_lease():
            physical = self._shared_backing.translate_index(self.pool_label, int(index))
            page = self._host_page_buffer_view[physical // self.page_size]
            return page.flatten() if flat else page.view(1, self.item_bytes)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.item_bytes,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def get_page_buffer_element_size(self, split_factor: int = 1) -> int:
        if split_factor != 1:
            raise ValueError(
                "Unified page-envelope host pools do not support split-head metadata."
            )
        return self.item_bytes

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        with self._shared_backing.layout_lease():
            physical = self._shared_backing.translate_index(self.pool_label, int(index))
            page = data_page.view(self.dtype).reshape(self.item_bytes)
            self._host_page_buffer_view[physical // self.page_size].copy_(page)

    def get_page_buffer_meta(self, indices):
        physical = self._shared_backing.translate_indices(self.pool_label, indices)
        pages = self._to_page_indices(physical).tolist()
        base_ptr = self._host_page_buffer_view.data_ptr()
        ptrs = [base_ptr + int(page) * self.item_bytes for page in pages]
        return ptrs, [self.item_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        buffer = self._host_page_buffer_view
        return (
            buffer.data_ptr() % page_size_bytes == 0
            and self.item_bytes % page_size_bytes == 0
        )
