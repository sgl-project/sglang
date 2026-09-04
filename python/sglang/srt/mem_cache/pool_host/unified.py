from __future__ import annotations

import threading
from collections.abc import Sequence

import torch

from sglang.srt.mem_cache.pool_host.base import HostKVCache, host_memory_budget_bytes
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    _cuda_host_unregister,
    get_allocator_from_storage,
)
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
if _is_cuda or _is_hip:
    from sgl_kernel.kvcacheio import transfer_kv_direct, transfer_kv_per_layer_mla


class _SharedPageEnvelopeHostBacking:
    class _Side:
        def __init__(
            self,
            *,
            page_size: int,
            page_bytes: int,
            page_num: int,
        ):
            self.page_size = page_size
            self.page_bytes = page_bytes
            self.page_num = page_num
            self.allocated_pages: set[int] = set()

    def __init__(
        self,
        *,
        device_pools: Sequence,
        pool_names: Sequence[str],
        host_to_device_ratio: float,
        host_size: float,
        pin_memory: bool,
        device: str,
        allocator_type: str,
    ):
        if len(device_pools) != len(pool_names) or not device_pools:
            raise ValueError("Unified host pool devices and names must be non-empty.")
        if len(device_pools) > 2:
            raise ValueError("Unified host backing supports at most two pools.")
        if len(set(pool_names)) != len(pool_names):
            raise ValueError(f"Unified host pool names must be unique: {pool_names}.")

        device_buffers = [pool.get_page_envelope_buffer() for pool in device_pools]
        registrations = [pool.get_contiguous_buf_infos() for pool in device_pools]
        for ptrs, lens, item_lens in registrations:
            if len(ptrs) != len(lens) or len(ptrs) != len(item_lens) or len(ptrs) != 1:
                raise ValueError(
                    "Unified page-envelope pools must expose one memory region."
                )
        shared_regions = {(ptrs[0], lens[0]) for ptrs, lens, _ in registrations}
        if len(shared_regions) != 1:
            raise ValueError("Unified full and SWA pools must share one device region.")

        page_bytes = [int(buffer.shape[1]) for buffer in device_buffers]
        if host_size > 0:
            token_capacities = [
                int(host_size * 1e9 * pool.page_size // item_bytes)
                for pool, item_bytes in zip(device_pools, page_bytes, strict=True)
            ]
        else:
            token_capacities = [
                int(pool.size * host_to_device_ratio) for pool in device_pools
            ]

        nominal_view_bytes = [
            (capacity // pool.page_size + 1) * item_bytes
            for pool, item_bytes, capacity in zip(
                device_pools, page_bytes, token_capacities, strict=True
            )
        ]

        if host_size > 0:
            total_bytes = max(nominal_view_bytes)
        else:
            device_total_bytes = next(iter(shared_regions))[1]
            total_bytes = max(
                int(device_total_bytes * host_to_device_ratio) + max(page_bytes),
                *nominal_view_bytes,
            )
        if total_bytes <= 0:
            raise ValueError(
                f"Unified host backing size must be positive: {total_bytes}"
            )

        available_bytes = host_memory_budget_bytes()
        if total_bytes > available_bytes:
            raise ValueError(
                "Not enough host memory for unified page-envelope backing. "
                f"Requesting {total_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        self.sides = {
            name: self._Side(
                page_size=pool.page_size,
                page_bytes=item_bytes,
                page_num=total_bytes // item_bytes,
            )
            for pool, name, item_bytes in zip(
                device_pools, pool_names, page_bytes, strict=True
            )
        }
        if any(side.page_num == 0 for side in self.sides.values()):
            raise ValueError(
                "Unified host backing must fit at least one page from every pool."
            )

        self.pin_memory = pin_memory
        self.allocator = get_allocator_from_storage(allocator_type)
        self.lock = threading.RLock()
        alloc_func = ALLOC_MEMORY_FUNCS[device_buffers[0].device]
        self.raw = alloc_func(
            (total_bytes,),
            dtype=torch.uint8,
            device=device,
            pin_memory=pin_memory,
            allocator=self.allocator,
        )
        self.capacity_bytes = total_bytes
        self.fd = getattr(self.allocator, "fd", None)
        self._free_extents = [(0, total_bytes)]
        self.registration_owner = pool_names[0]
        self._users: set[str] = set()
        self._destroyed = False

    def retain(self, name: str) -> None:
        if name in self._users:
            raise RuntimeError(f"Unified host pool {name!r} is already attached.")
        self._users.add(name)

    def release(self, name: str) -> None:
        with self.lock:
            self._users.discard(name)
            self.destroy_if_unused()

    def destroy_if_unused(self) -> None:
        with self.lock:
            if self._users or self._destroyed:
                return
            self._destroyed = True
            if self.pin_memory and (_is_cuda or _is_hip):
                _cuda_host_unregister(self.raw)
            self.raw = None

    def page_buffer(self, name: str) -> torch.Tensor:
        side = self.sides[name]
        return self.raw[: side.page_num * side.page_bytes].view(
            side.page_num, side.page_bytes
        )

    @staticmethod
    def _align_up(value: int, alignment: int) -> int:
        return (value + alignment - 1) // alignment * alignment

    @classmethod
    def _reserve_pages(
        cls, extents: list[tuple[int, int]], page_bytes: int, page_count: int
    ) -> list[int] | None:
        if page_count == 0:
            return []

        candidates = []
        for index, (start, end) in enumerate(extents):
            aligned_start = cls._align_up(start, page_bytes)
            capacity = (end - aligned_start) // page_bytes
            if capacity <= 0:
                continue
            candidates.append(
                (end - start, aligned_start - start, index, aligned_start, capacity)
            )

        remaining = page_count
        offsets = []
        replacements: dict[int, list[tuple[int, int]]] = {}
        for _, _, index, aligned_start, capacity in sorted(candidates):
            take = min(remaining, capacity)
            used_end = aligned_start + take * page_bytes
            start, end = extents[index]
            replacement = []
            if start < aligned_start:
                replacement.append((start, aligned_start))
            if used_end < end:
                replacement.append((used_end, end))
            replacements[index] = replacement
            offsets.extend(range(aligned_start, used_end, page_bytes))
            remaining -= take
            if remaining == 0:
                break

        if remaining:
            return None

        extents[:] = [
            replacement
            for index, extent in enumerate(extents)
            for replacement in replacements.get(index, (extent,))
        ]
        return offsets

    def _plan_allocations(
        self,
        requests: Sequence[tuple[str, int]],
        *,
        free_extents: Sequence[tuple[int, int]] | None = None,
    ) -> tuple[list[tuple[int, int]], list[list[int]]] | None:
        request_groups = []
        for request_index, (name, need_size) in enumerate(requests):
            side = self.sides[name]
            if need_size % side.page_size != 0:
                raise AssertionError(
                    "The requested size should be a multiple of the page size."
                )
            page_count = need_size // side.page_size
            request_groups.append((side.page_bytes, request_index, page_count, name))

        # Place large page envelopes first. Planning uses a copy so an allocation
        # is all-or-nothing even when the two sides need different page sizes.
        extents = list(self._free_extents if free_extents is None else free_extents)
        offsets = [[] for _ in requests]
        for _, request_index, page_count, name in sorted(
            request_groups, key=lambda group: group[0], reverse=True
        ):
            request_offsets = self._reserve_pages(
                extents, self.sides[name].page_bytes, page_count
            )
            if request_offsets is None:
                return None
            offsets[request_index] = request_offsets
        return extents, offsets

    @staticmethod
    def _merge_extents(extents: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged = []
        for start, end in sorted(extents):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _page_ids(indices: torch.Tensor, page_size: int) -> list[int]:
        indices = indices.to(dtype=torch.int64, device="cpu").flatten()
        if indices.numel() % page_size != 0:
            raise AssertionError("Host indices must contain whole pages.")
        groups = indices.view(-1, page_size)
        offsets = torch.arange(page_size, dtype=torch.int64)
        if groups.numel() and (
            bool(torch.any(groups[:, 0] % page_size != 0))
            or not torch.equal(groups, groups[:, :1] + offsets)
        ):
            raise AssertionError("Host indices must contain contiguous pages.")
        return (groups[:, 0] // page_size).tolist()

    def available_size(self, name: str) -> int:
        with self.lock:
            side = self.sides[name]
            physical_pages = sum(
                max(0, end - self._align_up(start, side.page_bytes)) // side.page_bytes
                for start, end in self._free_extents
            )
            return physical_pages * side.page_size

    def free_bytes(self) -> int:
        with self.lock:
            return sum(end - start for start, end in self._free_extents)

    @staticmethod
    def _expand_pages(pages: Sequence[int], page_size: int) -> torch.Tensor:
        if not pages:
            return torch.empty(0, dtype=torch.int64)
        page_ids = torch.tensor(pages, dtype=torch.int64)
        offsets = torch.arange(page_size, dtype=torch.int64)
        return (page_ids[:, None] * page_size + offsets).reshape(-1)

    def alloc(self, name: str, need_size: int) -> torch.Tensor | None:
        result = self.alloc_many(((name, need_size),))
        return None if result is None else result[0]

    def alloc_many(
        self, requests: Sequence[tuple[str, int]]
    ) -> list[torch.Tensor] | None:
        with self.lock:
            plan = self._plan_allocations(requests)
            if plan is None:
                return None
            extents, offsets = plan
            results = []
            for (name, _), request_offsets in zip(requests, offsets, strict=True):
                side = self.sides[name]
                pages = [offset // side.page_bytes for offset in request_offsets]
                side.allocated_pages.update(pages)
                results.append(self._expand_pages(pages, side.page_size))
            self._free_extents = extents
            return results

    def can_fit_many(self, requests: Sequence[tuple[str, int]]) -> bool:
        """Whether requests fit in an otherwise empty shared arena."""
        with self.lock:
            return (
                self._plan_allocations(
                    requests, free_extents=((0, self.capacity_bytes),)
                )
                is not None
            )

    def can_fit_many_then(
        self,
        requests: Sequence[tuple[str, int]],
        following_requests: Sequence[tuple[str, int]],
        *,
        empty: bool = False,
    ) -> bool:
        """Whether two allocation groups fit in order without mutating state."""
        with self.lock:
            free_extents = ((0, self.capacity_bytes),) if empty else self._free_extents
            first = self._plan_allocations(requests, free_extents=free_extents)
            if first is None:
                return False
            return (
                self._plan_allocations(following_requests, free_extents=first[0])
                is not None
            )

    def free(self, name: str, indices: torch.Tensor) -> int:
        with self.lock:
            side = self.sides[name]
            pages = self._page_ids(indices, side.page_size)
            if len(pages) != len(set(pages)):
                raise AssertionError("Host pages cannot be freed more than once.")
            if any(page not in side.allocated_pages for page in pages):
                raise AssertionError(f"Host pages are not allocated in {name!r}.")
            extents = list(self._free_extents)
            for page in pages:
                offset = page * side.page_bytes
                side.allocated_pages.remove(page)
                extents.append((offset, offset + side.page_bytes))
            self._free_extents = self._merge_extents(extents)
            return len(indices)

    def clear(self, name: str) -> None:
        with self.lock:
            side = self.sides[name]
            extents = list(self._free_extents)
            extents.extend(
                (page * side.page_bytes, (page + 1) * side.page_bytes)
                for page in side.allocated_pages
            )
            self._free_extents = self._merge_extents(extents)
            side.allocated_pages.clear()


class UnifiedPageEnvelopeHostPool(HostKVCache):
    """Host mirror that transfers complete unified-memory page envelopes."""

    stores_page_envelope = True

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
        pin_memory: bool = True,
        device: str = "cpu",
    ) -> tuple[HostKVCache, HostKVCache]:
        if mtp_draft_device_pools:
            raise NotImplementedError(
                "Unified page-envelope host pools do not support packed MTP draft pools."
            )
        if len(device_pools) != 2:
            raise ValueError("A hybrid-SWA host pool pair requires exactly two pools.")

        pool_names = ("full", "swa")
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

    def backup_from_device_all_layer(
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
        self._transfer_pages(
            self._host_page_buffer_view,
            self._device_page_buffer,
            self._to_page_indices(host_indices),
            self._to_page_indices(device_indices),
            io_backend,
        )

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        page = self._host_page_buffer_view[int(index) // self.page_size]
        return page.flatten() if flat else page.view(1, self.item_bytes)

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            self.item_bytes,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        page = data_page.view(self.dtype).reshape(self.item_bytes)
        self._host_page_buffer_view[int(index) // self.page_size].copy_(page)

    def get_page_buffer_meta(self, indices):
        pages = self._to_page_indices(indices).tolist()
        base_ptr = self._host_page_buffer_view.data_ptr()
        ptrs = [base_ptr + int(page) * self.item_bytes for page in pages]
        return ptrs, [self.item_bytes] * len(ptrs)

    def is_stride_page_aligned(self, page_size_bytes: int = 4096) -> bool:
        buffer = self._host_page_buffer_view
        return (
            buffer.data_ptr() % page_size_bytes == 0
            and self.item_bytes % page_size_bytes == 0
        )
