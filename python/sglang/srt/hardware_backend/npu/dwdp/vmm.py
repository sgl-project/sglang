# SPDX-License-Identifier: Apache-2.0
"""Ascend virtual weight views with resident local pages and remote-only slots."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.dwdp.layout import PageAlignedLayout
from sglang.srt.layers.moe.dwdp.page_pool import PagePool, compute_slot_sizes
from sglang.srt.utils.alignment import align_down, align_up


def _check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"NPU DWDP: {operation} failed with ACL error {ret}")


def tensor_from_pointer(pointer, nbytes, *, shape, dtype, device):
    import torch_npu

    storage = torch_npu._C._construct_storage_from_data_pointer(pointer, device, nbytes)
    stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
    return torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
        dict(
            size=shape,
            stride=stride,
            nbytes=nbytes,
            dtype=dtype,
            data_ptr=pointer,
            storage_offset=0,
            device=device,
            npu_format=2,
            layout=torch.strided,
            requires_grad=False,
        ),
        storage,
    )


class NPUWeightVMM:
    def __init__(self, acl, device):
        self.rt = acl.rt
        self.device = device
        self.prop = dict(
            handleType=0,
            allocationType=0,
            memAttr=4,  # ACL_HBM_MEM_HUGE
            location=dict(type=1, id=device.index),
            reserve=0,
        )
        self.granularity, ret = self.rt.mem_get_allocation_granularity(self.prop, 0)
        _check(ret, "mem_get_allocation_granularity")
        self._page_pool = None
        self._handles = []
        self._reservations = []
        self._mappings = []
        self._imports = []

    def _reserve(self, size):
        ptr, ret = self.rt.reserve_mem_address(size, 0, 0, 1)
        _check(ret, "reserve_mem_address")
        self._reservations.append(ptr)
        return ptr

    def _allocate(self, size):
        handle, ret = self.rt.malloc_physical(size, self.prop, 0)
        _check(ret, "malloc_physical")
        self._handles.append(handle)
        return handle

    def _map(self, ptr, size, handle):
        _check(self.rt.map_mem(ptr, size, 0, handle, 0), "map_mem")
        self._mappings.append(ptr)

    def _release_page(self, handle):
        _check(self.rt.free_physical(handle), "free pool page")
        self._handles.remove(handle)

    def layout(self, spec, rank):
        start, end = rank * spec.local_experts, (rank + 1) * spec.local_experts
        size = align_up(end * spec.expert_bytes, self.granularity) - align_down(
            start * spec.expert_bytes, self.granularity
        )
        return PageAlignedLayout.compute(
            spec.expert_bytes,
            spec.num_experts,
            start,
            end,
            self.granularity,
            size,
            pool_granularity=PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER * self.granularity,
        )

    def create_local(self, spec, rank):
        layout = self.layout(spec, rank)
        handle = self._allocate(layout.mnnvl_size)
        source = self._reserve(layout.mnnvl_size)
        self._map(source, layout.mnnvl_size, handle)
        return source + layout.data_offset, handle

    def create_weights(self, specs, rank, handles, assignments):
        """Map local shards and shared fixed-size pages after source migration."""
        layouts = {}
        for (li, name), spec in specs.items():
            layouts.setdefault(li, {})[name] = self.layout(spec, rank)
        self._page_pool = PagePool(
            compute_slot_sizes(layouts, assignments),
            self.device.index,
            granularity=self.granularity,
            allocate_page=self._allocate,
            release_page=self._release_page,
        )
        tensors = {}
        for li, weight_layouts in layouts.items():
            page_offset = 0
            for name, layout in weight_layouts.items():
                spec = specs[li, name]
                full = self._reserve(layout.total_size)
                self._map(full + layout.pre_size, layout.mnnvl_size, handles[li, name])
                reservation = SimpleNamespace(
                    map_existing=lambda offset, size, handle, base=full: self._map(
                        base + offset, size, handle
                    )
                )
                for offset, size in (
                    (0, layout.pre_size),
                    (layout.pre_size + layout.mnnvl_size, layout.post_size),
                ):
                    if size:
                        self._page_pool.map_pages(
                            assignments[li], reservation, offset, size, page_offset
                        )
                        page_offset += size // self._page_pool.page_size
                tensors[li, name] = tensor_from_pointer(
                    full + layout.pre_padding,
                    spec.num_experts * spec.expert_bytes,
                    shape=spec.full_shape,
                    dtype=spec.dtype,
                    device=self.device,
                )
        return tensors

    def import_shard(self, share, size, offset):
        handle, ret = self.rt.mem_import_from_shareable_handle(share, self.device.index)
        _check(ret, "mem_import_from_shareable_handle")
        # Keep imported lifetimes separate: all importers must release their
        # mappings/handles before a producer releases resident physical pages.
        self._imports.append([handle, None, False])
        entry = self._imports[-1]
        ptr, ret = self.rt.reserve_mem_address(size, 0, 0, 1)
        _check(ret, "reserve imported memory")
        entry[1] = ptr
        _check(self.rt.map_mem(ptr, size, 0, handle, 0), "map imported memory")
        entry[2] = True
        return ptr + offset

    def close_imports(self):
        while self._imports:
            handle, ptr, mapped = self._imports[-1]
            if mapped:
                _check(self.rt.unmap_mem(ptr), "unmap imported memory")
            if ptr is not None:
                _check(self.rt.release_mem_address(ptr), "release imported address")
            _check(self.rt.free_physical(handle), "free imported handle")
            self._imports.pop()

    def close(self):
        self.close_imports()
        while self._mappings:
            _check(self.rt.unmap_mem(self._mappings[-1]), "unmap weight memory")
            self._mappings.pop()
        while self._reservations:
            _check(
                self.rt.release_mem_address(self._reservations[-1]), "release address"
            )
            self._reservations.pop()
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None
        while self._handles:
            _check(
                self.rt.free_physical(self._handles[-1]), "free physical weight memory"
            )
            self._handles.pop()
