# SPDX-License-Identifier: Apache-2.0
"""Ascend virtual weight views with resident local pages and remote-only slots."""

from __future__ import annotations

import math

import torch

from sglang.srt.layers.moe.dwdp.layout import PageAlignedLayout
from sglang.srt.utils.alignment import align_down, align_up


def _check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"NPU DWDP: {operation} failed with ACL error {ret}")


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
        self.slots = {}
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

    def layout(self, spec, rank):
        start, end = rank * spec.local_experts, (rank + 1) * spec.local_experts
        size = align_up(end * spec.expert_bytes, self.granularity) - align_down(
            start * spec.expert_bytes, self.granularity
        )
        return PageAlignedLayout.compute(
            spec.expert_bytes, spec.num_experts, start, end, self.granularity, size
        )

    def create_weight(self, spec, rank, slot_key):
        """Return a full view, local source pointer, handle and page boundaries.

        Pages intersecting the local shard are private to this layer. Their
        remote edge bytes are filled once at setup. Only the pages completely
        outside the local shard reuse a slot and need runtime prefetches.
        """
        import torch_npu

        shape, dtype = spec.full_shape, spec.dtype
        nbytes = spec.num_experts * spec.expert_bytes
        layout = self.layout(spec, rank)
        local_size = layout.mnnvl_size
        handle = self._allocate(local_size)
        source = self._reserve(local_size)
        self._map(source, local_size, handle)
        full = self._reserve(layout.total_size)
        self._map(full + layout.pre_size, local_size, handle)
        for side, offset, size in (
            ("pre", 0, layout.pre_size),
            ("post", layout.pre_size + local_size, layout.post_size),
        ):
            if not size:
                continue
            # ACL requires mapping the entire allocation at offset zero. Pool
            # by exact size so heterogeneous layers never map a partial handle.
            key = (*slot_key, side, size)
            if key not in self.slots:
                self.slots[key] = self._allocate(size)
            self._map(full + offset, size, self.slots[key])
        storage = torch_npu._C._construct_storage_from_data_pointer(
            full, self.device, nbytes
        )
        stride = tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))
        tensor = torch_npu._C._construct_NPU_Tensor_From_Storage_And_Metadata(
            dict(
                size=shape,
                stride=stride,
                nbytes=nbytes,
                dtype=dtype,
                data_ptr=full,
                storage_offset=0,
                device=self.device,
                npu_format=2,
                layout=torch.strided,
                requires_grad=False,
            ),
            storage,
        )
        return tensor, source + layout.data_offset, handle, layout

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
        while self._handles:
            _check(
                self.rt.free_physical(self._handles[-1]), "free physical weight memory"
            )
            self._handles.pop()
        self.slots.clear()
