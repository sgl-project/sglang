"""Ascend virtual weight views with resident local pages and remote-only slots."""

from __future__ import annotations

import ctypes
import math
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.dwdp.layout import PageAlignedLayout
from sglang.srt.layers.moe.dwdp.page_pool import PagePool, compute_slot_sizes
from sglang.srt.utils.alignment import align_down, align_up


def _check(ret, operation):
    if ret != 0:
        raise RuntimeError(f"NPU DWDP: {operation} failed with ACL error {ret}")


class _FabricHandleAPI:
    """CANN V2 ABI; pyACL does not expose these APIs in all supported releases."""

    def __init__(self):
        self.lib = ctypes.CDLL("libascendcl.so")
        for name, args in (
            (
                "aclrtMemExportToShareableHandleV2",
                [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_void_p],
            ),
            (
                "aclrtMemSetPidToShareableHandleV2",
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int32),
                    ctypes.c_size_t,
                ],
            ),
            (
                "aclrtMemImportFromShareableHandleV2",
                [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_uint64,
                    ctypes.POINTER(ctypes.c_void_p),
                ],
            ),
        ):
            function = getattr(self.lib, name)
            function.argtypes, function.restype = args, ctypes.c_int

    def export_handle(self, handle, peer_pids):
        share = ctypes.create_string_buffer(128)  # aclrtMemFabricHandle.data[128]
        _check(
            self.lib.aclrtMemExportToShareableHandleV2(handle, 0, 2, share),
            "aclrtMemExportToShareableHandleV2(FABRIC)",
        )
        pids = (ctypes.c_int32 * len(peer_pids))(*peer_pids)
        _check(
            self.lib.aclrtMemSetPidToShareableHandleV2(share, 2, pids, len(pids)),
            "aclrtMemSetPidToShareableHandleV2(FABRIC)",
        )
        return share.raw

    def import_handle(self, share):
        if len(share) != 128:
            raise ValueError("NPU DWDP: invalid FABRIC handle size")
        share = ctypes.create_string_buffer(share, 128)
        handle = ctypes.c_void_p()
        _check(
            self.lib.aclrtMemImportFromShareableHandleV2(
                share, 2, 0, ctypes.byref(handle)
            ),
            "aclrtMemImportFromShareableHandleV2(FABRIC)",
        )
        return handle.value


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
        self._fabric = _FabricHandleAPI()

    def export_shard(self, handle, peer_pids):
        return self._fabric.export_handle(handle, peer_pids)

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
                reservation = SimpleNamespace(
                    map_existing=lambda offset, size, handle, base=full: self._map(
                        base + offset, size, handle
                    )
                )
                page_offset = self._page_pool.map_weight_pages(
                    assignments[li], reservation, layout, handles[li, name], page_offset
                )
                tensors[li, name] = tensor_from_pointer(
                    full + layout.pre_padding,
                    spec.num_experts * spec.expert_bytes,
                    shape=spec.full_shape,
                    dtype=spec.dtype,
                    device=self.device,
                )
        return tensors

    def import_shard(self, share, size, offset):
        handle = self._fabric.import_handle(share)
        # Release peer imports before producers free their physical pages.
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
            entry = self._imports[-1]
            handle, ptr, mapped = entry
            if mapped:
                _check(self.rt.unmap_mem(ptr), "unmap imported memory")
                entry[2] = False
            if ptr is not None:
                _check(self.rt.release_mem_address(ptr), "release imported address")
                entry[1] = None
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
