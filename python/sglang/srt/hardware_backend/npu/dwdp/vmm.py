"""ACL memory operations for the shared DWDP lifecycle."""

from __future__ import annotations

import ctypes
import logging
import math
from types import SimpleNamespace

import torch

logger = logging.getLogger(__name__)


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


class NPUBackend:
    def __init__(self, device_id, group):
        import acl

        self.rt = acl.rt
        self.device = torch.device("npu", device_id)
        self.device_module = torch.npu
        self.group = group
        self.exports = {}
        self._handles = []
        self._reservations = []
        self._mappings = []
        self._imports = []
        self._started = self._closed = False

    def initialize(self, num_handles):
        counts = self.group.all_gather_object(num_handles)
        if any(count != num_handles for count in counts):
            raise RuntimeError(
                f"Mismatched DWDP weight handle counts across ranks: {counts}"
            )
        pid, ret = self.rt.device_get_bare_tgid()
        _check(ret, "device_get_bare_tgid")
        self._peer_pids = [
            peer_pid
            for rank, peer_pid in enumerate(self.group.all_gather_object(pid))
            if rank != self.group.rank_in_group
        ]
        self._started, self._closed = True, False
        self.exports.clear()
        self.prop = dict(
            handleType=0,
            allocationType=0,
            memAttr=4,  # ACL_HBM_MEM_HUGE
            location=dict(type=1, id=self.device.index),
            reserve=0,
        )
        self.granularity, ret = self.rt.mem_get_allocation_granularity(self.prop, 0)
        _check(ret, "mem_get_allocation_granularity")
        self._fabric = _FabricHandleAPI()

    def reserve(self, size):
        ptr, ret = self.rt.reserve_mem_address(size, 0, 0, 1)
        _check(ret, "reserve_mem_address")
        self._reservations.append(ptr)
        return SimpleNamespace(
            base=ptr,
            map_existing=lambda offset, size, handle: self._map(
                ptr + offset, size, handle
            ),
        )

    def allocate(self, size):
        handle, ret = self.rt.malloc_physical(size, self.prop, 0)
        _check(ret, "malloc_physical")
        self._handles.append(handle)
        return handle

    def _map(self, ptr, size, handle):
        _check(self.rt.map_mem(ptr, size, 0, handle, 0), "map_mem")
        self._mappings.append(ptr)

    def release_page(self, handle):
        _check(self.rt.free_physical(handle), "free pool page")
        self._handles.remove(handle)

    def copy_local(self, param, size, data_offset):
        data = torch.ops.npu.npu_format_cast(param.data, 2).contiguous()
        self.device_module.current_stream(self.device).synchronize()
        param.data = data
        nbytes = param.numel() * param.element_size()
        handle = self.allocate(size)
        reservation = self.reserve(size)
        reservation.map_existing(0, size, handle)
        _check(
            self.rt.memcpy(
                reservation.base + data_offset, nbytes, param.data_ptr(), nbytes, 3
            ),
            "copy local shard",
        )
        self.device_module.synchronize(self.device)
        self.exports[handle] = self._fabric.export_handle(handle, self._peer_pids)
        param.untyped_storage().resize_(0)
        self.device_module.empty_cache()
        return handle

    def tensor(self, pointer, nbytes, *, shape, dtype):
        return tensor_from_pointer(
            pointer, nbytes, shape=shape, dtype=dtype, device=self.device
        )

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
        if self._closed:
            return
        self.close_imports()
        self.close_mappings()
        while self._handles:
            _check(
                self.rt.free_physical(self._handles[-1]), "free physical weight memory"
            )
            self._handles.pop()
        self._closed = True

    def close_mappings(self):
        while self._mappings:
            _check(self.rt.unmap_mem(self._mappings[-1]), "unmap weight memory")
            self._mappings.pop()
        while self._reservations:
            _check(
                self.rt.release_mem_address(self._reservations[-1]), "release address"
            )
            self._reservations.pop()

    def prepare_release(self, weight_buffer, peer_views):
        if not self._started or self._closed:
            return
        self.device_module.synchronize(self.device)
        self.group.barrier()
        peer_views.clear()
        if weight_buffer is not None:
            weight_buffer._remote_slices.clear()
        self.close_imports()
        self.group.barrier()

    def postprocess_tensor(self, name, full, experts):
        if (
            name in ("w13_weight_bias", "w2_weight_bias")
            and experts.w13_weight.dtype == torch.float16
        ):
            full = full.to(torch.float16)
        return full

    def ready(self, num_layers):
        self.device_module.synchronize(self.device)
        self.group.barrier()
        self.device_module.empty_cache()
        logger.info(
            "NPU DWDP ready: %d ranks, %d MoE layers, local VMM + remote-only double buffering",
            self.group.world_size,
            num_layers,
        )
