"""Immutable expert shards shared through ACL, copied on a caller's NPU stream.

Resident shards use shareable physical allocations. Local pool pages back the
remote regions of composite compute views; kernels never read remote memory.
"""

from __future__ import annotations

import ctypes
import socket
from dataclasses import dataclass


def _check(ret: int, operation: str) -> None:
    if ret != 0:
        raise RuntimeError(f"NPU DWDP: {operation} failed with ACL error {ret}")


class _FabricHandleAPI:
    """CANN V2 ABI, which is not exposed by all pyACL versions."""

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

    def export_handle(self, handle: int, peer_pids: list[int]) -> bytes:
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

    def import_handle(self, share: bytes) -> int:
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


@dataclass
class _Allocation:
    handle: int
    size: int
    pointer: int = 0
    mapped: bool = False
    data_offset: int = 0
    nbytes: int = 0


class NpuWeightTransport:
    """Publish once, then read immutable peer shards without peer forwards.

    FABRIC capability is checked by the actual ACL calls, independently of the
    SoC name. This backend requires a compatible driver, CANN and topology.
    """

    def __init__(self, device):
        import acl

        self.rt = acl.rt
        self.device = device
        self._fabric = _FabricHandleAPI()
        self._local: dict[tuple[int, str], _Allocation] = {}
        self._imports: list[_Allocation] = []
        self._exports: dict[tuple[int, str], bytes] = {}
        self._peer_shards: dict[tuple[int, int, str], tuple[int, int]] = {}
        self._allocations: list[_Allocation] = []
        self._prop = dict(
            handleType=0,
            allocationType=0,
            memAttr=4,  # ACL_HBM_MEM_HUGE
            location=dict(type=1, id=device.index),
            reserve=0,
        )
        self.granularity, ret = self.rt.mem_get_allocation_granularity(self._prop, 0)
        _check(ret, "mem_get_allocation_granularity")

    def identity(self) -> tuple[str, int, int]:
        pid, ret = self.rt.device_get_bare_tgid()
        _check(ret, "device_get_bare_tgid")
        return socket.gethostname(), self.device.index, pid

    def allocate(self, size: int) -> _Allocation:
        handle, ret = self.rt.malloc_physical(size, self._prop, 0)
        _check(ret, "malloc_physical")
        allocation = _Allocation(handle, size)
        self._allocations.append(allocation)
        return allocation

    def reserve(self, size: int) -> int:
        ptr, ret = self.rt.reserve_mem_address(size, 0, 0, 1)
        _check(ret, "reserve_mem_address")
        return ptr

    def map(self, pointer: int, size: int, handle: int) -> None:
        _check(self.rt.map_mem(pointer, size, 0, handle, 0), "map_mem")

    def unmap(self, pointer: int) -> None:
        _check(self.rt.unmap_mem(pointer), "unmap expert shard")

    def release_address(self, pointer: int) -> None:
        _check(self.rt.release_mem_address(pointer), "release expert address")

    def _map(self, allocation: _Allocation) -> None:
        ptr = self.reserve(allocation.size)
        allocation.pointer = ptr
        self.map(ptr, allocation.size, allocation.handle)
        allocation.mapped = True

    def publish(self, key, tensor, peer_pids: list[int], data_offset=0) -> None:
        """The caller supplies a ready, contiguous ND tensor at setup time."""
        nbytes = tensor.numel() * tensor.element_size()
        size = (
            (data_offset + nbytes + self.granularity - 1)
            // self.granularity
            * self.granularity
        )
        allocation = self.allocate(size)
        allocation.data_offset, allocation.nbytes = data_offset, nbytes
        self._map(allocation)
        _check(
            self.rt.memcpy(
                allocation.pointer + data_offset,
                size - data_offset,
                tensor.data_ptr(),
                nbytes,
                3,
            ),
            "copy resident expert shard",
        )
        self._exports[key] = self._fabric.export_handle(allocation.handle, peer_pids)
        self._local[key] = allocation

    def local_allocation(self, key) -> _Allocation:
        return self._local[key]

    def local_tensor(self, key, tensor):
        from .buffer import tensor_from_pointer

        allocation = self._local[key]
        return tensor_from_pointer(
            allocation.pointer + allocation.data_offset,
            allocation.nbytes,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            device=self.device,
        )

    def export_metadata(self) -> dict:
        return {
            key: (
                self._exports[key],
                allocation.size,
                allocation.data_offset,
                allocation.nbytes,
            )
            for key, allocation in self._local.items()
        }

    def import_peers(self, metadata: list[dict], rank: int, identities: list) -> None:
        host, device_id, _ = identities[rank]
        for peer, shards in enumerate(metadata):
            if peer == rank:
                continue
            peer_host, peer_device, _ = identities[peer]
            if peer_host == host and peer_device != device_id:
                can_access, ret = self.rt.device_can_access_peer(device_id, peer_device)
                _check(ret, "device_can_access_peer")
                if not can_access:
                    raise RuntimeError(
                        f"NPU DWDP: no peer access to device {peer_device}"
                    )
                _check(
                    self.rt.device_enable_peer_access(peer_device, 0),
                    "device_enable_peer_access",
                )
            for (layer, name), (share, size, offset, nbytes) in shards.items():
                allocation = _Allocation(self._fabric.import_handle(share), size)
                self._imports.append(allocation)
                self._map(allocation)
                self._peer_shards[peer, layer, name] = (
                    allocation.pointer + offset,
                    nbytes,
                )

    def copy_shard(
        self, rank: int, peer: int, key, destination: int, nbytes: int, stream
    ):
        if peer == rank:
            allocation = self._local[key]
            source = allocation.pointer + allocation.data_offset
            capacity = allocation.nbytes
        else:
            source, capacity = self._peer_shards[peer, *key]
        if not 0 < nbytes <= capacity:
            raise ValueError("NPU DWDP: expert copy exceeds the published shard")
        # A native ACL D2D copy is submitted to this exact NPU stream. Recording
        # a ready event on it therefore includes the actual remote read, unlike
        # recording an unrelated event after a host-side transfer request.
        _check(
            self.rt.memcpy_async(
                destination, nbytes, source, nbytes, 3, stream.npu_stream
            ),
            "memcpy_async expert shard",
        )

    def _release(self, allocations: list[_Allocation]) -> None:
        while allocations:
            allocation = allocations[-1]
            if allocation.mapped:
                _check(self.rt.unmap_mem(allocation.pointer), "unmap expert shard")
                allocation.mapped = False
            if allocation.pointer:
                _check(
                    self.rt.release_mem_address(allocation.pointer),
                    "release expert address",
                )
                allocation.pointer = 0
            _check(self.rt.free_physical(allocation.handle), "free expert handle")
            allocations.pop()

    def close_imports(self) -> None:
        self._release(self._imports)
        self._peer_shards.clear()

    def close(self) -> None:
        self.close_imports()
        self._release(self._allocations)
        self._local.clear()
        self._exports.clear()
