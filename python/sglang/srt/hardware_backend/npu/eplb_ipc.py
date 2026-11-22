"""Allowlisted, same-host expert-weight copies into peer staging memory."""

import ctypes
from typing import Callable

import torch
import torch.distributed as dist


_IPC_KEY_BYTES = 65
_MEMCPY_DEVICE_TO_DEVICE = 3
_MEM_MALLOC_HUGE_FIRST = 0
_IMPORT_ENABLE_PEER_ACCESS = 1


class AclEplbTransport:
    """Own the exported staging allocations, never the model's weight storage."""

    def __init__(self, wrap_storage: Callable):
        self._library = ctypes.CDLL("libascendcl.so")
        ptr = ctypes.c_void_p
        size = ctypes.c_size_t
        key = ctypes.c_char_p
        uint64 = ctypes.c_uint64
        pid_array = ctypes.POINTER(ctypes.c_int32)
        self._malloc = self._bind(
            "aclrtMalloc", ctypes.POINTER(ptr), size, ctypes.c_int
        )
        self._free = self._bind("aclrtFree", ptr)
        self._pid = self._bind("aclrtDeviceGetBareTgid", pid_array)
        self._export = self._bind(
            "aclrtIpcMemGetExportKey", ptr, size, key, size, uint64
        )
        self._allow = self._bind("aclrtIpcMemSetImportPid", key, pid_array, size)
        self._import = self._bind(
            "aclrtIpcMemImportByKey", ctypes.POINTER(ptr), key, uint64
        )
        self._close = self._bind("aclrtIpcMemClose", key)
        self._copy = self._bind(
            "aclrtMemcpyAsync", ptr, size, ptr, size, ctypes.c_int, ptr
        )
        self._wait = self._bind(
            "aclrtSynchronizeStreamWithTimeout", ptr, ctypes.c_int32
        )
        self._wrap_storage = wrap_storage
        self._allocations: list[int] = []
        self._exports: list[bytes] = []
        self._imports: list[bytes] = []
        self._peer_addresses: dict[int, tuple[int, ...]] = {}
        self._peer_events = {}
        self.ready_event = None

    def _bind(self, name, *types):
        function = getattr(self._library, name)
        function.argtypes = types
        function.restype = ctypes.c_int
        return function

    @staticmethod
    def _check(function, *args):
        status = function(*args)
        if status:
            raise RuntimeError(f"{function.__name__} failed with ACL status {status}")

    def allocate(self, shape, device):
        size = shape[0] * shape[1]
        address = ctypes.c_void_p()
        self._check(self._malloc, ctypes.byref(address), size, _MEM_MALLOC_HUGE_FIRST)
        self._allocations.append(address.value)
        return self._wrap_storage(address.value, size, shape, device)

    def _share(self, buffer, allowed_pids):
        key = ctypes.create_string_buffer(_IPC_KEY_BYTES)
        self._check(self._export, buffer.data_ptr(), buffer.nbytes, key, len(key), 0)
        self._exports.append(key.value)
        self._check(self._allow, key.value, allowed_pids, len(allowed_pids))

    def _open(self, key):
        address = ctypes.c_void_p()
        self._check(
            self._import, ctypes.byref(address), key, _IMPORT_ENABLE_PEER_ACCESS
        )
        self._imports.append(key)
        return address.value

    def open(self, staging, owner_stream, group):
        self._group = group
        self._rank = dist.get_rank(group)
        self._world_size = dist.get_world_size(group)
        self._row_bytes = tuple(buffer.size(1) for buffer in staging)

        def checked_exchange(action):
            try:
                result = (action(), None)
            except Exception as exc:
                result = (None, str(exc))
            results = [None] * self._world_size
            dist.all_gather_object(results, result, group=group)
            errors = [(rank, error) for rank, (_, error) in enumerate(results) if error]
            if errors:
                raise RuntimeError(f"Could not open EPLB IPC staging on ranks {errors}")
            return [value for value, _ in results]

        def get_pid():
            pid = ctypes.c_int32()
            self._check(self._pid, ctypes.byref(pid))
            return pid.value

        try:
            pids = checked_exchange(get_pid)
            allowed = [pid for rank, pid in enumerate(pids) if rank != self._rank]
            allowed_pids = (ctypes.c_int32 * len(allowed))(*allowed)

            def export_buffers():
                for buffer in staging:
                    self._share(buffer, allowed_pids)
                return tuple(self._exports)

            exported = checked_exchange(export_buffers)

            def import_buffers():
                for peer, keys in enumerate(exported):
                    if peer != self._rank:
                        self._peer_addresses[peer] = tuple(
                            self._open(key) for key in keys
                        )

            checked_exchange(import_buffers)

            def export_event():
                self.ready_event = torch.npu.Event(interprocess=True)
                self.ready_event.record(owner_stream)
                owner_stream.synchronize()
                self._event_handle = self.ready_event.ipc_handle()
                return self._event_handle

            handles = checked_exchange(export_event)

            def import_events():
                for peer, handle in enumerate(handles):
                    if peer != self._rank:
                        self._peer_events[peer] = torch.npu.Event.from_ipc_handle(
                            staging[0].device, handle
                        )

            checked_exchange(import_events)
        except Exception:
            self.close()
            raise

    def transfer(self, stream, weights, old, new, sends):
        local_slots = weights[0].size(0)
        old_offset = self._rank * local_slots
        by_address = {
            weight.data_ptr() + row * width: (field, old[old_offset + row])
            for field, (weight, width) in enumerate(
                zip(weights, self._row_bytes, strict=True)
            )
            for row in range(local_slots)
        }
        first_slots = [dict() for _ in range(self._world_size)]
        for index, expert in enumerate(new):
            peer, slot = divmod(index, local_slots)
            first_slots[peer].setdefault(expert, slot)

        pending = []
        destinations = set()
        for op in sends:
            source = op.tensor.data_ptr()
            field, expert = by_address[source]
            width = self._row_bytes[field]
            if op.tensor.nbytes != width:
                raise RuntimeError("EPLB send disagrees with the staging row size")
            # The updater receives the first copy of an expert into its first
            # destination slot. Additional replicas reuse that staging row.
            slot = first_slots[op.peer][expert]
            destination = self._peer_addresses[op.peer][field] + slot * width
            pending.append((destination, source, width))
            destinations.add(op.peer)

        # An incoming writer must wait until the peer's previous writeback
        # consumed its staging rows. The handle call drains the queued record
        # submission, and the barrier publishes that generation to every rank.
        if self.ready_event.ipc_handle() != self._event_handle:
            raise RuntimeError("EPLB IPC event handle changed")
        dist.barrier(group=self._group)
        for peer in sorted(destinations):
            stream.wait_event(self._peer_events[peer])
        # Flush the TorchNPU waits before issuing copies through the raw stream.
        handle = stream.npu_stream
        if not handle:
            raise RuntimeError("EPLB could not obtain its NPU migration stream")
        for destination, source, width in pending:
            self._check(
                self._copy,
                destination,
                width,
                source,
                width,
                _MEMCPY_DEVICE_TO_DEVICE,
                handle,
            )
        self._check(self._wait, handle, 60000)

    def close(self):
        self._peer_events.clear()
        for key in self._imports:
            self._check(self._close, key)
        self._imports.clear()
        dist.barrier(group=self._group)
        self.ready_event = None
        for key in self._exports:
            self._check(self._close, key)
        self._exports.clear()
        self._peer_addresses.clear()

    def free(self):
        # The caller drains device work and drops all staging tensor views first.
        for address in self._allocations:
            self._check(self._free, address)
        self._allocations.clear()
