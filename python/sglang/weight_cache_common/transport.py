# SPDX-License-Identifier: Apache-2.0
"""One-shot, storage-level Torch CUDA IPC deliveries.

This is deliberately separate from SRT's legacy replayable state_entries API.
Only descriptors are reusable. Every export below creates new counted sends.
Abandoned sends and fatal consumers may retain PyTorch bookkeeping until the
producer exits; a non-refundable generation budget bounds that exposure.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass

import torch
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef

from .descriptors import StateManifest
from .liveness import ProcessIdentity, ProducerWatchdog
from .mapping import import_state, validate_meta_schema
from .traversal import snapshot_module, storage_byte_views


class ExportBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ExportGeneration:
    producer: ProcessIdentity
    nonce: str
    manifest_digest: str
    device_uuid: str
    torch_version: str


@dataclass(frozen=True)
class StorageHandle:
    group: str
    handle: bytes | None
    nbytes: int
    allocation_offset: int
    counter_handle: bytes | None
    counter_offset: int
    event_handle: bytes | None
    event_sync_required: bool

    def validate(self, nbytes: int) -> None:
        if type(self.nbytes) is not int or self.nbytes != nbytes:
            raise ValueError(f"IPC storage size mismatch: {self.group}")
        for value in (self.allocation_offset, self.counter_offset):
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid IPC offset: {self.group}")
        if type(self.event_sync_required) is not bool:
            raise ValueError("Invalid IPC synchronization flag")
        if not nbytes:
            if self.handle is not None or self.counter_handle is not None:
                raise ValueError("Empty IPC storage must not own a send reference")
            return
        # This is a PyTorch allocator handle, NOT a raw cudaIpcMemHandle_t.
        # Recent Torch versions prefix its 64-byte CUDA payload with a format
        # tag. Keep it opaque and bounded; Torch validates its private format.
        if not isinstance(self.handle, bytes) or not 0 < len(self.handle) <= 4096:
            raise ValueError("Invalid CUDA allocation handle")
        if not isinstance(self.counter_handle, bytes) or not self.counter_handle:
            raise ValueError("Missing CUDA IPC send counter")
        if self.event_sync_required and (
            not isinstance(self.event_handle, bytes) or len(self.event_handle) != 64
        ):
            raise ValueError("Missing CUDA IPC synchronization event")


@dataclass(frozen=True)
class IpcDelivery:
    generation: ExportGeneration
    request_id: str
    storages: tuple[StorageHandle, ...]


def _validate_request_id(request_id: str) -> None:
    if not isinstance(request_id, str) or uuid.UUID(request_id).hex != request_id:
        raise ValueError("IPC request ID must be a canonical UUID hex string")


class CudaIpcExporter:
    """Own one immutable finalized module throughout a serving generation.

    The service must retain this object and its allocations until ALL consumers
    have stopped, including clients that disconnected from the control socket.
    No public close/free/update operation is provided: shutdown coordination
    belongs to the service. stop_admission() does not release any allocation.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        max_deliveries: int = 128,
        max_storage_exports: int = 65536,
    ):
        for limit in (max_deliveries, max_storage_exports):
            if type(limit) is not int or limit <= 0:
                raise ValueError("IPC generation budgets must be positive integers")
        self._snapshot = snapshot_module(module)
        self.manifest = self._snapshot.manifest
        devices = {tensor.device for tensor in self._snapshot.tensors.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError(
                "CUDA IPC export requires state on exactly one CUDA device"
            )
        self._device = next(iter(devices))
        # Refuse both environment and runtime allocator configuration; changing
        # settings via _set_allocator_settings() need not change the environment.
        for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            settings = os.environ.get(key, "").replace(" ", "").lower()
            if "expandable_segments:true" in settings:
                raise ValueError(
                    "Weight-cache Torch IPC does not support expandable_segments"
                )
        allocator_snapshot = torch.cuda.memory._snapshot()
        allocator_settings = allocator_snapshot["allocator_settings"]
        if allocator_settings.get("expandable_segments", False):
            raise ValueError(
                "Weight-cache Torch IPC does not support expandable_segments"
            )
        # Turning the setting off does not change already-created allocations.
        # Inspect the actual exported storage ranges as well as current policy.
        expandable_ranges = [
            (segment["address"], segment["address"] + segment["total_size"])
            for segment in allocator_snapshot["segments"]
            if segment["device"] == self._device.index
            and segment.get("is_expandable", False)
        ]
        for storage in self._snapshot.storages.values():
            if storage.nbytes() and any(
                start <= storage.data_ptr() < end for start, end in expandable_ranges
            ):
                raise ValueError(
                    "Exported allocation uses unsupported expandable_segments"
                )
        self._module = module
        self._views = storage_byte_views(self._snapshot)
        # Finalization may have used non-default streams. Publish no generation
        # until every producer-side write on this device has completed.
        torch.cuda.synchronize(self._device)
        self.generation = ExportGeneration(
            ProcessIdentity.read(os.getpid()),
            uuid.uuid4().hex,
            self.manifest.digest,
            str(torch.cuda.get_device_properties(self._device).uuid),
            str(torch.__version__),
        )
        self._max_deliveries = max_deliveries
        self._max_storage_exports = max_storage_exports
        self._requests: set[str] = set()
        self._storage_exports = 0
        self._failed_deliveries = 0
        self._event_exports = 0
        self._stopped = False
        self._lock = threading.Lock()

    def export(self, request_id: str, *, generation: ExportGeneration) -> IpcDelivery:
        """One request => fresh send references. Never retransmit its result.

        A transport retry must issue a NEW request ID. Accounting is reserved
        before the first Torch export and never refunded, even on exceptions.
        """
        _validate_request_id(request_id)
        # A lock held by another thread at fork can never be released in the
        # child. Reject the inherited exporter before touching that lock.
        if os.getpid() != self.generation.producer.pid:
            raise RuntimeError("An IPC exporter cannot be used after fork")
        with self._lock:
            if generation != self.generation:
                raise ValueError("IPC export generation mismatch")
            if request_id in self._requests:
                raise ValueError("IPC request replay: request a fresh delivery")
            count = len(self._views)
            if (
                self._stopped
                or len(self._requests) >= self._max_deliveries
                or self._storage_exports + count > self._max_storage_exports
            ):
                raise ExportBudgetExceeded(
                    "IPC generation budget exhausted/admission stopped; "
                    "drain consumers and restart the producer"
                )
            self._requests.add(request_id)
            self._storage_exports += count
            handles = []
            try:
                for group, view in self._views.items():
                    if not view.numel():
                        # Torch returns None flags for empty storage. There is
                        # no send reference or CUDA mapping to transfer.
                        handles.append(
                            StorageHandle(group, None, 0, 0, None, 0, None, False)
                        )
                        continue
                    # This API creates a new send reference even when the
                    # allocation handle itself is unchanged. Do not memoize it.
                    _, handle, size, offset, counter, counter_offset, event, sync = (
                        view.untyped_storage()._share_cuda_()
                    )
                    self._event_exports += int(bool(sync))
                    entry = StorageHandle(
                        group,
                        handle,
                        size,
                        offset,
                        counter,
                        counter_offset,
                        event,
                        sync,
                    )
                    entry.validate(view.numel())
                    handles.append(entry)
            except Exception:
                self._failed_deliveries += 1
                raise
            return IpcDelivery(self.generation, request_id, tuple(handles))

    def stop_admission(self) -> None:
        with self._lock:
            self._stopped = True

    def stats(self) -> dict:
        with self._lock:
            return {
                "unique_storage_bytes": self.manifest.unique_storage_bytes,
                "deliveries_reserved": len(self._requests),
                "storage_exports_reserved": self._storage_exports,
                "failed_deliveries": self._failed_deliveries,
                "event_exports": self._event_exports,
                "max_deliveries": self._max_deliveries,
                "max_storage_exports": self._max_storage_exports,
                "admission_stopped": self._stopped,
                "budget_exhausted": (
                    len(self._requests) >= self._max_deliveries
                    or self._storage_exports + len(self._views)
                    > self._max_storage_exports
                ),
            }


# Process-wide replay protection, including attempts through another importer.
# Reject rather than let bookkeeping grow without bound in a long-lived client.
_received: set[tuple[int, str, str]] = set()
_received_lock = threading.Lock()
_MAX_RECEIVED_DELIVERIES = 4096


def _open_storage(handle: StorageHandle, device: int) -> torch.UntypedStorage:
    return torch.UntypedStorage._new_shared_cuda(
        device,
        handle.handle,
        handle.nbytes,
        handle.allocation_offset,
        handle.counter_handle,
        handle.counter_offset,
        handle.event_handle,
        handle.event_sync_required,
    )


class CudaIpcImporter:
    """Generation-bound imports with liveness active before the first mapping."""

    def __init__(self, generation: ExportGeneration, manifest: StateManifest):
        manifest.validate()
        if generation.manifest_digest != manifest.digest:
            raise ValueError("IPC manifest digest mismatch")
        if generation.torch_version != str(torch.__version__):
            raise ValueError("Torch version mismatch for private CUDA IPC ABI")
        if generation.producer.pid == os.getpid():
            raise ValueError("CUDA IPC import must run in a separate consumer process")
        self.generation = generation
        self.manifest = manifest
        self._live_storages: list[StorageWeakRef] = []
        self._closed = False
        self._guard = ProducerWatchdog(generation.producer)
        try:
            matches = [
                index
                for index in range(torch.cuda.device_count())
                if str(torch.cuda.get_device_properties(index).uuid)
                == generation.device_uuid
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Producer physical GPU is not uniquely visible to client"
                )
            self._device = matches[0]
            self._guard.check_alive()
        except Exception:
            self._guard.close()
            raise

    def receive(
        self, delivery: IpcDelivery, module: nn.Module, *, request_id: str
    ) -> None:
        if self._closed:
            raise RuntimeError("IPC importer is closed")
        self._live_storages = [
            reference for reference in self._live_storages if not reference.expired()
        ]
        _validate_request_id(request_id)
        if delivery.generation != self.generation or delivery.request_id != request_id:
            raise ValueError("IPC delivery generation/request mismatch")
        validate_meta_schema(module, self.manifest)
        expected = {storage.group: storage.nbytes for storage in self.manifest.storages}
        if (
            len(delivery.storages) != len(expected)
            or {handle.group for handle in delivery.storages} != expected.keys()
        ):
            raise ValueError("IPC delivery storage groups differ")
        for handle in delivery.storages:
            handle.validate(expected[handle.group])
        allocation_views = [
            (handle.handle, handle.allocation_offset)
            for handle in delivery.storages
            if handle.nbytes
        ]
        if len(set(allocation_views)) != len(allocation_views):
            raise ValueError("Distinct storage groups repeat the same IPC storage")
        self._guard.check_alive()
        key = (os.getpid(), self.generation.nonce, request_id)
        with _received_lock:
            if key in _received:
                raise ValueError("IPC delivery was already consumed in this process")
            if len(_received) >= _MAX_RECEIVED_DELIVERIES:
                raise ExportBudgetExceeded("Consumer IPC delivery budget exhausted")
            # Burn the ID before mapping anything, including failed imports.
            _received.add(key)

        views = {}
        for handle in delivery.storages:
            self._guard.check_alive()
            if handle.nbytes:
                # A separate storage wrapper per counted send owns its one
                # release. Torch internally shares the CUDA allocation mapping.
                storage = _open_storage(handle, self._device)
                # Track immediately: a later import failure's traceback may
                # itself retain this mapping after receive() raises.
                self._live_storages.append(StorageWeakRef(storage))
                view = torch.empty(0, dtype=torch.uint8, device=self._device).set_(
                    storage, 0, (handle.nbytes,), (1,)
                )
            else:
                view = torch.empty(0, dtype=torch.uint8, device=self._device)
                self._live_storages.append(StorageWeakRef(view.untyped_storage()))
            views[handle.group] = view
        self._guard.check_alive()
        import_state(module, self.manifest, views)
        module.__dict__["_weight_cache_importer"] = self
        self._guard.check_alive()

    def close(self) -> None:
        if any(not reference.expired() for reference in self._live_storages):
            raise RuntimeError(
                "Release all imported storage/views before closing their guard"
            )
        self._guard.close()
        self._live_storages.clear()
        self._closed = True
