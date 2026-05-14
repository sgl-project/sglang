# SPDX-License-Identifier: Apache-2.0
"""Per-instance transfer manager for disaggregated diffusion roles."""

import logging
import threading
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    TransferTensorBuffer,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)

logger = logging.getLogger(__name__)


@dataclass
class StagedTransfer:
    request_id: str
    slot: SlotHandle
    manifest: dict
    scalar_fields: dict = field(default_factory=dict)


@dataclass
class PendingReceive:
    request_id: str
    slot: SlotHandle


class DiffusionTransferManager:
    """Manages tensor transfers for a single role instance.

    Owns a TransferTensorBuffer (memory pool) and a BaseTransferEngine (RDMA or mock).
    """

    def __init__(
        self,
        engine: BaseTransferEngine,
        buffer: TransferTensorBuffer,
    ):
        self._engine = engine
        self._buffer = buffer
        self._lock = threading.Lock()

        self._engine.register_buffer(self._buffer.pool_data_ptr, self._buffer.pool_size)

        self._staged: dict[str, StagedTransfer] = {}
        self._pending_receives: dict[str, PendingReceive] = {}

        logger.info(
            "DiffusionTransferManager initialized: session=%s, pool=%d bytes",
            self._engine.session_id,
            self._buffer.pool_size,
        )

    @property
    def session_id(self) -> str:
        return self._engine.session_id

    @property
    def pool_data_ptr(self) -> int:
        return self._buffer.pool_data_ptr

    @property
    def pool_size(self) -> int:
        return self._buffer.pool_size

    def stage_tensors(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> StagedTransfer | None:
        """Stage GPU tensors into the local TransferBuffer. Returns None on allocation failure."""
        total_size = 0
        for name, t in tensor_fields.items():
            if t is None:
                continue
            if isinstance(t, list):
                for ti in t:
                    total_size += ti.nelement() * ti.element_size()
            else:
                total_size += t.nelement() * t.element_size()

        if total_size == 0:
            staged = StagedTransfer(
                request_id=request_id,
                slot=None,
                manifest={},
                scalar_fields=scalar_fields or {},
            )
            with self._lock:
                self._staged[request_id] = staged
            return staged

        slot = self._buffer.allocate(total_size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate %d bytes for %s",
                total_size,
                request_id,
            )
            return None

        manifest = self._buffer.write_tensors_from_gpu(slot, tensor_fields, stream)

        if stream is not None:
            stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        staged = StagedTransfer(
            request_id=request_id,
            slot=slot,
            manifest=manifest,
            scalar_fields=scalar_fields or {},
        )
        with self._lock:
            self._staged[request_id] = staged

        logger.debug(
            "TransferManager: staged %s (%d bytes, offset=%d)",
            request_id,
            total_size,
            slot.offset,
        )
        return staged

    def stage_tensors_async(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[StagedTransfer | None, torch.cuda.Event | None]:
        """Stage GPU tensors, returning a CUDA event instead of blocking.

        Caller MUST wait on the event before reading buffer data.
        """
        total_size = 0
        for name, t in tensor_fields.items():
            if t is None:
                continue
            if isinstance(t, list):
                for ti in t:
                    total_size += ti.nelement() * ti.element_size()
            else:
                total_size += t.nelement() * t.element_size()

        if total_size == 0:
            staged = StagedTransfer(
                request_id=request_id,
                slot=None,
                manifest={},
                scalar_fields=scalar_fields or {},
            )
            with self._lock:
                self._staged[request_id] = staged
            return staged, None

        slot = self._buffer.allocate(total_size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate %d bytes for %s",
                total_size,
                request_id,
            )
            return None, None

        manifest = self._buffer.write_tensors_from_gpu(slot, tensor_fields, stream)

        d2h_event = None
        if stream is not None:
            d2h_event = torch.cuda.Event()
            d2h_event.record(stream)
        elif torch.cuda.is_available():
            d2h_event = torch.cuda.Event()
            d2h_event.record(torch.cuda.current_stream())

        staged = StagedTransfer(
            request_id=request_id,
            slot=slot,
            manifest=manifest,
            scalar_fields=scalar_fields or {},
        )
        with self._lock:
            self._staged[request_id] = staged

        logger.debug(
            "TransferManager: staged_async %s (%d bytes, offset=%d)",
            request_id,
            total_size,
            slot.offset,
        )
        return staged, d2h_event

    def load_tensors_async(
        self,
        request_id: str,
        manifest: dict,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], torch.cuda.Event | None]:
        """Load tensors from receive slot to GPU, returning a CUDA event.

        Caller MUST wait on the event before using the returned tensors.
        """
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )

        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot, manifest, device=device, stream=stream
        )

        load_event = None
        if stream is not None:
            load_event = torch.cuda.Event()
            load_event.record(stream)
        elif torch.cuda.is_available():
            load_event = torch.cuda.Event()
            load_event.record(torch.cuda.current_stream())

        logger.debug(
            "TransferManager: loaded_async %d tensor fields for %s to %s",
            len(tensors),
            request_id,
            device,
        )
        return tensors, load_event

    def push_to_peer(
        self,
        request_id: str,
        dest_session_id: str,
        dest_addr: int,
        transfer_size: int,
    ) -> bool:
        """Push staged data to a remote peer's buffer via RDMA. Returns True on success."""
        with self._lock:
            staged = self._staged.get(request_id)

        if staged is None:
            logger.error("TransferManager: no staged transfer for %s", request_id)
            return False

        if staged.slot is None:
            return True

        src_addr = self._buffer.pool_data_ptr + staged.slot.offset
        ret = self._engine.transfer_sync(
            dest_session_id, src_addr, dest_addr, transfer_size
        )

        if ret == 0:
            logger.debug(
                "TransferManager: pushed %s (%d bytes) to %s",
                request_id,
                transfer_size,
                dest_session_id,
            )
        else:
            logger.error(
                "TransferManager: RDMA push failed for %s (ret=%d)",
                request_id,
                ret,
            )

        return ret == 0

    def free_staged(self, request_id: str) -> None:
        with self._lock:
            staged = self._staged.pop(request_id, None)

        if staged and staged.slot is not None:
            self._buffer.free(staged.slot)
            logger.debug("TransferManager: freed staged slot for %s", request_id)

    def allocate_receive_slot(
        self, request_id: str, size: int
    ) -> PendingReceive | None:
        """Allocate a local buffer slot to receive incoming data."""
        slot = self._buffer.allocate(size, request_id)
        if slot is None:
            logger.warning(
                "TransferManager: failed to allocate receive slot (%d bytes) for %s",
                size,
                request_id,
            )
            return None

        pending = PendingReceive(request_id=request_id, slot=slot)
        with self._lock:
            self._pending_receives[request_id] = pending

        logger.debug(
            "TransferManager: allocated receive slot for %s (offset=%d, size=%d)",
            request_id,
            slot.offset,
            slot.size,
        )
        return pending

    def load_tensors(
        self,
        request_id: str,
        manifest: dict,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Load tensors from a receive slot into GPU memory."""
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )

        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot, manifest, device=device, stream=stream
        )

        if stream is not None:
            stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

        logger.debug(
            "TransferManager: loaded %d tensor fields for %s to %s",
            len(tensors),
            request_id,
            device,
        )
        return tensors

    def register_prealloc_as_receive(
        self, request_id: str, slot: "SlotHandle"
    ) -> "PendingReceive":
        """Register a pre-allocated slot as a pending receive (fast path)."""
        pending = PendingReceive(request_id=request_id, slot=slot)
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def free_receive_slot(self, request_id: str) -> None:
        with self._lock:
            pending = self._pending_receives.pop(request_id, None)

        if pending:
            self._buffer.free(pending.slot)
            logger.debug("TransferManager: freed receive slot for %s", request_id)

    def get_receive_slot_addr(self, request_id: str) -> int | None:
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None:
            return None
        return self._buffer.pool_data_ptr + pending.slot.offset

    def get_receive_slot_offset(self, request_id: str) -> int | None:
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None:
            return None
        return pending.slot.offset

    def get_staged_info(self, request_id: str) -> StagedTransfer | None:
        with self._lock:
            return self._staged.get(request_id)

    def free_slots_count(self, typical_size: int = 64 * 1024 * 1024) -> int:
        return self._buffer.free_slots_count(typical_size)

    def cleanup(self) -> None:
        self._engine.deregister_buffer(self._buffer.pool_data_ptr)
        logger.info("DiffusionTransferManager cleaned up")
