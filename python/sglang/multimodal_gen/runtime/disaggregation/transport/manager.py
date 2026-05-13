# SPDX-License-Identifier: Apache-2.0
"""Per-instance transfer manager for disaggregated diffusion roles."""

from __future__ import annotations

import concurrent.futures
import ctypes
import logging
import queue
import threading
import time
import zlib
from dataclasses import dataclass, field
from multiprocessing import shared_memory

import numpy as np
import torch
import zmq

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.disaggregation.transport.buffer import (
    SlotHandle,
    TransferMetaBuffer,
    TransferTensorBuffer,
    estimate_transfer_meta_bytes,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.engine import (
    BaseTransferEngine,
)
from sglang.multimodal_gen.runtime.disaggregation.transport.protocol import (
    TransferMsgType,
    decode_transfer_msg,
    encode_transfer_msg,
)
from sglang.multimodal_gen.runtime.utils.common import get_zmq_socket

logger = logging.getLogger(__name__)


@dataclass
class StagedTransfer:
    request_id: str
    slot: SlotHandle | None
    meta_slot: SlotHandle | None
    manifest: dict
    transfer_size: int = 0
    meta_size: int = 0
    scalar_fields: dict = field(default_factory=dict)
    ready: bool = False


@dataclass
class PendingReceive:
    request_id: str
    slot: SlotHandle | None
    meta_slot: SlotHandle | None
    slot_id: int | None = None


@dataclass
class PendingPeerSend:
    request_id: str
    dest_session_id: str = ""
    dest_addr: int = 0
    transfer_size: int = 0
    meta_dest_addr: int = 0
    meta_transfer_size: int = 0
    receiver_role: str = ""
    receiver_instance: int = -1
    receiver_control_endpoint: str = ""
    receiver_host_id: str = ""
    receiver_supports_local_copy: bool = False
    dest_shm_name: str | None = None
    dest_shm_offset: int = 0
    meta_dest_shm_name: str | None = None
    meta_dest_shm_offset: int = 0
    prealloc_slot_id: int | None = None
    send_attempts: int = 0
    max_send_retries: int = 2
    last_error: str | None = None
    state: str = "waiting_stage"


@dataclass
class SendCompletion:
    request_id: str
    peer_info: PendingPeerSend | None
    staged: StagedTransfer | None
    success: bool
    error_msg: str | None = None
    retryable: bool = True


@dataclass
class _SharedMemoryAttachment:
    shm: shared_memory.SharedMemory
    np_array: np.ndarray

    @property
    def data_ptr(self) -> int:
        return int(self.np_array.ctypes.data)


class DiffusionTransferManager:
    """Manages host-side tensor/meta transfers for one diffusion role instance."""

    def __init__(
        self,
        engine: BaseTransferEngine,
        buffer: TransferTensorBuffer,
        meta_buffer: TransferMetaBuffer,
        *,
        host_id: str = "",
        send_retry_limit: int = 2,
    ):
        self._engine = engine
        self._buffer = buffer
        self._meta_buffer = meta_buffer
        self._host_id = host_id
        self._send_retry_limit = max(0, int(send_retry_limit))
        self._lock = threading.Lock()

        self._engine.register_buffer(self._buffer.pool_data_ptr, self._buffer.pool_size)
        self._engine.register_buffer(
            self._meta_buffer.pool_data_ptr, self._meta_buffer.pool_size
        )

        self._staged: dict[str, StagedTransfer] = {}
        self._pending_receives: dict[str, PendingReceive] = {}
        self._pending_peer_sends: dict[str, PendingPeerSend] = {}
        self._context: zmq.Context | None = None
        self._control_endpoint: str | None = None
        self._control_pull: zmq.Socket | None = None
        self._control_push_sockets: dict[tuple[int, str], zmq.Socket] = {}
        self._send_queues: list[queue.Queue[str | None]] = []
        self._completion_queue: queue.Queue[SendCompletion] = queue.Queue()
        self._receive_thread: threading.Thread | None = None
        self._send_threads: list[threading.Thread] = []
        self._send_executors: list[concurrent.futures.ThreadPoolExecutor | None] = []
        self._send_futures: dict[str, concurrent.futures.Future] = {}
        self._terminal_send_states: dict[str, str] = {}
        self._running = False
        self._send_concurrency = 1
        self._send_queue_count = 0
        self._send_worker_counts: list[int] = []
        self._on_ready = None
        self._on_failed = None
        self._on_abort = None
        self._on_send_completion = None
        self._shared_memory_attachments: dict[str, _SharedMemoryAttachment] = {}
        self._aborted_requests: dict[str, float] = {}
        self._abort_tombstone_ttl_s = 300.0

        logger.info(
            "DiffusionTransferManager initialized: session=%s, data_pool=%d bytes, meta_pool=%d bytes",
            self._engine.session_id,
            self._buffer.pool_size,
            self._meta_buffer.pool_size,
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

    @property
    def meta_pool_ptr(self) -> int:
        return self._meta_buffer.pool_data_ptr

    @property
    def meta_pool_size(self) -> int:
        return self._meta_buffer.pool_size

    @property
    def data_shm_name(self) -> str | None:
        return self._buffer.shared_memory_name

    @property
    def meta_shm_name(self) -> str | None:
        return self._meta_buffer.shared_memory_name

    @property
    def control_endpoint(self) -> str | None:
        return self._control_endpoint

    @property
    def host_id(self) -> str:
        return self._host_id

    @staticmethod
    def _tensor_fields_use_cuda(
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    ) -> bool:
        for value in tensor_fields.values():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                return True
            if isinstance(value, list):
                for tensor in value:
                    if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                        return True
        return False

    @staticmethod
    def _estimate_transfer_size(
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
    ) -> int:
        total_size = 0
        for value in tensor_fields.values():
            if value is None:
                continue
            tensors = value if isinstance(value, list) else [value]
            for tensor in tensors:
                if tensor is None:
                    continue
                total_size += tensor.nelement() * tensor.element_size()
                total_size = (total_size + 511) & ~511
        return total_size

    @staticmethod
    def _distribute_workers(total_workers: int, queue_count: int) -> list[int]:
        base = total_workers // queue_count
        remainder = total_workers % queue_count
        return [base + (1 if idx < remainder else 0) for idx in range(queue_count)]

    def _resolve_send_runtime_config(
        self, fallback_total_workers: int
    ) -> tuple[int, int, list[int]]:
        queue_count_env = envs.SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE
        queue_count = 1 if queue_count_env is None else int(queue_count_env)
        if queue_count < 1:
            raise ValueError(
                "SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE must be greater than or equal to 1."
            )

        thread_pool_env = envs.SGLANG_DIFFUSION_DISAGG_SEND_THREAD_POOL_SIZE
        if thread_pool_env is None:
            total_workers = max(int(fallback_total_workers), queue_count)
        else:
            total_workers = int(thread_pool_env)

        if total_workers < 1:
            raise ValueError(
                "SGLANG_DIFFUSION_DISAGG_SEND_THREAD_POOL_SIZE must be greater than or equal to 1."
            )

        if thread_pool_env is not None and total_workers < queue_count:
            raise ValueError(
                "The environment variable "
                f"SGLANG_DIFFUSION_DISAGG_SEND_THREAD_POOL_SIZE={total_workers} "
                "must be greater than or equal to "
                f"SGLANG_DIFFUSION_DISAGG_SEND_QUEUE_SIZE={queue_count}."
            )

        worker_counts = self._distribute_workers(total_workers, queue_count)
        return queue_count, total_workers, worker_counts

    def _select_send_queue_idx(self, peer_info: PendingPeerSend) -> int:
        queue_count = len(self._send_queues)
        if queue_count <= 1:
            return 0

        downstream_key = f"{peer_info.receiver_instance}:{peer_info.dest_session_id}"
        if not peer_info.dest_session_id:
            downstream_key = (
                f"{peer_info.receiver_instance}:{peer_info.receiver_control_endpoint}"
            )
        if downstream_key.endswith(":"):
            downstream_key = peer_info.request_id
        return zlib.crc32(downstream_key.encode("utf-8")) % queue_count

    def start_background_loops(
        self,
        context: zmq.Context,
        control_endpoint: str,
        *,
        on_ready=None,
        on_failed=None,
        on_abort=None,
        on_send_completion=None,
        send_concurrency: int = 1,
        start_send_loop: bool = True,
    ) -> None:
        if self._running:
            return

        self._context = context
        self._control_endpoint = control_endpoint
        self._control_pull, _ = get_zmq_socket(
            context,
            zmq.PULL,
            control_endpoint,
            bind=True,
            max_bind_retries=5,
            same_port=True,
        )
        self._control_pull.setsockopt(zmq.RCVTIMEO, 1000)
        self._on_ready = on_ready
        self._on_failed = on_failed
        self._on_abort = on_abort
        self._on_send_completion = on_send_completion
        self._send_concurrency = max(1, int(send_concurrency))
        if start_send_loop:
            queue_count, total_workers, worker_counts = (
                self._resolve_send_runtime_config(self._send_concurrency)
            )
            self._send_queue_count = queue_count
            self._send_worker_counts = worker_counts
            self._send_queues = [queue.Queue() for _ in range(queue_count)]
            self._send_executors = [
                concurrent.futures.ThreadPoolExecutor(
                    max_workers=worker_count,
                    thread_name_prefix=(
                        f"transfer-worker-{self._engine.session_id}-{queue_idx}"
                    ),
                )
                for queue_idx, worker_count in enumerate(worker_counts)
            ]
            logger.info(
                "DiffusionTransferManager send runtime config: session=%s, queues=%d, total_workers=%d, worker_counts=%s",
                self._engine.session_id,
                queue_count,
                total_workers,
                worker_counts,
            )
        else:
            self._send_queue_count = 0
            self._send_worker_counts = []
            self._send_queues = []
            self._send_executors = []
        self._running = True

        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
            name=f"transfer-recv-{self._engine.session_id}",
        )
        self._receive_thread.start()
        if start_send_loop:
            self._send_threads = [
                threading.Thread(
                    target=self._send_loop,
                    args=(queue_idx,),
                    daemon=True,
                    name=f"transfer-send-{self._engine.session_id}-{queue_idx}",
                )
                for queue_idx in range(len(self._send_queues))
            ]
            for thread in self._send_threads:
                thread.start()
        else:
            self._send_threads = []

    def _get_shared_memory_attachment(self, name: str) -> _SharedMemoryAttachment:
        attachment = self._shared_memory_attachments.get(name)
        if attachment is not None:
            return attachment

        shm = shared_memory.SharedMemory(name=name)
        np_array = np.ndarray((shm.size,), dtype=np.uint8, buffer=shm.buf)
        attachment = _SharedMemoryAttachment(shm=shm, np_array=np_array)
        self._shared_memory_attachments[name] = attachment
        return attachment

    def _local_copy(
        self,
        src_addr: int,
        dst_shm_name: str | None,
        dst_shm_offset: int,
        length: int,
    ) -> bool:
        if not dst_shm_name or length <= 0:
            return True
        try:
            attachment = self._get_shared_memory_attachment(dst_shm_name)
            ctypes.memmove(
                attachment.data_ptr + int(dst_shm_offset), src_addr, int(length)
            )
            return True
        except Exception:
            logger.exception("TransferManager local shared-memory copy failed")
            return False

    def stage_tensors(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> StagedTransfer | None:
        """Compatibility wrapper for tests/debugging.

        Production role code uses stage_tensors_async() with background send
        completion so CUDA streams do not block the scheduler loop.
        """
        staged, ready_event = self.stage_tensors_async(
            request_id, tensor_fields, scalar_fields, stream
        )
        if staged is None:
            return None
        if ready_event is not None and hasattr(ready_event, "synchronize"):
            ready_event.synchronize()
        self.mark_staged_ready(request_id)
        return staged

    def stage_tensors_async(
        self,
        request_id: str,
        tensor_fields: dict[str, torch.Tensor | list[torch.Tensor] | None],
        scalar_fields: dict | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[StagedTransfer | None, object | None]:
        scalar_fields = scalar_fields or {}
        total_size = self._estimate_transfer_size(tensor_fields)
        data_slot = None
        manifest = {}

        if total_size > 0:
            data_slot = self._buffer.allocate(total_size, request_id)
            if data_slot is None:
                logger.warning(
                    "TransferManager: failed to allocate %d bytes for %s",
                    total_size,
                    request_id,
                )
                return None, None
            manifest = self._buffer.write_tensors_from_gpu(
                data_slot, tensor_fields, stream
            )

        meta_size = estimate_transfer_meta_bytes(manifest, scalar_fields)
        meta_slot = self._meta_buffer.allocate(request_id)
        if meta_slot is None:
            if data_slot is not None:
                self._buffer.free(data_slot)
            logger.warning(
                "TransferManager: failed to allocate meta slot for %s", request_id
            )
            return None, None

        try:
            meta_size = self._meta_buffer.write_metadata(
                meta_slot, manifest, scalar_fields
            )
        except Exception:
            if data_slot is not None:
                self._buffer.free(data_slot)
            self._meta_buffer.free(meta_slot)
            raise

        ready_event = None
        if stream is not None and self._tensor_fields_use_cuda(tensor_fields):
            ready_event = torch.cuda.Event()
            ready_event.record(stream)
        elif self._tensor_fields_use_cuda(tensor_fields) and torch.cuda.is_available():
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream())

        staged = StagedTransfer(
            request_id=request_id,
            slot=data_slot,
            meta_slot=meta_slot,
            manifest=manifest,
            transfer_size=total_size,
            meta_size=meta_size,
            scalar_fields=scalar_fields,
            ready=ready_event is None,
        )
        with self._lock:
            self._staged[request_id] = staged

        if ready_event is None:
            self.mark_staged_ready(request_id)

        logger.debug(
            "TransferManager: staged_async %s (data=%d bytes, meta=%d bytes)",
            request_id,
            total_size,
            meta_size,
        )
        return staged, ready_event

    def mark_staged_ready(self, request_id: str) -> None:
        with self._lock:
            staged = self._staged.get(request_id)
            if staged is None:
                return
            staged.ready = True
            self._maybe_enqueue_send_locked(request_id)

    def _load_received_transfer(
        self,
        request_id: str,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], dict, object | None]:
        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            raise ValueError(
                f"TransferManager: no pending receive slot for {request_id}"
            )

        manifest, scalar_fields = self._meta_buffer.read_metadata(pending.meta_slot)
        tensors = self._buffer.read_tensors_from_manifest(
            pending.slot,
            manifest,
            device=device,
            stream=stream,
        )

        load_event = None
        if stream is not None and str(device).startswith("cuda"):
            load_event = torch.cuda.Event()
            load_event.record(stream)
        elif str(device).startswith("cuda") and torch.cuda.is_available():
            load_event = torch.cuda.Event()
            load_event.record(torch.cuda.current_stream())

        return tensors, scalar_fields, load_event

    def load_tensors_async(
        self,
        request_id: str,
        manifest: dict | None = None,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], object | None]:
        tensors, _scalar_fields, load_event = self._load_received_transfer(
            request_id, device=device, stream=stream
        )
        return tensors, load_event

    def load_transfer_async(
        self,
        request_id: str,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor]], dict, object | None]:
        return self._load_received_transfer(request_id, device=device, stream=stream)

    def push_to_peer(
        self,
        request_id: str,
        dest_session_id: str,
        dest_addr: int,
        transfer_size: int,
    ) -> bool:
        """Compatibility wrapper for tests/debugging.

        Production role code queues sends through the background worker path.
        """
        with self._lock:
            staged = self._staged.get(request_id)
        if staged is None:
            logger.error("TransferManager: no staged transfer for %s", request_id)
            return False
        if staged.slot is None or transfer_size <= 0:
            return True
        src_addr = self._buffer.pool_data_ptr + staged.slot.offset
        ret = self._engine.transfer_sync(
            dest_session_id, src_addr, dest_addr, transfer_size
        )
        return ret == 0

    def free_staged(self, request_id: str) -> None:
        with self._lock:
            staged = self._staged.pop(request_id, None)
        if staged is None:
            return
        if staged.slot is not None:
            self._buffer.free(staged.slot)
        if staged.meta_slot is not None:
            self._meta_buffer.free(staged.meta_slot)

    def allocate_receive_slot(
        self, request_id: str, size: int, meta_size: int
    ) -> PendingReceive | None:
        slot = self._buffer.allocate(size, request_id) if size > 0 else None
        meta_slot = self._meta_buffer.allocate(request_id)
        if meta_slot is None:
            if slot is not None:
                self._buffer.free(slot)
            logger.warning(
                "TransferManager: failed to allocate receive meta slot for %s",
                request_id,
            )
            return None
        if meta_size > meta_slot.size:
            if slot is not None:
                self._buffer.free(slot)
            self._meta_buffer.free(meta_slot)
            logger.warning(
                "TransferManager: receive meta slot too small for %s (%d > %d)",
                request_id,
                meta_size,
                meta_slot.size,
            )
            return None
        pending = PendingReceive(request_id=request_id, slot=slot, meta_slot=meta_slot)
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def load_tensors(
        self,
        request_id: str,
        manifest: dict | None = None,
        device: torch.device | str = "cuda",
        stream: torch.cuda.Stream | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Compatibility wrapper for tests/debugging.

        Production role code uses load_transfer_async() to preserve scalar fields
        and defer CUDA synchronization to the caller.
        """
        del manifest
        tensors, _scalar_fields, load_event = self._load_received_transfer(
            request_id, device=device, stream=stream
        )
        if load_event is not None and hasattr(load_event, "synchronize"):
            load_event.synchronize()
        return tensors

    def register_prealloc_as_receive(
        self,
        request_id: str,
        slot: SlotHandle | None,
        meta_slot: SlotHandle,
        slot_id: int | None = None,
    ) -> PendingReceive:
        pending = PendingReceive(
            request_id=request_id,
            slot=slot,
            meta_slot=meta_slot,
            slot_id=slot_id,
        )
        with self._lock:
            self._pending_receives[request_id] = pending
        return pending

    def get_pending_receive(self, request_id: str) -> PendingReceive | None:
        with self._lock:
            return self._pending_receives.get(request_id)

    def free_receive_slot(self, request_id: str) -> None:
        with self._lock:
            pending = self._pending_receives.pop(request_id, None)
        if pending is None:
            return
        if pending.slot is not None:
            self._buffer.free(pending.slot)
        if pending.meta_slot is not None:
            self._meta_buffer.free(pending.meta_slot)

    def get_receive_slot_addr(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.slot is None:
            return None
        return self._buffer.pool_data_ptr + pending.slot.offset

    def get_receive_slot_offset(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.slot is None:
            return None
        return pending.slot.offset

    def get_receive_meta_addr(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.meta_slot is None:
            return None
        return self._meta_buffer.pool_data_ptr + pending.meta_slot.offset

    def get_receive_meta_offset(self, request_id: str) -> int | None:
        """Test/debug accessor; production transfer uses peer info messages."""
        with self._lock:
            pending = self._pending_receives.get(request_id)
        if pending is None or pending.meta_slot is None:
            return None
        return pending.meta_slot.offset

    def validate_receive_ready(
        self,
        request_id: str,
        *,
        dest_session_id: str | None = None,
        dest_slot_offset: int | None = None,
        dest_meta_slot_offset: int | None = None,
        data_size: int | None = None,
        meta_size: int | None = None,
    ) -> str | None:
        """Validate that a READY message targets the current pending receive slot.

        READY can arrive after a warmup-driven transfer-manager resize. In that
        case request_id alone is not enough: the new manager may have a fresh,
        zeroed slot for the same request while an older sender session is still
        finishing. Return a human-readable error instead of letting callers read
        unrelated metadata bytes.
        """
        if dest_session_id and dest_session_id != self.session_id:
            return (
                f"receiver session mismatch: ready={dest_session_id}, "
                f"current={self.session_id}"
            )

        with self._lock:
            pending = self._pending_receives.get(request_id)

        if pending is None:
            return f"no pending receive slot for {request_id}"

        if pending.slot is None:
            if data_size is not None and int(data_size) > 0:
                return f"data slot missing for non-empty transfer: ready={data_size}"
        else:
            if dest_slot_offset is not None and int(dest_slot_offset) != int(
                pending.slot.offset
            ):
                return (
                    f"data slot offset mismatch: ready={dest_slot_offset}, "
                    f"pending={pending.slot.offset}"
                )
            if data_size is not None and int(data_size) > int(pending.slot.size):
                return (
                    f"data size exceeds receive slot: ready={data_size}, "
                    f"slot={pending.slot.size}"
                )

        if dest_meta_slot_offset is not None and pending.meta_slot is not None:
            if int(dest_meta_slot_offset) != int(pending.meta_slot.offset):
                return (
                    f"metadata slot offset mismatch: ready={dest_meta_slot_offset}, "
                    f"pending={pending.meta_slot.offset}"
                )
        if meta_size is not None and pending.meta_slot is not None:
            if int(meta_size) > int(pending.meta_slot.size):
                return (
                    f"metadata size exceeds receive slot: ready={meta_size}, "
                    f"slot={pending.meta_slot.size}"
                )

        return None

    def get_staged_info(self, request_id: str) -> StagedTransfer | None:
        with self._lock:
            return self._staged.get(request_id)

    def has_active_transfers(self) -> bool:
        with self._lock:
            return bool(
                self._staged
                or self._pending_receives
                or self._pending_peer_sends
                or self._send_futures
            )

    def free_slots_count(self, typical_size: int = 64 * 1024 * 1024) -> int:
        return self._buffer.free_slots_count(typical_size)

    def send_direct_message(self, endpoint: str, msg) -> None:
        if self._context is None:
            raise RuntimeError(
                "TransferManager direct-control context is not initialized"
            )
        if not endpoint:
            raise ValueError("TransferManager direct-control endpoint is empty")

        key = (threading.get_ident(), endpoint)
        sock = self._control_push_sockets.get(key)
        if sock is None:
            sock, _ = get_zmq_socket(self._context, zmq.PUSH, endpoint, bind=False)
            self._control_push_sockets[key] = sock
        sock.send_multipart(encode_transfer_msg(msg))

    def _prune_aborted_locked(self) -> None:
        if not self._aborted_requests:
            return
        cutoff = time.monotonic() - self._abort_tombstone_ttl_s
        stale = [rid for rid, ts in self._aborted_requests.items() if ts < cutoff]
        for rid in stale:
            self._aborted_requests.pop(rid, None)

    def is_request_aborted(self, request_id: str) -> bool:
        with self._lock:
            self._prune_aborted_locked()
            return request_id in self._aborted_requests

    def abort_request(self, request_id: str) -> None:
        with self._lock:
            self._prune_aborted_locked()
            self._aborted_requests[request_id] = time.monotonic()
            self._terminal_send_states[request_id] = "aborted"
            staged = self._staged.pop(request_id, None)
            pending = self._pending_receives.pop(request_id, None)
            peer = self._pending_peer_sends.pop(request_id, None)
            future = self._send_futures.pop(request_id, None)
            if peer is not None:
                peer.state = "aborted"

        if future is not None:
            future.cancel()
        if staged is not None:
            if staged.slot is not None:
                self._buffer.free(staged.slot)
            if staged.meta_slot is not None:
                self._meta_buffer.free(staged.meta_slot)
        if pending is not None:
            if pending.slot is not None and pending.slot_id is None:
                self._buffer.free(pending.slot)
            if pending.meta_slot is not None and pending.slot_id is None:
                self._meta_buffer.free(pending.meta_slot)

    def _maybe_enqueue_send_locked(self, request_id: str) -> None:
        self._prune_aborted_locked()
        if request_id in self._aborted_requests:
            return
        if request_id in self._terminal_send_states:
            return
        staged = self._staged.get(request_id)
        peer_info = self._pending_peer_sends.get(request_id)
        if staged is None or peer_info is None:
            return
        if not staged.ready:
            return
        if peer_info.state in ("worker_queued", "inflight", "success", "failed"):
            return
        if not self._send_queues:
            logger.debug(
                "TransferManager: send queues not initialized, cannot enqueue %s",
                request_id,
            )
            return
        peer_info.state = "worker_queued"
        self._send_queues[self._select_send_queue_idx(peer_info)].put(request_id)

    def _register_peer_send(self, msg: dict) -> None:
        request_id = msg.get("request_id", "")
        if not request_id:
            return
        with self._lock:
            self._prune_aborted_locked()
            if request_id in self._aborted_requests:
                logger.debug(
                    "TransferManager: ignoring peer info for aborted %s", request_id
                )
                return
            if request_id in self._terminal_send_states:
                logger.debug(
                    "TransferManager: ignoring duplicate peer info for terminal %s",
                    request_id,
                )
                return
            existing = self._pending_peer_sends.get(request_id)
            if existing is not None and existing.state in ("worker_queued", "inflight"):
                logger.debug(
                    "TransferManager: ignoring duplicate peer info while %s is %s",
                    request_id,
                    existing.state,
                )
                return
            pending = PendingPeerSend(
                request_id=request_id,
                dest_session_id=msg.get("dest_session_id", ""),
                dest_addr=msg.get("dest_addr", 0),
                transfer_size=msg.get("transfer_size", 0),
                meta_dest_addr=msg.get("meta_dest_addr", 0),
                meta_transfer_size=msg.get("meta_transfer_size", 0),
                receiver_role=msg.get("receiver_role", ""),
                receiver_instance=msg.get("receiver_instance", -1),
                receiver_control_endpoint=msg.get("receiver_control_endpoint", ""),
                receiver_host_id=msg.get("receiver_host_id", ""),
                receiver_supports_local_copy=bool(
                    msg.get("receiver_supports_local_copy", False)
                ),
                dest_shm_name=msg.get("dest_shm_name"),
                dest_shm_offset=msg.get("dest_shm_offset", 0),
                meta_dest_shm_name=msg.get("meta_dest_shm_name"),
                meta_dest_shm_offset=msg.get("meta_dest_shm_offset", 0),
                prealloc_slot_id=msg.get("prealloc_slot_id"),
                max_send_retries=self._send_retry_limit,
                state="waiting_stage",
            )
            self._pending_peer_sends[request_id] = pending
            self._maybe_enqueue_send_locked(request_id)

    def _submit_send_task(self, request_id: str, queue_idx: int | None = None) -> None:
        with self._lock:
            self._prune_aborted_locked()
            if request_id in self._aborted_requests:
                return
            if request_id in self._terminal_send_states:
                return
            peer_info = self._pending_peer_sends.get(request_id)
            staged = self._staged.get(request_id)
            if peer_info is None:
                logger.debug(
                    "TransferManager send loop: missing peer info for %s", request_id
                )
                return
            if peer_info.state == "inflight" or request_id in self._send_futures:
                return
            if staged is None or not staged.ready:
                peer_info.state = "failed"
                self._completion_queue.put(
                    SendCompletion(
                        request_id=request_id,
                        peer_info=peer_info,
                        staged=staged,
                        success=False,
                        error_msg="missing staged payload",
                        retryable=False,
                    )
                )
                return

            if queue_idx is None:
                queue_idx = self._select_send_queue_idx(peer_info)
            executor = None
            if 0 <= queue_idx < len(self._send_executors):
                executor = self._send_executors[queue_idx]
            if executor is None:
                peer_info.state = "failed"
                self._completion_queue.put(
                    SendCompletion(
                        request_id=request_id,
                        peer_info=peer_info,
                        staged=staged,
                        success=False,
                        error_msg="send executor not initialized",
                        retryable=False,
                    )
                )
                return

            peer_info.send_attempts += 1
            peer_info.state = "inflight"
            future = executor.submit(self._execute_send, request_id, peer_info)
            self._send_futures[request_id] = future

        future.add_done_callback(
            lambda fut, req_id=request_id, info=peer_info, staged_info=staged: (
                self._enqueue_send_completion(req_id, info, staged_info, fut)
            )
        )

    def _execute_send(
        self, request_id: str, peer_info: PendingPeerSend
    ) -> tuple[bool, str | None]:
        try:
            with self._lock:
                staged = self._staged.get(request_id)
            if staged is None:
                return False, "missing staged payload"

            local_copy = (
                peer_info.receiver_supports_local_copy
                and peer_info.receiver_host_id
                and peer_info.receiver_host_id == self._host_id
            )

            if local_copy:
                ok_data = True
                if staged.slot is not None and peer_info.transfer_size > 0:
                    ok_data = self._local_copy(
                        self._buffer.pool_data_ptr + staged.slot.offset,
                        peer_info.dest_shm_name,
                        peer_info.dest_shm_offset,
                        peer_info.transfer_size,
                    )
                ok_meta = True
                if staged.meta_slot is not None and peer_info.meta_transfer_size > 0:
                    ok_meta = self._local_copy(
                        self._meta_buffer.pool_data_ptr + staged.meta_slot.offset,
                        peer_info.meta_dest_shm_name,
                        peer_info.meta_dest_shm_offset,
                        peer_info.meta_transfer_size,
                    )
                success = ok_data and ok_meta
                return success, None if success else "local shared-memory copy failed"

            src_addrs = []
            dst_addrs = []
            lengths = []
            if staged.slot is not None and peer_info.transfer_size > 0:
                src_addrs.append(self._buffer.pool_data_ptr + staged.slot.offset)
                dst_addrs.append(peer_info.dest_addr)
                lengths.append(peer_info.transfer_size)
            if staged.meta_slot is not None and peer_info.meta_transfer_size > 0:
                src_addrs.append(
                    self._meta_buffer.pool_data_ptr + staged.meta_slot.offset
                )
                dst_addrs.append(peer_info.meta_dest_addr)
                lengths.append(peer_info.meta_transfer_size)
            if not src_addrs:
                return True, None

            ret = self._engine.batch_transfer_sync(
                peer_info.dest_session_id,
                src_addrs,
                dst_addrs,
                lengths,
            )
            if ret == 0:
                return True, None
            return False, "transfer_sync failed"
        except Exception as exc:
            logger.exception("TransferManager send loop failed for %s", request_id)
            return False, str(exc)

    def _enqueue_send_completion(
        self,
        request_id: str,
        peer_info: PendingPeerSend,
        staged: StagedTransfer | None,
        future: concurrent.futures.Future,
    ) -> None:
        try:
            success, error_msg = future.result()
        except Exception as exc:
            success, error_msg = False, str(exc)
        self._completion_queue.put(
            SendCompletion(
                request_id=request_id,
                peer_info=peer_info,
                staged=staged,
                success=success,
                error_msg=error_msg,
                retryable=error_msg != "missing staged payload",
            )
        )

    def _process_send_completion(self, completion: SendCompletion) -> None:
        request_id = completion.request_id
        peer_info = completion.peer_info
        retry_request = False
        retry_queue_idx = 0

        with self._lock:
            self._prune_aborted_locked()
            if request_id in self._aborted_requests:
                self._send_futures.pop(request_id, None)
                return
            if request_id in self._terminal_send_states:
                self._send_futures.pop(request_id, None)
                return

            self._send_futures.pop(request_id, None)
            pending = self._pending_peer_sends.get(request_id)

            if completion.success:
                self._terminal_send_states[request_id] = "success"
                pending = self._pending_peer_sends.pop(request_id, None)
                if pending is not None:
                    pending.state = "success"
            else:
                retry_peer = pending or peer_info
                if retry_peer is not None:
                    retry_peer.last_error = completion.error_msg
                can_retry = (
                    retry_peer is not None
                    and completion.retryable
                    and retry_peer.send_attempts <= retry_peer.max_send_retries
                )
                if can_retry:
                    retry_request = True
                    retry_queue_idx = self._select_send_queue_idx(retry_peer)
                    retry_peer.state = "waiting_retry"
                else:
                    self._terminal_send_states[request_id] = "failed"
                    pending = self._pending_peer_sends.pop(request_id, None)
                    if pending is not None:
                        pending.state = "failed"
                        pending.last_error = completion.error_msg

        if retry_request:
            self._send_queues[retry_queue_idx].put(request_id)
            return

        if self._on_send_completion is not None and peer_info is not None:
            self._on_send_completion(
                request_id,
                peer_info,
                completion.staged,
                completion.success,
                completion.error_msg,
            )

        if completion.staged is not None:
            self.free_staged(request_id)

    def _drain_send_completions(self) -> bool:
        drained = False
        while True:
            try:
                completion = self._completion_queue.get_nowait()
            except queue.Empty:
                break
            self._process_send_completion(completion)
            drained = True
        return drained

    def _receive_loop(self) -> None:
        while self._running and self._control_pull is not None:
            try:
                frames = self._control_pull.recv_multipart()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if not self._running:
                    break
                logger.exception("TransferManager receive loop socket error")
                continue

            try:
                msg = decode_transfer_msg([bytes(f) for f in frames])
            except Exception:
                logger.exception(
                    "TransferManager receive loop failed to decode message"
                )
                continue

            msg_type = msg.get("msg_type", "")
            if msg_type == TransferMsgType.PEER_INFO:
                self._register_peer_send(msg)
            elif msg_type == TransferMsgType.READY:
                if self._on_ready is not None:
                    self._on_ready(msg)
            elif msg_type == TransferMsgType.FAILED:
                if self._on_failed is not None:
                    self._on_failed(msg)
            elif msg_type == TransferMsgType.ABORT:
                if self._on_abort is not None:
                    self._on_abort(msg)
            else:
                logger.warning(
                    "TransferManager receive loop: unsupported msg_type=%s", msg_type
                )

    def _send_loop(self, queue_idx: int) -> None:
        send_queue = self._send_queues[queue_idx]
        while self._running:
            handled_completion = self._drain_send_completions()
            try:
                request_id = send_queue.get(timeout=0.1)
            except queue.Empty:
                if handled_completion:
                    continue
                continue
            if request_id is None:
                break
            self._submit_send_task(request_id, queue_idx)

        executor = self._send_executors[queue_idx]
        self._send_executors[queue_idx] = None
        if executor is not None:
            executor.shutdown(wait=True)
        self._drain_send_completions()

    def cleanup(self) -> None:
        self._running = False
        for send_queue in self._send_queues:
            send_queue.put(None)
        receive_thread = self._receive_thread
        if (
            receive_thread is not None
            and threading.current_thread() is not receive_thread
        ):
            # Let the recv loop observe _running=False via RCVTIMEO before
            # closing the socket from another thread. Closing first can trip
            # libzmq assertions during warmup-triggered transfer reconfigure.
            receive_thread.join(timeout=5)
            if receive_thread.is_alive():
                logger.warning(
                    "DiffusionTransferManager cleanup timed out waiting for "
                    "receive thread to exit; forcing socket close"
                )
        if self._control_pull is not None:
            self._control_pull.close(linger=0)
            self._control_pull = None
        self._receive_thread = None
        for thread in self._send_threads:
            thread.join(timeout=5)
        self._send_threads = []
        for executor in self._send_executors:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        self._send_executors = []
        self._send_queues = []
        self._send_queue_count = 0
        self._send_worker_counts = []
        self._drain_send_completions()
        for sock in self._control_push_sockets.values():
            sock.close(linger=0)
        self._control_push_sockets.clear()
        for attachment in self._shared_memory_attachments.values():
            try:
                attachment.shm.close()
            except FileNotFoundError:
                pass
        self._shared_memory_attachments.clear()
        self._engine.deregister_buffer(self._buffer.pool_data_ptr)
        self._engine.deregister_buffer(self._meta_buffer.pool_data_ptr)
        self._buffer.cleanup()
        self._meta_buffer.cleanup()
        logger.info("DiffusionTransferManager cleaned up")
