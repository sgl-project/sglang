from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftClose,
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftSync,
    DraftTailStreamOutputBatch,
    VerifyCommit,
    iter_control_batch_messages,
)
from sglang.srt.speculative.decoupled_spec_trace import (
    DecoupledSpecTraceEvent,
    NullDecoupledSpecTracer,
    trace_decoupled_spec,
)
from sglang.srt.utils.network import get_zmq_socket

DraftControlMessage = DraftSync | VerifyCommit | DraftClose

TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S = 0.0005  # 0.5ms


@dataclass
class TokenSyncThread:
    """Drafter-side token sync thread for decoupled speculation IPC."""

    context: zmq.Context | None = None
    control_bind_endpoint: str | None = None
    verifier_result_endpoints: list[str] | tuple[str, ...] | None = None
    drafter_rank: int = 0
    _pending_controls: deque[DraftControlMessage] = field(default_factory=deque)
    # verifier -> drafter controls
    control_recv_socket: zmq.Socket | None = None
    # drafter -> verifier draft tokens
    result_send_sockets: dict[int, zmq.Socket] = field(default_factory=dict)
    # protects _pending_controls
    _pending_lock: threading.Lock = field(default_factory=threading.Lock)
    _outgoing_results: queue.SimpleQueue[DraftTailStreamOutputBatch] = field(
        default_factory=queue.SimpleQueue
    )
    _closed: threading.Event = field(default_factory=threading.Event)
    _wakeup: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    tracer: Any = None

    def __post_init__(self) -> None:
        if self.tracer is None:
            self.tracer = NullDecoupledSpecTracer()
        if (
            self.context is None
            or self.control_bind_endpoint is None
            or self.verifier_result_endpoints is None
        ):
            self._thread = threading.Thread(
                target=self._run,
                name="sglang-token-sync-thread",
                daemon=True,
            )
            return
        self.control_recv_socket = get_zmq_socket(
            self.context,
            zmq.PULL,
            self.control_bind_endpoint,
            True,
        )
        self.result_send_sockets = {
            verifier_rank: get_zmq_socket(
                self.context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for verifier_rank, endpoint in enumerate(self.verifier_result_endpoints)
        }
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-token-sync-thread",
            daemon=True,
        )

    def start(self) -> None:
        if self._thread is None:
            return
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.control_recv_socket is not None:
            self.control_recv_socket.close(linger=0)
        for socket in self.result_send_sockets.values():
            socket.close(linger=0)

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_CONTROL_SOCKET)
    def _drain_control_socket(self) -> bool | dict[str, Any]:
        pending_controls_before = self._pending_controls_size()
        num_control_batches = 0
        num_control_messages = 0
        did_work = False
        if self.control_recv_socket is None:
            return did_work

        while True:
            try:
                control_messages = self._recv_control_messages_from_socket()
            except zmq.error.ContextTerminated:
                raise
            except zmq.ZMQError:
                break
            did_work = True
            if control_messages is None:
                continue
            num_control_batches += 1
            num_control_messages += len(control_messages)
            if control_messages:
                with self._pending_lock:
                    self._pending_controls.extend(control_messages)
        if not did_work:
            return False
        return {
            "drafter_rank": int(self.drafter_rank),
            "pending_controls_before": pending_controls_before,
            "pending_controls_after": self._pending_controls_size(),
            "num_control_batches": num_control_batches,
            "num_control_messages": num_control_messages,
        }

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_RECV_CONTROL_BATCH)
    def _recv_control_messages_from_socket(self) -> list[DraftControlMessage] | None:
        message = self.control_recv_socket.recv_pyobj(zmq.NOBLOCK)
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        if (
            message.message_type != DraftMeshMessageType.CONTROL_BATCH
            or message.control_batch is None
        ):
            raise RuntimeError(f"Unexpected draft control message: {message}")
        control_batch = message.control_batch
        if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
            return None
        return iter_control_batch_messages(control_batch)

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_CONTROL_BATCH)
    def drain_control_messages(self) -> list[DraftControlMessage]:
        """Drain all pending control messages while preserving arrival order."""
        drained_messages: list[DraftControlMessage] = []
        with self._pending_lock:
            while self._pending_controls:
                drained_messages.append(self._pending_controls.popleft())
        return drained_messages

    @trace_decoupled_spec(
        DecoupledSpecTraceEvent.TOKEN_SYNC_ENQUEUE_DRAFT_RESULT_BATCH
    )
    def submit_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_OUTGOING_RESULTS)
    def _drain_outgoing_results(self) -> bool | dict[str, Any]:
        queue_size_before = self._outgoing_results_size()
        did_work = False
        num_result_batches = 0
        num_stream_outputs = 0
        while True:
            try:
                result_batch = self._outgoing_results.get_nowait()
            except queue.Empty:
                break
            did_work = True
            num_result_batches += 1
            num_stream_outputs += len(result_batch.outputs)
            self._send_draft_results(result_batch)
        if not did_work:
            return False
        return {
            "drafter_rank": int(self.drafter_rank),
            "queue_size_before": queue_size_before,
            "queue_size_after": self._outgoing_results_size(),
            "num_result_batches": num_result_batches,
            "num_stream_outputs": num_stream_outputs,
        }

    def _send_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return

        batches_by_verifier: dict[int, DraftTailStreamOutputBatch] = {}
        for output in result_batch.outputs:
            dst_verifier_rank = int(output.dst_verifier_rank)
            batches_by_verifier.setdefault(
                dst_verifier_rank,
                DraftTailStreamOutputBatch(),
            ).outputs.append(output)

        for dst_verifier_rank, send_batch in batches_by_verifier.items():
            self._send_result_batch(dst_verifier_rank, send_batch)

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_SEND_RESULT_BATCH)
    def _send_result_batch(
        self,
        dst_verifier_rank: int,
        send_batch: DraftTailStreamOutputBatch,
    ) -> None:
        socket = self.result_send_sockets.get(dst_verifier_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing result socket for dst_verifier_rank={dst_verifier_rank}"
            )
        socket.send_pyobj(DraftMeshMessage.from_tail_stream_output_batch(send_batch))

    def _outgoing_results_size(self) -> int:
        try:
            return int(self._outgoing_results.qsize())
        except (AttributeError, NotImplementedError):
            return -1

    def _pending_controls_size(self) -> int:
        with self._pending_lock:
            return len(self._pending_controls)

    @trace_decoupled_spec(DecoupledSpecTraceEvent.TOKEN_SYNC_IDLE_WAIT)
    def _idle_wait(self) -> dict[str, Any]:
        queue_size_before = self._outgoing_results_size()
        pending_controls_before = self._pending_controls_size()
        wakeup_set_before = self._wakeup.is_set()
        self._wakeup.wait(timeout=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S)
        wakeup_set_after = self._wakeup.is_set()
        queue_size_after = self._outgoing_results_size()
        pending_controls_after = self._pending_controls_size()
        self._wakeup.clear()
        return {
            "drafter_rank": int(self.drafter_rank),
            "wait_timeout_ms": TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S * 1_000,
            "wakeup_set_before_wait": wakeup_set_before,
            "wakeup_set_after_wait": wakeup_set_after,
            "queue_size_before_wait": queue_size_before,
            "queue_size_after_wait": queue_size_after,
            "pending_controls_before_wait": pending_controls_before,
            "pending_controls_after_wait": pending_controls_after,
        }

    def _run(self) -> None:
        while not self._closed.is_set():
            did_work = False
            try:
                did_work = bool(self._drain_outgoing_results()) or did_work
                did_work = bool(self._drain_control_socket()) or did_work
            except zmq.error.ContextTerminated:
                break

            if not did_work:
                self._idle_wait()
