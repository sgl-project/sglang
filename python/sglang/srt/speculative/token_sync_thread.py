from __future__ import annotations

import queue
import threading
import time
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

    def _drain_control_socket(self) -> bool:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        drain_start_ns = time.perf_counter_ns() if trace_enabled else 0
        pending_controls_before = self._pending_controls_size() if trace_enabled else 0
        num_control_batches = 0
        num_control_messages = 0
        did_work = False
        if self.control_recv_socket is None:
            return did_work

        while True:
            try:
                start_ns = time.perf_counter_ns() if trace_enabled else 0
                message = self.control_recv_socket.recv_pyobj(zmq.NOBLOCK)
                recv_duration_ms = (
                    (time.perf_counter_ns() - start_ns) / 1_000_000
                    if trace_enabled
                    else 0
                )
            except zmq.error.ContextTerminated:
                raise
            except zmq.ZMQError:
                break
            did_work = True
            if not isinstance(message, DraftMeshMessage):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            if (
                message.message_type != DraftMeshMessageType.CONTROL_BATCH
                or message.control_batch is None
            ):
                raise RuntimeError(f"Unexpected draft control message: {message}")
            control_batch = message.control_batch
            if int(control_batch.dst_drafter_rank) != int(self.drafter_rank):
                continue
            control_messages = iter_control_batch_messages(control_batch)
            num_control_batches += 1
            num_control_messages += len(control_messages)
            if control_messages:
                with self._pending_lock:
                    self._pending_controls.extend(control_messages)
            if trace_enabled:
                self._record_control_batch(
                    "recv_control_batch",
                    control_batch,
                    duration_ms=recv_duration_ms,
                )
        if trace_enabled and did_work:
            self.tracer.record(
                "token_sync_thread",
                "drain_control_socket",
                duration_ms=(time.perf_counter_ns() - drain_start_ns) / 1_000_000,
                drafter_rank=int(self.drafter_rank),
                pending_controls_before=pending_controls_before,
                pending_controls_after=self._pending_controls_size(),
                num_control_batches=num_control_batches,
                num_control_messages=num_control_messages,
            )
        return did_work

    def drain_sync_messages(self) -> list[DraftSync]:
        return [
            message
            for message in self._drain_pending_controls(
                DraftSync,
                "drain_sync_batch",
            )
            if isinstance(message, DraftSync)
        ]

    def drain_post_result_messages(self) -> list[VerifyCommit | DraftClose]:
        return [
            message
            for message in self._drain_pending_controls(
                (VerifyCommit, DraftClose),
                "drain_post_result_batch",
            )
            if isinstance(message, (VerifyCommit, DraftClose))
        ]

    def _drain_pending_controls(
        self,
        control_types: Any,
        op: str,
    ) -> list[DraftControlMessage]:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        drained_messages: list[DraftControlMessage] = []
        remaining_controls: deque[DraftControlMessage] = deque()
        with self._pending_lock:
            while self._pending_controls:
                message = self._pending_controls.popleft()
                if isinstance(message, control_types):
                    drained_messages.append(message)
                else:
                    remaining_controls.append(message)
            self._pending_controls = remaining_controls
        if trace_enabled and drained_messages:
            self._record_messages(
                op,
                drained_messages,
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            )
        return drained_messages

    def submit_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()
        if trace_enabled:
            self._record_draft_results(
                "enqueue_draft_result_batch",
                queued_batch,
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            )

    def _drain_outgoing_results(self) -> bool:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        drain_start_ns = time.perf_counter_ns() if trace_enabled else 0
        queue_size_before = self._outgoing_results_size() if trace_enabled else 0
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
        if trace_enabled and did_work:
            self.tracer.record(
                "token_sync_thread",
                "drain_outgoing_results",
                duration_ms=(time.perf_counter_ns() - drain_start_ns) / 1_000_000,
                drafter_rank=int(self.drafter_rank),
                queue_size_before=queue_size_before,
                queue_size_after=self._outgoing_results_size(),
                num_result_batches=num_result_batches,
                num_stream_outputs=num_stream_outputs,
            )
        return did_work

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
            socket = self.result_send_sockets.get(dst_verifier_rank)
            if socket is None:
                raise RuntimeError(
                    f"Missing result socket for dst_verifier_rank={dst_verifier_rank}"
                )

            trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
            start_ns = time.perf_counter_ns() if trace_enabled else 0
            socket.send_pyobj(
                DraftMeshMessage.from_tail_stream_output_batch(send_batch)
            )
            if trace_enabled:
                self._record_draft_results(
                    "send_result_batch",
                    send_batch,
                    dst_verifier_rank=dst_verifier_rank,
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
                )

    def _record_control_batch(
        self,
        op: str,
        batch: DraftControlBatch,
        *,
        duration_ms: float,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        messages = iter_control_batch_messages(batch)
        self._record_messages(op, messages, duration_ms=duration_ms)

    def _record_messages(
        self,
        op: str,
        messages: list[DraftControlMessage],
        *,
        duration_ms: float,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        fields: dict[str, Any] = {
            "duration_ms": duration_ms,
            "drafter_rank": int(self.drafter_rank),
            "batch_size": len(messages),
            "rids": [message.request_id for message in messages],
        }
        if op in ("recv_control_batch", "drain_sync_batch"):
            fields["num_sync"] = sum(
                isinstance(message, DraftSync) for message in messages
            )
        if op in ("recv_control_batch", "drain_post_result_batch"):
            fields["num_commit"] = sum(
                isinstance(message, VerifyCommit) for message in messages
            )
            fields["num_close"] = sum(
                isinstance(message, DraftClose) for message in messages
            )
        self.tracer.record("token_sync_thread", op, **fields)

    def _record_draft_results(
        self,
        op: str,
        result_batch: DraftTailStreamOutputBatch,
        *,
        duration_ms: float,
        dst_verifier_rank: int | None = None,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        counts_by_request: dict[str, int] = {}
        for output in result_batch.outputs:
            counts_by_request[output.request_id] = (
                counts_by_request.get(output.request_id, 0) + 1
            )
        request_ids = list(counts_by_request.keys())
        num_stream_outputs = len(result_batch.outputs)
        fields: dict[str, Any] = {
            "duration_ms": duration_ms,
            "drafter_rank": int(self.drafter_rank),
            "batch_size": len(request_ids),
            "rids": request_ids,
            "num_stream_outputs": num_stream_outputs,
            "emitted_token_lens_by_req": [
                counts_by_request[request_id] for request_id in request_ids
            ],
        }
        if op == "send_result_batch":
            fields["dst_verifier_rank"] = int(dst_verifier_rank)
        self.tracer.record("token_sync_thread", op, **fields)

    def _outgoing_results_size(self) -> int:
        try:
            return int(self._outgoing_results.qsize())
        except (AttributeError, NotImplementedError):
            return -1

    def _pending_controls_size(self) -> int:
        with self._pending_lock:
            return len(self._pending_controls)

    def _idle_wait(self) -> None:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        if not trace_enabled:
            self._wakeup.wait(timeout=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S)
            self._wakeup.clear()
            return

        queue_size_before = self._outgoing_results_size()
        pending_controls_before = self._pending_controls_size()
        wakeup_set_before = self._wakeup.is_set()
        start_ns = time.perf_counter_ns()
        self._wakeup.wait(timeout=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S)
        duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        wakeup_set_after = self._wakeup.is_set()
        queue_size_after = self._outgoing_results_size()
        pending_controls_after = self._pending_controls_size()
        self._wakeup.clear()
        self.tracer.record(
            "token_sync_thread",
            "idle_wait",
            duration_ms=duration_ms,
            drafter_rank=int(self.drafter_rank),
            wait_timeout_ms=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S * 1_000,
            wakeup_set_before_wait=wakeup_set_before,
            wakeup_set_after_wait=wakeup_set_after,
            queue_size_before_wait=queue_size_before,
            queue_size_after_wait=queue_size_after,
            pending_controls_before_wait=pending_controls_before,
            pending_controls_after_wait=pending_controls_after,
        )

    def _run(self) -> None:
        while not self._closed.is_set():
            did_work = False
            try:
                did_work = self._drain_outgoing_results() or did_work
                did_work = self._drain_control_socket() or did_work
            except zmq.error.ContextTerminated:
                break

            if not did_work:
                self._idle_wait()
