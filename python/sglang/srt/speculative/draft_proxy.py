from __future__ import annotations

import queue
import threading
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
)
from sglang.srt.speculative.decoupled_spec_trace import (
    DecoupledSpecTraceEvent,
    NullDecoupledSpecTracer,
    trace_decoupled_spec,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer
from sglang.srt.utils.network import get_zmq_socket


class DraftProxyThread:
    """
    Verifier-side proxy thread for decoupled speculation.

    Control batches from the verifier are first applied to the local
    DraftTailBuffer, then forwarded to the drafter. Draft tail stream batches
    from the drafter are appended to the same buffer.
    """

    def __init__(
        self,
        *,
        context: zmq.Context,
        verifier_rank: int,
        result_bind_endpoint: str,
        drafter_control_endpoints: list[str] | tuple[str, ...],
        draft_tail_buffer: DraftTailBuffer,
        tracer: Any = None,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer or NullDecoupledSpecTracer()
        # verifier -> drafter send control messages
        self.control_send_sockets: dict[int, zmq.Socket] = {
            drafter_rank: get_zmq_socket(
                context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for drafter_rank, endpoint in enumerate(drafter_control_endpoints)
        }
        self.result_recv_socket = get_zmq_socket(
            context,
            zmq.PULL,
            result_bind_endpoint,
            True,
        )
        self._send_queue: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-proxy",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.draft_tail_buffer.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        for socket in self.control_send_sockets.values():
            socket.close(linger=0)
        self.result_recv_socket.close(linger=0)

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        self._apply_control_batch(batch)
        self._send_queue.put(batch)

    @trace_decoupled_spec(
        DecoupledSpecTraceEvent.DRAFT_PROXY_APPLY_CONTROL_BATCH,
        inject_trace_enabled="collect_trace_stats",
    )
    def _apply_control_batch(
        self,
        batch: DraftControlBatch,
        *,
        collect_trace_stats: bool = False,
    ) -> dict[str, Any] | None:
        return self.draft_tail_buffer.apply_control_batch(
            batch,
            collect_stats=collect_trace_stats,
        )

    def _recv_tail_stream_output_batch(self) -> None:
        output_batch = self._recv_tail_stream_output_batch_from_socket()
        self._append_tail_stream_output_batch(output_batch)

    @trace_decoupled_spec(DecoupledSpecTraceEvent.DRAFT_PROXY_RECV_TAIL_STREAM_BATCH)
    def _recv_tail_stream_output_batch_from_socket(
        self,
    ) -> DraftTailStreamOutputBatch:
        message = self.result_recv_socket.recv_pyobj()
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")
        if (
            message.message_type != DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
            or message.tail_stream_output_batch is None
        ):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")

        output_batch = message.tail_stream_output_batch
        mismatched_outputs = [
            output
            for output in output_batch.outputs
            if int(output.dst_verifier_rank) != self.verifier_rank
        ]
        if mismatched_outputs:
            raise RuntimeError(
                "Draft proxy received a tail stream batch for the wrong verifier: "
                f"verifier_rank={self.verifier_rank} "
                f"dst_verifier_ranks={[int(output.dst_verifier_rank) for output in output_batch.outputs]} "
                f"request_ids={[output.request_id for output in output_batch.outputs]}"
            )
        return output_batch

    @trace_decoupled_spec(
        DecoupledSpecTraceEvent.DRAFT_PROXY_APPEND_TAIL_STREAM_BATCH,
        inject_trace_enabled="collect_trace_stats",
    )
    def _append_tail_stream_output_batch(
        self,
        output_batch: DraftTailStreamOutputBatch,
        *,
        collect_trace_stats: bool = False,
    ) -> dict[str, Any] | None:
        return self.draft_tail_buffer.append_draft_stream_batch(
            output_batch,
            collect_stats=collect_trace_stats,
        )

    @trace_decoupled_spec(DecoupledSpecTraceEvent.DRAFT_PROXY_SEND_CONTROL_BATCH)
    def _send_control_batch(self, batch: DraftControlBatch) -> None:
        dst_drafter_rank = int(batch.dst_drafter_rank)
        socket = self.control_send_sockets.get(dst_drafter_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing control socket for dst_drafter_rank={dst_drafter_rank}"
            )
        socket.send_pyobj(DraftMeshMessage.from_control_batch(batch))

    def _run(self) -> None:
        while not self._closed.is_set():
            while True:
                try:
                    batch = self._send_queue.get_nowait()
                except queue.Empty:
                    break
                self._send_control_batch(batch)

            try:
                if self.result_recv_socket.poll(timeout=1):
                    self._recv_tail_stream_output_batch()
            except zmq.error.ContextTerminated:
                break
