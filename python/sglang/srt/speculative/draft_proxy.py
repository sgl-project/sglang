from __future__ import annotations

import queue
import threading
import time
from typing import Any

import zmq

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshIpcConfig,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
    iter_control_batch_messages,
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
        ipc_config: DraftMeshIpcConfig,
        verifier_rank: int,
        draft_tail_buffer: DraftTailBuffer,
        tracer: Any = None,
    ) -> None:
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self.tracer = tracer
        # verifier -> drafter send control messages
        self.control_send_sockets: dict[int, zmq.Socket] = {
            drafter_rank: get_zmq_socket(
                context,
                zmq.PUSH,
                endpoint,
                False,
            )
            for drafter_rank, endpoint in sorted(ipc_config.control_endpoints.items())
        }
        self.result_recv_socket = get_zmq_socket(
            context,
            zmq.PULL,
            ipc_config.get_result_endpoint(self.verifier_rank),
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
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        apply_start_ns = time.perf_counter_ns() if trace_enabled else 0
        apply_stats = self.draft_tail_buffer.apply_control_batch(
            batch,
            collect_stats=trace_enabled,
        )
        if trace_enabled:
            self._record_control_batch(
                "apply_control_batch",
                batch,
                duration_ms=(time.perf_counter_ns() - apply_start_ns) / 1_000_000,
                apply_stats=apply_stats,
            )
        self._send_queue.put(batch)

    def _recv_tail_stream_output_batch(self) -> None:
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        recv_start_ns = time.perf_counter_ns() if trace_enabled else 0
        message = self.result_recv_socket.recv_pyobj()
        recv_duration_ms = (
            (time.perf_counter_ns() - recv_start_ns) / 1_000_000
            if trace_enabled
            else 0
        )
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
        if trace_enabled:
            self._record_tail_stream_event(
                "recv_tail_stream_batch",
                output_batch,
                duration_ms=recv_duration_ms,
            )
        append_start_ns = time.perf_counter_ns() if trace_enabled else 0
        append_stats = self.draft_tail_buffer.append_draft_stream_batch(
            output_batch,
            collect_stats=trace_enabled,
        )
        if trace_enabled:
            self._record_tail_stream_event(
                "append_tail_stream_batch",
                output_batch,
                duration_ms=(time.perf_counter_ns() - append_start_ns) / 1_000_000,
                append_stats=append_stats,
            )

    def _send_control_batch(self, batch: DraftControlBatch) -> None:
        dst_drafter_rank = int(batch.dst_drafter_rank)
        socket = self.control_send_sockets.get(dst_drafter_rank)
        if socket is None:
            raise RuntimeError(
                f"Missing control socket for dst_drafter_rank={dst_drafter_rank}"
            )
        trace_enabled = getattr(getattr(self, "tracer", None), "enabled", False)
        start_ns = time.perf_counter_ns() if trace_enabled else 0
        socket.send_pyobj(DraftMeshMessage.from_control_batch(batch))
        if trace_enabled:
            self._record_control_batch(
                "send_control_batch",
                batch,
                duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
            )

    def _record_control_batch(
        self,
        op: str,
        batch: DraftControlBatch,
        *,
        duration_ms: float,
        apply_stats: dict[str, Any] | None = None,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        messages = iter_control_batch_messages(batch)
        fields: dict[str, Any] = {
            "duration_ms": duration_ms,
            "verifier_rank": self.verifier_rank,
            "batch_size": len(messages),
            "rids": [message.request_id for message in messages],
            "num_sync": len(batch.sync_messages),
            "num_commit": len(batch.verify_commit_messages),
            "num_close": len(batch.close_messages),
        }
        if op != "apply_control_batch":
            fields["dst_drafter_rank"] = int(batch.dst_drafter_rank)
        else:
            fields.update(
                apply_stats
                or {
                    "commit_rids": [],
                    "pre_committed_lens_by_req": [],
                    "accepted_tail_lens_by_req": [],
                    "raw_tail_lens_before_by_req": [],
                    "bonus_token_ids_by_req": [],
                    "buffer_candidate_token_ids_by_req": [],
                    "bonus_match_by_req": [],
                    "preserved_suffix_lens_by_req": [],
                    "tail_lens_after_by_req": [],
                    "committed_lens_after_by_req": [],
                }
            )
        self.tracer.record("draft_proxy", op, **fields)

    def _record_tail_stream_event(
        self,
        op: str,
        output_batch: DraftTailStreamOutputBatch,
        *,
        duration_ms: float,
        append_stats: dict[str, Any] | None = None,
    ) -> None:
        if not getattr(getattr(self, "tracer", None), "enabled", False):
            return
        request_ids, draft_token_lens_by_req = self._count_outputs_by_request(
            output_batch
        )
        fields: dict[str, Any] = {
            "duration_ms": duration_ms,
            "verifier_rank": self.verifier_rank,
            "batch_size": len(request_ids),
            "rids": request_ids,
            "num_stream_outputs": len(output_batch.outputs),
            "draft_token_lens_by_req": draft_token_lens_by_req,
        }
        if op == "append_tail_stream_batch":
            fields.update(
                append_stats
                or {
                    "num_appended_outputs": 0,
                    "num_duplicate_outputs": 0,
                    "num_stale_base_outputs": 0,
                    "num_already_committed_outputs": 0,
                    "num_stale_gap_outputs": 0,
                    "num_unknown_request_outputs": 0,
                    "appended_token_lens_by_req": [0] * len(request_ids),
                    "tail_lens_after_by_req": [0] * len(request_ids),
                    "consumable_tail_lens_after_by_req": [0] * len(request_ids),
                    "committed_lens_after_by_req": [
                        self.draft_tail_buffer.get_committed_len(request_id) or 0
                        for request_id in request_ids
                    ],
                }
            )
        self.tracer.record("draft_proxy", op, **fields)

    def _count_outputs_by_request(
        self, output_batch: DraftTailStreamOutputBatch
    ) -> tuple[list[str], list[int]]:
        counts_by_request: dict[str, int] = {}
        for output in output_batch.outputs:
            counts_by_request[output.request_id] = (
                counts_by_request.get(output.request_id, 0) + 1
            )
        request_ids = list(counts_by_request.keys())
        return request_ids, [
            counts_by_request[request_id] for request_id in request_ids
        ]

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
