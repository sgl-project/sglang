"""Verifier-side IPC thread for decoupled speculative decoding.

Control batches from the verifier are applied to the local ``DraftTailBuffer``
and then forwarded to the drafter over an injected ``BaseDecoupledSpecTransport``;
draft tail stream batches received from the drafter are appended to the same
buffer. Message validation and rank routing live here; the wire lives in the
transport.

The loop body is factored into ``_step()`` so it can be driven directly (and
deterministically) by the fake-transport integration tests, while production
runs ``_run()`` on a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import threading

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    BaseDecoupledSpecTransport,
    TransportClosed,
)
from sglang.srt.speculative.draft_tail_buffer import DraftTailBuffer

logger = logging.getLogger(__name__)

# The proxy has no send-side wakeup, so a freshly submitted control waits up to
# this long before the loop services the send queue. This bounded (<=1ms) control
# latency is intentional (matches the PR's poll(1ms)).
DRAFT_PROXY_IDLE_WAIT_TIMEOUT_S = 0.001  # 1ms


class DraftProxyThread:
    """Verifier-side proxy thread for decoupled speculation.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down (and closes
    the ``DraftTailBuffer`` so any waiter is released).
    """

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        verifier_rank: int,
        draft_tail_buffer: DraftTailBuffer,
    ) -> None:
        self.transport = transport
        self.verifier_rank = int(verifier_rank)
        self.draft_tail_buffer = draft_tail_buffer
        self._send_queue: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-draft-proxy",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self.draft_tail_buffer.close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("Draft proxy thread did not exit within 1.0s of close()")
        self.transport.close()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        # Apply to the verifier's own mirror first, then forward to the drafter.
        self.draft_tail_buffer.apply_control_batch(batch)
        self._send_queue.put(batch)

    def _step(self) -> bool:
        """Run one drain cycle (outgoing controls + incoming tail tokens).

        Returns whether any work was done. Safe to call directly from tests.
        """
        did_work = self._drain_send_queue()
        did_work = self._drain_incoming() or did_work
        return did_work

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                if not self._step():
                    self.transport.wait_for_input(DRAFT_PROXY_IDLE_WAIT_TIMEOUT_S)
            except TransportClosed:
                break
            except Exception:
                # Without this, a routing error from _route_* escapes the loop
                # and silently kills the proxy for all requests. Die loudly;
                # phase 5c will quarantine the offending request instead.
                logger.exception("Draft proxy thread terminating on unexpected error")
                break

    def _drain_send_queue(self) -> bool:
        # verifier -> drafter controls
        did_work = False
        while True:
            try:
                batch = self._send_queue.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self.transport.send(
                int(batch.dst_drafter_rank),
                DraftMeshMessage.from_control_batch(batch),
            )
        return did_work

    def _drain_incoming(self) -> bool:
        # drafter -> verifier draft tokens
        did_work = False
        while (message := self.transport.try_recv()) is not None:
            did_work = True
            output_batch = self._route_tail_message(message)
            self.draft_tail_buffer.append_draft_stream_batch(output_batch)
        return did_work

    def _route_tail_message(
        self, message: DraftMeshMessage
    ) -> DraftTailStreamOutputBatch:
        """Validate + verifier-rank-check one tail stream batch.

        Raises on a malformed envelope or a batch for the wrong verifier; ``_run``
        catches that and terminates loudly (5c will quarantine instead).
        """
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")
        if (
            message.message_type != DraftMeshMessageType.TAIL_STREAM_OUTPUT_BATCH
            or message.tail_stream_output_batch is None
        ):
            raise RuntimeError(f"Unexpected draft proxy message: {message}")
        # TODO whether need to change this to any()
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
                f"dst_verifier_ranks={[int(o.dst_verifier_rank) for o in output_batch.outputs]} "
                f"request_ids={[o.request_id for o in output_batch.outputs]}"
            )
        return output_batch
