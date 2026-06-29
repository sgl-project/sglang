"""Drafter-side IPC thread for decoupled speculative decoding.

Owns the verifier->drafter control inbox and the drafter->verifier outgoing
result queue, moving ``DraftMeshMessage`` envelopes over an injected
``BaseDecoupledSpecTransport``. Message validation and rank routing live here;
the wire lives in the transport.

The loop body is factored into ``_step()`` so it can be driven directly (and
deterministically, no background thread) by the fake-transport integration
tests, while production runs ``_run()`` on a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftControlInbox,
    DraftMeshMessage,
    DraftMeshMessageType,
    DraftTailStreamOutputBatch,
    ReadyDraftControls,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    BaseDecoupledSpecTransport,
    TransportClosed,
)

logger = logging.getLogger(__name__)

# Idle floor only: the loop wakes immediately via _wakeup when a result is
# queued; this just bounds the fully-idle sleep before re-polling for controls.
TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S = 0.0005  # 0.5ms


class TokenSyncThread:
    """Drafter-side token sync thread for decoupled speculation IPC.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down.

    Plain class (not a dataclass): a thread controller, not a data container;
    mirrors the sibling ``DraftProxyThread``.
    """

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        drafter_rank: int = 0,
    ) -> None:
        self.transport = transport
        self.drafter_rank = int(drafter_rank)
        self._pending_control_inbox = DraftControlInbox()
        # Protects _pending_control_inbox (loop writes, scheduler reads).
        self._pending_lock = threading.Lock()
        self._outgoing_results: queue.SimpleQueue[DraftTailStreamOutputBatch] = (
            queue.SimpleQueue()
        )
        self._closed = threading.Event()
        # Wakes the idle loop the instant a result is queued (latency-critical send).
        self._wakeup = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-token-sync-thread",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning("Token sync thread did not exit within 1.0s of close()")
        self.transport.close()

    def collect_ready_draft_controls(
        self,
        collector: Callable[[DraftControlInbox], ReadyDraftControls],
    ) -> ReadyDraftControls:
        """Extract ready controls from the live inbox under the inbox lock."""
        with self._pending_lock:
            return collector(self._pending_control_inbox)

    def submit_draft_results(self, result_batch: DraftTailStreamOutputBatch) -> None:
        if not result_batch.outputs:
            return
        # Snapshot the outputs so later caller mutations can't race the queued batch.
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()

    def _step(self) -> bool:
        """Run one drain cycle (outgoing results + incoming controls).

        Returns whether any work was done. Safe to call directly from tests.
        """
        did_work = self._drain_outgoing_results()
        did_work = self._drain_incoming() or did_work
        return did_work

    def _run(self) -> None:
        while not self._closed.is_set():
            try:
                if not self._step():
                    self._wakeup.wait(timeout=TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S)
                    self._wakeup.clear()
            except TransportClosed:
                break
            except Exception:
                # Without this, a routing error from _route_* escapes the loop
                # and silently kills the thread for all requests. Die loudly;
                # phase 5c will quarantine the offending request instead.
                logger.exception("Token sync thread terminating on unexpected error")
                break

    def _drain_incoming(self) -> bool:
        # verifier -> drafter controls
        did_work = False
        while (message := self.transport.try_recv()) is not None:
            did_work = True
            control_batch = self._route_control_message(message)
            if control_batch is None:
                continue
            with self._pending_lock:
                self._pending_control_inbox.add_control_batch_locked(control_batch)
        return did_work

    def _route_control_message(
        self, message: DraftMeshMessage
    ) -> Optional[DraftControlBatch]:
        """Validate + rank-filter one control message.

        Returns the batch for this drafter, or ``None`` if addressed to another
        drafter rank (fan-out filtering, dropped quietly). Raises on a malformed
        envelope; ``_run`` catches that and terminates loudly (5c will quarantine).
        """
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
        return control_batch

    def _drain_outgoing_results(self) -> bool:
        # drafter -> verifier draft tokens
        did_work = False
        while True:
            try:
                result_batch = self._outgoing_results.get_nowait()
            except queue.Empty:
                break
            did_work = True
            self._send_draft_results(result_batch)
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
            self.transport.send(
                dst_verifier_rank,
                DraftMeshMessage.from_tail_stream_output_batch(send_batch),
            )
