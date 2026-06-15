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

import queue
import threading
from dataclasses import dataclass, field
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

TOKEN_SYNC_THREAD_IDLE_WAIT_TIMEOUT_S = 0.0005  # 0.5ms


@dataclass
class TokenSyncThread:
    """Drafter-side token sync thread for decoupled speculation IPC.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down.
    """

    transport: BaseDecoupledSpecTransport
    drafter_rank: int = 0
    _pending_control_inbox: DraftControlInbox = field(default_factory=DraftControlInbox)
    # protects _pending_control_inbox
    _pending_lock: threading.Lock = field(default_factory=threading.Lock)
    _outgoing_results: "queue.SimpleQueue[DraftTailStreamOutputBatch]" = field(
        default_factory=queue.SimpleQueue
    )
    _closed: threading.Event = field(default_factory=threading.Event)
    _wakeup: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def __post_init__(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-token-sync-thread",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if self._thread is not None and not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        self._wakeup.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.transport.close()

    # ---- scheduler-facing API ------------------------------------------------

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
        queued_batch = DraftTailStreamOutputBatch(outputs=list(result_batch.outputs))
        self._outgoing_results.put(queued_batch)
        self._wakeup.set()

    # ---- loop ----------------------------------------------------------------

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

    # ---- incoming: verifier -> drafter controls ------------------------------

    def _drain_incoming(self) -> bool:
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

    # ---- outgoing: drafter -> verifier draft tokens --------------------------

    def _drain_outgoing_results(self) -> bool:
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
