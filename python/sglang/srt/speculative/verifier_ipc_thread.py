"""Verifier-side IPC thread (the recv daemon) for decoupled enumeration spec.

Control batches from the verifier are forwarded to the drafter over an injected
``BaseDecoupledSpecTransport``; enumeration buffer blocks received from the
drafter are landed into the verifier's GPU ``DecoupledEnumBuffer`` (verifier
routing + staleness live in ``plan_landing`` / ``DecoupledEnumBuffer.land``,
keyed by the ``DecoupledSlotTable`` rid -> seat map the scheduler maintains).
Envelope validation lives here; the wire lives in the transport.

The loop body is factored into ``_step()`` so it can be driven directly (and
deterministically) by the fake-transport integration tests, while production
runs ``_run()`` on a daemon thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING

from sglang.srt.speculative.decoupled_spec_io import (
    DraftControlBatch,
    DraftEnumerationBufferBatch,
    DraftMeshMessage,
    DraftMeshMessageType,
)
from sglang.srt.speculative.decoupled_spec_transport import (
    BaseDecoupledSpecTransport,
    TransportClosed,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.decoupled_enum_buffer import DecoupledEnumBuffer
    from sglang.srt.speculative.decoupled_slot_table import DecoupledSlotTable

logger = logging.getLogger(__name__)

# The verifier IPC thread has no send-side wakeup, so a freshly submitted control
# waits up to this long before the loop services the send queue. This bounded
# (<=1ms) control latency is intentional (matches the PR's poll(1ms)).
VERIFIER_IPC_IDLE_WAIT_TIMEOUT_S = 0.001  # 1ms


class VerifierIpcThread:
    """Verifier-side IPC thread (recv daemon) for decoupled enumeration spec.

    The injected ``transport`` must be started before the loop runs; ``start()``
    starts it (and the daemon loop) and ``close()`` tears both down.
    """

    def __init__(
        self,
        *,
        transport: BaseDecoupledSpecTransport,
        enum_buffer: DecoupledEnumBuffer,
        slot_table: DecoupledSlotTable,
    ) -> None:
        self.transport = transport
        # The GPU landing buffer + its rid -> seat map. land() holds verifier_rank
        # and rejects a block routed to another verifier, so this thread does no
        # rank check of its own -- only envelope validation.
        self.enum_buffer = enum_buffer
        self.slot_table = slot_table
        self._send_queue: queue.SimpleQueue[DraftControlBatch] = queue.SimpleQueue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-verifier-ipc",
            daemon=True,
        )

    def start(self) -> None:
        self.transport.start()
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._closed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
            if self._thread.is_alive():
                logger.warning(
                    "Verifier IPC thread did not exit within 1.0s of close()"
                )
        self.transport.close()

    def submit_control_batch(self, batch: DraftControlBatch) -> None:
        # Verifier -> drafter only. The verifier keeps no control mirror in the
        # enumeration design: request lifecycle lives in the scheduler's slot
        # table (assign / remove) and committed length, not on this thread.
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
                    self.transport.wait_for_input(VERIFIER_IPC_IDLE_WAIT_TIMEOUT_S)
            except TransportClosed:
                break
            except Exception:
                # Without this, a routing error from _route_* escapes the loop
                # and silently kills the thread for all requests. Die loudly;
                # phase 5c will quarantine the offending request instead.
                logger.exception("Verifier IPC thread terminating on unexpected error")
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
        # drafter -> verifier enumeration buffer blocks
        did_work = False
        while (message := self.transport.try_recv()) is not None:
            did_work = True
            block = self._route_enumeration_message(message)
            # Verifier routing (wrong-verifier reject), validate(), and the
            # rid -> seat lookup all live in land() -> plan_landing; the SYNC
            # scatter runs on the current stream (6.3 moves it to a copy stream).
            self.enum_buffer.land(block, self.slot_table)
        return did_work

    def _route_enumeration_message(
        self, message: DraftMeshMessage
    ) -> DraftEnumerationBufferBatch:
        """Extract one enumeration buffer block from its envelope.

        Raises on a malformed envelope; ``_run`` catches that and terminates
        loudly (5c will quarantine instead). Semantic validation (verifier
        routing, duplicate rids, K/F dims) is deferred to ``land``.
        """
        if not isinstance(message, DraftMeshMessage):
            raise RuntimeError(
                f"Unexpected message on the verifier IPC thread: {message}"
            )
        if (
            message.message_type != DraftMeshMessageType.ENUMERATION_BUFFER_BATCH
            or message.enumeration_buffer_batch is None
        ):
            raise RuntimeError(
                f"Unexpected message on the verifier IPC thread: {message}"
            )
        return message.enumeration_buffer_batch
