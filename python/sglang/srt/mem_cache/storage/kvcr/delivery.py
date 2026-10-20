# SPDX-License-Identifier: Apache-2.0
"""Bound public delivery calls without changing the adapter's block mapping."""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from itertools import islice, pairwise

logger = logging.getLogger(__name__)


def validate_destination_spans(blocks, layouts):
    """Reject aliases across the entire call before any destination is writable."""
    components = {
        part.name: part for layout in layouts.values() for part in layout.components
    }
    spans = []
    for refs in blocks.values():
        for ref in refs:
            part = components[ref.label]
            start = part.address + ref.element_index * part.stride
            spans.append((start, start + part.size))
    spans.sort()
    for previous, current in pairwise(spans):
        if current[0] < previous[1]:
            raise ValueError("Overlapping KVCR delivery destination spans")


def block_chunks(blocks):
    """Keep every component of each pool-qualified block in the same chunk."""
    entries = iter(blocks.items())
    while chunk := dict(islice(entries, 512)):
        yield chunk


@dataclass
class Delivery:
    blocks: dict
    parent_operation: int | None = None
    operation: int | None = None
    prepared_epoch: float | None = None
    finished_epoch: float | None = None
    prepared: float | None = None
    submitted: float | None = None
    finished: float | None = None
    result: object = None


def run_window(store, deliveries, request_id, seen_handles):
    """Return raw completions only after ALL writes in this hint are terminal.

    The caller MUST hold the active hint scope. Submission itself may enqueue
    before raising, and even bookkeeping can lose a returned handle. There is
    no safe Python unwind in either case. Do not attempt an unbounded drain of
    known handles: it cannot establish quiescence of an unknown operation.
    """
    try:
        pending, completed = deque(), []
        remaining = iter(deliveries)
        exhausted = False
        while pending or not exhausted:
            while len(pending) < 4 and not exhausted:
                delivery = next(remaining, None)
                if delivery is None:
                    exhausted = True
                    break
                delivery.prepared_epoch = (
                    time.time() if store._diagnostic_telemetry else None
                )
                delivery.prepared = (
                    time.perf_counter() if store._diagnostic_telemetry else None
                )
                handle = store._runner.deliver(delivery.blocks, request_id=request_id)
                if type(handle) is not int or handle <= 0 or handle in seen_handles:
                    raise RuntimeError("Invalid or duplicate KVCR delivery handle")
                seen_handles.add(handle)
                delivery.operation = handle
                delivery.submitted = (
                    time.perf_counter() if store._diagnostic_telemetry else None
                )
                pending.append(delivery)
            if pending:
                delivery = pending[0]
                # The existing shared completion mailbox retains reverse-order
                # completions and completions belonging to concurrent callers.
                delivery.result = store._wait(delivery.operation)
                delivery.finished_epoch = (
                    time.time() if store._diagnostic_telemetry else None
                )
                delivery.finished = (
                    time.perf_counter() if store._diagnostic_telemetry else None
                )
                completed.append(delivery)
                pending.popleft()
        return completed
    except BaseException:  # noqa: BLE001 - even cancellation must not release live DMA slots
        # This must run INSIDE the hint. A logging failure must also terminate
        # without discarding hints or returning HiCache's still-exposed slots.
        try:
            logger.critical(
                "Unsafe KVCR delivery pipeline; terminating worker", exc_info=True
            )
        finally:
            os._exit(1)
        raise RuntimeError("KVCR fatal exit unexpectedly returned")


def requested_results(deliveries):
    """Interpret only this handle's requested identities, after phase completion."""
    return {
        key: delivery.result[key]
        for delivery in deliveries
        for key in delivery.blocks
        if key in delivery.result
    }
