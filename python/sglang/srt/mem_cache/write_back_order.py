"""Device-ordered HiCache write-back (SGLANG_HICACHE_WRITE_BACK_DEVICE_ORDERED, default off).

A write-back eviction copies its victims device-to-host and then demotes them, which returns
their device rows (KV pages, Mamba and indexer slots) to the allocators. The stock
``UnifiedRadixCache.writing_check(write_back=True)`` waits on the host for every copy's
finish event before that demote. The copy is fenced behind the in-flight forward
(``HiCacheController.start_writing``), so batch formation stalls until that forward ends and
then runs on an idle GPU.

Here the demote follows the submit. The copy's finish event is waited on by every stream that
can write a returned device row or read the host rows the copy fills, before the rows can be
handed out again:

* the caller's current stream (the scheduler's schedule stream during batch formation): the
  allocation, state-slot initialisation and copies that follow the eviction, and through
  ``_join_schedule_stream`` the next forward;
* the forward stream (``write_fence_stream``), in case anything launches there unjoined;
* the HiCache load stream: a load-back may write a returned row, or read the host copy of a
  node demoted here.

The one host-side reader of host rows on the kernel I/O path, the Mamba side-state leg of a
load (``MambaPoolHost._transfer_slot_states``), calls ``wait_landed`` first.

The waits are enqueued before the demote returns any row, so every later write to those rows
and every later read of the new host copy runs after the copy on the device. Rows go back to
the allocator's free list on the host earlier than before; nothing on the device can use them
earlier. The PD KV sender and the forward in flight read only locked rows, which eviction never
picks. Not used when a storage backend, buffer mode or an external linker is attached: those
read host copies from CPU threads.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Any, Deque, Iterable, Optional

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get(
    "SGLANG_HICACHE_WRITE_BACK_DEVICE_ORDERED", "0"
).strip().lower() in ("1", "true", "yes", "on")

# Durations of copies acknowledged on submit wait here until their events complete.
_MAX_PENDING_TIMINGS = 4096

STATS = {"ops": 0, "acks": 0, "nodes": 0, "bytes": 0, "host_waits": 0, "pinned_index_moves": 0,
         "logged": False}
# The last copy acknowledged on submit, per cache controller (copies of one controller finish in
# submit order on its device-to-host stream).
_PENDING: dict = {}


def _distinct(streams: Iterable[Optional[Any]]) -> list:
    out, seen = [], set()
    for stream in streams:
        if stream is None:
            continue
        handle = getattr(stream, "cuda_stream", id(stream))
        if handle in seen:
            continue
        seen.add(handle)
        out.append(stream)
    return out


def consumer_streams(cache_controller) -> list:
    """The streams that may write a row a write-back copy reads, or read the host rows it
    writes, after the copy's rows are returned to the allocators."""
    engine = cache_controller.l2_transfer_engine
    return _distinct(
        (
            torch.cuda.current_stream(),
            cache_controller.write_fence_stream,
            engine.host_to_device_stream,
        )
    )


def order_after(cache_controller, finish_event) -> None:
    """Order every consumer stream after one write-back copy's finish event."""
    for stream in consumer_streams(cache_controller):
        stream.wait_event(finish_event)


def note_pending(cache_controller, finish_event) -> None:
    _PENDING[id(cache_controller)] = finish_event


def wait_landed() -> None:
    """Wait on the host for every copy acknowledged on submit: a CPU read of host rows is not
    ordered by the stream waits."""
    for key, event in list(_PENDING.items()):
        if not event.query():
            STATS["host_waits"] += 1
            event.synchronize()
        _PENDING.pop(key, None)


def pin_indices(host_indices: torch.Tensor) -> torch.Tensor:
    """Pageable CPU host indices staged through pinned memory, so their copy to the device is
    only enqueued (a pageable one waits for the stream, which is fenced on the forward)."""
    if host_indices.device.type != "cpu" or host_indices.is_pinned():
        return host_indices
    STATS["pinned_index_moves"] += 1
    if STATS["pinned_index_moves"] == 1:
        logger.info(
            "SGLANG_HICACHE_WRITE_BACK_DEVICE_ORDERED: write-back host indices staged through "
            "pinned memory (first: %d indices)",
            host_indices.numel(),
        )
    return host_indices.pin_memory()


def note_op(num_acks: int, num_nodes: int, num_bytes: int) -> None:
    STATS["ops"] += 1
    STATS["acks"] += num_acks
    STATS["nodes"] += num_nodes
    STATS["bytes"] += num_bytes
    if not STATS["logged"]:
        STATS["logged"] = True
        logger.info(
            "SGLANG_HICACHE_WRITE_BACK_DEVICE_ORDERED: write-back acknowledged on submit; "
            "the schedule, forward and load streams wait on the copy instead of the host "
            "(first op: %d acks, %d nodes, %d bytes)",
            num_acks,
            num_nodes,
            num_bytes,
        )


class PendingTimings:
    """Backup durations of copies acknowledged before they finished, observed once their
    events complete (``elapsed_time`` needs both events done). Events on one stream complete
    in submit order, so the queue drains from its head."""

    def __init__(self) -> None:
        self._acks: Deque = deque()

    def add(self, ack) -> None:
        if len(self._acks) >= _MAX_PENDING_TIMINGS:
            self._acks.popleft()
        self._acks.append(ack)

    def drain(self, observe) -> None:
        while self._acks and self._acks[0].finish_event.query():
            ack = self._acks.popleft()
            observe(ack.start_event.elapsed_time(ack.finish_event) / 1000.0)
