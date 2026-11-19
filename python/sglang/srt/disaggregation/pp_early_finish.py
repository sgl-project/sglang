"""Early finish for pipeline-parallel PD prefill, on PP rank 0.

A PP PD prefill request finishes when the ring release consensus reaches PP
rank 0, one or two loop iterations after the last stage's KV sender succeeded.
Every stage's sender status already reaches PP rank 0 through its
PPConsensusStore, so a completion thread streams the finish as soon as every
stage reports Success. The KV release still follows the ring consensus on every
stage; the ring release then skips the output already streamed.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import TYPE_CHECKING, Collection, Iterable, List, Optional, Set

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.mode import FAKE_BOOTSTRAP_HOST
from sglang.srt.disaggregation.pp_consensus_store import PPConsensusStore
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.runtime_context import get_disagg, get_parallel

if TYPE_CHECKING:
    from sglang.srt.disaggregation.common.conn import CommonKVManager
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.scheduler_components.request_list import IndexedReqList

logger = logging.getLogger(__name__)

# Retry period of a room whose local staging chunks still hold its Success.
_RECHECK_S = 1e-3
# Period of the full scan that backs up the store's wake-ups.
_SCAN_S = 0.05
_MAX_EMIT_FAILURES = 3


class PPPrefillEarlyFinish:
    """Streams a request's finish once every PP stage's KV sender succeeded.

    Every structure is guarded by ``lock``, held across a whole early emit, so a
    request is streamed once: early, or by the ring release if it claims the
    request first (``released``).
    """

    def __init__(
        self,
        *,
        kv_manager: CommonKVManager,
        transfer_queue: IndexedReqList,
        output_streamer: SchedulerOutputStreamer,
    ) -> None:
        # PP rank 0's view of every stage's sender status.
        self.store: PPConsensusStore = kv_manager.request_status
        self.kv_manager = kv_manager
        self.transfer_queue = transfer_queue
        self.output_streamer = output_streamer
        self.lock = threading.RLock()
        self._emitted: Set[str] = set()
        self._ring_owned: Set[str] = set()
        self._recheck_rooms: Set[int] = set()
        self._seen_version: Optional[int] = None

    def emit(self, rooms: Optional[Collection[int]] = None) -> int:
        """Stream the finish of every queued request all stages completed.

        ``rooms`` limits the scan to those bootstrap rooms; None scans the whole
        queue when the store changed since the last such scan. The request stays
        queued until the ring release frees its KV.
        """
        with self.lock:
            ready = self._ready(self.transfer_queue.snapshot(), rooms)
            if not ready:
                return 0
            for req in ready:
                # The finish the ring release gives a transferred request.
                req.finish_outside_step(FINISH_LENGTH(length=0))
                req.time_stats.set_prefill_kv_transfer_finish_time()
                req.time_stats.set_completion_time()
            self.output_streamer.stream_output(
                ready, any(req.return_logprob for req in ready), None
            )
            self._emitted.update(req.rid for req in ready)
            return len(ready)

    def released(self, req: Req) -> bool:
        """The ring release concludes ``req``; True if it was streamed early.

        False means the ring release streams it, and no early emit will."""
        with self.lock:
            if req.rid in self._emitted:
                self._emitted.discard(req.rid)
                return True
            self._ring_owned.add(req.rid)
            return False

    def forget(self, rids: Iterable[str]) -> None:
        """The ring release is done with ``rids``."""
        with self.lock:
            self._ring_owned.difference_update(rids)

    def take_recheck_rooms(self) -> Set[int]:
        with self.lock:
            rooms, self._recheck_rooms = self._recheck_rooms, set()
            return rooms

    def has_recheck_rooms(self) -> bool:
        return bool(self._recheck_rooms)

    def _eligible(self, req: Req) -> bool:
        return not (
            req.rid in self._emitted
            or req.rid in self._ring_owned
            or req.pending_bootstrap
            or req.finished_output
            or isinstance(req.finished_reason, FINISH_ABORT)
            or req.bootstrap_host == FAKE_BOOTSTRAP_HOST
            # Routed experts are collected at the ring release.
            or req.return_routed_experts
        )

    def _ready(self, queued: List[Req], rooms: Optional[Collection[int]]) -> List[Req]:
        if rooms is None:
            version = self.store.version
            if version == self._seen_version:
                return []
            self._seen_version = version
            live = {req.rid for req in queued}
            self._emitted &= live
            self._ring_owned &= live
        out = []
        for req in queued:
            room = req.bootstrap_room
            if rooms is not None and room not in rooms:
                continue
            if not self._eligible(req):
                continue
            if any(status != KVPoll.Success for status in self.store.collect(room)):
                continue
            if self.kv_manager.staging_holds_success(room):
                self._recheck_rooms.add(room)
                continue
            self._recheck_rooms.discard(room)
            out.append(req)
        return out


class EarlyFinishThread:
    """PP rank 0's completion thread: streams early finishes off the main loop.

    The store calls ``_on_put`` on the writer's thread for every value it
    records; a Success wakes this thread, which emits that room at once. Rooms
    held by staging chunks are retried every ``_RECHECK_S``, and a full scan
    runs every ``_SCAN_S``. After ``_MAX_EMIT_FAILURES`` failed emits in a row
    the thread stops and the ring release streams every remaining finish.
    """

    def __init__(self, tracker: PPPrefillEarlyFinish) -> None:
        self.tracker = tracker
        self._dirty: collections.deque = collections.deque()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="pp-prefill-early-finish", daemon=True
        )

    def start(self) -> EarlyFinishThread:
        self.tracker.store.add_listener(self._on_put)
        self._thread.start()
        return self

    def alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _on_put(self, key, value) -> None:
        if value == KVPoll.Success:
            self._dirty.append(key)
            self._wake.set()

    def _take_rooms(self) -> Set[int]:
        rooms = set()
        while self._dirty:
            rooms.add(self._dirty.popleft())
        return rooms

    def _run(self) -> None:
        failures = 0
        next_scan = time.monotonic() + _SCAN_S
        while not self._stop.is_set():
            self._wake.wait(_RECHECK_S if self.tracker.has_recheck_rooms() else _SCAN_S)
            self._wake.clear()
            if self._stop.is_set():
                return
            rooms = self._take_rooms() | self.tracker.take_recheck_rooms()
            now = time.monotonic()
            try:
                if rooms:
                    self.tracker.emit(rooms)
                if now >= next_scan:
                    next_scan = now + _SCAN_S
                    self.tracker.emit()
                failures = 0
            except Exception:
                failures += 1
                logger.exception("PP prefill early finish: emit failed")
                if failures >= _MAX_EMIT_FAILURES:
                    logger.error(
                        "PP prefill early finish stopped; the ring release "
                        "streams every remaining finish"
                    )
                    return


def start_pp_prefill_early_finish(
    *,
    kv_manager: CommonKVManager,
    transfer_queue: IndexedReqList,
    output_streamer: SchedulerOutputStreamer,
) -> Optional[PPPrefillEarlyFinish]:
    """Start PP rank 0's early finish (SGLANG_OPT_PP_PREFILL_EARLY_FINISH) when
    its checks pass; None on the other ranks or when a check declines."""
    parallel = get_parallel()
    if parallel.pp_rank != 0:
        return None
    reason = None
    if get_disagg().disaggregation_transfer_backend != TransferBackend.MOONCAKE.value:
        reason = "the KV transfer backend is not Mooncake"
    elif parallel.attn_tp_size != 1 or parallel.attn_cp_size != 1:
        # PP rank 0's store holds one sender per stage only then.
        reason = "a stage has more than one attention TP or CP rank"
    elif not isinstance(kv_manager.request_status, PPConsensusStore):
        reason = "the KV sender status is not shared across PP stages"
    elif output_streamer.rust_server is not None:
        reason = "the output sink is not a zmq socket"
    if reason is not None:
        logger.info("SGLANG_OPT_PP_PREFILL_EARLY_FINISH declined: %s", reason)
        return None
    output_streamer.send_to_detokenizer.enable_thread_safety()
    tracker = PPPrefillEarlyFinish(
        kv_manager=kv_manager,
        transfer_queue=transfer_queue,
        output_streamer=output_streamer,
    )
    EarlyFinishThread(tracker).start()
    logger.info(
        "SGLANG_OPT_PP_PREFILL_EARLY_FINISH engaged: PP rank 0 streams a request's "
        "finish once every stage's KV sender reports Success; the KV release still "
        "follows the ring consensus"
    )
    return tracker
