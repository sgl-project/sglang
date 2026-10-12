"""Early send of prefill KV chunks.

Enabled by ``SGLANG_DISAGG_PREFILL_EARLY_SEND_KV``.

Under overlap scheduling the result of prefill batch N is processed only after
batch N+1 has been launched, and the page-index reads in ``send_kv_chunk`` run
on the schedule stream, which the WAR barrier fences behind N+1's forward. Every
KV chunk computed by N, including the last one that decode waits on, is
therefore handed to the transfer engine roughly one forward late.

``EarlyKVChunkSender`` sends the chunks of N from a worker thread as soon as N's
result copy (``copy_done``) completes, reading page indices on its own stream.

Each request's KV is sent by exactly one side, fixed at its first launched
batch (``Req.kv_send_owner``):

- ``EARLY``: the worker sends every chunk, in launch order, and appends the
  first output token with the last one. The scheduler sends nothing for it.
- ``SCHEDULER``: requests whose last chunk depends on the processed result
  (logprobs, grammar, sampling mask) or on a pending bootstrap keep the
  original path.

Paths that release an ``EARLY`` request's KV before its transfer concludes
hold ``lock`` so no worker send is reading it; the worker skips requests that
are aborted or no longer hold KV.
"""

from __future__ import annotations

import logging
import queue
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.disaggregation.utils import KVSendOwner, TransferBackend, is_aborted
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler

logger = logging.getLogger(__name__)


@dataclass
class _ChunkSend:
    batch_index: int
    req: Req
    # KV computed through this batch; snapshotted at launch since the next
    # chunk's scheduling moves req.extend_end.
    end_idx: int
    is_last: bool


@dataclass
class _BatchSend:
    copy_done: torch.cuda.Event
    next_token_ids: torch.Tensor
    chunks: List[_ChunkSend]


class EarlyKVChunkSender:
    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler
        self.lock = threading.Lock()
        # None stops the worker.
        self._queue: queue.SimpleQueue[Optional[_BatchSend]] = queue.SimpleQueue()
        # Worker-thread only: requests whose earlier chunk failed to send.
        self._failed_reqs: weakref.WeakSet[Req] = weakref.WeakSet()
        self._device = torch.cuda.current_device()
        # Must not be the schedule stream: the WAR barrier makes it wait for the
        # next batch's forward, which is the delay this class removes.
        self._stream = torch.cuda.Stream(device=self._device)
        threading.Thread(
            target=self._worker,
            name="prefill-early-kv-chunk-send",
            daemon=True,
        ).start()
        logger.info("Prefill early KV chunk send enabled")

    @classmethod
    def maybe_create(cls, scheduler: Scheduler) -> Optional[EarlyKVChunkSender]:
        if not envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_KV.get():
            return None
        if not cls.is_supported(scheduler):
            logger.warning(
                "SGLANG_DISAGG_PREFILL_EARLY_SEND_KV is not supported with this "
                "configuration; KV chunks are sent by the scheduler"
            )
            return None
        return cls(scheduler)

    @staticmethod
    def is_supported(scheduler: Scheduler) -> bool:
        return (
            torch.cuda.is_available()
            and scheduler.enable_overlap
            and not scheduler.enable_staging
            and scheduler.kv_checksum_computer is None
            and scheduler.spec_algorithm.is_none()
            and get_disagg().disaggregation_decode_allocation_policy
            != "prefill_complete"
            # Sends run off the scheduler thread; only Mooncake is verified.
            and scheduler.transfer_backend == TransferBackend.MOONCAKE
            # Compaction may move pages while the worker reads their ids.
            and not scheduler.enable_unified_memory
            # Extra state payloads (mamba, SWA, DSA, ...) are unverified.
            and not scheduler.disagg_prefill_bootstrap_queue.kv_manager.kv_args.state_types
        )

    def stop(self) -> None:
        self._queue.put(None)

    @staticmethod
    def is_eligible(req: Req) -> bool:
        return (
            req.bootstrap_room is not None
            and not req.pending_bootstrap
            and len(req.output_ids) == 0
            and not req.return_logprob
            and not req.return_sampling_mask
            and req.grammar is None
        )

    def submit(self, batch: ScheduleBatch, result: GenerationBatchResult) -> None:
        """Queue a launched batch; call once its result copy has been recorded.

        Fixes the KV send owner of requests seen for the first time.
        """
        can_send = result.copy_done is not None and isinstance(
            result.next_token_ids, torch.Tensor
        )
        chunks = []
        for i, req in enumerate(batch.reqs):
            if req.kv_send_owner == KVSendOwner.NONE:
                req.kv_send_owner = (
                    KVSendOwner.EARLY
                    if can_send and self.is_eligible(req)
                    else KVSendOwner.SCHEDULER
                )
            if req.kv_send_owner != KVSendOwner.EARLY:
                continue
            input_len = len(req.origin_input_ids)
            chunks.append(
                _ChunkSend(
                    batch_index=i,
                    req=req,
                    end_idx=min(req.extend_end, input_len),
                    is_last=req.extend_end >= input_len,
                )
            )
        if chunks:
            assert can_send, "EARLY-owned request in a batch without a result copy"
            self._queue.put(_BatchSend(result.copy_done, result.next_token_ids, chunks))

    def _worker(self) -> None:
        torch.cuda.set_device(self._device)
        while True:
            batch_send = self._queue.get()
            if batch_send is None:
                return
            try:
                with torch.cuda.stream(self._stream):
                    self._send_batch(batch_send)
            except Exception:
                logger.exception("Early KV chunk send failed")

    def _send_batch(self, batch_send: _BatchSend) -> None:
        batch_send.copy_done.synchronize()
        token_ids = None
        for chunk in batch_send.chunks:
            if chunk.is_last and token_ids is None:
                token_ids = batch_send.next_token_ids.tolist()
            with self.lock:
                self._send_chunk(
                    chunk, token_ids[chunk.batch_index] if chunk.is_last else None
                )

    def _send_chunk(self, chunk: _ChunkSend, token_id: int | None) -> None:
        req = chunk.req
        if req in self._failed_reqs or is_aborted(req) or not req.kv.holds_kv:
            # The scheduler concludes it through the inflight queue or abort.
            return
        try:
            if chunk.is_last:
                req.output_ids.append(token_id)
                self._scheduler.send_kv_chunk(req, last_chunk=True)
                req.time_stats.set_prefill_transfer_queue_entry_time()
            else:
                self._scheduler.send_kv_chunk(
                    req, last_chunk=False, end_idx=chunk.end_idx
                )
        except Exception:
            # A missing range cannot be resent safely; fail the transfer so the
            # scheduler concludes the request through the inflight queue.
            logger.exception("Early KV chunk send failed for %s", req.rid)
            if not chunk.is_last:
                self._failed_reqs.add(req)
            req.disagg_kv_sender.abort()
