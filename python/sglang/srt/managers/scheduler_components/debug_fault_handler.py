"""Dev-only fault handling for the scheduler event loop (see ``--debug-mode``).

By default a serving-time exception tears the process down, so every reproduction
pays full weight load + CUDA graph capture again. Debug mode instead **discards the
failed batch**: the requests that batch owns are aborted and their resources are
released through the same primitives the normal finish path uses, and the event loop
continues with the rest of the scheduler state untouched.

Scoping the fault to the batch is what keeps this maintainable. A whole-scheduler
reset has to enumerate every loop-carried field (queues, chunked slots, pending
health-check IPCs, dLLM staging, ...) and silently rots as fields are added; the
same argument is already written down for ``pause_generation(mode="in_place")``,
which deliberately leaves scheduler state alone so "the normal event loop handles
last_batch merge, chunked_req cleanup ... through the standard code paths. This
avoids duplicating batch manipulation logic and the accounting bugs that come with
it." A batch, by contrast, carries its own request list, so the blast radius is
defined by data that already exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.managers.scheduler_components.ipc_channels import SchedulerIpcChannels
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.common import release_kv_cache

if TYPE_CHECKING:
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch

logger = logging.getLogger(__name__)

ABORT_MESSAGE = "Aborted by --debug-mode after a serving-time exception."


class SchedulerDebugFaultHandler:
    """Tears down the requests of a batch whose forward pass raised."""

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        ipc_channels: SchedulerIpcChannels,
        # Callable, not a bool: attach/detach_hicache_storage flips this at runtime.
        enable_hicache_storage: Callable[[], bool],
    ) -> None:
        self._tree_cache = tree_cache
        self._hisparse_coordinator = hisparse_coordinator
        self._ipc_channels = ipc_channels
        self._enable_hicache_storage = enable_hicache_storage

    def discard_batch(self, batch: ScheduleBatch) -> List[str]:
        """Abort every unfinished request in ``batch`` and free its resources.

        Returns the aborted rids. Requests that already carry a finish reason are
        skipped: the normal path streamed their final output before the exception,
        so notifying them again would send an abort for a completed rid.
        """
        aborted_rids = []
        for req in batch.reqs:
            if req.finished():
                continue
            self._abort_req(req)
            aborted_rids.append(req.rid)
        return aborted_rids

    def _abort_req(self, req: Req) -> None:
        """Finish ``req`` as aborted without another forward pass.

        Mirrors ``Scheduler.process_pending_chunked_abort``, the existing path for a
        request that must be torn down before its forward pass can complete. Ordering
        matters: HiSparse retraction and the KV release read state that
        ``prepare_abort`` invalidates, so they bracket it the same way ``release_req``
        does.
        """
        if self._hisparse_coordinator is not None:
            self._hisparse_coordinator.retract_req(req)
        prepare_abort(req, ABORT_MESSAGE)
        req.time_stats.trace_ctx.abort(abort_info={"reason": ABORT_MESSAGE})
        # prepare_abort only populates finish metadata; clear the staged finish
        # reason so nothing re-applies it after the request is gone.
        req.to_finish = None
        if self._enable_hicache_storage():
            self._tree_cache.release_aborted_request(req.rid)
        release_kv_cache(req, self._tree_cache, is_insert=False)
        self._ipc_channels.send_to_tokenizer.send_output(
            AbortReq(rid=req.rid, abort_message=ABORT_MESSAGE), req
        )
        logger.debug(f"Discarded request from the failed batch. {req.rid=}")
