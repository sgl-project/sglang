"""Dev-only handling for a batch whose forward pass raised (see ``--debug-mode``).

Tears down the requests the failed batch owns so the event loop can continue with the
weights and captured CUDA graphs still in memory. The teardown is scoped to the batch:
the waiting queue, the memory pools and the rest of the scheduler state are untouched,
and the aborted requests are left ``finished()`` so ``filter_batch`` drops them from
``running_batch`` on the next iteration through the normal path.
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
        skipped: the normal path streamed their final output and released their KV
        before the exception, so touching them again would send an abort for a
        completed rid and double-free.
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

        Mirrors ``Scheduler.process_pending_chunked_abort``. HiSparse retraction runs
        first because it is only valid while the request is unfinished, the same
        ordering ``release_req`` uses.
        """
        if self._hisparse_coordinator is not None:
            self._hisparse_coordinator.retract_req(req)
        prepare_abort(req, ABORT_MESSAGE)
        req.time_stats.trace_ctx.abort(abort_info={"reason": ABORT_MESSAGE})
        # prepare_abort populates the finish metadata; drop the staged finish reason
        # so nothing re-applies it after the request is gone.
        req.to_finish = None
        if self._enable_hicache_storage():
            self._tree_cache.release_aborted_request(req.rid)
        release_kv_cache(req, self._tree_cache, is_insert=False)
        self._ipc_channels.send_to_tokenizer.send_output(
            AbortReq(rid=req.rid, abort_message=ABORT_MESSAGE), req
        )
        logger.debug(f"Discarded request from the failed batch. {req.rid=}")
