from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.exit_ledger import record_exit
from sglang.srt.managers.utils import make_abort_req
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import (
    backup_kv_cache,
    discard_kv_cache_backup,
    release_kv_cache,
)
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.io_struct import AbortReq, FinishReasonDict
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.models.unlimited_ocr_v2_support.scheduler_runtime import (
        UocrSchedulerRuntime,
    )


class RequestRetirer:
    """Ends requests and releases what they hold: `abort` for one no launched
    step reports (sends the abort output), `retire_unadmitted` for one rejected
    before it was queued (streams its output), `release` for one its caller
    ended and sends. `abort`, `retire_unadmitted` and `abort_retracted` take
    the request out of the list it is still in: a request ended on its way
    between lists leaves for none."""

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        beam_coordinator: BeamCoordinator,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        uocr_runtime: Optional[UocrSchedulerRuntime],
        send_abort_output: Callable[[AbortReq, Req], None],
        stream_output: Callable[..., None],
    ):
        self.tree_cache = tree_cache
        self.beam_coordinator = beam_coordinator
        self.hisparse_coordinator = hisparse_coordinator
        self.uocr_runtime = uocr_runtime
        self.send_abort_output = send_abort_output
        self.stream_output = stream_output

    def abort(
        self,
        req: Req,
        *,
        reason: Optional[FINISH_ABORT] = None,
        finished_reason: Optional[FinishReasonDict] = None,
        checkpoint: bool = False,
    ) -> None:
        """Abort a request no launched step reports: one aborted in a queue or
        the chunked slot, or refused at the waiting-queue door. ``reason`` is the
        request's own finish reason and ``finished_reason`` what the tokenizer
        receives."""
        _leave_owner_list(req)
        # Finished in place, so a later abort in the same intake skips it.
        if req.finished_reason is None:
            req.finish_outside_step(reason or FINISH_ABORT())
        self.send_abort_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )
        self.release(req, checkpoint=checkpoint)

    def retire_unadmitted(self, req: Req) -> None:
        """Finish a request rejected before it was queued. It holds no device KV,
        but the release still frees its streaming session, its cache prefetch
        retry and a retracted request's host KV backup."""
        _leave_owner_list(req)
        # An already set `finished_reason` is what the client receives.
        reason = req.finished_reason or req.to_finish
        req.time_stats.trace_ctx.abort(abort_info={"reason": reason.message})
        req.update_finish_state()
        self.release(req, checkpoint=False)
        # Like `find_active_slot`, the detached request also leaves its session.
        if req.session is not None and req.session.streaming:
            req.session = None
        self.stream_output([req], req.return_logprob)

    def release(self, req: Req, *, checkpoint: bool) -> None:
        """Give back what an ended request holds; the caller finishes and streams it."""
        record_exit(req, "release")
        if isinstance(req.finished_reason, FINISH_ABORT):
            self._detach(req)
            # A checkpoint hands the KV to the cache, whose store settles the attempt.
            if not checkpoint:
                self.tree_cache.finish(
                    req.cache_request_handle, CacheRequestOutcome.ABORT
                )
            # No-op once the group retired; member rows are not freed here.
            self.beam_coordinator.retire_group(req)
        discard_kv_cache_backup(req, self.tree_cache)
        # HiSparse clears its mapping of the row before the row is freed.
        if req.kv.holds_kv and self.hisparse_coordinator is not None:
            self.hisparse_coordinator.release_request(req)
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, checkpoint=checkpoint)

    def retract(self, req: Req, *, offload_kv: bool) -> bool:
        """Release a retracted request's memory uncheckpointed; return whether it can resume."""
        if req.kv.holds_kv and self.hisparse_coordinator is not None:
            self.hisparse_coordinator.release_request(req)

        # PD decode backs the KV up to host to restore it without recompute; a caller
        # that will rebootstrap passes offload_kv=False to skip the copy.
        if get_disagg().disaggregation_mode != "decode":
            resumable = req.resumable_by_reprefill
        elif offload_kv:
            resumable = backup_kv_cache(
                req,
                self.tree_cache,
                self.tree_cache.req_to_token_pool,
                self.tree_cache.token_to_kv_pool_allocator,
                get_disagg().disaggregation_decode_retraction_backup,
            )
        else:
            resumable = req.resumable_by_rebootstrap
        # TODO (csy): for preempted requests, we may want to insert into the tree
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, checkpoint=False)
        return resumable

    def abort_retracted(self, req: Req, *, reason: FINISH_ABORT) -> None:
        """Abort a retracted request that will not be requeued. Not finished in place:
        a step launched before the retraction may still carry it and report it again."""
        _leave_owner_list(req)
        record_exit(req, "abort_retracted")
        self._detach(req)
        # retract released the KV without a checkpoint.
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        self.beam_coordinator.retire_group(req)
        self.send_abort_output(
            make_abort_req(req, finished_reason=reason.to_json()), req
        )

    def _detach(self, req: Req) -> None:
        """Drop an aborted request from the components tracking it and free its
        media, but no KV or rows. `release` runs it before the KV goes: uOCR's
        drop tells unadmitted prefix slots from row-owned ones by `holds_kv`."""
        # Counted before the media goes: the output sent after still reports it.
        req.get_mm_token_counts()
        req.release_mm_inputs_on_abort()
        if req.session is not None and req.session.streaming:
            req.session.abort_req()
        if self.uocr_runtime is not None:
            self.uocr_runtime.on_req_dropped(req)


def _leave_owner_list(req: Req) -> None:
    if req.owner_list is not None:
        req.owner_list.leave([req])
