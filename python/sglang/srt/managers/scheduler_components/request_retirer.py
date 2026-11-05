from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.utils import make_abort_req
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import discard_kv_cache_backup, release_kv_cache

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
    """Ends requests and releases what they hold.

    abort ends a request no launched step carries and tells the tokenizer
    once; release frees one its caller ended, in a step or out of one, and
    streams, detaching it first when aborted."""

    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        beam_coordinator: BeamCoordinator,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        uocr_runtime: Optional[UocrSchedulerRuntime],
        send_abort_output: Callable[[AbortReq, Req], None],
    ):
        self.tree_cache = tree_cache
        self.beam_coordinator = beam_coordinator
        self.hisparse_coordinator = hisparse_coordinator
        self.uocr_runtime = uocr_runtime
        self.send_abort_output = send_abort_output

    def abort(
        self,
        req: Req,
        *,
        reason: Optional[FINISH_ABORT] = None,
        finished_reason: Optional[FinishReasonDict] = None,
        checkpoint: bool = False,
    ) -> None:
        """Abort a request no launched step carries: one aborted in a queue, or
        refused at the waiting-queue door. ``reason`` is the request's own finish
        reason and ``finished_reason`` what the tokenizer receives."""
        # Finished in place, so a later abort in the same intake skips it.
        if req.finished_reason is None:
            req.finish_outside_step(reason or FINISH_ABORT())
        self.beam_coordinator.retire_group(req)
        self.send_abort_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )
        # It never ran a step, so only the abandon settles its cache attempt.
        self._release(req, checkpoint=checkpoint, abandon_attempt=True)

    def release(self, req: Req, *, checkpoint: bool) -> None:
        """Give back what an ended request holds:
        an aborted one is first dropped from the components tracking it, then
        its KV backup and its memory go. The caller finishes and streams it."""
        # A checkpoint hands the KV to the cache, whose store settles the
        # attempt; an abandon first would end LMCache's session before it.
        self._release(req, checkpoint=checkpoint, abandon_attempt=not checkpoint)

    def _release(self, req: Req, *, checkpoint: bool, abandon_attempt: bool) -> None:
        if isinstance(req.finished_reason, FINISH_ABORT):
            self._detach(req, abandon_attempt=abandon_attempt)
        discard_kv_cache_backup(req, self.tree_cache)
        # HiSparse clears its mapping of the row before the row is freed.
        if req.kv.holds_kv and self.hisparse_coordinator is not None:
            self.hisparse_coordinator.request_finished(req)
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, checkpoint=checkpoint)

    def _detach(self, req: Req, *, abandon_attempt: bool) -> None:
        """Drop an aborted request from the components tracking it; frees no
        memory, so it runs while the request still holds its KV."""
        # Counted before the media goes: the output sent after still reports it.
        req.get_mm_token_counts()
        req.release_mm_inputs_on_abort()
        if abandon_attempt:
            self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        if req.session is not None and req.session.streaming:
            req.session.abort_req()
        if self.uocr_runtime is not None:
            self.uocr_runtime.on_req_dropped(req)
