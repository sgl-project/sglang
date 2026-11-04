"""The ways a request leaves the scheduler before any launched step carries it.

Each entry point names one exit and runs the cleanup steps that exit takes, in
its order: the finish reason, the media and cache-side state of an aborted
request, the beam group, HiSparse rows, the KV, and one send to the tokenizer.
Exits of the same kind still differ in which steps they take; each entry keeps
its own sequence.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING, Callable, Optional

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.prefill import maybe_release_metadata_buffer
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.utils import make_abort_req
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import discard_kv_cache_backup, release_kv_cache
from sglang.srt.runtime_context import get_disagg

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.io_struct import AbortReq, FinishReasonDict
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.output_sender import SenderWrapper
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.models.unlimited_ocr_v2_support.scheduler_runtime import (
        UocrSchedulerRuntime,
    )


class RequestRetirer:
    def __init__(
        self,
        *,
        tree_cache: BasePrefixCache,
        beam_coordinator: BeamCoordinator,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        uocr_runtime: Optional[UocrSchedulerRuntime],
        output_streamer: SchedulerOutputStreamer,
        send_abort_output: Callable[[AbortReq, Req], None],
        send_to_tokenizer: SenderWrapper,
    ):
        self.tree_cache = tree_cache
        self.beam_coordinator = beam_coordinator
        self.hisparse_coordinator = hisparse_coordinator
        self.uocr_runtime = uocr_runtime
        self.output_streamer = output_streamer
        self.send_abort_output = send_abort_output
        # The raw tokenizer channel, bypassing `send_abort_output`.
        self.send_to_tokenizer = send_to_tokenizer

    # ------------------------------------------------------------------
    # Queued aborts: a request aborted in a queue that has started nothing.
    # ------------------------------------------------------------------
    def retire_aborted_waiting(
        self,
        req: Req,
        *,
        finished_reason: Optional[FinishReasonDict],
        disaggregation_mode: DisaggregationMode,
    ) -> None:
        """A request aborted in the waiting queue. A PD prefill request also
        releases its sender and KV, which the caller does after this."""
        # A waiting request has started nothing, so it is aborted here; the
        # tokenizer still needs the abort output to clean up its state.
        # Finished in place, so a later abort in the same intake skips it.
        if req.finished_reason is None:
            req.finish_outside_step(FINISH_ABORT())
        self.release_aborted_request(req)
        self.beam_coordinator.retire_group(req)
        # Without the initiator's reason the tokenizer falls back to a
        # generic abort message.
        self.send_abort_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )
        # For disaggregation decode mode, the request in the waiting queue has KV cache allocated.
        if disaggregation_mode == DisaggregationMode.DECODE:
            if get_disagg().disaggregation_decode_host_receive_threshold < 1:
                discard_kv_cache_backup(req, self.tree_cache, "host_pool")
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.request_finished(req)
            release_kv_cache(req, self.tree_cache, checkpoint=True)

        # For mamba radix cache
        if req.kv.holds_mamba and disaggregation_mode not in (
            DisaggregationMode.PREFILL,
            DisaggregationMode.DECODE,
        ):
            release_kv_cache(req, self.tree_cache, checkpoint=False)

    def retire_evicted_waiting(self, req: Req, *, message: str, incoming: Req) -> None:
        """A waiting request evicted for the higher-priority ``incoming`` one
        when the queue is full; the caller has taken it off the queue."""
        self.release_aborted_request(req)
        self.beam_coordinator.retire_group(req)
        # Finished in place, so a later abort in the same intake skips it.
        req.finish_outside_step(FINISH_ABORT(message, HTTPStatus.SERVICE_UNAVAILABLE))
        self._send_queue_abort(req, message=message)
        # An evicted request sharing the incoming rid rejects the incoming one
        # too, which then only drops its cache-side state.
        if req.rid == incoming.rid:
            self.release_aborted_request(incoming)

    def retire_aborted_dllm_waiting(self, req: Req) -> None:
        """A dLLM request aborted in its staging queue."""
        # Finished in place, so a later abort in the same intake skips it.
        if req.finished_reason is None:
            req.finish_outside_step(FINISH_ABORT())
        self.release_aborted_request(req)
        self.send_abort_output(make_abort_req(req), req)
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, checkpoint=False)

    def retire_aborted_doc_window(
        self, req: Req, *, finished_reason: Optional[FinishReasonDict]
    ) -> None:
        """A document window aborted while held off the waiting queue."""
        self.release_aborted_request(req)
        self.send_to_tokenizer.send_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )

    # ------------------------------------------------------------------
    # Intake rejects: a request refused before it is queued, or at a door.
    # ------------------------------------------------------------------
    def reject_request(
        self, req: Req, error_msg: str, *, status_code: Optional[HTTPStatus]
    ) -> None:
        """Finish a request with an abort and stream it, without running it."""
        prepare_abort(req, error_msg, status_code=status_code)
        self.output_streamer.stream_output([req], req.return_logprob)

    def reject_unqueued_request(
        self,
        req: Req,
        *,
        finished_reason: FinishReasonDict,
        release_aborted: bool,
    ) -> None:
        """Send an abort for a request refused before it is queued; its own
        finish reason stays unset. ``release_aborted`` also drops its media
        and cache-side state."""
        abort_req = make_abort_req(req, finished_reason=finished_reason)
        req.time_stats.trace_ctx.abort(abort_info=abort_req.finished_reason)
        self.send_abort_output(abort_req, req)
        if release_aborted:
            self.release_aborted_request(req)

    def reject_queue_full(self, req: Req, *, message: str) -> None:
        """An incoming request refused because the waiting queue is full."""
        self._send_queue_abort(req, message=message)
        self.release_aborted_request(req)

    def retire_unadmitted_request(self, req: Req) -> None:
        """Finish a request the disaggregation queues rejected at their door."""
        # `create_req` marks a streaming session in-flight, and the pre-abort
        # detach lives in `StreamingSession.find_active_slot`, which only runs
        # while scheduling; a session left in-flight rejects every later request.
        if req.session is not None and req.session.streaming:
            req.release_mm_inputs_on_abort()
            req.session.abort_req()
            req.session = None
        # `beam_coordinator.validate_and_init` counts the group in ahead of the
        # checks that reject; no-op when the request has no group.
        self.beam_coordinator.retire_group(req)
        # PREFILL runs `_prefetch_kvcache` before its door, so even the
        # one-token stub is registered with the cache by now:
        # `prefetch_from_storage` arms the paced-retry set for this attempt's
        # cache handle. Only a `finish`/ABORT and a `waiting_queue` sweep clear
        # that, and a retired request reaches neither.
        self.release_aborted_request(req)
        # `update_finish_state` returns early once `finished()`, so an already
        # set `finished_reason` is what the client receives; report the same.
        reason = req.finished_reason or req.to_finish
        req.time_stats.trace_ctx.abort(abort_info={"reason": reason.message})
        req.update_finish_state()
        self.output_streamer.stream_output([req], req.return_logprob)

    # ------------------------------------------------------------------
    # PD queue exits: a request leaving a PD queue before any step ran it.
    # ------------------------------------------------------------------
    def abort_queued_attempt(self, req: Req) -> None:
        """A request aborted in the PD bootstrap queue: its media and session
        go and its cache request ends now; the queue retires it once the abort
        reaches its sender."""
        self.release_aborted_request(req)

    def retire_aborted_held(
        self, req: Req, *, finished_reason: Optional[FinishReasonDict]
    ) -> None:
        """A PD decode request held for rebootstrap, aborted; it holds no KV."""
        req.finish_outside_step(FINISH_ABORT())
        self.release_aborted_request(req)
        self.send_abort_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )

    def retire_aborted_retracted(self, req: Req) -> None:
        """A retracted PD decode request, aborted; it holds a KV backup."""
        discard_kv_cache_backup(
            req,
            self.tree_cache,
            get_disagg().disaggregation_decode_retraction_backup,
        )
        self.send_abort_output(make_abort_req(req), req)

    def retire_bootstrap_failure(
        self,
        req: Req,
        error_message: str,
        *,
        metadata_buffer_allocator: ReqToMetadataIdxAllocator,
    ) -> None:
        """A PD prefill request whose bootstrap failed."""
        if req.kv.holds_kv or req.kv.holds_mamba:
            release_kv_cache(req, self.tree_cache, checkpoint=False)
        maybe_release_metadata_buffer(req, metadata_buffer_allocator)
        req.pending_bootstrap = False
        self.reject_request(
            req, error_message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR
        )
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)

    # ------------------------------------------------------------------
    # Steps.
    # ------------------------------------------------------------------
    # Public while `Scheduler.release_aborted_request` still delegates here.
    def release_aborted_request(self, req: Req) -> None:
        """Drop the cache-side state an aborted request left behind."""
        req.release_mm_inputs_on_abort()
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        if req.session is not None and req.session.streaming:
            req.session.abort_req()
        if self.uocr_runtime is not None:
            self.uocr_runtime.on_req_dropped(req)

    def _send_queue_abort(self, req: Req, *, message: str) -> None:
        self.send_abort_output(
            make_abort_req(
                req,
                finished_reason={
                    "type": "abort",
                    "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                    "message": message,
                },
            ),
            req,
        )
        req.time_stats.trace_ctx.abort(abort_info={"reason": message})
