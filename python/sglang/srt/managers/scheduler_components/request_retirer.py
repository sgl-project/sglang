"""Finish or reject a request that never ran."""

from __future__ import annotations

import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Optional

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.managers.io_struct import AbortReq, FinishReasonDict
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.runtime_context import get_serving
from sglang.srt.utils.weight_versions import compute_weight_version_spans

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler_components.bundles import KvResources
    from sglang.srt.managers.scheduler_components.ipc_channels import (
        SchedulerIpcChannels,
    )
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )

logger = logging.getLogger(__name__)


class RequestRetirer:
    def __init__(
        self,
        *,
        kv: KvResources,
        beam_coordinator: BeamCoordinator,
        output_streamer: SchedulerOutputStreamer,
        ipc_channels: SchedulerIpcChannels,
    ):
        self.tree_cache = kv.tree_cache
        self.beam_coordinator = beam_coordinator
        self.output_streamer = output_streamer
        self.ipc_channels = ipc_channels

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

    def release_aborted_request(self, req: Req) -> None:
        """Drop the cache-side state an aborted request left behind."""
        req.release_mm_inputs_on_abort()
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        if req.session is not None and req.session.streaming:
            req.session.abort_req()

    def send_abort_output(
        self, req: Req, finished_reason: Optional[FinishReasonDict] = None
    ) -> None:
        self.ipc_channels.send_to_tokenizer.send_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )

    def reject_sampling_mask_request(self, req: Req, error_msg: str) -> None:
        """Return a sampling-mask validation error without running the model."""
        logger.error(f"{error_msg}, {req.rid=}")
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
        self.output_streamer.stream_output([req], req.return_logprob)


def make_abort_req(
    req: Req, finished_reason: Optional[FinishReasonDict] = None
) -> AbortReq:
    return AbortReq(
        rid=req.rid,
        finished_reason=finished_reason,
        weight_versions=compute_weight_version_spans(
            req.weight_version_events,
            current_version=get_serving().weight_version,
            num_output_tokens=len(req.output_ids),
        ),
    )
