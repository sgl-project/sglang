"""Every way a request leaves the scheduler, or gives its KV back to be
requeued.

A request that leaves releases up to five parts. Each entry point runs the
parts its outcome needs, in this order:

1. media and streaming session, with the end of the cache request (an aborted
   outcome);
2. beam group (an aborted outcome);
3. retraction backup (a retraction; it reads the device KV);
4. HiSparse rows;
5. KV row and request slot; the slot and the pages a pending step may still
   write go back at the host commit of the last pending step that carries
   the request.

One entry keeps a documented exception: a PD transfer that failed after its
prefill checkpoints the KV before it ends the cache request.

Only this module calls the part functions (`release_kv_cache`,
`tree_cache.finish`, `beam_coordinator.retire_group`, the HiSparse release
calls and the media release); a CPU test enforces it.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from http import HTTPStatus
from typing import TYPE_CHECKING, Iterable, Optional

from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import AbortReq, FinishReasonDict
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import (
    HeldKv,
    backup_kv_cache,
    evict_from_tree_cache,
    free_held_kv,
    free_req_slot,
    release_kv_cache,
)
from sglang.srt.runtime_context import get_disagg, get_serving
from sglang.srt.utils.weight_versions import compute_weight_version_spans

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.bundles import KvResources
    from sglang.srt.managers.scheduler_components.output_streamer import (
        SchedulerOutputStreamer,
    )
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps

logger = logging.getLogger(__name__)


class RequestRetirer:
    def __init__(
        self,
        *,
        kv: KvResources,
        beam_coordinator: BeamCoordinator,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        decode_offload_manager: Optional[DecodeKVCacheOffloadManager],
        pending: PendingSteps,
        output_streamer: SchedulerOutputStreamer,
    ):
        self.kv = kv
        self.tree_cache = kv.tree_cache
        self.beam_coordinator = beam_coordinator
        self.hisparse_coordinator = hisparse_coordinator
        self.decode_offload_manager = decode_offload_manager
        self.pending = pending
        self.output_streamer = output_streamer

    # ------------------------------------------------------------------
    # Entry points, one per outcome.
    # ------------------------------------------------------------------
    def retire_finished(self, req: Req, *, checkpoint: bool) -> None:
        """A request its commit finished (or a PD handoff finished at once):
        HiSparse rows, then the KV row (with ``checkpoint``, the tree first
        keeps what it can key). A successful cache request needs no finish."""
        self._release_hisparse_rows(req)
        self._release_kv(req, checkpoint=checkpoint)

    def retire_aborted_waiting(self, req: Req) -> None:
        """A request aborted in the waiting queue: media and session with the
        cache request, the beam group, then the mamba state a mamba cache may
        have allocated ahead of its KV (a KV row it shows is borrowed from its
        streaming session, which keeps it)."""
        self.release_aborted_request(req)
        self.beam_coordinator.retire_group(req)
        if req.kv.holds_mamba:
            self._release_kv(req, checkpoint=False)

    def retire_aborted(self, req: Req, *, checkpoint: bool = False) -> None:
        """A request aborted once it may hold KV (running, chunked, or in a PD
        queue that allocated it): media and session with the cache request,
        the beam group, HiSparse rows, then whatever KV or mamba state it
        holds."""
        self.release_aborted_request(req)
        self.beam_coordinator.retire_group(req)
        self._release_hisparse_rows(req)
        if req.kv.holds_kv or req.kv.holds_mamba:
            self._release_kv(req, checkpoint=checkpoint)

    def retire_transfer_failed(self, req: Req) -> None:
        """A PD prefill request whose KV transfer failed after its prefill: the
        prefill KV is valid, so the tree keeps it first; then media and session
        with the cache request."""
        self._release_kv(req, checkpoint=True)
        self.release_aborted_request(req)

    def retire_offloaded(self, req: Req) -> None:
        """A finished PD decode request whose KV was offloaded to host: once
        its last offload copy landed, the rest of its row (the tail the copies
        did not cover) goes back without an insert; until then, nothing."""
        if self.decode_offload_manager.has_inflight_offload(req):
            return
        # Defensive guard: ReqToTokenPool.free sets req_pool_idx to None,
        # so a previously-released request must be skipped here to avoid
        # non-idempotent side effects (double free, double unlock).
        if req.kv.req_pool_idx is None or req.kv.req_pool_idx == -1:
            return

        # Released only at request finish; a mid-decode free races with
        # concurrent admission over live slots. The prefix the prealloc match
        # took from the tree stays with the tree: not freed, only unlocked.
        row = (req.kv.cache_protected_len, req.kv.kv_allocated_len)
        step = self.pending.last_step_with(req)
        if step is None:
            self.tree_cache.free_kv_row(req.kv, [row])
        self.tree_cache.unlock(req.lock)
        req.lock = None
        if step is None:
            free_req_slot(self.kv.req_to_token_pool, req)
        else:
            # A pending step still writes the row: all of it waits.
            held = HeldKv(
                kv=dataclasses.replace(
                    req.kv,
                    component_evicted_seqlens=dict(req.kv.component_evicted_seqlens),
                ),
                held_from=row[0],
                row_frees=(row,),
            )
            req.kv.req_pool_idx = None
            req.kv.penalty_row = None
            self.pending.hold_until_commit(step, held)
        req.kv.mark_kv_released()
        req.disagg.offload = None

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

    def reject_at_door(
        self,
        req: Req,
        error_msg: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST,
        err_type: str = "BadRequestError",
    ) -> None:
        """A request the intake rejects before it is queued: its media goes
        now and it is marked aborted; it then runs as a one-token stub whose
        commit finishes it (the beam commit retires a beam group)."""
        req.set_finish_with_abort(error_msg, status_code=status_code, err_type=err_type)

    def abort_queued_attempt(self, req: Req) -> None:
        """An abort of a request still in a PD queue that owns its exit: its
        media and session go and its cache request ends now; the queue retires
        it when the abort reaches its transfer."""
        self.release_aborted_request(req)

    def requeue_attempt(self, req: Req) -> None:
        """A PD prefill request requeued for another attempt: this attempt's
        cache request ends and its KV goes back without an insert; its media
        and session stay for the retry."""
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        self._release_kv(req, checkpoint=False)

    def retract(
        self, req: Req, *, num_remaining_reqs: int, offload_kv: bool = True
    ) -> bool:
        """Give a decoding request's KV back so it can be requeued; returns
        False when the KV backup failed and the request cannot be resumed."""
        # In decode disaggregation the retracted KV is offloaded to host so it can be
        # restored later without recompute (see resume_retracted_reqs/load_kv_cache).
        # Callers that will recompute the KV instead (PD true-retraction rebootstrap)
        # pass offload_kv=False to skip the wasteful device->host copy.
        backup_saved = True
        # The config bag reflects role flips; server_args keeps the launch role.
        if get_disagg().disaggregation_mode == "decode" and offload_kv:
            backup_saved = backup_kv_cache(
                req,
                self.tree_cache,
                self.kv.req_to_token_pool,
                self.kv.token_to_kv_pool_allocator,
                get_disagg().disaggregation_decode_retraction_backup,
            )
        # HiSparse frees the rows and device buffers the backup above reads.
        hisparse_coordinator = self.hisparse_coordinator
        if hisparse_coordinator is not None and not req.finished():
            hisparse_coordinator.retract_req(req)
        # TODO (csy): for preempted requests, we may want to insert into the tree
        self._release_kv(req, checkpoint=False)
        # NOTE(lsyin): we should use the newly evictable memory instantly.
        num_tokens = num_remaining_reqs * envs.SGLANG_RETRACT_DECODE_STEPS.get()
        evict_from_tree_cache(self.tree_cache, num_tokens)

        req.reset_for_retract()
        return backup_saved

    def retire_retracted_abort(self, req: Req) -> None:
        """A retracted request that cannot resume is aborted; its KV is gone,
        so only its beam group leaves."""
        self.beam_coordinator.retire_group(req)

    def release_finished_features(self, req: Req) -> None:
        """Drop a finished request's media features early to save memory; a
        session keeps them for its next request."""
        if req.multimodal_inputs is not None and req.session is None:
            req.multimodal_inputs.release_features()

    def clear_finished_media(self, reqs: Iterable[Req]) -> None:
        """After a commit, drop the media of the requests it finished; a
        session keeps its media unless the request was aborted."""
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        for req in reqs:
            if not req.finished() or not (mm_inputs := req.multimodal_inputs):
                continue
            # For session requests, keep mm_inputs for the next request
            if req.session:
                if isinstance(req.finished_reason, FINISH_ABORT):
                    req.release_mm_inputs_on_abort()
                continue
            # For non-session requests, clear features and mm_inputs
            mm_inputs.release_features()
            req.multimodal_inputs = None

    def release_held(self, step: ScheduleBatch) -> None:
        """At step's host commit, free the KV pages and slots released
        requests left to it."""
        for held in self.pending.take_held(step):
            free_held_kv(held, self.tree_cache)

    # ------------------------------------------------------------------
    # Outputs of a leaving request.
    # ------------------------------------------------------------------
    def send_abort_output(
        self, req: Req, finished_reason: Optional[FinishReasonDict] = None
    ) -> None:
        self.output_streamer.send_abort_output(
            make_abort_req(req, finished_reason=finished_reason), req
        )

    def reject_sampling_mask_request(self, req: Req, error_msg: str) -> None:
        """Return a sampling-mask validation error without running the model."""
        logger.error(f"{error_msg}, {req.rid=}")
        req.time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
        self.output_streamer.stream_output([req], req.return_logprob)

    # ------------------------------------------------------------------
    # Parts.
    # ------------------------------------------------------------------
    def release_aborted_request(self, req: Req) -> None:
        """Drop the media and the cache-side state an aborted request left
        behind, and detach its streaming session."""
        req.release_mm_inputs_on_abort()
        self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)
        if req.session is not None and req.session.streaming:
            req.session.abort_req()

    def _release_kv(self, req: Req, *, checkpoint: bool) -> None:
        """The KV part: the row and slot go back to their pools, after the
        commit of the last pending step that may still write them."""
        step = self.pending.last_step_with(req)
        if step is None:
            release_kv_cache(req, self.tree_cache, checkpoint=checkpoint)
            return
        hold = functools.partial(self.pending.hold_until_commit, step)
        release_kv_cache(req, self.tree_cache, checkpoint=checkpoint, hold=hold)

    def _release_hisparse_rows(self, req: Req) -> None:
        hisparse = self.hisparse_coordinator
        if hisparse is not None and req.kv.holds_kv and hisparse.holds_rows(req):
            hisparse.request_finished(req)


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
