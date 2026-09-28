"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, List, Optional

import msgspec
import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import pdmux_prefill_tp_group
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex.pdmux_context import (
    get_current_stream_idx,
    get_sm_counts,
    get_stream_groups,
    initialize_stream_groups,
    load_pdmux_config,
    set_current_stream_idx,
)
from sglang.srt.runtime_context import get_device, get_disagg, get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.utils import GenerationBatchResult

logger = logging.getLogger(__name__)


class PdmuxPrefillInflight(msgspec.Struct):
    """One PDMux prefill work item submitted on the prefill lane."""

    batch: Any
    result: Any
    keep_alive: Optional[List[Any]] = None
    done_work: Any = None
    done_flags: Any = None
    submit_s: float = 0.0


class SchedulerMultiplexMixin:
    def init_pdmux(self: Scheduler):
        # The current split prefill batch (layer_split mode)
        self.split_prefill_batch: Optional[ScheduleBatch] = None

        self._pdmux_prefill_pending: Optional[ScheduleBatch] = None
        self._pdmux_prefill_inflight: Optional[PdmuxPrefillInflight] = None
        self.pdmux_prefill_stream = None
        self.pdmux_prefill_copy_stream = torch.cuda.Stream(get_device().gpu_id)
        self.pdmux_prefill_copy_stream_ctx = torch.cuda.stream(
            self.pdmux_prefill_copy_stream
        )

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(get_disagg().pdmux_config_path)
        initialize_stream_groups(get_device().gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    def _extra_inflight_batches(self: Scheduler) -> List[ScheduleBatch]:
        """PDMux prefill batches that live outside running_batch / last_batch.

        Between formation and finalize the standard lane holds its work item in
        its own field, so `abort_request` would not see those requests and
        `is_fully_idle` would call the scheduler idle while a forward is still
        running. Only the standard lane reports here: the layer_split loop keeps
        its existing behaviour.
        """
        if not self.pdmux_standard:
            return []
        batches = []
        if self._pdmux_prefill_pending is not None:
            batches.append(self._pdmux_prefill_pending)
        if self._pdmux_prefill_inflight is not None:
            batches.append(self._pdmux_prefill_inflight.batch)
        return batches

    def _update_decode_attn_backends(self: Scheduler, stream_idx: int) -> None:
        """Point the decode-side attention backends at this stream group.

        The target runner has always been switched here. The standard lane also
        switches the draft runners: a graph replay indexes the group directly,
        but the eager TARGET_VERIFY fallback and DSpark's draft block resolve
        `decode_attn_backend`, which would otherwise stay pinned to group 0.
        Reached only with no prefill in flight and after both streams have been
        drained, so no forward can be reading these fields.
        """
        model_worker = getattr(self, "model_worker", None)
        if model_worker is not None and hasattr(
            model_worker, "update_pdmux_decode_attn_backend"
        ):
            model_worker.update_pdmux_decode_attn_backend(stream_idx)
            return

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
        if not self.pdmux_standard or self.draft_worker is None:
            return
        for runner in self.draft_worker._draft_model_runners():
            if runner.decode_attn_backend_group:
                runner.update_decode_attn_backend(stream_idx)

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler, running_batch: ScheduleBatch, has_prefill: bool
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        """Pick the stream group for the next step.

        `has_prefill` is an explicit argument rather than a read of
        `split_prefill_batch`: the standard lane keeps its work item in a
        different field, and at the moment of the switch that item may be formed
        but not yet submitted -- which still needs a shared group, because it is
        about to run alongside decode.
        """
        if not running_batch.is_empty() and has_prefill:
            decode_bs = running_batch.batch_size()
            manual_divisions = self.pdmux_config.manual_divisions
            if manual_divisions:
                # A decode batch under every configured threshold still has to
                # land on a shared group: index 0 is the prefill-only stream,
                # which would starve decode entirely.
                stream_idx = 1
                for i in range(len(manual_divisions)):
                    _, _, threshold = manual_divisions[i]
                    if decode_bs >= threshold:
                        stream_idx = i + 1
            else:
                stream_idx = max(
                    1,
                    min(
                        self.real_sm_group_num - 2,
                        decode_bs
                        * (self.real_sm_group_num - 2)
                        // self.pdmux_config.decode_bs_divisor,
                    ),
                )
            set_current_stream_idx(stream_idx)
        elif not running_batch.is_empty():
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self._update_decode_attn_backends(stream_idx)
        return stream_idx, self.stream_groups[stream_idx]

    def update_split_prefill_batch(
        self: Scheduler, sm_count: int, running_batch: ScheduleBatch
    ) -> tuple[bool, ScheduleBatch]:
        if self.split_prefill_batch is not None:
            return False, running_batch

        # No split forward is in flight here, which matches the normal loop's
        # "top of the scheduling step" safe point for tearing down an aborted
        # chunked request before its next chunk is formed.
        self.process_pending_chunked_abort()

        # add new request
        prefill_plan = self.get_new_batch_prefill(running_batch)
        batch = prefill_plan.batch_to_run
        running_batch = prefill_plan.running_batch
        # PDMux forms batches outside get_next_batch_to_run(), so prepare the
        # DP/MLP metadata here. Passing None is intentional: peer DP ranks may
        # have prefill work and require this rank to run an idle participant.
        batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(batch)
        if batch is not None:
            # Preserve IDLE so speculative workers can skip final prefill-only
            # post-processing while still participating in every split layer.
            if not batch.forward_mode.is_idle():
                batch.forward_mode = ForwardMode.SPLIT_PREFILL
            batch.split_index = 0
            batch.split_prefill_finished = False
            batch.split_forward_count = 1
            batch.split_forward_batch = None
            self.split_prefill_batch = batch
            return True, running_batch
        return False, running_batch

    def _get_split_forward_count(
        self: Scheduler, decode_batch: Optional[ScheduleBatch]
    ) -> int:
        remaining_layers = (
            self.model_config.num_hidden_layers - self.split_prefill_batch.split_index
        )

        # Splitting only benefits decode work that can run between prefill
        # intervals. maybe_prepare_mlp_sync_batch() returns a real or IDLE
        # decode batch on every DP rank when any rank has decode work, so this
        # decision is rank-consistent. Using the local running batch here can
        # make one rank run all remaining prefill layers while a peer runs only
        # a segment, which mismatches their model collectives and deadlocks.
        decode_global_num_tokens = (
            decode_batch.scheduler_global_num_tokens
            if decode_batch is not None
            else None
        )
        if decode_batch is None or (
            decode_global_num_tokens is not None
            and max(decode_global_num_tokens, default=0) <= 0
        ):
            return remaining_layers

        # The DP metadata gather installs the same token-count vector on both
        # active and IDLE prefill batches. Size segments from its maximum so an
        # IDLE rank (whose local extend_num_tokens is zero) advances by exactly
        # the same layer count as the busiest active rank.
        global_num_tokens = self.split_prefill_batch.scheduler_global_num_tokens
        prefill_num_tokens = (
            max(global_num_tokens, default=0)
            if global_num_tokens is not None
            else self.split_prefill_batch.extend_num_tokens
        )
        if prefill_num_tokens <= 0:
            return remaining_layers

        forward_count = max(
            1,
            self.pdmux_config.split_forward_token_budget // prefill_num_tokens,
        )
        return min(forward_count, remaining_layers)

    def init_pdmux_prefill_plan_limit(
        self: Scheduler, attn_backend: AttentionBackend
    ) -> None:
        """Resolve the backend's hard prefill-plan token limit for admission.

        Without chunked prefill, PDMux cannot split an oversized request, so
        request validation and admission are clamped to the limit. With chunked
        prefill, every forward's aggregate extend tokens are capped by the
        chunk budget instead, which must itself fit one plan.
        """
        self.pdmux_max_prefill_plan_tokens = (
            attn_backend.max_prefill_plan_tokens if self.enable_pdmux else None
        )
        if self.pdmux_max_prefill_plan_tokens is None:
            return
        logger.info(
            "PDMux prefill planner hard limit: %s tokens",
            self.pdmux_max_prefill_plan_tokens,
        )
        if self.chunked_prefill_size is not None:
            SchedulerMultiplexMixin._validate_pdmux_chunked_prefill_size(self)
        else:
            self.max_req_input_len = SchedulerMultiplexMixin._get_max_req_input_len(
                self, self.max_req_input_len
            )

    def _validate_pdmux_chunked_prefill_size(self: Scheduler) -> None:
        # Dynamic chunking could raise a batch's chunk size past the static
        # value, but it is PP-only and PDMux asserts pp_size == 1, so the
        # configured size is the true per-forward bound.
        hard_limit = self.pdmux_max_prefill_plan_tokens
        aligned_limit = hard_limit - hard_limit % self.page_size
        if self.chunked_prefill_size > aligned_limit:
            raise ValueError(
                f"--chunked-prefill-size ({self.chunked_prefill_size}) exceeds "
                f"the attention backend's prefill plan limit of {aligned_limit} "
                f"tokens (page-aligned). Set --chunked-prefill-size to at most "
                f"{aligned_limit}."
            )

    def _get_pdmux_prefill_token_limit(
        self: Scheduler, max_prefill_tokens: int
    ) -> Optional[int]:
        hard_limit = self.pdmux_max_prefill_plan_tokens
        if not (self.enable_pdmux and hard_limit is not None):
            return None
        if self.chunked_prefill_size is not None:
            # The chunk budget caps every forward's aggregate extend tokens
            # below the planner limit (validated at init), so neither request
            # validation nor admission needs the hard clamp.
            return None

        # PrefillAdder accounts input tokens in page-aligned units. Align the
        # backend's raw-token limit down so every accepted request can consume
        # the admission budget instead of remaining in the waiting queue.
        limit = min(max_prefill_tokens, hard_limit)
        return limit - limit % self.page_size

    def _get_prefill_admission_config(
        self: Scheduler, max_prefill_tokens: int
    ) -> tuple[int, bool]:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, max_prefill_tokens
        )
        if effective_limit is None:
            return max_prefill_tokens, False
        return effective_limit, True

    def _get_max_req_input_len(self: Scheduler, max_req_input_len: int) -> int:
        effective_limit = SchedulerMultiplexMixin._get_pdmux_prefill_token_limit(
            self, self.max_prefill_tokens
        )
        if effective_limit is None:
            return max_req_input_len
        # Request validation rejects lengths >= max_req_input_len.
        return min(max_req_input_len, effective_limit + 1)

    def _merge_finished_prefill_batch(
        self: Scheduler,
        prefill_result,
        prefill_stream,
        decode_stream,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        running_batch = SchedulerMultiplexMixin._merge_completed_prefill_batch(
            self,
            batch=self.split_prefill_batch,
            prefill_result=prefill_result,
            running_batch=running_batch,
            all_segments_run=True,
        )
        self.split_prefill_batch = None

        # merge_batch and the chunk stash enqueue tensor work (concatenations,
        # radix-cache inserts, page frees) on the prefill stream. The next loop
        # prepares decode before the stream-group synchronization, so publish
        # the dependency even when nothing was merged — decode may reallocate
        # pages the stash just freed.
        merge_done = prefill_stream.record_event()
        decode_stream.wait_event(merge_done)
        return running_batch

    def _merge_completed_prefill_batch(
        self: Scheduler,
        *,
        batch: ScheduleBatch,
        prefill_result,
        running_batch: ScheduleBatch,
        all_segments_run: bool,
    ) -> ScheduleBatch:
        """process -> chunk stash -> filter -> merge for one completed prefill.

        The single owner of a prefill batch's result handling in both PDMux
        modes: the loop never assigns `last_batch`, so `get_next_batch_to_run`'s
        merge path is not involved and this runs exactly once per work item on
        the normal path. The caller clears its own in-flight state afterwards
        and publishes the merge dependency; an exception here propagates with
        that state still set, which prevents a second finalize -- it is not a
        retry mechanism, and nothing re-enters this for the same batch.
        """
        self.process_batch_result(batch, prefill_result)

        if all_segments_run:
            assert batch.split_prefill_finished
            # The persistent ForwardBatch owns the token inputs, intermediate
            # mHC hidden state and DSpark auxiliary captures across segments.
            # Release it before this ScheduleBatch can become the long-lived
            # decode batch; otherwise those prefill activations remain pinned
            # for the lifetime of the running batch and eventually cause OOM.
            batch.split_forward_batch = None
            batch.split_index = 0
            batch.split_forward_count = 1
            batch.split_prefill_finished = False

        # Mirror get_next_batch_to_run's chunked bookkeeping: a request that
        # only finished a middle chunk must stay out of the decode batch, and
        # its chunk KV must be stashed so the next chunk extends the cached
        # prefix instead of recomputing it.
        chunked_req_to_exclude = set()
        if self.chunked_req is not None:
            chunked_req_to_exclude.add(self.chunked_req)
            # Stash only when this chunk produced new KV beyond what is
            # already cached. A parked chunk (add_chunked_req hybrid-SWA
            # early-return) has nothing new to cache.
            if self.chunked_req.extend_range.end > len(self.chunked_req.prefix_indices):
                # The stash rewrites the request's req_to_token row, which the
                # sparse-prefill scaffolding cache snapshots at segment 0, so
                # it is only legal once every split segment has run. A standard
                # prefill has no segments -- its single forward has already
                # completed by the time this runs.
                self.stash_chunked_request(self.chunked_req)
        if batch.chunked_req is not None:
            chunked_req_to_exclude.add(batch.chunked_req)

        # Mirror get_next_batch_to_run: filter unconditionally, not only when
        # a chunked request is excluded -- the filter also drops requests that
        # FINISHED during prefill, whose KV and req slots process_batch_result
        # already released. Merging them into the decode batch leaves a freed
        # req_pool_idx registered as a live owner until the next filter.
        last_bs = batch.batch_size()
        batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
        if batch.batch_size() < last_bs:
            running_batch.batch_is_full = False

        if not batch.is_empty():
            if running_batch and not running_batch.is_empty():
                running_batch.merge_batch(batch)
            else:
                running_batch = batch

        self.running_batch = running_batch
        return running_batch

    # Pump the HiCache event drain every Nth in-flight iteration instead of
    # every iteration: each pump pays a TP-wide gloo all-reduce plus ack
    # bookkeeping (~10ms/iteration measured on a busy 8-rank host), while the
    # acks it retires are latency-insensitive background accounting -- the
    # transfers themselves are ordered by CUDA events, not by the pump. The
    # tick advances under a rank-consistent condition, so every rank pumps on
    # the same iterations and the pump's collectives stay aligned.
    HICACHE_PUMP_INTERVAL = 16

    def event_loop_pdmux(self: Scheduler):
        """Enter the PD-multiplexing loop for the configured prefill mode."""
        if self.pdmux_standard:
            return self.event_loop_pdmux_standard()
        return self.event_loop_pdmux_layer_split()

    @torch.inference_mode()
    def event_loop_pdmux_layer_split(self: Scheduler):
        """A scheduler loop for pd multiplexing."""
        decode_done = False
        prefill_done = False
        wait_prefill_kernel_done = False
        adjust_stream_group = False
        self._hicache_pump_tick = 0
        stream_idx = get_current_stream_idx()
        stream_group = self.stream_groups[stream_idx]
        prefill_stream = stream_group[0]
        decode_stream = stream_group[1]
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing...")

        while True:
            with torch.cuda.stream(decode_stream):
                self.ingest_requests()
                running_batch = self.running_batch

            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                sm_count = self.sm_counts[stream_idx][0]
                formation_done = None
                # Batch formation is the only other caller of the HiCache pump,
                # and it is skipped for every iteration a split prefill occupies
                # -- which is most of them under the long prefills PDMux exists
                # to overlap. Pump exactly on those iterations: skipping starves
                # HiCache of ack processing and host-lock release for the whole
                # prefill, while doubling up desyncs the pump's collective
                # all-reduces across TP ranks, which deadlocks rather than
                # degrades. The pump's cache actions can free device KV
                # segments and zero full-to-SWA mapping rows on this stream, so
                # it needs the same dependency publication as batch formation
                # -- decode allocates from that free list and the decode graph
                # re-reads the mapping on replay.
                had_inflight_split = self.split_prefill_batch is not None
                if wait_prefill_kernel_done or had_inflight_split:
                    self._hicache_pump_tick += 1
                    if self._hicache_pump_tick % self.HICACHE_PUMP_INTERVAL == 0:
                        # Publish a dependency only when the pump reports it
                        # may have enqueued device work (write_back frees,
                        # storage-queue actions): an event recorded here lands
                        # after the in-flight split segments and serializes
                        # decode behind the whole prefill's completion, so a
                        # host-only ack drain must not pay it.
                        if self.check_hicache_events_if_enabled():
                            formation_done = prefill_stream.record_event()
                if not wait_prefill_kernel_done:
                    created, running_batch = self.update_split_prefill_batch(
                        sm_count, running_batch=running_batch
                    )
                    self.running_batch = running_batch
                    adjust_stream_group = created or adjust_stream_group
                    if not had_inflight_split:
                        # Batch formation enqueued radix-cache and allocator
                        # work (prefix-match concatenations, evictions, KV
                        # allocation) on the prefill stream, rebinding the
                        # free-page list that decode-side allocation slices.
                        # Publish the dependency before decode prepares its
                        # next step. Record ONLY when formation actually ran
                        # (or the pump above enqueued device work): an event
                        # recorded on an idle iteration lands after the
                        # in-flight split segments and serializes every decode
                        # step behind the whole prefill's completion -- a
                        # ~50% TPOT regression under prefill-heavy load, for
                        # no ordering benefit.
                        formation_done = prefill_stream.record_event()

            with torch.cuda.stream(decode_stream):
                if formation_done is not None:
                    decode_stream.wait_event(formation_done)
                running_batch = self.update_running_batch(running_batch)
                self.running_batch = running_batch
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and running_batch.is_empty()
                )
                if running_batch.is_empty() and self.split_prefill_batch is None:
                    self._sched_idled = True
                    self.on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups(
                    running_batch=running_batch,
                    has_prefill=self.split_prefill_batch is not None,
                )
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                # process decode batch
                decode_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                    running_batch if not running_batch.is_empty() else None
                )
                if decode_batch is not None:
                    decode_result = self.run_batch(decode_batch)
                    decode_done = True
                else:
                    decode_done = False
            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                if (
                    self.split_prefill_batch is not None
                    and not wait_prefill_kernel_done
                ):
                    prefill_done = True
                    forward_count = self._get_split_forward_count(decode_batch)
                    next_split_index = min(
                        self.split_prefill_batch.split_index + forward_count,
                        self.model_config.num_hidden_layers,
                    )
                    forward_count = (
                        next_split_index - self.split_prefill_batch.split_index
                    )

                    self.split_prefill_batch.split_forward_count = forward_count
                    prefill_result = self.run_batch(self.split_prefill_batch)
                    if next_split_index == self.model_config.num_hidden_layers:
                        self.split_prefill_batch.split_prefill_finished = True
                        prefill_exe_done = prefill_stream.record_event()
                    self.split_prefill_batch.split_index = next_split_index

                elif wait_prefill_kernel_done:
                    prefill_done = True
                else:
                    prefill_done = False

            with torch.cuda.stream(decode_stream):
                decode_stream.synchronize()
                if decode_done:
                    self.process_batch_result(decode_batch, decode_result)

            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                if prefill_done and self.split_prefill_batch.split_prefill_finished:
                    wait_prefill_kernel_done = True
                    prefill_exe_done_flag = prefill_exe_done.query()
                    flags = (
                        torch.ones(1, device="cpu", dtype=torch.int32)
                        if prefill_exe_done_flag
                        else torch.zeros(1, device="cpu", dtype=torch.int32)
                    )

                    self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM).wait()
                    if flags.item() == get_parallel().tp_size:
                        running_batch = self._merge_finished_prefill_batch(
                            prefill_result,
                            prefill_stream,
                            decode_stream,
                            running_batch,
                        )
                        wait_prefill_kernel_done = False
                        adjust_stream_group = True

    # ------------------------------------------------------------------
    # Standard prefill lane
    # ------------------------------------------------------------------

    def _submit_standard_prefill(self: Scheduler, batch: ScheduleBatch) -> None:
        """Submit one prefill work item on the prefill lane.

        The identity marker is set only around the `run_batch` call: after
        finalize this same object is merged into the running (decode) batch, and
        a marker that outlived the call would make the next decode step take the
        prefill branch.
        """
        previous_marker = self._pdmux_prefill_batch
        self._pdmux_prefill_batch = batch
        submit_start = time.perf_counter()
        try:
            batch_result: GenerationBatchResult = self.run_batch(batch)
        finally:
            self._pdmux_prefill_batch = previous_marker
        self._pdmux_prefill_inflight = PdmuxPrefillInflight(
            batch=batch,
            result=batch_result,
            keep_alive=batch_result.extra_keep_alive_refs,
            submit_s=time.perf_counter() - submit_start,
        )

    def _advance_standard_prefill(
        self: Scheduler,
        *,
        running_batch: ScheduleBatch,
        prefill_stream,
        decode_stream,
        decode_done,
    ) -> tuple[ScheduleBatch, bool]:
        """One iteration of the completion pipeline for the in-flight prefill.

        Step 1 consumes the vote issued last iteration and finalizes when every
        rank saw the result ready. Step 2 samples `copy_done` again and issues
        the next vote. So each iteration performs exactly one allreduce for the
        work item, every rank votes on the same iteration, and detection lags
        actual completion by at most one iteration.

        A rank-local `copy_done.query()` must never decide finalize on its own:
        ranks would finalize on different iterations and their next decode
        batches would diverge, which under TP shows up as a hang rather than a
        wrong answer. Returns the (possibly new) running batch and whether the
        work item was finalized this iteration.
        """
        inflight = self._pdmux_prefill_inflight
        if inflight.done_work is not None:
            wait_start = time.perf_counter()
            inflight.done_work.wait()
            self._pdmux_completion_wait_s += time.perf_counter() - wait_start
            self._pdmux_completion_votes += 1
            ready_ranks = int(inflight.done_flags.item())
            # Drop the consumed vote before finalizing, so an exception below
            # cannot leave a Work whose flags are read again next iteration.
            inflight.done_work = None
            inflight.done_flags = None
            if ready_ranks == self.pdmux_done_group_size:
                # Finalize frees pages, inserts into the radix cache and
                # rewrites req_to_token rows. Order it after the decode work
                # this iteration already enqueued.
                prefill_stream.wait_event(decode_done)
                running_batch = self._finalize_standard_prefill(
                    inflight=inflight,
                    running_batch=running_batch,
                    prefill_stream=prefill_stream,
                    decode_stream=decode_stream,
                )
                return running_batch, True

        # Fresh sample for this iteration's vote. A new tensor each time: the
        # previous one may still be owned by a Work that has not been waited on.
        flags = torch.zeros(1, device="cpu", dtype=torch.int32)
        if inflight.result.copy_done.query():
            flags[0] = 1
        inflight.done_flags = flags
        inflight.done_work = self.tp_cpu_group.allreduce(flags, dist.ReduceOp.SUM)
        return running_batch, False

    def _finalize_standard_prefill(
        self: Scheduler,
        *,
        inflight: PdmuxPrefillInflight,
        running_batch: ScheduleBatch,
        prefill_stream,
        decode_stream,
    ) -> ScheduleBatch:
        """Consume one completed prefill and merge it into the decode batch."""
        # Report the cost of the completion pipeline itself: how long the votes
        # blocked the scheduler thread, and how many iterations elapsed between
        # submitting this work item and detecting it done (>= 1 by construction,
        # since a vote is issued after the submit and consumed the next
        # iteration). Neither number is predicted anywhere; they are measured.
        logger.debug(
            "PDMux prefill finalized: CPU submit %.3f ms, %d completion votes, "
            "%.3f ms spent waiting on them",
            inflight.submit_s * 1e3,
            self._pdmux_completion_votes,
            self._pdmux_completion_wait_s * 1e3,
        )
        self._pdmux_completion_votes = 0
        self._pdmux_completion_wait_s = 0.0

        running_batch = self._merge_completed_prefill_batch(
            batch=inflight.batch,
            prefill_result=inflight.result,
            running_batch=running_batch,
            all_segments_run=False,
        )
        # Only now may the keep-alive bundle be dropped: the result's CPU copies
        # have been consumed and the batch has been merged.
        self._pdmux_prefill_inflight = None

        # The merge, the chunk stash and the result processing enqueued tensor
        # work (radix inserts, page frees, req_to_token rewrites) on the prefill
        # stream. Publish it before decode prepares its next step: decode may
        # reallocate pages this just freed.
        merge_done = prefill_stream.record_event()
        decode_stream.wait_event(merge_done)
        return running_batch

    def _pump_hicache_events_inflight(self: Scheduler) -> None:
        """Drain HiCache events while a standard prefill occupies the lane.

        Batch formation is the only other pump caller and it is skipped for
        every iteration a prefill is in flight, so without this HiCache would be
        starved of ack processing and host-lock release for the whole prefill.
        The tick advances under a rank-consistent condition, so every rank pumps
        on the same iterations and the pump's gloo all-reduces stay aligned;
        doubling up or skipping on one rank deadlocks rather than degrades.

        Runs in the decode lane's stream context, unlike the layer_split loop:
        the standard lane submits the whole prefill at once, so any event
        recorded on the prefill stream while it is in flight would land after
        the entire forward and serialize every decode step behind it. The pump's
        device work (write-back frees, storage-queue actions) therefore lands on
        the decode stream and is covered by that iteration's decode_done.
        """
        self._hicache_pump_tick += 1
        if self._hicache_pump_tick % self.HICACHE_PUMP_INTERVAL == 0:
            self.check_hicache_events_if_enabled()

    @torch.inference_mode()
    def event_loop_pdmux_standard(self: Scheduler):
        """PDMux loop that submits each prefill as one standard EXTEND.

        One CPU thread, two lanes. The prefill lane submits a whole work item
        and never blocks on it; the decode lane keeps its existing synchronous
        shape. Cross-lane ordering is published with four events:

            E0 input_done      D -> P   input handling's device work
            E1 formation_done  P -> D   batch formation and its HiCache pump
            E2 merge_done      P -> D   finalize's frees, inserts and rewrites
            E3 decode_done     D -> P   decode result processing, retract, pump

        Scheduling invariant: between submitting a prefill forward and its
        copy_done being ready, the prefill stream records no event that the
        decode lane waits on. Without it every decode step would queue behind
        the whole prefill. It does not say decode never waits on prefill --
        formation (E1) and finalize (E2) both publish dependencies decode honors.
        """
        self._hicache_pump_tick = 0
        self._pdmux_completion_votes = 0
        self._pdmux_completion_wait_s = 0.0
        # The rank count of the group the completion vote reduces over. Read
        # here rather than in init_pdmux: tp_cpu_group is assigned later in the
        # scheduler's __init__.
        self.pdmux_done_group_size = dist.get_world_size(group=self.tp_cpu_group)

        stream_idx = get_current_stream_idx()
        prefill_stream, decode_stream = self.stream_groups[stream_idx]
        self.pdmux_prefill_stream = prefill_stream
        adjust_stream_group = False
        decode_done = None
        torch.cuda.empty_cache()

        logger.debug("Starting event loop for pd multiplexing (standard prefill)...")

        while True:
            with torch.cuda.stream(decode_stream):
                recv_reqs = self.request_receiver.recv_requests()
                self.process_input_requests(recv_reqs)
                running_batch = self.running_batch
                # E0: prefetch's prefix matching concatenates on this stream.
                input_done = decode_stream.record_event()

            inflight = self._pdmux_prefill_inflight

            formation_done = None
            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                if inflight is None:
                    prefill_stream.wait_event(input_done)
                    if decode_done is not None:
                        prefill_stream.wait_event(decode_done)
                    # No prefill forward is in flight, which matches the normal
                    # loop's safe point for tearing down an aborted chunked
                    # request before its next chunk is formed.
                    self.process_pending_chunked_abort()
                    prefill_plan = self.get_new_batch_prefill(running_batch)
                    running_batch = prefill_plan.running_batch
                    self.running_batch = running_batch
                    new_batch = prefill_plan.batch_to_run
                    new_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                        new_batch
                    )
                    if new_batch is not None:
                        self._pdmux_prefill_pending = new_batch
                        adjust_stream_group = True
                    # E1: record even when formation produced no batch. It ran
                    # the HiCache pump, and admission's host-to-device load-back
                    # allocates and can evict, both on this stream.
                    formation_done = prefill_stream.record_event()

            with torch.cuda.stream(decode_stream):
                if formation_done is not None:
                    decode_stream.wait_event(formation_done)
                if inflight is not None:
                    self._pump_hicache_events_inflight()
                running_batch = self.update_running_batch(running_batch)
                self.running_batch = running_batch
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and running_batch.is_empty()
                )
                if running_batch.is_empty() and not self._extra_inflight_batches():
                    self.on_idle()

            # Never switch with a forward in flight: the switch drains both
            # streams and rebinds the decode attention backends. A pending
            # prefill is fine -- it is submitted below, on the new lane.
            if adjust_stream_group and self._pdmux_prefill_inflight is None:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups(
                    running_batch=running_batch,
                    has_prefill=self._pdmux_prefill_pending is not None,
                )
                prefill_stream, decode_stream = stream_group
                self.pdmux_prefill_stream = prefill_stream
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: "
                    f"{self.sm_counts[stream_idx][0]}, decode sm: "
                    f"{self.sm_counts[stream_idx][1]}"
                )

            decode_result = None
            with torch.cuda.stream(decode_stream):
                decode_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                    running_batch
                    if running_batch is not None and not running_batch.is_empty()
                    else None
                )
                if decode_batch is not None:
                    decode_result = self.run_batch(decode_batch)

            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                if self._pdmux_prefill_pending is not None:
                    self._submit_standard_prefill(self._pdmux_prefill_pending)
                    self._pdmux_prefill_pending = None

            with torch.cuda.stream(decode_stream):
                decode_stream.synchronize()
                if decode_result is not None:
                    self.process_batch_result(decode_batch, decode_result)
                # E3: covers this iteration's decode result handling, the
                # retract/free inside update_running_batch, and the pump above.
                decode_done = decode_stream.record_event()

            with torch.cuda.stream(prefill_stream), pdmux_prefill_tp_group():
                if self._pdmux_prefill_inflight is not None:
                    running_batch, merged = self._advance_standard_prefill(
                        running_batch=running_batch,
                        prefill_stream=prefill_stream,
                        decode_stream=decode_stream,
                        decode_done=decode_done,
                    )
                    if merged:
                        adjust_stream_group = True
