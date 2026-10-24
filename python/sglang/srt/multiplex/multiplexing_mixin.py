"""
Mixin class providing multiplexing scheduling logic
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.cuda.streams import ExternalStream

from sglang.srt.distributed.parallel_state import pdmux_prefill_tp_group
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
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
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerMultiplexMixin:
    def init_pdmux(self: Scheduler):
        # The current layerwise prefill batch.
        self.split_prefill_batch: Optional[ScheduleBatch] = None
        self._pdmux_pending_chunk_stash_req = None

        # for pd_multiplexing, Init stream_groups, exclude normal stream for prefill only and decode only
        self.pdmux_config = load_pdmux_config(get_disagg().pdmux_config_path)
        initialize_stream_groups(get_device().gpu_id, self.pdmux_config)
        self.stream_groups = get_stream_groups()
        self.sm_counts = get_sm_counts()
        self.real_sm_group_num = len(self.stream_groups)
        logger.info(
            f"PD-Multiplexing enabled with {self.real_sm_group_num} stream groups, sm_counts (prefill_sm, decode_sm): {self.sm_counts}"
        )

    @staticmethod
    def _global_decode_size(decode_batch: Optional[ScheduleBatch]) -> int:
        if decode_batch is None:
            return 0
        # The DP sync creates IDLE participants and publishes the same vector
        # on every rank. Local batch sizes cannot select a shared stream group.
        global_num_tokens = getattr(decode_batch, "scheduler_global_num_tokens", None)
        if global_num_tokens is not None:
            return max(global_num_tokens, default=0)
        return 0 if decode_batch.is_empty() else decode_batch.batch_size()

    # TODO(jason-fxz): This is a temporary demo
    def adjust_stream_groups(
        self: Scheduler, decode_batch: Optional[ScheduleBatch]
    ) -> tuple[int, tuple[ExternalStream, ExternalStream]]:
        """Pick the stream group for the next step."""
        decode_bs = SchedulerMultiplexMixin._global_decode_size(decode_batch)
        if decode_bs > 0 and self.split_prefill_batch is not None:
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
        elif decode_bs > 0:
            set_current_stream_idx(self.real_sm_group_num - 1)
        else:
            set_current_stream_idx(0)

        stream_idx = get_current_stream_idx()

        self.tp_worker.model_runner.update_decode_attn_backend(stream_idx)
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

        if SchedulerMultiplexMixin._resume_pending_chunk_stash(self):
            # add new request
            prefill_plan = self.get_new_batch_prefill(running_batch)
            batch = prefill_plan.batch_to_run
            running_batch = prefill_plan.running_batch
        else:
            batch = None
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
        if self.pdmux_config.max_split_forward_layers:
            forward_count = min(
                forward_count, self.pdmux_config.max_split_forward_layers
            )
        return min(forward_count, remaining_layers)

    def _merge_finished_prefill_batch(
        self: Scheduler,
        prefill_result,
        prefill_stream,
        decode_stream,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        running_batch = self._merge_completed_prefill_batch(
            batch=self.split_prefill_batch,
            prefill_result=prefill_result,
            running_batch=running_batch,
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
    ) -> ScheduleBatch:
        """process -> chunk stash -> filter -> merge for one completed prefill.

        The PDMux loop never assigns `last_batch`, so its normal merge path
        does not handle the prefill result.
        """
        self.process_batch_result(batch, prefill_result)

        assert batch.split_prefill_finished
        # The persistent ForwardBatch owns token inputs and hidden state across
        # segments. Release it before this becomes the long-lived decode batch.
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
                # Stash rewrites req_to_token, which the sparse-prefill
                # scaffolding snapshots at segment 0. All segments have now run.
                SchedulerMultiplexMixin._stash_chunked_request_when_ready(
                    self, self.chunked_req
                )
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

    def _mamba_stash_slot_ready(self: Scheduler, req) -> bool:
        """Check the extra state slot a middle chunk needs before caching it.

        Keep every attention-TP rank for this request on the same path:
        eviction and radix insertion both contain collectives. DP-attention
        shards can have different chunked requests, so the full TP group is
        too broad. Check capacity without allocating a temporary GPU slot.
        """
        pool = getattr(self, "req_to_token_pool", None)
        allocator = getattr(pool, "mamba_allocator", None)
        if allocator is None or not self.tree_cache.supports_mamba():
            return True

        def ready_on_all_ranks() -> bool:
            flag = torch.tensor(
                int(
                    getattr(getattr(req, "kv", None), "mamba_cache_reserve_slot", None)
                    is not None
                    or allocator.schedulable_available_size() >= 1
                ),
                device="cpu",
                dtype=torch.int32,
            )
            self.attn_tp_cpu_group.allreduce(flag, dist.ReduceOp.MIN).wait()
            return bool(flag.item())

        if ready_on_all_ranks():
            return True
        # The normal loop drains HiCache just before stashing a chunk. PDMux
        # merges the result earlier, so retire completed write-through acks
        # here before asking the cache to evict an otherwise protected state.
        self._process_hicache_events()
        # All ranks must enter this rank-consensus eviction, even if only one
        # rank ran out of slots. A backup still in flight may keep the state
        # non-evictable; in that case wait for its ack or decode completion.
        self.tree_cache.evict_for_alloc(EvictParams(num_tokens=0, mamba_num=1))
        return ready_on_all_ranks()

    def _stash_chunked_request_when_ready(self: Scheduler, req) -> bool:
        if not SchedulerMultiplexMixin._mamba_stash_slot_ready(self, req):
            self._pdmux_pending_chunk_stash_req = req
            now = time.monotonic()
            if now - getattr(self, "_last_mamba_stash_warning_ts", 0.0) >= 10.0:
                logger.warning(
                    "Deferring chunked prefill: no Mamba state slot is available "
                    "to stash the completed chunk"
                )
                self._last_mamba_stash_warning_ts = now
            return False
        self.stash_chunked_request(req)
        self._pdmux_pending_chunk_stash_req = None
        return True

    def _resume_pending_chunk_stash(self: Scheduler) -> bool:
        req = getattr(self, "_pdmux_pending_chunk_stash_req", None)
        if req is None:
            return True
        if self.chunked_req is not req:
            # An abort at this between-chunks safe point released the request.
            self._pdmux_pending_chunk_stash_req = None
            return True
        return SchedulerMultiplexMixin._stash_chunked_request_when_ready(self, req)

    # Pump the HiCache event drain every Nth in-flight iteration instead of
    # every iteration: each pump pays a TP-wide gloo all-reduce plus ack
    # bookkeeping (~10ms/iteration measured on a busy 8-rank host), while the
    # acks it retires are latency-insensitive background accounting -- the
    # transfers themselves are ordered by CUDA events, not by the pump. The
    # tick advances under a rank-consistent condition, so every rank pumps on
    # the same iterations and the pump's collectives stay aligned.
    HICACHE_PUMP_INTERVAL = 16

    @torch.inference_mode()
    def event_loop_pdmux(self: Scheduler):
        """Run PDMux with layerwise prefill."""
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
                    if not had_inflight_split:
                        # The normal scheduler drains HiCache before every
                        # prefill admission. Without this, consecutive short
                        # chunks can keep their write-through Mamba states
                        # protected until the active pool is exhausted.
                        self._process_hicache_events()
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
                # Gather DP metadata before selecting streams: an empty local
                # decode batch may still have active peers and run an IDLE
                # participant. This is the existing gather, moved ahead of the
                # stream switch so every rank makes the same decision.
                decode_batch = self.dp_attn_adapter.maybe_prepare_mlp_sync_batch(
                    running_batch if not running_batch.is_empty() else None
                )
                has_decode = (
                    SchedulerMultiplexMixin._global_decode_size(decode_batch) > 0
                )
                adjust_stream_group = adjust_stream_group or (
                    stream_idx > 0 and not has_decode
                )
                if not has_decode and self.split_prefill_batch is None:
                    self.on_idle()

            if adjust_stream_group:
                prefill_stream.synchronize()
                decode_stream.synchronize()
                stream_idx, stream_group = self.adjust_stream_groups(
                    decode_batch=decode_batch,
                )
                prefill_stream = stream_group[0]
                decode_stream = stream_group[1]
                adjust_stream_group = False
                logger.debug(
                    f"Adjusting stream groups: {stream_idx}, prefill sm: {self.sm_counts[stream_idx][0]}, decode sm: {self.sm_counts[stream_idx][1]}"
                )

            with torch.cuda.stream(decode_stream):
                # process decode batch
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
