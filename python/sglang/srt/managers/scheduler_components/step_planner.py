"""Decides what each scheduler step runs: the prefill admission, the decode
selection and retraction, and the retract/reset half of pause and flush."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.min_free_slots_delayer import create_min_free_slots_delayer
from sglang.srt.managers.prefill_delayer import (
    PrefillDelayerSinglePassExecutor,
    RecentPrefillBatchSizeTracker,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    check_decode_mem,
    new_tokens_required_next_decode,
    retract_all,
    retract_decode,
)
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    priority_preemption_enabled,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import PrefillStats
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_lora,
    get_memory,
    get_parallel,
    get_schedule,
    get_serving,
    get_spec,
    hicache_storage_enabled,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.constrained.grammar_manager import GrammarManager
    from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
        DecodeKVCacheOffloadManager,
    )
    from sglang.srt.managers.prefill_delayer import PrefillDelayer
    from sglang.srt.managers.request_lists import CoreLists
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.schedule_policy import SchedulePolicy
    from sglang.srt.managers.scheduler_components.bundles import (
        KvCapacity,
        KvResources,
        RequestLimits,
        Workers,
    )
    from sglang.srt.managers.scheduler_components.dynamic_chunk_sizer import (
        DynamicChunkSizer,
    )
    from sglang.srt.managers.scheduler_components.engine_state import (
        EngineFlags,
        LaunchCounters,
    )
    from sglang.srt.managers.scheduler_components.lora_admission import (
        LoraAdmission,
    )
    from sglang.srt.managers.scheduler_components.metrics_reporter import MetricsSink
    from sglang.srt.managers.scheduler_components.mode_owners import ModeOwners
    from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
    from sglang.srt.managers.scheduler_components.pending_steps import PendingSteps
    from sglang.srt.managers.scheduler_components.pool_stats_observer import (
        SchedulerPoolStatsObserver,
    )
    from sglang.srt.managers.scheduler_components.request_intake import (
        RequestIntake,
    )
    from sglang.srt.managers.scheduler_components.request_registry import (
        RequestRegistry,
    )
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.managers.scheduler_components.step_builder import StepBuilder
logger = logging.getLogger(__name__)

# Test retract decode for debugging purposes
TEST_RETRACT = envs.SGLANG_TEST_RETRACT.get()
TEST_RETRACT_INTERVAL = envs.SGLANG_TEST_RETRACT_INTERVAL.get()
TEST_RETRACT_NO_PREFILL_BS = envs.SGLANG_TEST_RETRACT_NO_PREFILL_BS.get()


@dataclass(frozen=True, slots=True)
class _NoKvRoom:
    """A pass that stopped on a waiting request finding no KV room, and the
    state it saw. Later passes are skipped while no request has left the slot
    (its running requests still lead its list; admitted ones join behind
    them), no request slot has been freed, the request that found no room is
    still the queue head, under priority scheduling, which sorts a
    higher-priority arrival first, nothing has arrived, and the KV reserved
    for a mixed step's decode part has not shrunk (see admission_skip_reason)."""

    running: Tuple[Req, ...]
    num_free_req_slots: int
    head: Req
    tail: Optional[Req]
    decode_kv_tokens: int

    def holds(
        self,
        running_reqs: List[Req],
        num_free_req_slots: int,
        waiting_queue: List[Req],
        priority_scheduling: bool,
    ) -> bool:
        return (
            tuple(running_reqs[: len(self.running)]) == self.running
            and num_free_req_slots <= self.num_free_req_slots
            and waiting_queue[0] is self.head
            and (not priority_scheduling or waiting_queue[-1] is self.tail)
        )


@dataclass(frozen=True, slots=True)
class DecodePart:
    """The decoding requests a mixed step runs as one-token extend rows after
    its prefill rows. It is reserved before prefill admission as two
    quantities: its forward tokens, one per row, and the KV its rows allocate
    (page- and spec-aware, as the decode step's own check counts it)."""

    reqs: List[Req]
    kv_tokens: int

    @property
    def num_tokens(self) -> int:
        return len(self.reqs)


@dataclass(frozen=True, slots=True)
class PrefillPart:
    """The prefill rows of a step: the admitted requests, in row order, and
    the request left mid-chunk after it."""

    reqs: List[Req]
    chunked_req: Optional[Req]


class CompositionPolicy:
    """How a mixed step composes its two parts: the decode part that rides,
    reserved ahead of prefill admission, given the chunk budget (None without
    chunked prefill). Admission then sees only what the reservation leaves."""

    def decode_part_to_reserve(
        self, decode_part: DecodePart, chunk_budget: Optional[int]
    ) -> Optional[DecodePart]:
        raise NotImplementedError


class DecodeFirst(CompositionPolicy):
    """The whole decode part rides; prefill gets what its rows leave of the
    budgets, possibly nothing, and the step is then a plain decode step."""

    def decode_part_to_reserve(
        self, decode_part: DecodePart, chunk_budget: Optional[int]
    ) -> Optional[DecodePart]:
        return decode_part


class StepPlanner:
    """Decides what a step runs: which waiting requests a prefill admits, which
    decoding requests a decode step takes, and which it retracts when the KV
    pool runs short. StepBuilder builds the step from the requests picked."""

    def __init__(
        self,
        *,
        core_lists: CoreLists,
        pending: PendingSteps,
        flags: EngineFlags,
        counters: LaunchCounters,
        kv: KvResources,
        capacity: KvCapacity,
        policy: SchedulePolicy,
        new_token_ratio_tracker: NewTokenRatioTracker,
        pool_stats_observer: SchedulerPoolStatsObserver,
        lora_admission: LoraAdmission,
        request_registry: RequestRegistry,
        beam_coordinator: BeamCoordinator,
        step_builder: StepBuilder,
        request_retirer: RequestRetirer,
        grammar_manager: GrammarManager,
        metrics_sink: MetricsSink,
        workers: Workers,
        decode_offload_manager: Optional[DecodeKVCacheOffloadManager],
        mode_owners: ModeOwners,
        request_intake: RequestIntake,
        prefill_delayer: Optional[PrefillDelayer],
        dynamic_chunk_sizer: Optional[DynamicChunkSizer],
        limits: RequestLimits,
        prefill_adder_class: type[PrefillAdder],
    ):
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        prefill_decode_interval = get_schedule().prefill_decode_interval or 0
        enable_priority_preemption = priority_preemption_enabled()
        prefill_bs_tracker = RecentPrefillBatchSizeTracker(
            window_size=envs.SGLANG_PREFILL_DELAYER_MAX_PREFILL_BS_WINDOW_SIZE.get()
        )
        # DFlash auto-enables the legacy formula; other workloads opt in via
        # --min-free-slots-delay. Built independently of the prefill delayer.
        min_free_slots_delayer = create_min_free_slots_delayer(
            get_schedule().min_free_slots_delay,
            capacity.max_running_requests,
            is_dflash_family=spec_algorithm.is_dflash_family(),
        )
        self.core_lists = core_lists
        self.pending = pending
        self.flags = flags
        self.counters = counters
        self.req_to_token_pool = kv.req_to_token_pool
        self.token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        self.tree_cache = kv.tree_cache
        self.max_running_requests = capacity.max_running_requests
        self.policy = policy
        self.new_token_ratio_tracker = new_token_ratio_tracker
        self.pool_stats_observer = pool_stats_observer
        self.lora_admission = lora_admission
        self.beam_coordinator = beam_coordinator
        self.step_builder = step_builder
        self.hisparse_coordinator = mode_owners.hisparse_coordinator
        self.request_retirer = request_retirer
        self.grammar_manager = grammar_manager
        self.metrics_sink = metrics_sink
        self.tp_worker = workers.tp_worker
        self.draft_worker = workers.draft_worker
        self.decode_offload_manager = decode_offload_manager
        self.request_registry = request_registry
        # A PD role switch replaces the owners in place.
        self.mode_owners = mode_owners
        self.request_intake = request_intake
        self.prefill_adder_class = prefill_adder_class
        self.prefill_delayer = prefill_delayer
        self.prefill_bs_tracker = prefill_bs_tracker
        self.min_free_slots_delayer = min_free_slots_delayer
        self.dynamic_chunk_sizer = dynamic_chunk_sizer
        self.spec_algorithm = spec_algorithm
        self.is_hybrid_swa = capacity.is_hybrid_swa
        self.require_mlp_sync = require_mlp_sync()
        self.chunked_prefill_size = limits.chunked_prefill_size
        self.is_mixed_chunk = limits.is_mixed_chunk
        self.prefill_decode_interval = prefill_decode_interval
        self.prefill_decode_interval_min_bs = (
            get_schedule().prefill_decode_interval_min_bs
        )
        self.max_prefill_tokens = limits.max_prefill_tokens
        self.enable_priority_preemption = enable_priority_preemption
        self.truncation_align_size = limits.truncation_align_size
        # The largest recent prefill batch, which the prefill delayer reads.
        self.max_prefill_bs: int = 0
        # Steps left to decode before the next prefill may run.
        self._prefill_decode_interval_remaining = 0
        # Per slot, the inputs of its last pass that found no KV room.
        self._no_kv_room: Dict[Optional[int], _NoKvRoom] = {}
        # How a mixed step's decode part shares the step with prefill.
        self.composition_policy: CompositionPolicy = DecodeFirst()

    def should_defer_prefill(self, slot: ScheduleSlot) -> bool:
        if self._prefill_decode_interval_remaining == 0:
            return False

        # A small running batch admits prefill every round so it can grow;
        # the interval only protects decode once the batch is large enough.
        running_reqs = slot.filter_decoding(self.core_lists.decoding)
        if len(running_reqs) < self.prefill_decode_interval_min_bs:
            self._prefill_decode_interval_remaining = 0
            return False

        self._prefill_decode_interval_remaining -= 1
        return True

    def arm_prefill_decode_interval(self, batch: Optional[ScheduleBatch]) -> None:
        if self.prefill_decode_interval == 0 or batch is None:
            return

        # DP attention synchronizes this flag across ranks. This keeps every
        # rank on the same prefill/decode cadence even when only one rank has
        # local prefill work. Non-DP scheduling can use the local mode directly.
        is_extend = (
            batch.is_extend_in_batch
            if self.require_mlp_sync
            else batch.forward_mode.is_extend()
        )
        if is_extend:
            self._prefill_decode_interval_remaining = self.prefill_decode_interval

    def admission_skip_reason(
        self, slot: ScheduleSlot, running_reqs: List[Req]
    ) -> Optional[str]:
        """Why a prefill admission for slot next to running_reqs would admit no
        request now, or None to run it."""
        if self.core_lists.chunked_req is not None:
            # The chunked request continues regardless: under PP it can start in
            # one slot and end in another, so a slot's request cap is not strict.
            return None
        if len(self.core_lists.waiting_queue) == 0:
            return "no waiting request"
        if not self.enable_priority_preemption and not self._room_for_one(running_reqs):
            # Every waiting request meets the request cap and the row budget
            # before it is added; with priority preemption it may free a
            # running request's row.
            return "no request slot"
        no_kv_room = self._no_kv_room.get(slot.index)
        if no_kv_room is not None:
            if no_kv_room.holds(
                running_reqs,
                self.req_to_token_pool.available_size(),
                self.core_lists.waiting_queue,
                self.policy.enable_priority_scheduling,
            ) and (
                # The decode part's KV need moves with page boundaries, with
                # requests finishing in flight and with whether it rides; a
                # smaller one may leave the head room.
                no_kv_room.decode_kv_tokens == 0
                or self._decode_kv_tokens(slot) >= no_kv_room.decode_kv_tokens
            ):
                return "no KV room"
            del self._no_kv_room[slot.index]
        return None

    def _room_for_one(self, running_reqs: List[Req]) -> bool:
        """Whether one more request fits next to running_reqs. A waiting
        request that already holds a row (a session borrow) needs no free
        one."""
        num_reqs = len(running_reqs) + 1
        if self.beam_coordinator.has_room(num_reqs, 1):
            return True
        # Only a streaming session lends a waiting request its row (at the
        # request's match); without them no waiting request holds one.
        return (
            get_serving().enable_streaming_session
            and self.beam_coordinator.has_room(num_reqs, 0)
            and any(req.kv.holds_kv for req in self.core_lists.waiting_queue)
        )

    def get_new_batch_prefill(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        prefill_delayer_single_pass = None
        if self.prefill_delayer:
            # Get max usage across all pools for prefill delay decision
            max_pool_usage = (
                self.pool_stats_observer.get_pool_stats().get_max_pool_usage()
            )
            prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
                self.prefill_delayer, token_usage=max_pool_usage
            )

        ret = self._get_new_batch_prefill_raw(
            slot,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
        )

        if self.prefill_delayer:
            observed_prefill_bs = prefill_delayer_single_pass.finalize(
                actual_prefill_bs=ret.batch_size() if ret is not None else 0
            )
            if observed_prefill_bs > 0:
                self.max_prefill_bs = self.prefill_bs_tracker.observe_attempt(
                    observed_prefill_bs
                )

        return ret

    def _get_new_batch_prefill_raw(
        self,
        slot: ScheduleSlot,
        prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor],
    ) -> Optional[ScheduleBatch]:
        # Under PP only the scheduled slot's requests: a slot's batch size
        # is capped, and preemption must not pick a request in flight.
        running_reqs = slot.filter_decoding(self.core_lists.decoding)
        if self.admission_skip_reason(slot, running_reqs) is not None:
            return None

        running_bs = len(running_reqs)
        # Skipped during a chunked prefill: that pass must proceed regardless.
        if (
            self.min_free_slots_delayer is not None
            and self.core_lists.chunked_req is None
            and self.min_free_slots_delayer.should_delay(
                running_bs=running_bs,
                num_allocatable_reqs=min(
                    get_parallel().pp_max_micro_batch_size - running_bs,
                    self.beam_coordinator.row_budget(),
                ),
            )
        ):
            return None

        # Get priority queue
        disagg_prefill = self.mode_owners.disagg_prefill
        self.policy.calc_priority(
            self.core_lists.waiting_queue,
            running_reqs,
            processed_tokens=self.counters.processed_tokens_counter,
            bootstrap_queue=(
                disagg_prefill.bootstrap_queue.queue
                if disagg_prefill is not None
                else []
            ),
        )

        if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
            # If we are testing retraction and the decoding batch size exceeds
            # TEST_RETRACT_NO_PREFILL_BS, we skip the prefill to keep the requests
            # in the waiting queue.
            return None

        # Determine chunked_prefill_size for this batch
        chunked_prefill_size = self.chunked_prefill_size
        if (
            self.core_lists.chunked_req is not None
            and self.dynamic_chunk_sizer is not None
        ):
            history_len = self.core_lists.chunked_req.prefix_len
            dynamic_size = self.dynamic_chunk_sizer.predict(history_len)
            if dynamic_size is not None:
                chunked_prefill_size = dynamic_size

        # Prefill policy
        # Get BLOCK_M from the backend for tile-budget admission logic
        attn_backend = self.tp_worker.model_runner.attn_backend
        if hasattr(attn_backend, "extend_attention_block_m"):
            prefill_tile_block_m = attn_backend.extend_attention_block_m
        else:
            prefill_tile_block_m = 64  # Fallback for non-Triton backends

        # A mixed step reserves its decode part before prefill admission.
        decode_part = self._plan_decode_part(slot, chunked_prefill_size)
        adder = self.prefill_adder_class(
            get_schedule().page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_reqs,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            chunked_prefill_size,
            decode_part.num_tokens if decode_part is not None else 0,
            get_schedule().priority_scheduling_preemption_threshold,
            max_prefill_bs=int(self.max_prefill_bs),
            max_running_requests=self.max_running_requests,
            prefill_max_requests=get_schedule().prefill_max_requests,
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            waiting_queue_len=len(self.core_lists.waiting_queue),
            prefill_tile_block_m=prefill_tile_block_m,
            request_retirer=self.request_retirer,
            mixed_decode_kv_tokens=(
                decode_part.kv_tokens if decode_part is not None else 0
            ),
        )

        # The chunked request after this step; the launch moves it into the
        # chunked holder (see join_launched_step).
        chunked_req = self.core_lists.chunked_req
        if chunked_req is not None:
            chunked_req.init_next_round_input()
            adder.chunked_req_limit = self.policy.shortest_prefill_chunk_limit(
                chunked_req,
                self.core_lists.waiting_queue,
                adder.rem_chunk_tokens or 0,
                get_schedule().page_size,
            )
            chunked_req = adder.add_chunked_req(chunked_req)

        if get_lora().enable_lora:
            running_loras = self.request_registry.lora_ids_in_use()
            # Account for LoRAs that are already loaded in the adder, such as chunked requests
            running_loras.update(req.lora_id for req in adder.can_run_list)

            if self.lora_admission.lora_drainer:
                self.lora_admission.lora_drainer.update_draining_state(
                    self.core_lists.waiting_queue,
                    self.core_lists.decoding,
                )

        mamba_allocator = getattr(self.req_to_token_pool, "mamba_allocator", None)
        if mamba_allocator is not None:
            mamba_allocator.alloc_group_begin(len(self.core_lists.waiting_queue))
        buffer_pipeline = self.tree_cache.buffer_pipeline
        # Once set, every later request must preempt to be admitted.
        is_full = False
        no_kv_room = False
        # The new request rows the admitted requests take (see num_rows).
        num_admitted_rows = sum(map(self.beam_coordinator.num_rows, adder.can_run_list))
        # Get requests from the waiting queue to a new prefill batch
        for req in self.core_lists.waiting_queue:
            # A spent pass budget admits no one, even before the first
            # candidate: the resumed chunk or the decode part can spend it.
            if adder.budget_state() == AddReqResult.OTHER:
                break
            if get_lora().enable_lora and not self.lora_admission.can_schedule_lora_req(
                req, running_loras
            ):
                continue

            # A forward batch runs one pooling mode, so setwise readout requests
            # (token_indices_to_pool) cannot share a batch with last-token ones.
            # Admit only the first request's mode; the other stays queued.
            if adder.can_run_list and (
                (req.token_indices_to_pool is not None)
                != (adder.can_run_list[0].token_indices_to_pool is not None)
            ):
                continue

            # Preemption below shrinks the requests still decoding and frees
            # their rows.
            running_bs = len(running_reqs) - len(adder.preempt_list)
            rows_with_req = num_admitted_rows + self.beam_coordinator.num_rows(req)
            if not self.beam_coordinator.has_room(
                running_bs + len(adder.can_run_list) + 1, rows_with_req
            ):
                is_full = True

            if is_full and not self.enable_priority_preemption:
                break

            if hicache_storage_enabled() or get_memory().enable_lmcache:
                prefetch_done = self.tree_cache.check_prefetch_progress(
                    req.cache_request_handle
                )
                req.time_stats.observe_storage_prefetch_ready(prefetch_done)
                if not prefetch_done:
                    # skip staging requests that are ongoing prefetch
                    continue
                # Pop the L3-loaded span. Unified cache exposes its absolute
                # start so cache-mode L2/L3 attribution survives L3-tail eviction.
                loaded_tokens, loaded_start = self.tree_cache.pop_prefetch_loaded_span(
                    req.cache_request_handle
                )
                if loaded_tokens > 0:
                    req.storage_hit_length = loaded_tokens
                    req.storage_hit_start = loaded_start
                    # Cache-mode host memory is a resident L2 tier. Buffer mode
                    # marks the staged span below once it is surfaced.
                    req.host_hit_is_storage = False

            req.init_next_round_input(self.tree_cache)
            # Preemption sizes the request by this match: its fill ids and
            # cached prefix are stale or empty before it.
            if is_full and not adder.preempt_to_schedule(req):
                self._revert_match(req)
                break
            if is_full and not self.beam_coordinator.has_room(
                len(running_reqs)
                - len(adder.preempt_list)
                + len(adder.can_run_list)
                + 1,
                rows_with_req,
            ):
                # A preempted request's row is held until the pending step
                # that carries it commits; the request waits for it there.
                self._revert_match(req)
                break
            if hicache_storage_enabled() and (
                self.request_intake.prefetch_after_device_hit_loss(req)
            ):
                self._revert_match(req)
                continue
            if hicache_storage_enabled() and buffer_pipeline is not None:
                staging_ready = buffer_pipeline.prepare_staged_prefetch(req)
                req.time_stats.observe_storage_staging_ready(staging_ready)
                if not staging_ready:
                    self._revert_match(req)
                    continue
            res = adder.add_one_req(
                req,
                has_chunked_req=(chunked_req is not None),
                truncation_align_size=self.truncation_align_size,
            )

            if get_lora().enable_lora:
                running_loras.add(req.lora_id)
            added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
            if added:
                num_admitted_rows = rows_with_req

            if res != AddReqResult.CONTINUE:
                no_kv_room = res == AddReqResult.NO_TOKEN
                if not added:
                    self._revert_match(req)
                break

        if mamba_allocator is not None:
            mamba_allocator.alloc_group_end()

        # Remember a pass that stopped on a request finding no KV room, after
        # admitting others or none: until a request leaves, a request slot
        # frees or the queue head changes (or, under priority scheduling, a
        # request arrives), later passes would stop there too. Estimates that
        # drift between those events (the new token ratio, prefix matches,
        # the evictable tree size) do not reopen it. With nothing running no
        # release is coming, so nothing is remembered. Hybrid SWA frees KV as
        # windows slide and HiSparse admits as host copies land; both run
        # every pass.
        if (
            no_kv_room
            and running_reqs
            and not adder.preempt_list
            and not self.is_hybrid_swa
            and self.hisparse_coordinator is None
        ):
            # As the launch leaves them: the admitted requests out of the
            # queue, and the new rows they take (see num_rows) gone from the
            # free ones. An admission that used up the budget may have taken
            # the last waiting request; the next pass then skips on the empty
            # queue.
            admitted = set(adder.can_run_list)
            waiting_queue = self.core_lists.waiting_queue
            head = next((r for r in waiting_queue if r not in admitted), None)
            if head is not None:
                num_new_rows = sum(
                    map(self.beam_coordinator.num_rows, adder.can_run_list)
                )
                self._no_kv_room[slot.index] = _NoKvRoom(
                    running=tuple(running_reqs),
                    num_free_req_slots=(
                        self.req_to_token_pool.available_size() - num_new_rows
                    ),
                    head=head,
                    tail=waiting_queue[-1],
                    decode_kv_tokens=(
                        decode_part.kv_tokens if decode_part is not None else 0
                    ),
                )

        # Preemption already released these requests; requeue them even when
        # the request they made room for was not admitted.
        if adder.preempt_list:
            preempted = set(adder.preempt_list)
            self.core_lists.decoding = [
                r for r in self.core_lists.decoding if r not in preempted
            ]
            running_reqs = slot.filter_decoding(self.core_lists.decoding)
            admitted = set(adder.can_run_list)
            for req in adder.preempt_list:
                self.request_intake.add_request_to_queue(req, admitted=admitted)

        # The admitted requests stay in the waiting queue until the launch.
        can_run_list: List[Req] = adder.can_run_list
        if len(can_run_list) == 0:
            return None
        if adder.new_chunked_req is not None:
            assert chunked_req is None
            chunked_req = adder.new_chunked_req
        prefill_part = PrefillPart(reqs=can_run_list, chunked_req=chunked_req)

        retries = self.tree_cache.storage_prefetch_retries
        if hicache_storage_enabled() and retries is not None:
            for req in can_run_list:
                retries.cancel(req.rid)

        set_time_batch(can_run_list, "set_forward_entry_time")

        new_batch = self.step_builder.build_extend_step(
            prefill_part.reqs, chunked_req=prefill_part.chunked_req
        )

        # Record prefill stats for logging after forward. The tokens pending
        # prefill once this step launches: the waiting requests it does not
        # admit, and the chunked request's tokens past this chunk.
        admitted = set(can_run_list)
        num_pending_tokens = sum(
            req.seqlen for req in self.core_lists.waiting_queue if req not in admitted
        )
        if chunked_req is not None:
            num_pending_tokens += (
                chunked_req.seqlen - chunked_req.prefix_len - chunked_req.extend_len
            )
        new_batch.prefill_stats = PrefillStats.from_adder(
            adder,
            running_reqs,
            get_schedule().enable_priority_scheduling,
            num_pending_tokens=num_pending_tokens,
        )

        # The reserved decode part rides this step as one-token extend rows
        # after the prefill rows, unless the prefill part keeps it out.
        if decode_part is not None and self._may_mix([], prefill_part.reqs):
            decode_reqs = decode_part.reqs
            if adder.preempt_list:
                # Preemption released some of them.
                preempted = set(adder.preempt_list)
                decode_reqs = [r for r in decode_reqs if r not in preempted]
            # Admission estimated the KV the decode rows need; the allocator
            # decides. Rows that do not fit sit out this step and decode on
            # the next decode step, which fits them; a mixed step does not
            # retract them.
            if decode_reqs and check_decode_mem(
                decode_reqs,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
                spec_algorithm=self.spec_algorithm,
            ):
                new_batch.mix_decode_rows(
                    self.step_builder.new_decode_step(decode_reqs)
                )

        new_batch.sampling_info = self.step_builder.build_sampling_info(
            new_batch.reqs, new_batch.req_pool_indices
        )
        return new_batch

    @staticmethod
    def _may_mix(decode_reqs: List[Req], prefill_reqs: List[Req]) -> bool:
        """Whether decode rows of decode_reqs may share a step with prefill
        rows of prefill_reqs. These are the per-step conditions; the startup
        ones (chunked prefill on, and a speculative algorithm, dLLM, HiSparse
        or other mode that supports it) argument validation settles into
        is_mixed_chunk."""
        return (
            # TODO (lianmin): support return_logprob + mixed chunked prefill
            not any(r.return_logprob for r in decode_reqs)
            and not any(r.return_logprob for r in prefill_reqs)
            # The decode rows carry no input_embeds; shapes would mismatch.
            and all(r.input_embeds is None for r in prefill_reqs)
            # Beam member rows are not supported inside a mixed extend batch.
            and all(r.beam_group is None for r in decode_reqs)
        )

    def _plan_decode_part(
        self, slot: ScheduleSlot, chunk_budget: Optional[int]
    ) -> Optional[DecodePart]:
        """The decode part a mixed step reserves before prefill admission, or
        None when the step does not mix."""
        if not self.is_mixed_chunk:
            return None
        reqs = self._select_decode_reqs(slot)
        if not reqs or not self._may_mix(reqs, []):
            return None
        decode_part = DecodePart(
            reqs=reqs,
            kv_tokens=new_tokens_required_next_decode(
                reqs, self.token_to_kv_pool_allocator.page_size, self.spec_algorithm
            ),
        )
        return self.composition_policy.decode_part_to_reserve(decode_part, chunk_budget)

    def _decode_kv_tokens(self, slot: ScheduleSlot) -> int:
        """The KV a pass now would reserve for the decode part."""
        decode_part = self._plan_decode_part(slot, self.chunked_prefill_size)
        return decode_part.kv_tokens if decode_part is not None else 0

    def _revert_match(self, req: Req) -> None:
        """Undo the mamba state init_next_round_input staged for a request the
        pass does not admit, so its freshly allocated slot does not leak. A
        session holds its slot across requests and frees it itself."""
        req.kv.mamba_cow_src_index = None
        req.kv.mamba_needs_clear = False
        if req.kv.holds_mamba and not getattr(req, "session", None):
            self.tree_cache.req_to_token_pool.mamba_allocator.free(
                req.kv.mamba_pool_idx.unsqueeze(-1)
            )
            req.kv.mamba_pool_idx = None

    def _select_decode_reqs(self, slot: ScheduleSlot) -> List[Req]:
        """The decoding requests one decode step (or a mixed step's decode
        rows) runs: not prefill-only, and not already reaching max_new_tokens
        in the steps still in flight."""
        candidates = [
            req
            for req in slot.filter_decoding(self.core_lists.decoding)
            if not req.is_prefill_only
        ]
        num_pending = len(self.pending)
        if not self.flags.enable_skip_finishing_decode or num_pending == 0:
            return candidates
        # Each pending step commits at most one output to a request, so only a
        # request within num_pending outputs of max_new_tokens can finish in
        # flight; the rest skip the exact count.
        return [
            req
            for req in candidates
            if len(req.output_ids) + num_pending < req.sampling_params.max_new_tokens
            or not self._finishes_inflight(req)
        ]

    def _finishes_inflight(self, req: Req) -> bool:
        """Whether the uncommitted outputs of req reach max_new_tokens."""
        if req.beam_group is not None:
            return False
        num_inflight = self.pending.num_inflight_outputs(req)
        return (
            num_inflight > 0
            and len(req.output_ids) + num_inflight >= req.sampling_params.max_new_tokens
        )

    def get_decode_step(self, slot: ScheduleSlot) -> Optional[ScheduleBatch]:
        """Build the decode step of the decoding requests, retracting some when
        the step does not fit the KV pool; None skips decode this step."""
        reqs = self._select_decode_reqs(slot)
        if not reqs:
            return None

        kv_full_retract_flag = not check_decode_mem(
            reqs,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            spec_algorithm=self.spec_algorithm,
        )
        if kv_full_retract_flag and len(reqs) < sum(
            not req.is_prefill_only
            for req in slot.filter_decoding(self.core_lists.decoding)
        ):
            # The pending result frees the finishing requests' KV at the end of
            # this step; decode the rest next step instead of retracting them.
            return None

        # Check if decode out of memory
        if kv_full_retract_flag or (
            TEST_RETRACT and self.counters.forward_ct % TEST_RETRACT_INTERVAL == 0
        ):
            reqs = self._retract_decode_reqs(reqs, kv_full_retract_flag)
        else:
            self.new_token_ratio_tracker.decay_step()

        if not reqs:
            return None
        return self.step_builder.build_decode_step(reqs)

    def _retract_decode_reqs(
        self, reqs: List[Req], kv_full_retract_flag: bool
    ) -> List[Req]:
        """Move decoding requests back to the waiting queue until the rest fit
        one decode step; return the requests that stay."""
        if (
            self.mode_owners.disagg_decode is not None
            and self.decode_offload_manager is not None
        ):
            # Pending offload copies may still read the device KV that retraction frees.
            self.mode_owners.disagg_decode.drain_offloads_before_retraction()
        old_available_tokens = self.token_to_kv_pool_allocator.available_size()
        old_ratio = self.new_token_ratio_tracker.current
        mamba_allocator = getattr(
            self.tree_cache.req_to_token_pool, "mamba_allocator", None
        )
        old_mamba_available = (
            mamba_allocator.available_size() if mamba_allocator is not None else None
        )
        kept_reqs, retracted_reqs, reqs_to_abort = retract_decode(
            reqs,
            request_retirer=self.request_retirer,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            spec_algorithm=self.spec_algorithm,
        )
        new_token_ratio = NewTokenRatioTracker.estimate_new_token_ratio_after_retract(
            kept_reqs
        )
        # The kept requests decode in the retraction order from now on.
        kept = set(kept_reqs)
        left = set(retracted_reqs) | set(reqs_to_abort)
        self.core_lists.decoding = kept_reqs + [
            r for r in self.core_lists.decoding if r not in kept and r not in left
        ]
        new_available_tokens = self.token_to_kv_pool_allocator.available_size()
        new_token_gained = new_available_tokens - old_available_tokens
        mamba_num_gained = (
            mamba_allocator.available_size() - old_mamba_available
            if mamba_allocator is not None
            else None
        )

        self.metrics_sink.num_retracted_reqs = len(retracted_reqs)
        if self.metrics_sink.enable_metrics and len(retracted_reqs) > 0:
            self.metrics_sink.metrics_collector.increment_retracted_reqs(
                num_retracted_reqs=len(retracted_reqs),
                num_retracted_input_tokens=sum(
                    len(r.origin_input_ids) for r in retracted_reqs
                ),
                num_retracted_output_tokens=sum(
                    len(r.output_ids) for r in retracted_reqs
                ),
            )
        self.new_token_ratio_tracker.current = new_token_ratio
        for req in reqs_to_abort:
            abort_reason: FINISH_ABORT = req.to_finish
            self.request_retirer.send_abort_output(
                req, finished_reason=abort_reason.to_json()
            )
        for req in reqs_to_abort:
            # Member rows were freed inside retract_decode; the group only
            # has to leave the live set.
            self.request_retirer.retire_retracted_abort(req)

        msg_prefix = (
            "KV cache pool is full. Retract requests. "
            if kv_full_retract_flag
            else "Testing retraction. "
        )
        msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
        if mamba_num_gained is not None:
            msg_details += f", #mamba_num_gained: {mamba_num_gained}"
        if kv_full_retract_flag:
            msg_details += (
                f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
            )
        logger.warning(msg_prefix + msg_details)

        for req in retracted_reqs:
            self.request_intake.add_request_to_queue(req, is_retracted=True)
        return kept_reqs

    def retract_for_pause(self) -> None:
        """The retract half of a retract-mode pause: every live request
        leaves the device, back to the waiting queue or, on PD decode, held
        for a rebootstrap."""
        lists = self.request_registry.request_lists()
        retract_reqs = [
            req
            for entry in lists.values()
            if entry.retract_on_pause
            for req in entry.reqs
            if not req.finished()
        ]

        self.counters.cur_batch_for_debug = None

        if (
            self.mode_owners.disagg_decode is not None
            and self.decode_offload_manager is not None
        ):
            # Pending offload copies may still read the device KV that retraction frees,
            # and the paused decode loop never drains them.
            self.mode_owners.disagg_decode.drain_offloads_before_retraction()
        if retract_reqs:
            # Decode-side retract always rebootstraps (recomputes the KV from
            # the prefill), so skip the device->host KV offload that release_req
            # would otherwise do; the offloaded copy would be immediately
            # discarded. Non-decode modes ignore offload_kv (they never offload).
            retract_all(
                reqs=retract_reqs,
                request_retirer=self.request_retirer,
                offload_kv=False,
            )
        self.core_lists.decoding = []
        if self.mode_owners.disagg_decode is not None:
            self.mode_owners.disagg_decode.hold_for_rebootstrap(retract_reqs)
        else:
            for req in retract_reqs:
                self.request_intake.add_request_to_queue(req)
        # In disagg-PREFILL, keep a live mid-chunk chunked_req rather than retract it:
        # freeing its KV under a live disagg KV-sender crashes pop_bootstrapped or
        # sends freed/reused KV to decode. Kept, it resumes prefill after the pause.
        # TODO(disagg-prefill-retract): tear the sender down (abort + release metadata
        # buffer + reset pending_bootstrap) before freeing KV, then retract for real.
        # Until then a weight-update pause leaves stale-weight prefix KV (off-policy).
        if lists["chunked"].retract_on_pause:
            self.core_lists.chunked_req = None

    def flush_cache(self, empty_cache: bool = True):
        """Flush memory pools (e.g., KV cache, Mamba cache) and optionally empty device allocator cache."""
        if self.request_registry.is_fully_idle(
            ignore_waiting=self.flags._engine_paused
        ):
            self.counters.cur_batch_for_debug = None
            self.tree_cache.reset()
            self.req_to_token_pool.clear()
            self.token_to_kv_pool_allocator.clear()
            self.req_to_token_pool.reset_aux_cache_allocator()
            self.grammar_manager.clear()
            self.metrics_sink.reset_metrics()

            if self.draft_worker:
                self.draft_worker.clear_cache_pool()

            if empty_cache:
                current_platform.empty_cache()
            # Per-DP-group leader logs once: ranks within a DP group are
            # state-synchronous, but DP groups may diverge.
            if self.metrics_sink.is_stats_logging_rank:
                logger.info("Cache flushed successfully!")
            success = True
        else:
            logging.warning(
                f"Cache not flushed because there are pending requests. "
                f"#queue-req: {len(self.request_registry.queued_reqs())}, "
                f"#running-req: {len(self.request_registry.running_reqs())}"
            )
            success = False
        return success
