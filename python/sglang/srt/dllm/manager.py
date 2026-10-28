"""The dLLM mode owner: its request lists and the stages the scheduler calls."""

from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Tuple

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.prefill_adder import DllmPrefillAdder
from sglang.srt.dllm.req_state import DllmReqPhase, DllmReqState
from sglang.srt.managers.request_lists import RequestList
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    SchedulePolicy,
    priority_preemption_enabled,
)
from sglang.srt.mem_cache.common import checkpoint_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.beam_search.coordinator import BeamCoordinator
    from sglang.srt.managers.io_struct import AbortReq
    from sglang.srt.managers.scheduler_components.bundles import (
        KvResources,
        RequestLimits,
        SchedulerServices,
    )
    from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
        NewTokenRatioTracker,
    )
    from sglang.srt.managers.scheduler_components.request_retirer import (
        RequestRetirer,
    )
    from sglang.srt.managers.scheduler_components.step_builder import StepBuilder
    from sglang.srt.managers.utils import GenerationBatchResult


class DllmManager:
    """The dLLM mode owner, and its three disjoint lists of the requests it
    took from the core waiting list (at most `max_running_requests` in all):
    `waiting` (no step launched them yet), `staging` (in the launched step,
    until its commit), and `between_blocks` (left unfinished by a committed
    step). A step selects from `waiting` and `between_blocks` in the order the
    requests were taken. dLLM requests never decode.
    """

    def __init__(
        self,
        *,
        dllm_config: DllmConfig,
        kv: KvResources,
        services: SchedulerServices,
        policy: SchedulePolicy,
        new_token_ratio_tracker: NewTokenRatioTracker,
        limits: RequestLimits,
        context_len: int,
        beam_coordinator: BeamCoordinator,
        step_builder: StepBuilder,
        request_retirer: RequestRetirer,
    ):
        page_size = get_schedule().page_size
        priority_scheduling_preemption_threshold = (
            get_schedule().priority_scheduling_preemption_threshold
        )
        enable_priority_preemption = priority_preemption_enabled()
        enable_priority_scheduling = get_schedule().enable_priority_scheduling
        tree_cache = kv.tree_cache
        req_to_token_pool = kv.req_to_token_pool
        token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        metrics_sink = services.metrics_sink
        output_streamer = services.output_streamer
        self.dllm_config = dllm_config
        self.tree_cache = tree_cache
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.policy = policy
        self.new_token_ratio_tracker = new_token_ratio_tracker
        self.page_size = page_size
        self.max_prefill_tokens = limits.max_prefill_tokens
        self.chunked_prefill_size = limits.chunked_prefill_size
        self.priority_scheduling_preemption_threshold = (
            priority_scheduling_preemption_threshold
        )
        self.enable_priority_preemption = enable_priority_preemption
        self.enable_priority_scheduling = enable_priority_scheduling
        self.truncation_align_size = limits.truncation_align_size
        self.context_len = context_len
        self.beam_coordinator = beam_coordinator
        self.step_builder = step_builder
        self.metrics_sink = metrics_sink
        self.output_streamer = output_streamer
        self.request_retirer = request_retirer

        self.waiting: List[Req] = []
        self.staging: List[Req] = []
        self.between_blocks: List[Req] = []
        self._next_intake_seq = 0

    def lists(self) -> List[RequestList]:
        return [
            RequestList(
                name="dllm_waiting",
                reqs=self.waiting,
                abort=self.abort_waiting,
                health_check=True,
                waiting=True,
            ),
            # A launched step runs them: abort marks them to_finish.
            RequestList(
                name="dllm_staging",
                reqs=self.staging,
                inflight=True,
                health_check=True,
                holds_kv=True,
                pins_lora=True,
                counts_running=True,
            ),
            # Started requests between two of their blocks: the running
            # timeout reads them. Abort pops them before the scheduler marks
            # the list's members to_finish.
            RequestList(
                name="dllm_between_blocks",
                reqs=self.between_blocks,
                inflight=True,
                abort=self.abort_between_blocks,
                health_check=True,
                holds_kv=True,
                pins_lora=True,
                counts_running=True,
            ),
        ]

    def is_empty(self) -> bool:
        return (
            len(self.waiting) == 0
            and len(self.staging) == 0
            and len(self.between_blocks) == 0
        )

    def validate_request(self, req: Req) -> Optional[str]:
        return self.dllm_config.validate_request(req)

    def abort_waiting(self, recv_req: AbortReq) -> None:
        """Pop and release the waiting requests recv_req names."""
        self.waiting = self._pop_aborted(self.waiting, recv_req)

    def abort_between_blocks(self, recv_req: AbortReq) -> None:
        """Pop and release the requests between blocks that recv_req names."""
        # In place: the scheduler then reads the same list object for its
        # to_finish pass, which must not reach the requests released here.
        self.between_blocks[:] = self._pop_aborted(self.between_blocks, recv_req)

    def _pop_aborted(self, reqs: List[Req], recv_req: AbortReq) -> List[Req]:
        """Release the members of reqs that recv_req names; return the rest."""
        aborted = [
            req
            for req in reqs
            if recv_req.abort_all or req.rid.startswith(recv_req.rid)
        ]
        if not aborted:
            return reqs
        for req in aborted:
            self.request_retirer.retire_aborted(req)
            self.request_retirer.send_abort_output(req)
            logger.debug(f"Abort dLLM queued request. {req.rid=}")
        return [req for req in reqs if req not in aborted]

    def schedule(
        self, waiting_queue: List[Req]
    ) -> Tuple[Optional[ScheduleBatch], List[Req]]:
        """Build this iteration's dLLM step; return it and what stays in
        waiting_queue.

        The step takes requests from the core waiting list only as far as
        max_running_requests allows: a request that holds no KV yet stays
        there, as in normal mode and PD prefill, so the priority sort, the
        queue limits and the waiting timeout see it unchanged.
        """
        # Requests taken already finished (a rejected multimodal input).
        self.waiting = [req for req in self.waiting if not req.finished()]
        self.between_blocks = [req for req in self.between_blocks if not req.finished()]

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(waiting_queue):
            return None, waiting_queue

        self.policy.calc_priority(waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder()

        # Transfer requests to the DLLM manager
        waiting_queue = self._fetch_waiting_reqs(waiting_queue)

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None, waiting_queue

        # Record metrics
        set_time_batch(can_run_list, "set_forward_entry_time")

        # Create and prepare batch
        new_batch = self._create_dllm_batch(can_run_list, forward_mode, adder=adder)
        return new_batch, waiting_queue

    def on_launch(self, step: ScheduleBatch) -> None:
        """The step's requests move to staging until its commit."""
        launched = set(step.reqs)
        self.waiting = [req for req in self.waiting if req not in launched]
        self.between_blocks = [
            req for req in self.between_blocks if req not in launched
        ]
        self.staging.extend(step.reqs)

    def commit_extend(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ) -> None:
        """The host commit of a dLLM step: write each request's block, finish
        the block, and move the requests still running between blocks."""
        self._write_blocks(batch, result)

        launched = set(batch.reqs)
        self.staging = [req for req in self.staging if req not in launched]
        live_reqs = [req for req in batch.reqs if not req.finished()]
        for req in live_reqs:
            self.finish_dllm_forward(req)
            req.dllm.init_next_round_input(req)
        self._move_between_blocks(live_reqs)

    def _write_blocks(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if (
            self.dllm_config.requires_separate_context_encoding
            and not batch.forward_mode.is_dllm_extend()
        ):
            return

        fdfo_mode = self.dllm_config.first_done_first_out_mode
        assert not fdfo_mode or result.accept_length_per_req_cpu is not None, (
            "FDFO dLLM result is missing accept lengths."
        )

        # FDFO also commits unresolved blocks so their KV can be reused.
        if fdfo_mode or result.next_token_ids:
            block_size = self.dllm_config.block_size
            algo_states = result.dllm_algo_state

            self.token_to_kv_pool_allocator.free_group_begin()
            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                if not fdfo_mode:
                    next_token_ids = result.next_token_ids[idx].tolist()
                    new_tokens = len(next_token_ids)
                    if new_tokens == 0:
                        continue

                    req.full_untruncated_fill_ids[
                        req.extend_end - new_tokens : req.extend_end
                    ] = array("q", next_token_ids)
                    self.metrics_sink.num_generated_tokens += new_tokens

                    req.output_ids.extend(next_token_ids)
                    req.update_finish_state(new_accepted_len=new_tokens)
                    self._finish_dllm_request_if_needed(req)
                    continue

                next_token_ids = result.next_token_ids[idx]
                assert len(next_token_ids) == block_size

                if result.accept_length_per_req_cpu[idx] == 0:
                    # Unresolved: keep partial state and KV for the next FDFO round.
                    req.dllm.incomplete_ids = array("q", next_token_ids)
                    req.dllm.algo_state = (
                        algo_states[idx] if algo_states is not None else None
                    )
                    continue

                req.dllm.incomplete_ids = array("q")
                req.dllm.algo_state = None

                # Mirror the resolved block into the committed fill ids so the
                # prefix cache keys on the real tokens, not the mask block, next
                # round. Index relative to extend_end (the truncated/
                # committed length), which can be shorter than
                # full_untruncated_fill_ids when the staging adder truncates the
                # block to the KV budget.
                req.full_untruncated_fill_ids[
                    req.extend_end - block_size : req.extend_end
                ] = array("q", next_token_ids)

                len_input = len(req.origin_input_ids)
                len_fill = req.extend_end
                if len_fill <= len_input:
                    continue

                if len_fill - len(next_token_ids) < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                self.metrics_sink.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))
                self._finish_dllm_request_if_needed(req)

            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
            self.token_to_kv_pool_allocator.free_group_end()

    def _finish_dllm_request_if_needed(self, req: Req) -> None:
        if (
            not req.finished()
            and req.seqlen + self.dllm_config.block_size > self.context_len
        ):
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))

        if req.finished():
            self.request_retirer.retire_finished(
                req,
                checkpoint=not self.dllm_config.requires_separate_context_encoding,
            )
            req.time_stats.set_completion_time()

    def _stash_dllm_context(self, req: Req) -> None:
        context_len = req.dllm.block_offset
        block_size = self.dllm_config.block_size
        assert req.extend_end == context_len + block_size
        assert req.kv is not None and req.kv.req_pool_idx is not None

        allocated_len = req.kv.kv_allocated_len
        page_size = self.token_to_kv_pool_allocator.page_size
        free_start = -(-context_len // page_size) * page_size
        if free_start < allocated_len:
            slots = self.req_to_token_pool.req_to_token[
                req.kv.req_pool_idx, free_start:allocated_len
            ]
            self.token_to_kv_pool_allocator.free_segment(slots, start_pos=free_start)

        assert req.kv.max_evicted_seqlen <= context_len
        req.kv.truncate(context_len)
        req.extend_end = context_len
        checkpoint_kv_cache(req, self.tree_cache)

    def finish_dllm_forward(self, req: Req) -> None:
        """Commit or discard KV after one scheduler-visible dLLM forward."""
        fdfo_mode = self.dllm_config.first_done_first_out_mode
        if fdfo_mode and req.dllm.incomplete_ids:
            return

        if req.dllm.is_prefill():
            checkpoint_kv_cache(req, self.tree_cache)
        elif self.dllm_config.requires_separate_context_encoding:
            self._stash_dllm_context(req)
        else:
            # The row stays with the request between blocks: a later abort
            # releases its KV and tree lock through it.
            checkpoint_kv_cache(req, self.tree_cache)

    def _move_between_blocks(self, reqs: List[Req]) -> None:
        """Put reqs into between_blocks, which keeps the order requests were
        taken in."""
        self.between_blocks = sorted(
            [*self.between_blocks, *reqs], key=lambda req: req.dllm.intake_seq
        )

    def _selectable_reqs(self) -> List[Req]:
        """The requests a step may select, in the order they were taken."""
        return sorted(
            [*self.waiting, *self.between_blocks],
            key=lambda req: req.dllm.intake_seq,
        )

    def _fetch_waiting_reqs(self, waiting_queue: List[Req]) -> List[Req]:
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = (
            self.dllm_config.max_running_requests
            - len(self.waiting)
            - len(self.staging)
            - len(self.between_blocks)
        )
        num_requests_to_add = min(max_dllm_capacity, len(waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = waiting_queue[:num_requests_to_add]
            for req in requests_to_add:
                req.dllm = DllmReqState.for_req(
                    req, self.dllm_config, intake_seq=self._next_intake_seq
                )
                self._next_intake_seq += 1
            self.add_waiting_reqs(requests_to_add)
            waiting_queue = waiting_queue[num_requests_to_add:]
        return waiting_queue

    def _should_skip_prefill(self, waiting_queue: List[Req]) -> bool:
        """Check if DLLM prefill should be skipped."""
        if not waiting_queue and self.is_empty():
            return True

        # No request decodes in dLLM mode.
        if (
            not self.beam_coordinator.has_room(1, 1)
            and self.is_empty()
            and not self.enable_priority_preemption
        ):
            return True

        return False

    def _create_dllm_prefill_adder(self) -> DllmPrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return DllmPrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            [],
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self, adder: DllmPrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        # Try prefill batch first
        prefill_reqs = self.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
            if adder.can_run_list:
                return (
                    ForwardMode.EXTEND
                    if self.dllm_config.requires_separate_context_encoding
                    else ForwardMode.DLLM_EXTEND
                )
        # Fall back to decode batch: also when no prefill fits, since the
        # requests between blocks hold the KV it waits for until they finish.
        decode_reqs = self.get_decode_requests()
        self._process_batch_by_phase(
            adder,
            decode_reqs,
            DllmReqPhase.STAGING_DECODE,
            DllmReqPhase.INCOMING_DECODE,
        )
        return ForwardMode.DLLM_EXTEND

    def _process_batch_by_phase(
        self,
        adder: DllmPrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm.phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm.phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _create_dllm_batch(
        self,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: DllmPrefillAdder,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        # An FDFO block left unresolved rewrites the KV its request kept.
        new_batch = self.step_builder.build_extend_step(
            can_run_list,
            kv_reuse_lens=[
                len(req.dllm.incomplete_ids) if req.kv.holds_kv else 0
                for req in can_run_list
            ],
        )
        new_batch.forward_mode = forward_mode
        if forward_mode.is_dllm_extend():
            block_size = self.dllm_config.block_size
            new_batch.extend_positions = [
                i
                for block_offset in (req.dllm.block_offset for req in can_run_list)
                for i in range(block_offset, block_offset + block_size)
            ]
            if any(req.dllm.has_literal_mask for req in can_run_list):
                # Committed outputs are immutable too when a retracted request
                # rebuilds its KV.
                new_batch.dllm_committed_lens = [
                    len(req.origin_input_ids) + len(req.output_ids)
                    for req in can_run_list
                ]

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, [], self.enable_priority_scheduling
        )
        new_batch.sampling_info = self.step_builder.build_sampling_info(
            can_run_list, new_batch.req_pool_indices
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self, adder: DllmPrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        # Once set, every later request must preempt to be admitted.
        is_full = False
        # The new request rows the admitted requests take; a staging request
        # already holds its row.
        num_admitted_rows = sum(map(self.beam_coordinator.num_rows, adder.can_run_list))
        for req in reqs:
            # No request decodes in dLLM mode.
            rows_with_req = num_admitted_rows + self.beam_coordinator.num_rows(req)
            if not self.beam_coordinator.has_room(
                len(adder.can_run_list) + 1, rows_with_req
            ):
                is_full = True

            # Try preemption if batch is full
            if is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            # Prepare and add request
            req.dllm.init_next_round_input(req, self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )
            if adder.can_run_list and req is adder.can_run_list[-1]:
                num_admitted_rows = rows_with_req

            if res != AddReqResult.CONTINUE:
                break

        return res

    def process_dllm_staging_reqs(
        self, adder: DllmPrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests a step may select."""
        return [req for req in self._selectable_reqs() if req.dllm.is_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests a step may select."""
        return [req for req in self._selectable_reqs() if not req.dllm.is_prefill()]

    def add_waiting_reqs(self, reqs: List[Req]) -> None:
        """Add requests to waiting queue with redundancy check."""
        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting.extend(reqs)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in the dLLM lists."""
        existing_rids: Set[str] = {
            r.rid for r in [*self.waiting, *self.staging, *self.between_blocks]
        }
        return any(req.rid in existing_rids for req in reqs)
