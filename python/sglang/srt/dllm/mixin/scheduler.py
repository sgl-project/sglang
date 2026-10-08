from __future__ import annotations

import logging
from array import array
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.runtime_context import get_exec, get_schedule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if get_exec().dllm.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def validate_dllm_request(self: Scheduler, req: Req) -> Optional[str]:
        if self.dllm_config is None:
            return None
        return self.dllm_config.validate_request(req)

    def get_new_batch_dllm(
        self: Scheduler, running_batch: ScheduleBatch
    ) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.enable_priority_preemption:
            running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill(running_batch=running_batch):
            return None

        running_bs = len(running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs, running_batch=running_batch)

        # Clear the previous batch and fetch waiting requests
        self.dllm_manager.staging_queue = []
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder, running_batch=running_batch)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(
            can_run_list, forward_mode, adder=adder, running_batch=running_batch
        )
        return new_batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        if result.copy_done is not None:
            result.copy_done.synchronize()

        if (
            self.dllm_config.requires_separate_context_encoding
            and not batch.forward_mode.is_dllm_extend()
        ):
            for req in batch.reqs:
                if not req.finished() and not req.dllm_block_done:
                    self._complete_dllm_block(req)
            self.metrics_reporter.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=result.can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            return

        block_tokens = result.next_token_ids.tolist()
        fdfo_mode = self.dllm_config.first_done_first_out_mode
        assert not fdfo_mode or result.dllm_block_done is not None, (
            "FDFO dLLM result is missing dllm_block_done."
        )
        block_done = result.dllm_block_done.tolist() if fdfo_mode else None
        block_size = self.dllm_config.block_size
        algo_states = result.dllm_algo_state
        has_new_tokens = False

        assert result.dllm_block_ids is not None
        assert len(result.dllm_block_ids) == len(batch.reqs)

        self.token_to_kv_pool_allocator.free_group_begin()
        for idx, req in enumerate(batch.reqs):
            if (
                req.finished()
                or result.dllm_block_ids[idx] != req.dllm_block_id
                or req.dllm_block_done
            ):
                continue

            next_token_ids = block_tokens[idx]
            assert len(next_token_ids) == block_size

            # Keep the unresolved block for the next FDFO step.
            if fdfo_mode and not block_done[idx]:
                req.dllm_incomplete_ids = array("q", next_token_ids)
                req.dllm_algo_state = (
                    algo_states[idx] if algo_states is not None else None
                )
                continue

            req.dllm_incomplete_ids = array("q")
            req.dllm_algo_state = None

            # extend_end accounts for KV-budget truncation.
            len_fill = req.extend_end
            req.full_untruncated_fill_ids[len_fill - block_size : len_fill] = array(
                "q", next_token_ids
            )

            len_input = len(req.origin_input_ids)
            if len_fill > len_input:
                if len_fill - block_size < len_input:
                    next_token_ids = next_token_ids[len_input - len_fill :]

                has_new_tokens = True
                self.metrics_reporter.num_generated_tokens += len(next_token_ids)
                req.output_ids.extend(next_token_ids)
                req.update_finish_state(new_accepted_len=len(next_token_ids))
                self._finish_dllm_request_if_needed(req)

            self._complete_dllm_block(req)

        if fdfo_mode or has_new_tokens:
            self.output_streamer.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.metrics_reporter.report_prefill_stats(
            batch=batch,
            prefill_stats=batch.prefill_stats,
            can_run_cuda_graph=result.can_run_cuda_graph,
            dp_cooperation_info=batch.dp_cooperation_info,
        )

    def _complete_dllm_block(self: Scheduler, req: Req) -> None:
        """Finish a block once; keep its ID/done until the next block is scheduled."""
        if req.dllm_block_done:
            return
        req.dllm_block_done = True
        if req.finished() or not self.dllm_config.requires_separate_context_encoding:
            return
        if self.dllm_config.first_done_first_out_mode:
            self._clear_dllm_future(req)
        self.finish_dllm_forward(req)
        req.init_next_round_input()

    def _finish_dllm_request_if_needed(self: Scheduler, req: Req) -> None:
        if (
            not req.finished()
            and req.seqlen + self.dllm_config.block_size > self.model_config.context_len
        ):
            req.finished_reason = FINISH_LENGTH(length=len(req.output_ids))

        if req.finished():
            if self.dllm_config.first_done_first_out_mode:
                self._clear_dllm_future(req)
            release_kv_cache(
                req,
                self.tree_cache,
                checkpoint=not self.dllm_config.requires_separate_context_encoding,
            )
            req.time_stats.set_completion_time()

    def _stash_dllm_context(self: Scheduler, req: Req) -> None:
        context_len = req.dllm_block_offset
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

        req.kv.kv_committed_len = context_len
        req.kv.kv_allocated_len = context_len
        assert req.kv.max_evicted_seqlen <= context_len
        req.extend_end = context_len
        self.stash_chunked_request(req)

    def finish_dllm_forward(self: Scheduler, req: Req) -> None:
        """Commit or discard KV after one scheduler-visible dLLM forward."""
        fdfo_mode = self.dllm_config.first_done_first_out_mode
        if fdfo_mode and req.dllm_incomplete_ids:
            return

        if req.is_dllm_prefill():
            self.stash_chunked_request(req)
        elif self.dllm_config.requires_separate_context_encoding:
            self._stash_dllm_context(req)
        else:
            # The row stays with the request between blocks: a later abort
            # releases its KV and tree lock through it.
            self.stash_chunked_request(req)

    def _fetch_waiting_reqs(self: Scheduler):
        # Calculate how many requests can be added to DLLM manager
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))

        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler, running_batch: ScheduleBatch) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_preemption
        ):
            running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(
        self: Scheduler, running_bs: int, running_batch: ScheduleBatch
    ) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            running_batch,
            self.new_token_ratio_tracker.current,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=get_schedule().prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(
        self: Scheduler, adder: PrefillAdder, running_batch: ScheduleBatch
    ) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
                running_batch=running_batch,
            )
            return (
                ForwardMode.EXTEND
                if self.dllm_config.requires_separate_context_encoding
                else ForwardMode.DLLM_EXTEND
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
                running_batch=running_batch,
            )
            return ForwardMode.DLLM_EXTEND

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
        running_batch: ScheduleBatch,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(
                adder, incoming_reqs, running_batch=running_batch
            )

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_inflight_middle_chunks()

    def _create_dllm_batch(
        self: Scheduler,
        can_run_list: List[Req],
        forward_mode: ForwardMode,
        adder: PrefillAdder,
        running_batch: ScheduleBatch,
    ) -> ScheduleBatch:
        """Create and prepare a new DLLM batch."""
        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )
        new_batch.prepare_for_extend()
        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        # Record prefill stats for logging after forward
        from sglang.srt.managers.scheduler_components.metrics_reporter import (
            PrefillStats,
        )

        new_batch.prefill_stats = PrefillStats.from_adder(
            adder, running_batch.reqs, self.enable_priority_scheduling
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler,
        adder: PrefillAdder,
        reqs: List[Req],
        running_batch: ScheduleBatch,
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                running_batch.batch_is_full = True

            # Try preemption if batch is full
            if running_batch.batch_is_full:
                if not self.enable_priority_preemption or not adder.preempt_to_schedule(
                    req
                ):
                    break

            # Prepare and add request
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process staging DLLM requests with resource allocation."""
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res

        return AddReqResult.CONTINUE


class DllmManager:
    """
    Manager for Diffusion LLM request scheduling.

    Maintains two queues:
    - waiting_queue: The requests waiting to be scheduled with max running requests limit
    - staging_queue: Requests allocated resources by PrefillAdder
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        """Get all prefill requests from waiting queue."""
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        """Get all decode requests from waiting queue."""
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to waiting queue with redundancy check."""
        assert self.dllm_config is not None, "Diffusion LLM config is not set."

        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]

        # Check for duplicate request IDs
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")

        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        """Add requests to staging queue (allocated by PrefillAdder)."""
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        """Check if any request ID already exists in waiting queue."""
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        """Check if there are requests in staging queue."""
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        """Check if both queues are empty or DLLM is not configured."""
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_inflight_middle_chunks(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.inflight_middle_chunks += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def pop_aborted_reqs(self, abort_all: bool, rid: str) -> List[Req]:
        aborted_reqs: List[Req] = []
        seen: Set[int] = set()

        for queue_name in ("waiting_queue", "staging_queue"):
            queue = getattr(self, queue_name)
            kept_queue = []
            for req in queue:
                if abort_all or req.rid.startswith(rid):
                    req_id = id(req)
                    if req_id not in seen:
                        aborted_reqs.append(req)
                        seen.add(req_id)
                else:
                    kept_queue.append(req)
            setattr(self, queue_name, kept_queue)

        return aborted_reqs
