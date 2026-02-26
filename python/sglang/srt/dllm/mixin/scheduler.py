from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Set, Union

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler


class SchedulerDllmMixin:
    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Generate a new batch for DLLM (Diffusion LLM) scheduling."""
        if self.try_preemption:
            self.running_batch.batch_is_full = False

        # Early exit if batch is full or no requests available
        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        # Create prefill adder with resource constraints
        adder = self._create_dllm_prefill_adder(running_bs)

        # Initialize DLLM manager and transfer requests
        self.dllm_manager.init_next_round()
        self._fetch_waiting_reqs()

        # Process batches
        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        # Record metrics and update state
        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        # Create and prepare batch
        new_batch = self._create_dllm_batch(can_run_list, forward_mode)
        return new_batch

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

    def _should_skip_prefill(self: Scheduler) -> bool:
        """Check if DLLM prefill should be skipped."""
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True

        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.try_preemption
        ):
            self.running_batch.batch_is_full = True
            return True

        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        """Create a prefill adder configured for DLLM scheduling."""
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Process prefill or decode batches for DLLM."""
        forward_mode = ForwardMode.DLLM_EXTEND

        # Try prefill batch first
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            # Fall back to decode batch
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return forward_mode

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        """Process a batch, separating staging and incoming requests."""
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            staging_result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if staging_result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        """Update state for the batch."""

        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
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
        from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats

        new_batch.prefill_stats = PrefillStats(
            log_input_tokens=self.adder.log_input_tokens,
            log_hit_tokens=self.adder.log_hit_tokens,
            new_token_ratio=self.adder.new_token_ratio,
            running_bs=len(self.running_batch.reqs),
            num_new_seqs=len(can_run_list),
        )

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        """Process incoming DLLM requests with resource allocation and preemption."""
        res = AddReqResult.CONTINUE
        for req in reqs:
            # Check if batch is full
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            # Try preemption if batch is full
            if self.running_batch.batch_is_full:
                if not self.try_preemption or not adder.preempt_to_schedule(
                    req, self.server_args
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
                    self.running_batch.batch_is_full = True
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

    def increment_chunked_count(self) -> None:
        """Increment chunked count for all staging requests."""
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        """Remove finished requests from both queues."""
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]

    def init_next_round(self) -> None:
        """Initialize staging requests for next round and clear staging queue."""
        for req in self.staging_queue:
            req.init_next_round_input()
        self.staging_queue = []
