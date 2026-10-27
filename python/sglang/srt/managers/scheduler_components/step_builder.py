"""Builds the scheduler's extend and decode steps and their sampling info."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.managers.schedule_batch import (
    DecodeRows,
    Req,
    ScheduleBatch,
    req_pool_indices_of,
)
from sglang.srt.runtime_context import get_memory, get_schedule, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.managers.scheduler_components.bundles import KvResources, Workers
    from sglang.srt.managers.scheduler_components.engine_state import (
        LaunchCounters,
    )


class StepBuilder:
    """Builds a step of the requests the scheduler picked: the batch, its KV
    and slots prepared for the forward, and its sampling info."""

    def __init__(
        self,
        *,
        kv: KvResources,
        future_map: FutureMap,
        counters: LaunchCounters,
        workers: Workers,
        hisparse_coordinator: Optional[HiSparseCoordinator],
        model_config: ModelConfig,
        device: str,
    ):
        spec_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.req_to_token_pool = kv.req_to_token_pool
        self.token_to_kv_pool_allocator = kv.token_to_kv_pool_allocator
        self.tree_cache = kv.tree_cache
        self.penalty_store = kv.penalty_store
        self.future_map = future_map
        self.counters = counters
        self.tp_worker = workers.tp_worker
        self.model_worker = workers.model_worker
        # HiSparse spec decoding allocates each new request slot's scratch.
        self.hisparse_coordinator = hisparse_coordinator
        self.model_config = model_config
        self.spec_algorithm = spec_algorithm
        self.device = device
        # The last decode step's request inputs, shared with the next decode
        # step of the same requests.
        self.decode_rows: Optional[DecodeRows] = None

    def build_extend_step(
        self,
        can_run_list: List[Req],
        chunked_req: Optional[Req] = None,
        kv_reuse_lens: Optional[List[int]] = None,
    ) -> ScheduleBatch:
        """The extend step of the admitted requests: allocated, and each
        request's chunk admitted."""
        new_batch = self.new_step(
            can_run_list, chunked_req=chunked_req, kv_reuse_lens=kv_reuse_lens
        )

        new_batch.contains_last_prefill_chunk = (
            chunked_req is None or len(can_run_list) != 1
        )

        if (
            get_memory().enable_hierarchical_cache
            or get_memory().enable_unified_cache_external_linker
        ):
            # todo (zhiqiang): disable cuda graph execution if hicache loading triggered
            new_batch.hicache_consumer_index = (
                self.tree_cache.ready_to_load_host_cache()
            )

        new_slot_reqs = [req for req in can_run_list if not req.kv.holds_kv]
        new_batch.prepare_for_extend(self.hisparse_coordinator)
        if new_slot_reqs and not self.spec_algorithm.is_none():
            # A slot taken now may still name its previous request's draft
            # state; forget it before this step's forward publishes the slot.
            self.future_map.release_slots(
                torch.tensor([req.kv.req_pool_idx for req in new_slot_reqs])
            )
        # Admission of each request's chunk; the build above read the state
        # before it (a retracted request's encoder replay). The penalty store
        # (re)sets a penalized request's history row from its outputs.
        for req in can_run_list:
            req.admit_prefill_chunk(self.device)
            self.penalty_store.alloc(req)
        new_batch.admit_mamba_track()

        if self.tp_worker.model_runner.prefill_aware_swa:
            for req in can_run_list:
                req.kv.shield_swa_prefill(req.extend_end)

        return new_batch

    def new_step(
        self,
        reqs: List[Req],
        chunked_req: Optional[Req] = None,
        kv_reuse_lens: Optional[List[int]] = None,
    ) -> ScheduleBatch:
        """An empty step of reqs, numbered as the next launch."""
        return ScheduleBatch.init_new(
            reqs,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            (not get_schedule().disable_overlap_schedule),
            self.spec_algorithm,
            chunked_req=chunked_req,
            kv_reuse_lens=kv_reuse_lens,
            forward_iter=self.counters.forward_ct + 1,
        )

    def new_decode_step(
        self,
        reqs: List[Req],
        req_pool_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> ScheduleBatch:
        step = self.new_step(reqs)
        step.prepare_for_decode(req_pool_indices)
        self.model_worker.prepare_decode_step(step)
        return step

    def build_decode_step(self, reqs: List[Req]) -> ScheduleBatch:
        """The decode step of reqs with its sampling info; the last decode
        step's request inputs are reused while its requests keep their slots."""
        slots = [r.kv.req_pool_idx for r in reqs]
        rows = self.decode_rows
        if rows is None or rows.reqs != reqs or rows.slots != slots:
            req_pool_indices, req_pool_indices_cpu = req_pool_indices_of(
                reqs, self.device
            )
            rows = self.decode_rows = DecodeRows(
                reqs=list(reqs),
                slots=slots,
                req_pool_indices=req_pool_indices,
                req_pool_indices_cpu=req_pool_indices_cpu,
                sampling_info=self.build_sampling_info(reqs, req_pool_indices),
            )
        step = self.new_decode_step(
            reqs, (rows.req_pool_indices, rows.req_pool_indices_cpu)
        )
        step.sampling_info = rows.sampling_info
        return step

    def build_sampling_info(
        self, reqs: List[Req], req_pool_indices: torch.Tensor
    ) -> SamplingBatchInfo:
        return SamplingBatchInfo.build(
            reqs,
            self.model_config.vocab_size,
            self.device,
            self.penalty_store,
            req_pool_indices,
        )

    def clear_decode_rows(self) -> None:
        self.decode_rows = None
