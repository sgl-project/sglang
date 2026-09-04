from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_memory, get_schedule

if TYPE_CHECKING:
    from sglang.srt.managers.io_struct import (
        UpdateWeightFromDiskReqInput,
        UpdateWeightsFromIPCReqInput,
    )
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import (
        ModelRunner,
        SamplingPrewarmResult,
    )
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class HiCacheDraftMode(str, Enum):
    NONE = "none"
    PACKED = "packed"
    SIDECAR = "sidecar"


@dataclass(frozen=True, slots=True)
class HiCacheDraftPlan:
    mode: HiCacheDraftMode = HiCacheDraftMode.NONE
    device_pools: tuple[object, ...] = ()


def _can_pack_hicache_mtp(
    spec_algorithm: SpeculativeAlgorithm,
    draft_runners: tuple[ModelRunner, ...],
) -> bool:
    is_nextn_mtp = (
        spec_algorithm.is_eagle()
        and not spec_algorithm.is_eagle3()
        and all(
            runner.model_config.num_nextn_predict_layers for runner in draft_runners
        )
    )
    is_dspark_dsv4 = (
        spec_algorithm.is_dspark()
        and draft_runners[0].model_config.hf_config.architectures[0]
        == "DeepseekV4ForCausalLMDSpark"
    )
    return is_nextn_mtp or is_dspark_dsv4


class EagleDraftWorkerBase(ABC):
    # topk=1 chain constants for draft_forward's fast path; None when topk > 1.
    _topk1_parents_prealloc: Optional[torch.Tensor] = None
    _topk1_score_indices_prealloc: Optional[torch.Tensor] = None

    def __init__(self) -> None:
        self._specialized_graph_memory_usage: dict[str, float] = {}
        self._specialized_graph_time_usage: dict[str, float] = {}

    @abstractmethod
    def draft():
        pass

    @abstractmethod
    def draft_extend():
        pass

    @property
    def draft_runners(self) -> list[ModelRunner]:
        """All draft model runners; multi-layer eagle overrides with its
        per-step runner list."""
        return [self.draft_runner]

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        return merge_graph_memory_usage(
            *(runner.graph_memory_usage for runner in self.draft_runners),
            self._specialized_graph_memory_usage,
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        return merge_graph_time_usage(
            *(runner.graph_time_usage for runner in self.draft_runners),
            self._specialized_graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        return sum(runner.weight_load_time for runner in self.draft_runners)

    @property
    def preloaded_weights_bytes(self) -> int:
        return sum(runner.preloaded_weights_bytes for runner in self.draft_runners)

    def alloc_memory_pool(self, **kwargs):
        pass

    def init_attention_backends(self):
        """Subclasses wrap this with their context managers (draft_tp_context,
        speculative_moe_backend_context, etc.) rather than reimplementing it."""
        self.draft_worker.init_attention_backends()
        self.init_attention_backend()

    def init_cuda_graphs(self):
        """Capture draft graphs (decode disabled on the draft TpModelWorker)."""
        self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
        self._capture_cuda_graphs()

    def _rebuild_topk1_chain_buffers(self) -> None:
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        decode_max_bs = (
            get_exec().graph.cuda_graph_config.decode.max_bs
            if get_exec().graph.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            get_schedule().max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)


class BaseSpecWorker(ABC):
    _hicache_draft_plan = HiCacheDraftPlan()

    def __init__(self) -> None:
        self._additional_graph_memory_usage: dict[str, float] = {}
        self._additional_graph_time_usage: dict[str, float] = {}

    @property
    def hicache_draft_plan(self) -> HiCacheDraftPlan:
        return self._hicache_draft_plan

    def _draft_model_runners(self) -> tuple[ModelRunner, ...]:
        spec_algorithm = self.target_worker.model_runner.spec_algorithm
        draft_worker = self.draft_worker
        if (
            draft_worker is None
            or spec_algorithm.is_ngram()
            or spec_algorithm.is_frozen_kv_mtp()
        ):
            return ()
        if spec_algorithm.is_dflash_family():
            return (draft_worker.model_runner,)
        return tuple(draft_worker.draft_runners)

    @property
    def primary_draft_kv_pool(self) -> Optional[object]:
        draft_runners = self._draft_model_runners()
        return draft_runners[0].token_to_kv_pool if draft_runners else None

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> Optional[EagleDraftWorkerBase | TpModelWorker]:
        # dflash / dspark drive the draft model through a plain TpModelWorker;
        # ngram has no draft worker at all (returns None via its override).
        return self._draft_worker

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        if self.draft_worker is None:
            draft_memory_usage = None
        else:
            draft_memory_usage = self.draft_worker.graph_memory_usage
        return merge_graph_memory_usage(
            draft_memory_usage,
            self._additional_graph_memory_usage,
        )

    @property
    def graph_time_usage(self) -> dict[str, float]:
        if self.draft_worker is None:
            draft_time_usage = None
        else:
            draft_time_usage = self.draft_worker.graph_time_usage
        return merge_graph_time_usage(
            draft_time_usage,
            self._additional_graph_time_usage,
        )

    @property
    def weight_load_time(self) -> float:
        if self.draft_worker is None:
            return 0.0
        return self.draft_worker.weight_load_time

    def prewarm_sampling(self) -> SamplingPrewarmResult:
        return self.target_worker.model_runner.prewarm_sampling()

    @property
    def preloaded_weights_bytes(self) -> int:
        if self.draft_worker is None:
            return 0
        return self.draft_worker.preloaded_weights_bytes

    @property
    def last_shared_read_runner(self):
        # The runner that runs the step's LAST shared-buffer-reading phase --
        # it owns the read-done event the scheduler's WAR barrier waits on.
        # Default is the target runner; override if the last phase runs
        # elsewhere (eagle's draft_extend runs on the draft runner).
        return self.target_worker.model_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        """Attn backends touched by spec_v2 forward; OR-ed by decide_needs_cpu_seq_lens.
        Default returns target only; subclasses extend with draft backends."""
        return (self.target_worker.model_runner.attn_backend,)

    def clear_cache_pool(self):
        """Default no-op: the allocator and kv cache pool are shared with the
        target worker and cleared by the scheduler."""
        # TODO: move this method to BaseTpWorker and call through self.model_runner
        pass

    def _build_hicache_draft_plan(self) -> HiCacheDraftPlan:
        target_model_runner = self.target_worker.model_runner
        target_model_runner.mtp_draft_device_pools = ()
        spec_algorithm = target_model_runner.spec_algorithm
        if not (
            get_memory().enable_hierarchical_cache
            or get_disagg().disaggregation_decode_retraction_backup == "host_pool"
        ):
            return HiCacheDraftPlan()

        draft_runners = self._draft_model_runners()
        if not draft_runners:
            return HiCacheDraftPlan()
        draft_pools = tuple(runner.token_to_kv_pool for runner in draft_runners)
        if (
            "InklingForConditionalGenerationMTP"
            in draft_runners[0].model_config.hf_config.architectures
        ):
            raise NotImplementedError(
                "HiCache does not support Inkling MTP draft state yet."
            )

        if _can_pack_hicache_mtp(spec_algorithm, draft_runners):
            target_model_runner.mtp_draft_device_pools = draft_pools
            return HiCacheDraftPlan(
                mode=HiCacheDraftMode.PACKED,
                device_pools=draft_pools,
            )

        return HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,
            # Preserve the legacy non-packed HiCache behavior: multi-layer
            # EAGLE registers only the first draft runner as the sidecar.
            device_pools=draft_pools[:1],
        )

    def init_hicache_draft_plan(self) -> None:
        self._hicache_draft_plan = self._build_hicache_draft_plan()

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        if self.draft_worker is not None:
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        if self.draft_worker is not None:
            self.draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        if self.draft_worker is not None:
            self.draft_worker.init_cuda_graphs()

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        for runner in self.draft_worker.draft_runners:
            success, message = runner.weight_updater.update_weights_from_disk(
                recv_req.model_path,
                recv_req.load_format,
                recapture_cuda_graph=recv_req.recapture_cuda_graph,
            )
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        for runner in self.draft_worker.draft_runners:
            success, message = runner.weight_updater.update_weights_from_ipc(recv_req)
            if not success:
                return success, message
        return True, "Succeeded to update model weights."

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU→CPU sync in the worker hot path.
        """
        pass

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None:
        """Hook called by the batch-result processor when a request finishes.

        Default no-op. DSpark overrides this to settle / censor its
        block-accept estimator state for the finished request.
        """
        pass

    def activate_step_by_batch(self, batch_size: int) -> None:
        """Activate the optimal adaptive step for the current batch size.

        Default no-op. Adaptive-aware workers override this to switch
        the runtime state before each draft round.
        """
        pass
