from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Optional

import torch

from sglang.srt.configs.draft_parallel import DraftParallelLayout
from sglang.srt.managers.overlap_utils import resolve_forward_inputs
from sglang.srt.model_executor.graph_memory_usage import (
    merge_graph_memory_usage,
    merge_graph_time_usage,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_memory, get_schedule
from sglang.srt.speculative.forward_batch_view import worker_forward_batch
from sglang.srt.speculative.spec_utils import spec_prepare_for_decode

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap, StepPublish
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler_components.pending_steps import PendingStep
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.model_runner import (
        ModelRunner,
        SamplingPrewarmResult,
    )
    from sglang.srt.speculative.spec_info import SpecInput, SpeculativeAlgorithm


class HiCacheDraftMode(str, Enum):
    NONE = "none"
    FUSED = "fused"
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
    draft_parallel_layout: DraftParallelLayout | None = None

    # The relay each forward writes its draft state into and the next step's
    # forward gathers it from; None under the pipeline-parallel relay.
    future_map: Optional[FutureMap] = None
    # The current forward's publish, between its new lengths and its commit.
    _step_publish: Optional[StepPublish] = None
    # DSpark's verify budget, prepared from the relayed confidence on the
    # forward's view before its inputs resolve; None for other workers.
    _confidence_budget_prepare: Optional[Callable[[ScheduleBatch, FutureMap], None]] = (
        None
    )

    def __init__(self) -> None:
        self._additional_graph_memory_usage: dict[str, float] = {}
        self._additional_graph_time_usage: dict[str, float] = {}

    @property
    def hicache_draft_plan(self) -> HiCacheDraftPlan:
        return self._hicache_draft_plan

    @property
    def draft_swa_window(self) -> int:
        """Tokens back from the committed length that the draft reads in the
        target's swa sub-pool; 0 when the target's own window covers them."""
        return 0

    def weight_update_runners(self) -> list[tuple[str, ModelRunner]]:
        """(role, runner) pairs weight ops apply to; [] when the draft owns no weights."""
        # NGRAM and UNO have no draft model
        if self.draft_worker is None:
            return []
        return [("draft", runner) for runner in self.draft_worker.draft_runners]

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
        from sglang.srt.mem_cache.unified_draft_pool import fused_draft_host_allocator

        draft_runners = self._draft_model_runners()
        if not draft_runners:
            return None
        pool = draft_runners[0].token_to_kv_pool
        if fused_draft_host_allocator(pool) is not None:
            return None
        return pool

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
        target_model_runner.mtp_draft_mamba_pools = ()
        spec_algorithm = target_model_runner.spec_algorithm
        if not (
            get_memory().enable_hierarchical_cache
            or get_memory().enable_unified_cache_external_linker
            or get_disagg().disaggregation_decode_retraction_backup == "host_pool"
        ):
            return HiCacheDraftPlan()

        draft_runners = self._draft_model_runners()
        if not draft_runners:
            return HiCacheDraftPlan()
        draft_pools = tuple(runner.token_to_kv_pool for runner in draft_runners)
        from sglang.srt.mem_cache.unified_draft_pool import (
            fused_draft_host_allocator,
        )

        if any(fused_draft_host_allocator(pool) is not None for pool in draft_pools):
            from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
                validate_fused_hicache_draft,
            )

            validate_fused_hicache_draft(
                target_config=target_model_runner.kv_cache_configurator,
                draft_device_pools=draft_pools,
            )
            # The target's page envelopes already carry every draft layer.
            # Packing or adding a sidecar would back up the same bytes twice.
            return HiCacheDraftPlan(
                mode=HiCacheDraftMode.FUSED, device_pools=draft_pools
            )
        if _can_pack_hicache_mtp(spec_algorithm, draft_runners):
            target_model_runner.mtp_draft_device_pools = draft_pools
            target_mamba_pool = getattr(
                target_model_runner.req_to_token_pool, "mamba_pool", None
            )
            draft_mamba_pools = tuple(
                getattr(runner.req_to_token_pool, "mamba_pool", None)
                for runner in draft_runners
            )
            # Shared state is already backed up with the target. Separate
            # draft state follows the target's Mamba slot indices.
            if any(
                pool is not None and pool is not target_mamba_pool
                for pool in draft_mamba_pools
            ):
                assert target_mamba_pool is not None, (
                    "Packed draft state requires target Mamba slot indices"
                )
                assert all(
                    pool is not None and pool is not target_mamba_pool
                    for pool in draft_mamba_pools
                ), "Packed MTP requires separate state for every draft layer"
                target_model_runner.mtp_draft_mamba_pools = draft_mamba_pools
            return HiCacheDraftPlan(
                mode=HiCacheDraftMode.PACKED,
                device_pools=draft_pools,
            )

        if get_memory().enable_unified_cache_external_linker and len(draft_pools) > 1:
            raise NotImplementedError(
                "The external linker supports one non-packed draft KV pool."
            )

        return HiCacheDraftPlan(
            mode=HiCacheDraftMode.SIDECAR,
            # Preserve the legacy non-packed HiCache behavior: multi-layer
            # EAGLE registers only the first draft runner as the sidecar.
            device_pools=draft_pools[:1],
        )

    def init_hicache_draft_plan(self) -> None:
        self._hicache_draft_plan = self._build_hicache_draft_plan()

    def _wait_for_fused_hicache_load(self, batch) -> None:
        if (
            self.hicache_draft_plan.mode == HiCacheDraftMode.FUSED
            and batch.hicache_consumer_index >= 0
        ):
            counter = self.target_worker.hicache_layer_transfer_counter
            # A draft's layer ids do not identify target transfer events. Wait
            # before dispatch so CUDA graph replay observes this dependency too.
            counter.events[batch.hicache_consumer_index].wait(counter.num_layers - 1)

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

    def attach_future_map(self, future_map: FutureMap, *, enable_overlap: bool) -> None:
        """Hand the worker the FutureMap its forwards commit their draft state
        to, when draft state travels through it (draft_state_in_future_map)."""
        from sglang.srt.managers.overlap_utils import draft_state_in_future_map

        if draft_state_in_future_map(enable_overlap):
            self.future_map = future_map

    def prepare_decode_step(self, step: ScheduleBatch) -> None:
        """The worker's part of building a decode step: key its draft input to
        its rows' slots when the draft state travels through FutureMap (under
        the pipeline-parallel relay the launch builds the verify input), then
        the algorithm's decode preparation."""
        if self.future_map is not None:
            # The rows' draft state is in their FutureMap slots; the worker
            # gathers it at forward entry.
            step.spec_info = self.keyed_draft_input(
                step.req_pool_indices, step.req_pool_indices_cpu
            )
        spec_prepare_for_decode(step)
        self._prepare_decode(step)

    def _prepare_decode(self, step: ScheduleBatch) -> None:
        """The algorithm's decode preparation: EAGLE's allocation of the
        verify and draft KV; a worker whose draft input prepares itself
        overrides it."""
        from sglang.srt.speculative.eagle_utils import eagle_prepare_for_decode

        eagle_prepare_for_decode(step)

    def keyed_draft_input(
        self, req_pool_indices: torch.Tensor, req_pool_indices_cpu: torch.Tensor
    ) -> SpecInput:
        """The draft input of a decode step over these rows, built from their
        slots instead of carried from the previous result: it keys the slots
        the worker gathers the rows' draft state from, and says whether that
        state lets the draft replay its graphs."""
        draft_input = self._new_draft_input(req_pool_indices)
        draft_input.future_indices = req_pool_indices
        draft_input.cuda_graph_compatible = self.future_map.cuda_graph_compatible(
            req_pool_indices_cpu
        )
        return draft_input

    def _new_draft_input(self, req_pool_indices: torch.Tensor) -> SpecInput:
        """An empty draft input of the type this worker's forwards produce;
        workers carrying another type override it."""
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        return EagleDraftInput()

    def forward_batch_generation(
        self,
        step: ScheduleBatch,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        pending_grammar: Optional[Iterable[PendingStep]] = None,
    ) -> GenerationBatchResult:
        """Run one step. `step` is read only: the forward runs on the worker's
        own batches, and its draft state goes to FutureMap at commit. A verify
        runs the grammar commit of the `pending_grammar` steps right before it
        builds its vocab mask."""
        batch = self.forward_input(step)
        on_publish = partial(self.publish_new_seq_lens, batch)
        result = self._forward(batch, on_publish, pp_proxy_tensors, pending_grammar)
        return self.commit_forward(batch, result)

    @abstractmethod
    def _forward(
        self,
        batch: ScheduleBatch,
        on_publish: Callable[[torch.Tensor], None],
        pp_proxy_tensors: Optional[PPProxyTensors],
        pending_grammar: Optional[Iterable[PendingStep]],
    ) -> GenerationBatchResult:
        """The forward of one step on the worker's batch. It calls
        `on_publish` with the step's new lengths as soon as they are known."""

    def forward_input(self, step: ScheduleBatch) -> ScheduleBatch:
        """The batch this worker forwards for `step`, which it never writes.

        Its input ids are resolved from FutureMap on a view of the step; under
        the pipeline-parallel relay the ring hands over a view it resolved. A
        step that keys its draft input into FutureMap gets the draft state and
        the committed lengths of its rows gathered from their slots, after the
        steps that last wrote them published.
        """
        if self.future_map is None:
            return worker_forward_batch(step)
        view = worker_forward_batch(step)
        if self._confidence_budget_prepare is not None:
            self._confidence_budget_prepare(view, self.future_map)
        resolve_forward_inputs(view, self.future_map)
        carried = view.spec_info
        if carried is None or carried.future_indices is None:
            return worker_forward_batch(view)
        indices_cpu = view.req_pool_indices_cpu
        seq_lens, seq_lens_cpu, seq_lens_sum = self.future_map.gather_seq_lens(
            carried.future_indices, indices_cpu
        )
        return worker_forward_batch(
            view,
            spec_info=self.future_map.gather_draft_input(carried, indices_cpu),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=seq_lens_sum,
        )

    def publish_new_seq_lens(
        self,
        batch: ScheduleBatch,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        """Write the forward's new lengths into its slots as soon as they are
        known (verify end), so the next step's host read of them overlaps the
        rest of this forward."""
        if self.future_map is None:
            return
        self._step_publish = self.future_map.publish(
            batch.req_pool_indices,
            new_seq_lens,
            confidence,
            indices_cpu=batch.req_pool_indices_cpu,
        )

    def commit_forward(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> GenerationBatchResult:
        """The device commit of a forward: write its next draft input into the
        slots of its rows and seal its publish. Leaves `result` complete: the
        next draft input keys its rows, and the result carries the publish."""
        publish, self._step_publish = self._step_publish, None
        if self.future_map is None:
            return result
        next_draft_input = result.next_draft_input
        if next_draft_input is not None:
            self.future_map.stash(
                batch.req_pool_indices,
                next_draft_input.relay_payload(),
                indices_cpu=batch.req_pool_indices_cpu,
            )
            next_draft_input.future_dsa_topk_indices_available = (
                next_draft_input.dsa_topk_indices is not None
            )
            next_draft_input.future_indices = batch.req_pool_indices
        self.future_map.seal(publish)
        result.step_publish = publish
        return result
