from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@dataclass
class SpecResourceContext:
    """Resource + config shared across the spec-worker layers (coordinator and
    inner draft executor).

    Constructed once at coordinator `__init__` and passed (or re-read) by the
    inner draft executor. Replaces the duplicated `self.topk = ...` /
    `self.speculative_num_steps = ...` / ... boilerplate copied across nine
    worker `__init__` methods.

    Rank coordinates (gpu_id / tp_rank / dp_rank / ...) are intentionally
    *not* held here:
    - V1 monolithic workers inherit `TpModelWorker`, which stores those as
      instance attributes itself.
    - V2 workers store them as instance attributes too, since they pass them
      through to the inner `TpModelWorker`.
    Centralizing them here would clash with `TpModelWorker.__init__`'s own
    assignments (the property would refuse the instance assignment).
    """

    server_args: "ServerArgs"
    target_worker: "TpModelWorker"
    speculative_algorithm: "SpeculativeAlgorithm"

    # Spec config derived from server_args
    topk: int
    speculative_num_steps: int
    speculative_num_draft_tokens: int
    page_size: int

    # Memory pool refs (shared with target_worker)
    req_to_token_pool: "ReqToTokenPool"
    token_to_kv_pool_allocator: "BaseTokenToKVPoolAllocator"

    @classmethod
    def from_server_args(
        cls,
        server_args: "ServerArgs",
        target_worker: "TpModelWorker",
    ) -> "SpecResourceContext":
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        req_to_token_pool, allocator = target_worker.get_memory_pool()
        return cls(
            server_args=server_args,
            target_worker=target_worker,
            speculative_algorithm=SpeculativeAlgorithm.from_string(
                server_args.speculative_algorithm
            ),
            topk=server_args.speculative_eagle_topk,
            speculative_num_steps=server_args.speculative_num_steps,
            speculative_num_draft_tokens=server_args.speculative_num_draft_tokens,
            page_size=server_args.page_size,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
        )


class DraftExecutor(ABC):
    """Contract for the draft-execution layer of speculative decoding.

    Implementations either run the draft model on `self` (V1 monolithic
    `EAGLEWorker(TpModelWorker, DraftExecutor)`) or wrap an inner draft
    `TpModelWorker` (V2 `EagleDraftWorker`). `draft_runner` is the canonical
    accessor; shape classmethods on `EagleDraftInput` / `EagleDraftExtendInput`
    read it directly.

    The `_ctx` field is the single source of truth for shared config; the
    properties below forward to it. Properties cover only attributes that
    `TpModelWorker.__init__` does *not* itself set (otherwise V1 workers
    would fail at `super().__init__()` because the property has no setter).
    `eagle_use_aux_hidden_state` stays an instance attribute because it is
    derived later, after the draft model is loaded.
    """

    _ctx: SpecResourceContext
    # Whether the draft model consumes EAGLE3 auxiliary hidden states. Always
    # False outside EAGLE3 (e.g. Frozen-KV MTP, standalone).
    eagle_use_aux_hidden_state: bool

    @property
    def target_worker(self) -> "TpModelWorker":
        return self._ctx.target_worker

    @property
    def speculative_algorithm(self) -> "SpeculativeAlgorithm":
        return self._ctx.speculative_algorithm

    @property
    def topk(self) -> int:
        return self._ctx.topk

    @property
    def speculative_num_steps(self) -> int:
        return self._ctx.speculative_num_steps

    @speculative_num_steps.setter
    def speculative_num_steps(self, value: int) -> None:
        # Adaptive controller temporarily overrides this during cuda graph
        # capture (see `_override_worker_state` in V1 / `_AdaptiveSnapshot` in V2).
        self._ctx.speculative_num_steps = value

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self._ctx.speculative_num_draft_tokens

    @speculative_num_draft_tokens.setter
    def speculative_num_draft_tokens(self, value: int) -> None:
        # Same adaptive-override path as `speculative_num_steps`.
        self._ctx.speculative_num_draft_tokens = value

    @property
    def page_size(self) -> int:
        return self._ctx.page_size

    @property
    @abstractmethod
    def draft_runner(self) -> Optional["ModelRunner"]:
        """Primary draft `ModelRunner`. `None` for algorithms with no draft model
        (e.g. NGRAM); for multi-layer algorithms, points to layer 0."""

    @abstractmethod
    def init_attention_backend(self) -> None: ...

    @abstractmethod
    def init_cuda_graphs(self) -> None: ...

    # NOTE: V2 draft executors (`EagleDraftWorker`, `MultiLayerEagleDraftWorker`)
    # additionally expose `draft(...)` and `draft_extend(...)` as their per-phase
    # entry points. V1 monolithic workers (`EAGLEWorker` & subclasses) drive both
    # phases internally from `forward_batch_generation`; they do not expose those
    # methods. Formalizing the per-phase signature is left to a later step where
    # V1 is split into coordinator + executor.


class NullDraftExecutor(DraftExecutor):
    """Concrete no-op `DraftExecutor` for algorithms that derive draft tokens
    without running a draft model (e.g. `NGRAM` corpus lookup).

    The host `SpecCoordinator` keeps a reference to a `NullDraftExecutor` so the
    coordinator/executor type contract holds uniformly across all spec
    algorithms — the draft side is simply absent rather than duck-typed.
    """

    def __init__(self, ctx: SpecResourceContext):
        self._ctx = ctx

    @property
    def draft_runner(self) -> None:
        return None

    @property
    def eagle_use_aux_hidden_state(self) -> bool:
        return False

    def init_attention_backend(self) -> None:
        pass

    def init_cuda_graphs(self) -> None:
        pass


# Backward-compat alias; existing V2 subclasses continue to inherit through this name.
BaseDraftWorker = DraftExecutor


class SpecCoordinator(ABC):
    """Contract for the spec-pipeline-coordinating layer.

    A `SpecCoordinator` drives the draft / verify / extend pipeline. It holds a
    `target_worker` (target model) and a `draft_worker` (a `DraftExecutor`,
    possibly `self` for V1 monolithic workers, or `NullDraftExecutor` for
    no-draft-model algorithms).

    The `_ctx` field is shared with the inner `DraftExecutor` (same object,
    not a copy). Forwarding properties match `DraftExecutor` so coordinator
    code reads `self.topk` / `self.target_worker` / ... uniformly.
    """

    _ctx: SpecResourceContext

    @property
    def target_worker(self) -> "TpModelWorker":
        return self._ctx.target_worker

    @property
    def speculative_algorithm(self) -> "SpeculativeAlgorithm":
        return self._ctx.speculative_algorithm

    @property
    def topk(self) -> int:
        return self._ctx.topk

    @property
    def speculative_num_steps(self) -> int:
        return self._ctx.speculative_num_steps

    @speculative_num_steps.setter
    def speculative_num_steps(self, value: int) -> None:
        self._ctx.speculative_num_steps = value

    @property
    def speculative_num_draft_tokens(self) -> int:
        return self._ctx.speculative_num_draft_tokens

    @speculative_num_draft_tokens.setter
    def speculative_num_draft_tokens(self, value: int) -> None:
        self._ctx.speculative_num_draft_tokens = value

    @property
    def page_size(self) -> int:
        return self._ctx.page_size

    @property
    @abstractmethod
    def draft_worker(self) -> DraftExecutor: ...

    @abstractmethod
    def clear_cache_pool(self) -> None:
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        ...

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU -> CPU sync in the worker hot path.
        """
        pass

    # NOTE: `forward_batch_generation(...)` is the pipeline entry point but its
    # signature is split (`ScheduleBatch` on V1 / `NGRAM`, `ModelWorkerBatch` on
    # V2 / DFlash). Formalizing it as abstract is left to a later step that
    # unifies the input type.


# Backward-compat alias; pre-existing call sites keep working.
BaseSpecWorker = SpecCoordinator
