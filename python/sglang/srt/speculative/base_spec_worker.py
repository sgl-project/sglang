from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class DraftExecutor(ABC):
    """Contract for the draft-execution layer of speculative decoding.

    Implementations either run the draft model on `self` (V1 monolithic
    `EAGLEWorker(TpModelWorker, DraftExecutor)`) or wrap an inner draft
    `TpModelWorker` (V2 `EagleDraftWorker`). `draft_runner` is the canonical
    accessor; shape classmethods on `EagleDraftInput` / `EagleDraftExtendInput`
    read it directly.

    The fields below (`target_worker`, `speculative_algorithm`,
    `eagle_use_aux_hidden_state`) are declared as type annotations rather than
    `@abstractmethod` properties because subclasses set them as instance attrs
    in `__init__`; Python ABC would reject instance attrs as a valid override
    of a strict abstract property.
    """

    target_worker: "TpModelWorker"
    speculative_algorithm: "SpeculativeAlgorithm"
    # Whether the draft model consumes EAGLE3 auxiliary hidden states. Always
    # False outside EAGLE3 (e.g. Frozen-KV MTP, standalone).
    eagle_use_aux_hidden_state: bool

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

    def __init__(
        self,
        target_worker: "TpModelWorker",
        speculative_algorithm: "SpeculativeAlgorithm",
    ):
        self._target_worker = target_worker
        self._speculative_algorithm = speculative_algorithm

    @property
    def draft_runner(self) -> None:
        return None

    @property
    def target_worker(self) -> "TpModelWorker":
        return self._target_worker

    @property
    def speculative_algorithm(self) -> "SpeculativeAlgorithm":
        return self._speculative_algorithm

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

    `target_worker` and `speculative_algorithm` are declared as type annotations
    rather than abstract properties — see `DraftExecutor` for the same rationale.
    """

    target_worker: "TpModelWorker"
    speculative_algorithm: "SpeculativeAlgorithm"

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
