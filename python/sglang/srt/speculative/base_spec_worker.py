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
    """

    @property
    @abstractmethod
    def draft_runner(self) -> Optional["ModelRunner"]:
        """Primary draft `ModelRunner`. `None` for algorithms with no draft model
        (e.g. NGRAM); for multi-layer algorithms, points to layer 0."""

    @property
    @abstractmethod
    def target_worker(self) -> "TpModelWorker": ...

    @property
    @abstractmethod
    def speculative_algorithm(self) -> "SpeculativeAlgorithm": ...

    @property
    @abstractmethod
    def eagle_use_aux_hidden_state(self) -> bool:
        """Whether the draft model consumes EAGLE3 auxiliary hidden states.
        Always False outside EAGLE3 (e.g. Frozen-KV MTP, standalone)."""

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


# Backward-compat alias; existing V2 subclasses continue to inherit through this name.
BaseDraftWorker = DraftExecutor


class BaseSpecWorker(ABC):
    @property
    @abstractmethod
    def target_worker(self) -> TpModelWorker:
        pass

    @property
    @abstractmethod
    def draft_worker(self) -> DraftExecutor:
        pass

    @abstractmethod
    def clear_cache_pool(self):
        # TODO: move this abstract method to BaseTpWorker and call through self.model_runner
        pass

    def on_verify_complete_cpu(self, num_correct_drafts_per_req: list[int]) -> None:
        """Hook called after verify finishes and accept counts are on CPU.

        Default no-op. Adaptive-aware workers override this to feed the
        controller without forcing a GPU -> CPU sync in the worker hot path.
        """
        pass
