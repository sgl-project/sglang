"""Explicit model capabilities used by context-parallel runners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Protocol, Tuple, Union, runtime_checkable

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.logits_processor import LogitsMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class LanguageModelInputPreparation:
    """Opt in to preparing global model inputs before CP sharding.

    The default is the identity preparation used by text-only models.
    """

    def prepare_language_model_inputs(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return input_ids, input_embeds


def prepare_language_model_inputs(
    model,
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch only to models that explicitly declare input preparation."""
    if isinstance(model, LanguageModelInputPreparation):
        return model.prepare_language_model_inputs(
            input_ids, forward_batch, input_embeds
        )
    return input_ids, input_embeds


@dataclass
class CPLogitsInputs:
    input_ids: torch.Tensor
    logits_metadata: Union[ForwardBatch, LogitsMetadata]
    hidden_states_token_indices: Optional[torch.Tensor] = None


class CPBoundedTailLogits(ABC):
    """Opt in to adapting global CP outputs to a model's bounded logits tail."""

    @abstractmethod
    def prepare_cp_logits_inputs(
        self, input_ids: torch.Tensor, forward_batch: ForwardBatch
    ) -> CPLogitsInputs:
        """Return model-owned logits inputs and the optional output row mapping."""
        raise NotImplementedError


@runtime_checkable
class ModelForwardPreparation(Protocol):
    def prepare_for_forward(self, forward_batch: ForwardBatch) -> None: ...


@runtime_checkable
class CPOutputPostprocessing(Protocol):
    def postprocess_cp_output(self, output, forward_batch: ForwardBatch): ...


@runtime_checkable
class DecoderTailGraphCapture(Protocol):
    def capture_decoder_tail_graphs(
        self, model_runner, eager_runner, prefill_runner
    ) -> tuple[float, float]: ...
