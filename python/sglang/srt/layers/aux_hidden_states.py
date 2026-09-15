"""Aux hidden states captured for Eagle3/DFlash draft models."""

from typing import List, Optional, Union

import torch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.
    """

    # ``append`` copies, so producers need not clone a tensor they later mutate.
    copies_on_append = True

    def __init__(self, num_captures: int) -> None:
        self._num_captures = int(num_captures)
        self._buffer: Optional[torch.Tensor] = None
        self._feature_size: Optional[int] = None
        self._idx = 0

    def append(self, hidden: torch.Tensor) -> None:
        feature_size = int(hidden.shape[-1])
        if self._buffer is None:
            self._feature_size = feature_size
            self._buffer = hidden.new_empty(
                (*hidden.shape[:-1], feature_size * self._num_captures)
            )
        start = self._idx * self._feature_size
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        if self._buffer is None or self._idx != self._num_captures:
            raise RuntimeError(
                f"captured {self._idx} of {self._num_captures} aux hidden states"
            )
        return self._buffer


# What a model hands down the capture path: a plain list, or a packer writing in place.
AuxHiddenStateAccumulator = Union[List[torch.Tensor], AuxHiddenStatePacker]


# Runtime-attached capture is used by models whose decoder stack follows the
# standard SGLang residual-stream contract but does not implement a model-local
# DFlash/DSpark capture hook. The attributes are deliberately private: they
# are an executor/model boundary, not checkpoint configuration.
RUNTIME_AUX_HIDDEN_STATES_ATTR = "_sglang_runtime_aux_hidden_states"
RUNTIME_AUX_CAPTURE_LOGITS_ATTR = "_sglang_use_runtime_aux_hidden_states"


def resolve_runtime_aux_hidden_states(
    logits_processor,
    hidden_states,
    logits_metadata,
    aux_hidden_states: Optional[AuxHiddenStates],
):
    """Unpack aux states produced by a runtime-attached decoder stack.

    Body-only prefill graphs return ``(hidden_states, aux_hidden_states)`` so
    every captured tensor remains a graph output. Ordinary and split-prefill
    forwards can instead carry the same list on ``ForwardBatch``.
    """
    if not getattr(logits_processor, RUNTIME_AUX_CAPTURE_LOGITS_ATTR, False):
        return hidden_states, aux_hidden_states

    if isinstance(hidden_states, tuple):
        if len(hidden_states) != 2:
            raise RuntimeError(
                "Runtime auxiliary hidden-state capture expected a two-item "
                "transformer output."
            )
        hidden_states, runtime_aux_hidden_states = hidden_states
        if aux_hidden_states is None:
            aux_hidden_states = runtime_aux_hidden_states
    elif aux_hidden_states is None:
        aux_hidden_states = getattr(
            logits_metadata, RUNTIME_AUX_HIDDEN_STATES_ATTR, None
        )

    return hidden_states, aux_hidden_states


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)
