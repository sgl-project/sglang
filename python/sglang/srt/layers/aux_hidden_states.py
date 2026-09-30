"""Aux hidden states captured for Eagle3/DFlash draft models."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStateList(list):
    """Retain aux snapshots independently of storage reused by later layers."""

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Copy borrowed storage, or adopt a value whose ownership is transferred.

        Views and reusable communication buffers are borrowed even when they
        are different tensor objects from the source.
        """
        self.append(hidden if owned else hidden.clone())


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.

    ``for_batch`` writes into a decode graph runner's buffer when the batch
    carries one, so every graph size aliases it instead of pinning its own.
    """

    def __init__(self, num_captures: int, out: Optional[torch.Tensor] = None) -> None:
        self._num_captures = int(num_captures)
        self._buffer = out
        self._feature_size: Optional[int] = None
        self._idx = 0

    @classmethod
    def for_batch(
        cls, forward_batch: ForwardBatch, num_captures: int
    ) -> AuxHiddenStatePacker:
        return cls(num_captures, out=forward_batch.aux_hidden_states_buffer)

    def append(self, hidden: torch.Tensor) -> None:
        feature_size = int(hidden.shape[-1])
        if self._feature_size is None:
            self._feature_size = feature_size
            shape = (*hidden.shape[:-1], feature_size * self._num_captures)
            if self._buffer is None:
                self._buffer = hidden.new_empty(shape)
            elif self._buffer.shape != shape or self._buffer.dtype != hidden.dtype:
                raise ValueError(
                    f"aux hidden output buffer is {tuple(self._buffer.shape)} "
                    f"{self._buffer.dtype}, captures need {shape} {hidden.dtype}"
                )
        start = self._idx * self._feature_size
        self._buffer[..., start : start + self._feature_size].copy_(hidden)
        self._idx += 1

    def capture(self, hidden: torch.Tensor, *, owned: bool = False) -> None:
        """Write directly to the final packed buffer, without an intermediate copy."""
        self.append(hidden)

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        if self._feature_size is None or self._idx != self._num_captures:
            raise RuntimeError(
                f"captured {self._idx} of {self._num_captures} aux hidden states"
            )
        return self._buffer


# Both collectors accept borrowed values through capture; storage belongs to aux.
AuxHiddenStateAccumulator = Union[AuxHiddenStateList, AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)
