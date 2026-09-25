"""Aux hidden states captured for Eagle3/DFlash draft models."""

from typing import List, Optional, Protocol, Union, runtime_checkable

import torch

# Two representations coexist: models migrated to AuxHiddenStatePacker pass one
# packed [tokens, K * hidden] tensor, the rest still pass a list of K tensors.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStatePacker:
    """Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into.

    Each ``.append()`` writes into one preallocated ``[tokens, K * hidden]``
    buffer, avoiding the list path's transient ~2x HBM at ``torch.cat``.
    Assumes all captures share leading shape and feature size.

    ``out`` lets a CUDA graph runner supply the destination, so graphs of
    different sizes can alias one buffer instead of each pinning its own.
    """

    # ``append`` copies, so producers need not clone a tensor they later mutate.
    copies_on_append = True

    def __init__(self, num_captures: int, out: Optional[torch.Tensor] = None) -> None:
        self._num_captures = int(num_captures)
        self._buffer = out
        self._feature_size: Optional[int] = None
        self._idx = 0

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

    def __len__(self) -> int:
        return self._idx

    def finalize(self) -> torch.Tensor:
        """Return the packed buffer; callers guard the empty case on ``len()``."""
        if self._idx == 0 or self._idx != self._num_captures:
            raise RuntimeError(
                f"captured {self._idx} of {self._num_captures} aux hidden states"
            )
        return self._buffer


@runtime_checkable
class SupportsSharedAuxHiddenStates(Protocol):
    """Model whose packer accepts ``ForwardBatch.aux_hidden_states_buffer``."""

    def get_packed_aux_hidden_size(self) -> int:
        """Width of the packed output; 0 when nothing is captured."""
        ...


# What a model hands down the capture path: a plain list, or a packer writing in place.
AuxHiddenStateAccumulator = Union[List[torch.Tensor], AuxHiddenStatePacker]


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    if isinstance(aux_hidden_states, torch.Tensor):
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return aux_hidden_states[0]
    return torch.cat(aux_hidden_states, dim=-1)
