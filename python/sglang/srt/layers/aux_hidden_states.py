"""
Aux hidden states captured for Eagle3/DFlash draft models.
"""

from typing import List, Optional, Union

import torch

# Aux hidden states arrive as either a list of K [tokens, hidden] tensors (concatenated
# here -> transient ~2x HBM) or a pre-packed [tokens, K * hidden] tensor written in place
# by AuxHiddenStatePacker (no cat). LogitsProcessor branches on which; both paths stay
# live since models that haven't opted into packing still pass a list.
AuxHiddenStates = Union[torch.Tensor, List[torch.Tensor]]


class AuxHiddenStatePacker:
    """Append-compatible accumulator that packs aux hidden states in place.

    Drop-in for the ``[]`` a model collects Eagle3/DFlash captures into: each
    ``.append()`` writes straight into one preallocated ``[tokens, K * hidden]``
    buffer and ``.finalize()`` returns it. Avoids the legacy list path's transient
    ~2x HBM (K separate tensors plus their ``torch.cat`` in ``LogitsProcessor``).
    Assumes all captures share leading shape and feature size.
    """

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

    def finalize(self) -> Optional[torch.Tensor]:
        """Return the packed buffer, narrowed if fewer layers were captured than
        ``num_captures``. Returns ``None`` when nothing was captured."""
        if self._buffer is None:
            return None
        if self._idx != self._num_captures:
            return self._buffer[..., : self._idx * self._feature_size]
        return self._buffer


def pack_aux_hidden_states(aux_hidden_states: AuxHiddenStates) -> torch.Tensor:
    # The point where the two representations converge to one packed tensor.
    if isinstance(aux_hidden_states, torch.Tensor):
        # Already packed in place by the producer -- no copy.
        return aux_hidden_states
    # Legacy list path: concatenate K tensors, transiently ~2x aux HBM (see AuxHiddenStates).
    return torch.cat(aux_hidden_states, dim=-1)
