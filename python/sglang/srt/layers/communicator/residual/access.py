# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Access to a layer output outside the next stage's prepare call."""

from typing import Optional, Tuple, Union

import torch

from sglang.srt.distributed import GroupCoordinator
from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


def buffer(
    hidden_states: Union[torch.Tensor, UnreducedOutput, HandoffOutput],
) -> Optional[torch.Tensor]:
    """Storage that can be reused after prepare consumes the input. A finalize
    handoff has no reusable layer-output tensor. This does not complete or read
    the value held in the storage."""
    if isinstance(hidden_states, UnreducedOutput):
        return hidden_states.partial
    if isinstance(hidden_states, HandoffOutput):
        return None
    return hidden_states


def add_to_output(hidden_states, residual, extra: torch.Tensor) -> Tuple:
    """Complete the output before adding an extra contribution exactly once.
    Leave its residual update for the next prepare call."""
    hidden_states = reduce_output(hidden_states)
    hidden_states.add_(extra)
    return hidden_states, residual


def fold(hidden_states, residual) -> Tuple[torch.Tensor, None]:
    """Finish a plain layer output and fold its residual into a complete value.
    The following stage receives it with no outstanding residual addition."""
    hidden_states = reduce_output(hidden_states)
    if residual is not None:
        hidden_states = hidden_states + residual
    return hidden_states, None


def finish_layer_stack(
    hidden_states: Union[torch.Tensor, UnreducedOutput, HandoffOutput],
    residual: Optional[torch.Tensor],
    forward_batch: ForwardBatch,
    *,
    final_norm_takes_handoff: bool = False,
) -> Tuple[Union[torch.Tensor, HandoffOutput], Optional[torch.Tensor]]:
    """Complete what this layer left for a next layer. Call it on the last
    layer of this rank before its output reaches the final norm, the next
    pipeline rank, or any other consumer outside the layers. A final norm
    that does a producer's handoff together with its own work
    (``final_norm_takes_handoff``) receives it as it is."""
    if final_norm_takes_handoff and isinstance(hidden_states, HandoffOutput):
        return hidden_states, residual
    return reduce_output(hidden_states), residual


def from_pp(
    tensors: PPProxyTensors,
    *,
    residual_in_hidden: bool = False,
    allow_missing_residual: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Receive the layer-stack state without reducing a declared partial sum.
    MHC carries its written streams in hidden_states; ordinary layers receive
    a separate residual tensor."""
    if residual_in_hidden:
        residual = None
    elif allow_missing_residual:
        residual = tensors.tensors.get("residual")
    else:
        residual = tensors["residual"]
    return tensors["hidden_states"], residual


def snapshot(
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
    *,
    group: Optional[GroupCoordinator] = None,
) -> torch.Tensor:
    """Copy a complete output with its plain residual. A statically declared
    sum is reduced on a copy; the main output and residual remain unchanged."""
    if group is not None:
        hidden_states = group.all_reduce(hidden_states.clone())
    return hidden_states.clone() if residual is None else hidden_states + residual
