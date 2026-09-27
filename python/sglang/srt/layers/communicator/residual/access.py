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

from sglang.srt.layers.communicator.output import (
    HandoffOutput,
    UnreducedOutput,
    reduce_output,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


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
