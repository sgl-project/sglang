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

from sglang.srt.layers.layer_boundary.output import DeferredFinalize, UnreducedOutput
from sglang.srt.layers.layer_boundary.residual.stream import OwedOutput
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def buffer(
    hidden_states: Union[torch.Tensor, UnreducedOutput, DeferredFinalize],
) -> Optional[torch.Tensor]:
    """Storage that can be reused after prepare consumes the input. A finalize
    handoff has no reusable layer-output tensor. This does not complete or read
    the value held in the storage."""
    if isinstance(hidden_states, OwedOutput):
        return hidden_states.contribution.value
    if isinstance(hidden_states, UnreducedOutput):
        return hidden_states.partial
    if isinstance(hidden_states, DeferredFinalize):
        return None
    return hidden_states


def final_norm_pair(hidden_states, residual, norm, capture=None, **read_kwargs):
    """The final norm and an optional capture of the same updated residual.

    Keep add and norm together: normalizing a separately rounded residual is
    not numerically equivalent to the fused kernel's FP32 accumulation. The
    capture callback owns retention of the borrowed residual storage.
    """
    if residual is None:
        if capture is not None:
            capture(hidden_states)
        return norm(hidden_states)
    hidden_states, residual = norm(hidden_states, residual, **read_kwargs)
    if capture is not None:
        capture(residual)
    return hidden_states


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
