# Copyright 2023-2026 SGLang Team
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

"""CP-v2 token-layout transitions for Kimi-Linear decoder layers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.srt.layers.cp.utils import get_cp_strategy, is_cp_v2_active

if TYPE_CHECKING:
    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class KimiLinearCPV2LayerCommunicator:
    """Convert Kimi-Linear layer inputs between KDA and MLA token layouts."""

    def __init__(
        self,
        *,
        is_kda_layer: bool,
        previous_is_kda_layer: Optional[bool],
    ) -> None:
        self._gather_before_attn = is_kda_layer and (previous_is_kda_layer is not True)
        self._shard_before_attn = not is_kda_layer and previous_is_kda_layer is True

    def prepare_attn(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not is_cp_v2_active(forward_batch):
            return hidden_states, residual

        strategy = get_cp_strategy()
        assert strategy is not None
        if self._gather_before_attn:
            hidden_states = strategy.gather_hidden_states(
                hidden_states, forward_batch, stream
            )
        elif self._shard_before_attn:
            hidden_states = strategy.shard_hidden_states(hidden_states, forward_batch)
            if residual is not None:
                residual = strategy.shard_hidden_states(residual, forward_batch)
        return hidden_states, residual
