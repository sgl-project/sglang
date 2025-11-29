# Copyright 2023-2025 SGLang Team
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

"""FusedMoE layer with LoRA support."""

import torch
from torch import nn

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.lora.backend.base_backend import BaseLoRABackend


class FusedMoEWithLoRA(nn.Module):
    """
    Wrapper around FusedMoE that adds parallel LoRA computation.

    Design: Base MoE and LoRA Delta run independently and merge at the end.
    This preserves SGLang's existing 3-stage MoE architecture unchanged.
    """

    def __init__(
        self,
        base_moe: FusedMoE,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_moe = base_moe
        self.lora_backend = lora_backend
        self.lora_enabled = False

        # LoRA tensors will be set by LoRAManager
        self.lora_a_weights = None
        self.lora_b_weights = None

    def set_lora_info(
        self,
        lora_a_weights: torch.Tensor,
        lora_b_weights: torch.Tensor,
    ):
        """Set LoRA weight tensors from memory pool."""
        self.lora_enabled = True
        self.lora_a_weights = lora_a_weights
        self.lora_b_weights = lora_b_weights

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
        """
        Forward pass with parallel LoRA computation.

        Flow:
        1. Base MoE forward
        2. Parallel LoRA delta computation (if enabled, added in-place)
        3. Return modified base_output
        """
        # Run base MoE
        base_output = self.base_moe.forward(hidden_states, topk_output, **kwargs)

        # If LoRA is enabled, compute delta and add in-place for memory efficiency
        if self.lora_enabled and self.lora_a_weights is not None:
            self._compute_lora_delta(hidden_states, topk_output, base_output)

        return base_output

    def _compute_lora_delta(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        base_output: torch.Tensor,
    ) -> None:
        """
        Compute LoRA delta using per-expert LoRA weights and add to base_output in-place.

        Dispatch tokens to experts and compute per-expert deltas.
        """
        from sglang.srt.lora.moe_dispatch import moe_dispatch
        from sglang.srt.lora.triton_ops.per_expert_lora_moe import (
            per_expert_lora_forward,
        )

        # Get dispatch info from TopKOutput
        topk_ids = topk_output.topk_ids  # [num_tokens, top_k]
        topk_weights = topk_output.topk_weights  # [num_tokens, top_k]

        # Get LoRA batch info from backend
        batch_info = self.lora_backend.batch_info
        lora_ranks = batch_info.lora_ranks  # [num_loras]
        scalings = batch_info.scalings  # [num_loras]

        # Use precomputed per-token LoRA indices from forward batch
        lora_indices = self.lora_backend.forward_batch.token_lora_indices

        num_experts = self.base_moe.num_experts
        num_loras = self.lora_a_weights.shape[0]

        # Dispatch tokens to experts
        token_ids, expert_ids, _, lora_ids = moe_dispatch(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            lora_indices=lora_indices,
            num_experts=num_experts,
            num_loras=num_loras,
        )



        # Compute per-expert LoRA forward (adds to base_output in-place)
        per_expert_lora_forward(
            hidden_states=hidden_states,
            lora_a_weights=self.lora_a_weights,
            lora_b_weights=self.lora_b_weights,
            token_ids=token_ids,
            expert_ids=expert_ids,
            lora_ids=lora_ids,
            lora_ranks=lora_ranks,
            lora_scalings=scalings,
            num_experts=num_experts,
            base_output=base_output,
        )
