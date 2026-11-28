# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
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

"""Qwen2 action expert of LingBot-VLA 2.0: token-routed MoE layers and time-conditioned AdaRMSNorm."""

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2MLP,
    Qwen2RMSNorm,
)


class Qwen2MoeSharedExpertMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2FusedExperts(nn.Module):
    """Packed expert weights and padded sparse execution with reference routing arithmetic."""

    def __init__(
        self, num_experts, hidden_size, intermediate_size, initializer_range=0.02
    ):
        super().__init__()
        self.num_experts = num_experts
        self.intermediate_size = intermediate_size
        self.initializer_range = initializer_range
        self.gate_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.up_proj = nn.Parameter(
            torch.empty(num_experts, intermediate_size, hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size)
        )
        self.register_buffer("_gate_up_proj_cache", None, persistent=False)
        self.register_buffer("_sparse_wd_cache", None, persistent=False)
        self._sparse_cache_key = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.gate_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.up_proj, mean=0.0, std=self.initializer_range)
        nn.init.normal_(self.down_proj, mean=0.0, std=self.initializer_range)
        self.clear_inference_cache()

    def clear_inference_cache(self):
        self._gate_up_proj_cache = None
        self._sparse_wd_cache = None
        self._sparse_cache_key = None

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.clear_inference_cache()

    def _sparse_packed_weights(self):
        """Cache packed inference weights by parameter version."""
        cache_key = (
            self.gate_proj._version,
            self.up_proj._version,
            self.down_proj._version,
        )
        if self._gate_up_proj_cache is None or self._sparse_cache_key != cache_key:
            with torch.no_grad():
                self._gate_up_proj_cache = (
                    torch.cat([self.gate_proj, self.up_proj], dim=1)
                    .transpose(1, 2)
                    .contiguous()
                )
                self._sparse_wd_cache = self.down_proj.transpose(1, 2).contiguous()
            self._sparse_cache_key = cache_key
        return self._gate_up_proj_cache, self._sparse_wd_cache

    def _sparse_forward(
        self,
        routing_weights: torch.Tensor,
        selected_experts: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Group tokens by expert, run padded batched GEMMs, and accumulate routed outputs."""
        num_tokens, hidden_size = hidden_states.shape
        num_experts, intermediate_size = self.gate_proj.shape[:2]
        top_k = selected_experts.shape[-1]

        flat_expert_ids = selected_experts.reshape(-1)
        route_order = torch.argsort(flat_expert_ids)
        sorted_expert_ids = flat_expert_ids[route_order]
        sorted_token_ids = route_order // top_k
        # note (0xtoward): bincount needs a host synchronization that breaks graph capture.
        expert_counts = torch.zeros(
            num_experts, dtype=torch.long, device=hidden_states.device
        )
        expert_counts.scatter_add_(0, flat_expert_ids, torch.ones_like(flat_expert_ids))
        expert_offsets = torch.cumsum(expert_counts, 0) - expert_counts
        expert_slots = (
            torch.arange(num_tokens * top_k, device=hidden_states.device)
            - expert_offsets[sorted_expert_ids]
        )

        expert_inputs = torch.zeros(
            num_experts,
            num_tokens,
            hidden_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_inputs[sorted_expert_ids, expert_slots] = hidden_states[sorted_token_ids]
        gate_up_weights, down_weights = self._sparse_packed_weights()
        gate_up = torch.bmm(expert_inputs, gate_up_weights)
        intermediate = (
            F.silu(gate_up[..., :intermediate_size]) * gate_up[..., intermediate_size:]
        )
        padded_outputs = torch.bmm(intermediate, down_weights)

        expert_outputs = padded_outputs[sorted_expert_ids, expert_slots]
        # note (0xtoward): disjoint route slots avoid a floating-point atomic reduction.
        routed = torch.empty_like(expert_outputs).index_copy_(
            0, route_order, expert_outputs
        )
        routed = routed.reshape(num_tokens, top_k, hidden_size).float()
        routed = routed * routing_weights.unsqueeze(-1).float()
        # note (0xtoward): expert order matches deterministic index_add and preserves near-tie routing.
        expert_order = (
            selected_experts.argsort(dim=1).unsqueeze(-1).expand(-1, -1, hidden_size)
        )
        output, *remaining = routed.gather(1, expert_order).unbind(1)
        for contribution in remaining:
            output = output + contribution
        return output.to(hidden_states.dtype)

    def forward(self, routing_weights, selected_experts, hidden_states):
        """Run the routed experts with packed weights."""
        return self._sparse_forward(routing_weights, selected_experts, hidden_states)


class FixQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        FixQwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class AdaRMSNorm(nn.Module):
    """RMSNorm followed by a FiLM modulation from the flow-matching time embedding."""

    def __init__(self, hidden_size, cond_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.gamma = nn.Linear(cond_dim, hidden_size)
        self.beta = nn.Linear(cond_dim, hidden_size)

        # note (0xtoward): zero modulation preserves the initial RMSNorm behavior.
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, hidden_states, cond):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        hidden_states = self.weight * hidden_states
        cond = cond.to(torch.float32)
        gamma = self.gamma(cond).unsqueeze(1)  # [B, 1, H]
        beta = self.beta(cond).unsqueeze(1)  # [B, 1, H]
        hidden_states = (1 + gamma.to(torch.float32)) * hidden_states + beta.to(
            torch.float32
        )
        return hidden_states.to(input_dtype)


def replace_lnorm_with_adanorm(module, hidden_size, cond_dim, final_norm_adanorm):
    """Swap the expert's layer RMSNorms, and its final norm when final_norm_adanorm is set, for AdaRMSNorm."""
    norm_types = (
        (Qwen2RMSNorm, FixQwen2RMSNorm) if final_norm_adanorm else (Qwen2RMSNorm,)
    )
    for name, child in module.named_children():
        if isinstance(child, norm_types):
            setattr(module, name, AdaRMSNorm(hidden_size, cond_dim))
        else:
            replace_lnorm_with_adanorm(child, hidden_size, cond_dim, final_norm_adanorm)


class Qwen2TokenMoeBlock(nn.Module):
    """Token-level routing MoE block with all-to-all computation for torch.compile compatibility."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # note (0xtoward): inference uses the routing bias learned by the checkpoint.
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(config.num_experts),
            persistent=True,
        )

        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen2FusedExperts(
            self.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            initializer_range=config.initializer_range,
        )

        self.shared_expert = Qwen2MoeSharedExpertMLP(
            config, intermediate_size=config.shared_expert_intermediate_size
        )
        self._router_activation = config.router_activation
        self.routed_scaling_factor = config.routed_scaling_factor
        self._use_shared_expert_gate = config.use_shared_expert_gate
        if self._use_shared_expert_gate:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route each token and combine routed and shared expert outputs."""
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_flat = hidden_states.reshape(-1, hidden_dim)
        # note (0xtoward): bf16 rounding can change the selected experts on near-ties.
        with torch.amp.autocast(hidden_flat.device.type, enabled=False):
            router_logits = F.linear(hidden_flat.float(), self.gate.weight.float())

        if self._router_activation == "sigmoid":
            routing_scores = router_logits.sigmoid()
        else:
            routing_scores = F.softmax(router_logits, dim=1, dtype=torch.float)

        scores_for_choice = routing_scores + self.e_score_correction_bias.unsqueeze(0)
        _, selected_experts = torch.topk(scores_for_choice, self.top_k, dim=-1)
        routing_weights = routing_scores.gather(1, selected_experts)
        if self.norm_topk_prob:
            routing_weights = routing_weights / (
                routing_weights.sum(dim=-1, keepdim=True) + 1e-20
            )
        if self.routed_scaling_factor != 1.0:
            routing_weights = routing_weights * self.routed_scaling_factor
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            routing_weights=routing_weights,
            selected_experts=selected_experts,
            hidden_states=hidden_flat,
        )

        if final_hidden_states.dtype != hidden_flat.dtype:
            final_hidden_states = final_hidden_states.to(hidden_flat.dtype)
        shared_expert_output = self.shared_expert(hidden_flat)
        if self._use_shared_expert_gate:
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_flat)) * shared_expert_output
            )
        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int | None = 0,
        end: int | None = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        ada_cond: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)
        if ada_cond is not None:
            ada_cond = ada_cond.to(param_dtype)

        if compute_kqv:
            if ada_cond is not None:
                hidden_states = self.input_layernorm(hidden_states, ada_cond)
            else:
                hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)

            query_state = self.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = self.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)

            return query_state, key_state, value_state

        elif output_atten:
            assert att_output is not None
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])

            out_emb += hidden_states
            residual = out_emb
            if ada_cond is not None:
                out_emb = self.post_attention_layernorm(out_emb, ada_cond)
            else:
                out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb += residual

            return out_emb

        else:
            raise ValueError(
                f"Invalid operation: compute_kqv={compute_kqv}, output_atten={output_atten}"
            )


class Qwen2PreTrainedModel(PreTrainedModel):
    config: Qwen2Config
    base_model_prefix = "model"

    def _init_weights(self, module) -> None:
        """Weights come from the checkpoint; skip Hugging Face initialization."""
        return


class Qwen2Model(Qwen2PreTrainedModel):
    """Expert decoder stack, driven layer by layer by the LingBot dual-stream attention."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = FixQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class Qwen2ActionExpert(Qwen2PreTrainedModel):
    """Keeps the checkpoint layout (qwen_expert.model.*); the expert has no embeddings or LM head."""

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.post_init()
