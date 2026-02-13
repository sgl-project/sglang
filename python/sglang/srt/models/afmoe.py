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
"""Inference-only AfMoE model compatible with HuggingFace weights.

AfMoE is a Mixture-of-Experts model with:
- Gated attention with sigmoid gating
- Q/K normalization with RMSNorm
- Dual normalization (pre/post for both attention and MLP)
- Sliding window attention for local layers
- muP (maximal update parameterization) scaling support
"""

from __future__ import annotations

import functools
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton import fused_moe
from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix


def get_attention_sliding_window_size(config: PretrainedConfig) -> Optional[int]:
    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        return None
    if sliding_window <= 0:
        return None
    # Align with other local attention implementations (see gpt_oss).
    return sliding_window - 1


class AfmoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class AfmoeMoE(nn.Module):

    @staticmethod
    def _custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        *,
        score_func: str,
        expert_bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = gating_output.to(torch.float32)
        if score_func == "sigmoid":
            scores = torch.sigmoid(logits)
            if expert_bias is not None:
                bias = expert_bias.to(scores.device, dtype=scores.dtype)
                scores_for_choice = scores + bias
                topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
                topk_weights = scores.gather(dim=-1, index=topk_ids)
            else:
                topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
        else:
            if expert_bias is not None:
                logits = logits + expert_bias.to(logits.device, dtype=logits.dtype)
            probs = F.softmax(logits, dim=-1)
            topk_weights, topk_ids = torch.topk(probs, k=topk, dim=-1)

        if renormalize:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            topk_weights = topk_weights / denom

        return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.n_routed_experts = getattr(config, "num_experts", None)
        if self.n_routed_experts is None:
            raise ValueError("AfmoeConfig must define `num_experts`.")
        self.top_k = config.num_experts_per_tok
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {self.n_routed_experts}."
            )

        self.score_func = getattr(config, "score_func", "softmax")
        self.route_norm = getattr(config, "route_norm", True)
        self.route_scale = float(getattr(config, "route_scale", 1.0))
        self.n_group = getattr(config, "n_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)
        self.use_grouped_topk = self.n_group is not None and self.n_group > 1
        self.num_shared_experts = getattr(config, "num_shared_experts", 0)

        self.gate = ReplicatedLinear(
            config.hidden_size,
            self.n_routed_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )

        self.expert_bias = nn.Parameter(
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )

        self.experts = nn.ModuleList(
            [
                AfmoeMLP(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    reduce_results=False,
                    prefix=add_prefix(f"experts.{idx}", prefix),
                )
                for idx in range(self.n_routed_experts)
            ]
        )
        self.pack_params()

        if self.num_shared_experts:
            intermediate_size = config.moe_intermediate_size * self.num_shared_experts
            self.shared_experts = AfmoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )
        else:
            self.shared_experts = None

        custom_routing_fn = None
        correction_bias = None
        if self.use_grouped_topk:
            correction_bias = self.expert_bias
        elif self.score_func == "sigmoid":
            custom_routing_fn = functools.partial(
                AfmoeMoE._custom_routing_function,
                score_func=self.score_func,
                expert_bias=self.expert_bias,
            )

        renormalize = self.route_norm if self.score_func == "sigmoid" else False
        self.topk = TopK(
            top_k=self.top_k,
            renormalize=renormalize,
            use_grouped_topk=self.use_grouped_topk,
            num_expert_group=self.n_group if self.use_grouped_topk else None,
            topk_group=self.topk_group if self.use_grouped_topk else None,
            custom_routing_function=custom_routing_fn,
            correction_bias=correction_bias,
            routed_scaling_factor=self.route_scale,
        )

    def pack_params(self) -> None:
        w1: list[torch.Tensor] = []
        w2: list[torch.Tensor] = []
        for expert in self.experts:
            w1.append(expert.gate_up_proj.weight)
            w2.append(expert.down_proj.weight)
        self.w1 = torch._utils._flatten_dense_tensors(w1)
        w1s = torch._utils._unflatten_dense_tensors(self.w1, w1)
        for data, param in zip(w1s, w1):
            param.data = data
        self.w1 = self.w1.view(len(w1), *w1s[0].shape)

        self.w2 = torch._utils._flatten_dense_tensors(w2)
        w2s = torch._utils._unflatten_dense_tensors(self.w2, w2)
        for data, param in zip(w2s, w2):
            param.data = data
        self.w2 = self.w2.view(len(w2), *w2s[0].shape)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        shared_output = None
        if self.shared_experts is not None:
            shared_output = self.shared_experts(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = fused_moe.fused_moe(
            hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_output=topk_output,
            moe_runner_config=MoeRunnerConfig(
                inplace=True,
                routed_scaling_factor=self.route_scale,
            ),
        )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(num_tokens, hidden_dim)


class AfmoeAttention(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.rotary_dim = int(self.head_dim * partial_rotary_factor)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        layer_types = getattr(config, "layer_types", None)
        self.is_local_attention = (
            layer_types is not None and layer_types[layer_id] == "sliding_attention"
        )
        sliding_window = (
            get_attention_sliding_window_size(config) if self.is_local_attention else -1
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.rotary_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)
        self.sliding_window = sliding_window

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_heads = self.q_norm(q.reshape(-1, self.head_dim))
        k_heads = self.k_norm(k.reshape(-1, self.head_dim))
        q = q_heads.view(q.shape)
        k = k_heads.view(k.shape)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)

        if self.is_local_attention:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        gate_vals, _ = self.gate_proj(hidden_states)
        attn_output = attn_output * torch.sigmoid(gate_vals)
        output, _ = self.o_proj(attn_output)
        return output


class AfmoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        self.self_attn = AfmoeAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        use_moe = False
        if hasattr(config, "num_dense_layers"):
            use_moe = layer_id >= config.num_dense_layers
        elif (
            getattr(config, "num_experts", None) is not None
            and hasattr(config, "first_k_dense_replace")
            and hasattr(config, "moe_layer_freq")
        ):
            base = config.first_k_dense_replace
            freq = config.moe_layer_freq
            use_moe = layer_id >= base and (layer_id - base) % freq == 0

        if use_moe:
            self.mlp = AfmoeMoE(
                config=config,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = AfmoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )

        eps = getattr(config, "rms_norm_eps", 1e-5)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.pre_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        attn_residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = attn_residual + hidden_states

        mlp_residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = mlp_residual + hidden_states

        return hidden_states


class AfmoeModel(nn.Module):

    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = nn.ModuleList(
            [
                AfmoeDecoderLayer(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        if getattr(self.config, "mup_enabled", False):
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        for layer in self.layers:
            hidden_states = layer(positions, hidden_states, forward_batch)
        hidden_states = self.norm(hidden_states)
        return hidden_states

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens


class AfmoeForCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = AfmoeModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_attention_sliding_window_size(self) -> Optional[int]:
        return get_attention_sliding_window_size(self.config)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # Skip rotary embedding inverse frequencies
            if "rotary_emb.inv_freq" in name:
                continue

            # Remap router gate weights: HF uses .mlp.router.gate., SGLang uses .mlp.gate.
            if ".mlp.router.gate." in name:
                name = name.replace(".mlp.router.gate.", ".mlp.gate.")

            # Handle stacked params (qkv_proj, gate_up_proj)
            handled = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                # Skip gate_proj/up_proj stacking for self_attn (attention uses separate gate_proj)
                if ".self_attn." in name and weight_name in {"gate_proj", "up_proj"}:
                    continue

                new_name = name.replace(weight_name, param_name)
                # Skip if parameter doesn't exist (e.g., bias for layers without bias)
                if new_name not in params_dict:
                    handled = True
                    break

                param = params_dict[new_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                handled = True
                break

            if handled:
                continue

            # Load remaining weights directly
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = AfmoeForCausalLM
