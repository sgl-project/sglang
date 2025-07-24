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
# Copyright 2024 Cohere and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/arctic.py

"""Inference-only Snowflake Arctic model."""

import logging
from typing import Iterable, List, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.arctic import ArcticConfig
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_native import fused_moe_forward_native
from sglang.srt.layers.moe.topk import select_experts
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import RotaryEmbedding, get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma3_causal import extract_layer_index
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class ArcticMLP(nn.Module):
    def __init__(
        self,
        config: ArcticConfig,
        expert_id: int = -1,
        is_residual_mlp: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.expert_id = expert_id

        self.ffn_dim = (
            config.intermediate_size if not is_residual_mlp else self.hidden_size
        )

        self.w13 = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.ffn_dim] * 2,
            bias=False,
            quant_config=quant_config,
        )
        self.w2 = RowParallelLinear(
            self.ffn_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        gate_up, _ = self.w13(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.w2(hidden_states)
        return hidden_states


class ArcticMoE(nn.Module):
    """
    Model-parallel implementation of Arctic MoE Layer.
    """

    def __init__(
        self,
        config: ArcticConfig,
        tp_size: Optional[int] = None,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()

        layer_id = extract_layer_index(prefix)
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_local_experts
        self.layer_id = layer_id
        self.top_k = config.num_experts_per_tok
        self.intermediate_size = config.intermediate_size // self.tp_size

        self.is_moe_layer = (layer_id + 1) % config.moe_layer_frequency == 0
        self.is_quant = quant_config is not None
        self.reduce_results = reduce_results
        # Some other parameters
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        if not self.is_moe_layer:
            self.mlp = ArcticMLP(
                config=config,
                quant_config=quant_config,
                reduce_results=reduce_results,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.gate = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.num_experts,
                bias=False,
                params_dtype=self.params_dtype,
                quant_config=quant_config,
                prefix=f"{prefix}.gate",
            )
            if self.is_quant:
                raise NotImplementedError("Quantization is not supported yet.")
            else:
                self.ws = nn.Parameter(
                    data=torch.empty(
                        self.num_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        device="cuda",
                        dtype=self.params_dtype,
                    )
                )
                self.w2s = nn.Parameter(
                    data=torch.empty(
                        self.num_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        device="cuda",
                        dtype=self.params_dtype,
                    )
                )
            # SGLang handles weight loading through the parameter's __dict__
            setattr(self.ws, "weight_loader", self.weight_loader)
            setattr(self.w2s, "weight_loader", self.weight_loader)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        expert_id: int,
    ) -> None:
        tp_rank: int = get_tensor_model_parallel_rank()
        param_data: torch.Tensor = param.data
        shard_size: int = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith("w1.weight"):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith("w3.weight"):
            param_data[expert_id, shard_size : 2 * shard_size, :] = loaded_weight[
                shard, :
            ]
        if weight_name.endswith("w2.weight"):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor):
        if self.is_moe_layer:
            num_tokens, hidden_size = hidden_states.shape
            hidden_states = hidden_states.view(-1, self.hidden_size)
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            do_normalize: bool = self.top_k > 1

            topk_weights, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                top_k=self.top_k,
                use_grouped_topk=False,
                renormalize=do_normalize,
                torch_native=True,
            )

            final_hidden_states = fused_moe_forward_native(
                layer=self,
                x=hidden_states,
                use_grouped_topk=False,
                top_k=self.top_k,
                router_logits=router_logits,
                renormalize=do_normalize,
                inplace=True,
            )

            if self.reduce_results and self.tp_size > 1:
                final_hidden_states: torch.Tensor = tensor_model_parallel_all_reduce(
                    final_hidden_states
                )
            return final_hidden_states.view(num_tokens, hidden_size)
        else:
            return self.mlp(hidden_states)


class ArcticAttention(nn.Module):
    def __init__(
        self,
        config: ArcticConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size: int = config.hidden_size
        layer_idx: int = extract_layer_index(prefix=prefix)

        tp_size: int = get_tensor_model_parallel_world_size()
        self.total_num_heads: int = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads: int = self.total_num_heads // tp_size
        self.total_num_kv_heads: int = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size=self.hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
        )

        self.rotary_emb: RotaryEmbedding = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=int(self.rope_theta),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class ArcticDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ArcticConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        layer_idx: int = extract_layer_index(prefix=prefix)
        is_moe_layer: bool = (layer_idx + 1) % config.moe_layer_frequency == 0
        self.use_residual = config.use_residual and is_moe_layer
        self.self_attn = ArcticAttention(
            config, quant_config=quant_config, prefix=f"{prefix}.self_attn"
        )
        self.block_sparse_moe = ArcticMoE(
            config=config,
            quant_config=quant_config,
            reduce_results=(not self.use_residual),
            prefix=f"{prefix}.block_sparse_moe",
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps
        )

        if self.use_residual:
            self.residual_layernorm = RMSNorm(
                hidden_size=config.hidden_size, eps=config.rms_norm_eps
            )
            self.residual_mlp = ArcticMLP(
                config=config,
                is_residual_mlp=True,
                reduce_results=False,
                prefix=f"{prefix}.residual_mlp",
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual_input = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual_input + hidden_states

        residual_attn = hidden_states
        if self.use_residual:
            hidden_states = self.residual_layernorm(hidden_states)
            hidden_states = self.residual_mlp(hidden_states)
            residual_mlp = hidden_states
            hidden_states = self.post_attention_layernorm(residual_input)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_mlp + hidden_states
            hidden_states = tensor_model_parallel_all_reduce(hidden_states)
            hidden_states = residual_attn + hidden_states
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.block_sparse_moe(hidden_states)
            hidden_states = residual_attn + hidden_states
        return hidden_states


class ArcticModel(nn.Module):
    def __init__(
        self,
        *,
        config: ArcticConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=config.hidden_size,
            org_num_embeddings=self.vocab_size,
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            num_hidden_layers=config.num_hidden_layers,
            layer_fn=lambda layer_id, prefix: ArcticDecoderLayer(
                config=config, quant_config=quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self._attn_implementation = config._attn_implementation
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is not None:
            hidden_states = input_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids=input_ids)

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class ArcticForCausalLM(nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    def __init__(
        self,
        *,
        config: ArcticConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config: ArcticConfig = config
        self.supports_torch_tp = True
        self.model = ArcticModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix(name="model", prefix=prefix),
        )
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            num_embeddings=self.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=quant_config,
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.unpadded_vocab_size = config.vocab_size
        self.logits_processor = LogitsProcessor(self.config)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ) -> LogitsProcessorOutput:
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        mlp_params_mapping: List[Tuple[str, str, int]] = []
        expert_params_mapping: List[Tuple[str, str, int]] = []
        num_layers = self.config.num_hidden_layers

        for layer in range(num_layers):
            mlp_params_mapping.append(
                (
                    f"layers.{layer}.residual_mlp.w13.weight",
                    f"layers.{layer}.residual_mlp.w1.weight",
                    0,
                )
            )
            mlp_params_mapping.append(
                (
                    f"layers.{layer}.residual_mlp.w13.weight",
                    f"layers.{layer}.residual_mlp.w3.weight",
                    1,
                )
            )
            if (layer + 1) % self.config.moe_layer_frequency != 0:
                # MLP layers
                mlp_params_mapping.append(
                    (
                        f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                        f"layers.{layer}.block_sparse_moe.mlp.w1.weight",
                        0,
                    )
                )
                mlp_params_mapping.append(
                    (
                        f"layers.{layer}.block_sparse_moe.mlp.w13.weight",
                        f"layers.{layer}.block_sparse_moe.mlp.w3.weight",
                        1,
                    )
                )
            else:
                # MoE layers
                for expert_id in range(self.config.num_local_experts):
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w1.weight", expert_id)
                    )
                    expert_params_mapping.append(
                        ("w2s", f"experts.{expert_id}.w2.weight", expert_id)
                    )
                    expert_params_mapping.append(
                        ("ws", f"experts.{expert_id}.w3.weight", expert_id)
                    )

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        logger.info(
            "It will take ~10 minutes loading from the 16-bit weights. "
            "Alternatively, use the prequantized 8-bit weights of arctic "
            "and set load-format to `sharded_state`, which will accelerate loading significantly."
        )
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, shard_id in mlp_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight, shard_id)
                    break
                else:
                    for param_name, weight_name, shard_id in expert_params_mapping:
                        if weight_name not in name:
                            continue
                        name = name.replace(weight_name, param_name)
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(
                            param, loaded_weight, weight_name, expert_id=shard_id
                        )
                        break
                    else:
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        param = params_dict[name]

                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
