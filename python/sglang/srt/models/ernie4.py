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

""" Inference-only Ernie4.5 model compatible with baidu/ERNIE-4.5-*-PT weights. """

from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import (
    Ernie4_5_MoeConfig,
)

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import enable_moe_dense_fully_dp
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Ernie4MLP
from sglang.srt.models.llama import LlamaAttention as Ernie4Attention
from sglang.srt.utils import add_prefix, make_layers


class MoEGate(nn.Module):
    def __init__(
        self,
        config,
        prefix: str = "",
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty((config.moe_num_experts, config.hidden_size))
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((1, config.moe_num_experts))
        )

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class Ernie4Moe(nn.Module):
    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_num_shared_experts = getattr(config, "moe_num_shared_experts", 0)

        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )

        self.gate = MoEGate(config=config, prefix=add_prefix("gate", prefix))

        self.topk = TopK(
            top_k=config.moe_k,
            renormalize=True,
            use_grouped_topk=False,
            correction_bias=self.gate.e_score_correction_bias,
        )

        self.experts = get_moe_impl_class(quant_config)(
            num_experts=config.moe_num_experts,
            top_k=config.moe_k,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            layer_id=self.layer_id,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
        )

        if self.moe_num_shared_experts > 0:
            intermediate_size = (
                config.moe_intermediate_size * config.moe_num_shared_experts
            )
            # disable tp for shared experts when enable deepep moe
            self.shared_experts = Ernie4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward_normal(hidden_states)

    def forward_normal(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shared_output = (
            self.shared_experts(hidden_states)
            if self.moe_num_shared_experts > 0
            else None
        )
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        final_hidden_states = self.experts(
            hidden_states=hidden_states, topk_output=topk_output
        )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states


class Ernie4DecoderLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_mtp: bool = False,
    ):
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = getattr(config, "rope_is_neox_style", False)
        # Self attention.
        self.self_attn = Ernie4Attention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=config.use_bias,
        )
        moe_layer_start_index = getattr(
            config, "moe_layer_start_index", config.num_hidden_layers
        )
        moe_layer_end_index = getattr(
            config, "moe_layer_end_index", config.num_hidden_layers - 1
        )
        # MLP
        if (not is_mtp) and (
            moe_layer_start_index <= layer_id <= moe_layer_end_index
            and (layer_id - moe_layer_start_index) % config.moe_layer_interval == 0
        ):
            self.mlp = Ernie4Moe(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = Ernie4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Ernie4Model(nn.Module):
    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Ernie4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix="model.layers",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


class Ernie4_5_ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    stacked_params_mapping = [
        # (param_name, weight_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]

    def __init__(
        self,
        config: Ernie4_5_MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config: Ernie4_5_MoeConfig = config
        self.quant_config = quant_config
        self.model = Ernie4Model(config, quant_config, add_prefix("model", prefix))
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix="lm_head",
            )
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    raise KeyError(f"Parameter '{name}' not found in model.")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


class Ernie4_5_MoeForCausalLM(Ernie4_5_ForCausalLM):
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.moe_num_experts,
        )
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            if name.startswith("model.mtp_"):
                continue
            if "moe_statics.e_score_correction_bias" in name:
                name = name.replace("moe_statics", "gate")
            for param_name, weight_name, shard_id in self.stacked_params_mapping:
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    else:
                        raise KeyError(
                            f"Parameter '{name}'(replaced) not found in model."
                        )
                    break
                else:
                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        raise KeyError(f"Parameter '{name}' not found in model.")


EntryClass = [Ernie4_5_MoeForCausalLM, Ernie4_5_ForCausalLM]
