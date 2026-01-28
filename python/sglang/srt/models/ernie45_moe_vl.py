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

""" Inference-only Ernie4.5 VL model compatible with baidu/ERNIE-4.5-VL-*-PT weights. """

import logging
from itertools import islice
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import Ernie4_5_VLRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Ernie4_5_VLMoeMLP
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class Ernie4_5_VLMoeAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        rope_is_neox_style: bool = True,
        freq_allocation: int = 20,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1)
        self.rotary_dim = int(partial_rotary_factor * self.head_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        # 3D rope
        t_rope = freq_allocation
        h_rope = (self.head_dim // 2 - freq_allocation) // 2
        w_rope = (self.head_dim // 2 - freq_allocation) // 2

        self.rotary_emb = Ernie4_5_VLRotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=False,
            dtype=torch.get_default_dtype(),
            mrope_section=[h_rope, w_rope, t_rope],
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
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


class Ernie4_5_VLMoeMoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.tp_size = get_tensor_model_parallel_world_size()
        self.moe_num_shared_experts = getattr(config, "moe_num_shared_experts", 0)
        self.hidden_size = config.hidden_size

        moe_num_experts = config.moe_num_experts
        max_moe_num_experts = max(moe_num_experts)

        if self.tp_size > max_moe_num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {moe_num_experts}."
            )

        moe_layer_start_index = config.moe_layer_start_index
        text_moe_layer_start_index = moe_layer_start_index[0]
        vision_moe_layer_start_index = moe_layer_start_index[1]
        moe_layer_end_index = config.moe_layer_end_index
        moe_layer_end_index = getattr(
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        text_moe_layer_end_index = moe_layer_end_index[0]
        vision_moe_layer_end_index = moe_layer_end_index[1]

        assert config.moe_num_experts[0] == config.moe_num_experts[1]
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(2, config.moe_num_experts[0], dtype=torch.float32)
        )

        assert text_moe_layer_start_index <= text_moe_layer_end_index

        if (
            layer_id >= text_moe_layer_start_index
            and layer_id <= text_moe_layer_end_index
        ):
            self.text_experts_gate = ReplicatedLinear(
                config.hidden_size,
                config.moe_num_experts[0],
                bias=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix=add_prefix("text_experts_gate", prefix),
            )

            self.text_experts_topk = TopK(
                top_k=config.moe_k,
                renormalize=True,
                use_grouped_topk=False,
                correction_bias=self.e_score_correction_bias[0],
            )

            self.text_experts = get_moe_impl_class(quant_config)(
                num_experts=config.moe_num_experts[0],
                top_k=config.moe_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size[0],
                layer_id=self.layer_id,
                quant_config=quant_config,
                prefix=add_prefix("text_experts", prefix),
            )

        assert vision_moe_layer_start_index <= vision_moe_layer_end_index
        if (
            layer_id >= vision_moe_layer_start_index
            and layer_id <= vision_moe_layer_end_index
        ):

            self.vision_experts_gate = ReplicatedLinear(
                config.hidden_size,
                config.moe_num_experts[1],
                bias=False,
                params_dtype=torch.float32,
                quant_config=quant_config,
                prefix=add_prefix("vision_experts_gate", prefix),
            )

            self.vision_experts_topk = TopK(
                top_k=config.moe_k,
                renormalize=True,
                use_grouped_topk=False,
                correction_bias=self.e_score_correction_bias[1],
            )

            self.vision_experts = get_moe_impl_class(quant_config)(
                num_experts=config.moe_num_experts[1],
                top_k=config.moe_k,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size[1],
                layer_id=self.layer_id,
                quant_config=quant_config,
                prefix=add_prefix("vision_experts", prefix),
            )

        if self.moe_num_shared_experts > 0:
            intermediate_size = (
                config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )
            self.shared_experts = Ernie4_5_VLMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_experts", prefix),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_token_mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        shared_output = (
            self.shared_experts(hidden_states)
            if self.moe_num_shared_experts > 0
            else None
        )

        orig_shape = hidden_states.shape
        hidden_dim = hidden_states.shape[-1]
        hidden_states = hidden_states.view(-1, hidden_dim)

        capturing = torch.cuda.is_current_stream_capturing()

        if visual_token_mask is not None and not capturing:
            all_visual = visual_token_mask.all()
            any_visual = visual_token_mask.any()
        else:
            # During CUDA Graph capture, all set false
            all_visual = False
            any_visual = False

        if all_visual:
            # vision modal input processing directly
            vision_router_logits, _ = self.vision_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )
            vision_topk_output = self.vision_experts_topk(
                hidden_states, vision_router_logits
            )
            final_hidden_states = self.vision_experts(
                hidden_states=hidden_states, topk_output=vision_topk_output
            )
        elif any_visual:
            visual_token_mask = visual_token_mask.repeat(1, self.hidden_size).bool()
            text_token_mask = ~visual_token_mask
            final_hidden_states = torch.zeros_like(hidden_states)

            text_hidden_states = hidden_states[text_token_mask].reshape(
                -1, self.hidden_size
            )
            vision_hidden_states = hidden_states[visual_token_mask].reshape(
                -1, self.hidden_size
            )

            text_router_logits, _ = self.text_experts_gate(
                text_hidden_states.to(dtype=torch.float32)
            )
            text_topk_output = self.text_experts_topk(
                text_hidden_states, text_router_logits
            )
            final_hidden_states[text_token_mask] = self.text_experts(
                hidden_states=text_hidden_states, topk_output=text_topk_output
            ).flatten()

            vision_router_logits, _ = self.vision_experts_gate(
                vision_hidden_states.to(dtype=torch.float32)
            )
            vision_topk_output = self.vision_experts_topk(
                vision_hidden_states, vision_router_logits
            )
            final_hidden_states[visual_token_mask] = self.vision_experts(
                hidden_states=vision_hidden_states, topk_output=vision_topk_output
            ).flatten()

        else:
            # text modal input processing directly
            text_router_logits, _ = self.text_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )
            topk_output = self.text_experts_topk(hidden_states, text_router_logits)
            final_hidden_states = self.text_experts(
                hidden_states=hidden_states, topk_output=topk_output
            )

        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output

        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class Ernie4_5_VLMoeDecoderLayer(nn.Module):
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
    ):
        super().__init__()
        rope_theta = getattr(config, "rope_theta", 500000)
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_is_neox_style = getattr(config, "rope_is_neox_style", False)
        freq_allocation = getattr(config, "freq_allocation", 20)
        max_position_embeddings = getattr(config, "max_position_embeddings", 131072)
        # Self attention.
        self.self_attn = Ernie4_5_VLMoeAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_is_neox_style=rope_is_neox_style,
            freq_allocation=freq_allocation,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            bias=config.use_bias,
        )

        # MoE
        moe_layer_start_index = config.moe_layer_start_index
        min_moe_layer_start_index = min(moe_layer_start_index)
        moe_layer_end_index = getattr(
            config,
            "moe_layer_end_index",
            [config.num_hidden_layers - 1, config.num_hidden_layers - 1],
        )
        max_moe_layer_end_index = max(moe_layer_end_index)
        assert min_moe_layer_start_index <= max_moe_layer_end_index
        moe_num_experts = config.moe_num_experts
        max_moe_num_experts = max(moe_num_experts)
        moe_layer_interval = getattr(config, "moe_layer_interval", 1)
        use_moe = getattr(config, "use_moe", max_moe_num_experts > 0)
        # MLP
        if (
            use_moe
            and ((layer_id + 1) % moe_layer_interval == 0)
            and layer_id >= min_moe_layer_start_index
            and layer_id <= max_moe_layer_end_index
        ):
            self.mlp = Ernie4_5_VLMoeMoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Ernie4_5_VLMoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
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
        visual_token_mask: torch.Tensor | None,
        **kwargs: object,
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
        if isinstance(self.mlp, Ernie4_5_VLMoeMoE):
            hidden_states = self.mlp(hidden_states, visual_token_mask, **kwargs)
        else:
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# only used as text backbone for ernie4.5 vl
class Ernie4_5_VLMoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Ernie4_5_VLMoeDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

    def get_input_embeddings(self) -> torch.Tensor:
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        visual_token_mask: torch.Tensor | None = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:

        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
                visual_token_mask,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states
