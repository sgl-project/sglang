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

import logging
from itertools import islice
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import Ernie4_5_VLRotaryEmbedding
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2MLP as Ernie4_5_VLMoeMLP

# from sglang.srt.models.llama import LlamaAttention as Ernie4_5_VLMoeAttention
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

        # TODO 文本专家和视觉专家
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
                correction_bias=self.e_score_correction_bias[
                    0
                ],  # TODO 这个bias最后要确定下是否正确
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
        # else:
        #     self.text_experts = Ernie4_5_VLMoeMLP(
        #         hidden_size=config.hidden_size,
        #         intermediate_size=config.intermediate_size,
        #         hidden_act=config.hidden_act,
        #         use_bias=getattr(config, "use_bias", False),
        #         quant_config=quant_config,
        #         prefix=add_prefix("mlp", prefix),
        #     )

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
                correction_bias=self.e_score_correction_bias[
                    1
                ],  # TODO 这个bias最后要确定下是否正确
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
        # else:
        #     self.vision_experts = Ernie4_5_VLMoeMLP(
        #         hidden_size=config.hidden_size,
        #         intermediate_size=config.intermediate_size,
        #         hidden_act=config.hidden_act,
        #         use_bias=getattr(config, "use_bias", False),
        #         quant_config=quant_config,
        #         prefix=add_prefix("mlp", prefix),
        #     )

        if self.moe_num_shared_experts > 0:
            intermediate_size = (
                config.moe_intermediate_size[0] * config.moe_num_shared_experts
            )
            # disable tp for shared experts when enable deepep moe
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
            # Safe to compute boolean value on CPU
            all_visual = visual_token_mask.all().item()
            any_visual = visual_token_mask.any().item()
        else:
            # During CUDA Graph capture, forbid CPU sync ops
            # Fallback: treat as "not all visual"
            all_visual = False
            any_visual = visual_token_mask is not None  # mixed tokens path
            any_visual = False
        # print("orig:", orig_shape)
        # print("hs after flatten:", hidden_states.shape)
        # print("visual mask:", visual_token_mask.shape)
        # print("any_visual:", any_visual, "all_visual:", all_visual)

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
            # Mixed tokens (or we are in capture and mask exists) — avoid boolean indexing.
            # Compute both experts on the full hidden_states and then select per-token results with torch.where.
            # This avoids hidden_states[mask] style indexing which may not be allowed in CUDA Graph capture.
            # Make sure mask has shape [N, 1] for broadcasting
            mask = visual_token_mask.view(-1).bool().unsqueeze(-1)  # [N, 1]

            # Text branch computed on full sequence
            text_router_logits, _ = self.text_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )
            text_topk_output = self.text_experts_topk(hidden_states, text_router_logits)
            text_out = self.text_experts(
                hidden_states=hidden_states, topk_output=text_topk_output
            )  # [N, H]

            # Vision branch computed on full sequence
            vision_router_logits, _ = self.vision_experts_gate(
                hidden_states.to(dtype=torch.float32)
            )
            vision_topk_output = self.vision_experts_topk(
                hidden_states, vision_router_logits
            )
            vision_out = self.vision_experts(
                hidden_states=hidden_states, topk_output=vision_topk_output
            )  # [N, H]

            # Merge per-token outputs: for visual tokens take vision_out, else text_out
            final_hidden_states = torch.where(mask, vision_out, text_out)
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


class Ernie4_5_VLMoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size
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


# only used as text backbone for ernie4.5 vl
class Ernie4_5_VLMoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        self.model = Ernie4_5_VLMoeModel(
            config, quant_config, add_prefix("model", prefix)
        )
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
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

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
            if "mtp" in name or "vision_model" in name or "resampler_model" in name:
                continue

            # if "moe_statics.e_score_correction_bias" in name:
            #     name = name.replace("moe_statics", "gate")
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:

                # Distinguish between vision experts and text experts
                if "mlp.experts" in name:
                    moe_offset = int(name.split(".")[-3])
                    vision_expert_start_idx = self.config.moe_num_experts[0]
                    is_text_expert = moe_offset <= vision_expert_start_idx - 1
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(
                            f".experts.{moe_offset}",
                            f".vision_experts.{moe_offset - vision_expert_start_idx}",
                        )

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Distinguish between vision experts and text experts
                    moe_offset = int(name.split(".")[-3])
                    is_text_expert = moe_offset <= self.config.moe_num_experts[0] - 1

                    name = name.replace(weight_name, param_name)
                    if is_text_expert:
                        name = name.replace(".experts.", ".text_experts.")
                    else:
                        name = name.replace(".experts.", ".vision_experts.")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

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
                        logger.warning(f"Parameter {name} not found in params_dict")
                    break
                else:
                    # Distinguish between vision expert gate
                    # and text expert gate
                    if name.endswith("mlp.gate.weight"):
                        name = name.replace("gate.weight", "text_experts_gate.weight")
                        loaded_weight = loaded_weight.T
                    elif name.endswith("mlp.gate.weight_1"):
                        name = name.replace(
                            "gate.weight_1", "vision_experts_gate.weight"
                        )
                        loaded_weight = loaded_weight.T

                    if "e_score_correction_bias" in name:
                        name = name.replace(".moe_statics.", ".")

                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


EntryClass = [Ernie4_5_VLMoeForCausalLM]
