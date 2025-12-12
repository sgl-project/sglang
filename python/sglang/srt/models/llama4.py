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

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/model_executor/models/llama4.py
"""Inference-only LLaMA model compatible with HuggingFace weights."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Llama4TextConfig

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.models.llama import LlamaForCausalLM, LlamaMLP
from sglang.srt.utils import (
    add_prefix,
    fast_topk,
    get_compiler_backend,
    is_cuda,
    make_layers,
)
from sglang.srt.utils.common import get_current_device_stream_fast

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


class Llama4MoE(nn.Module):

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    @staticmethod
    def custom_routing_function(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        router_scores_aK, router_indices_aK = fast_topk(gating_output, topk, dim=-1)
        router_scores_aK = torch.sigmoid(router_scores_aK.float()).to(
            hidden_states.dtype
        )
        return (
            router_scores_aK.view(-1).reshape(router_scores_aK.shape),
            router_indices_aK.to(torch.int32),
        )

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.device_module = torch.get_device_module()

        intermediate_size_moe = config.intermediate_size
        self.router = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("router", prefix),
        )

        self.topk = TopK(
            top_k=self.top_k,
            renormalize=False,
            custom_routing_function=Llama4MoE.custom_routing_function,
        )

        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            layer_id=layer_id,
            reduce_results=False,
            quant_config=quant_config,
            apply_router_weight_on_input=True,
            prefix=add_prefix("experts", prefix),
        )

        self.shared_expert = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size_moe,
            hidden_act="silu",
            quant_config=quant_config,
            prefix=add_prefix("shared_expert", prefix),
            reduce_results=False,  # We need to do scatter before reduce
        )

    def forward(
        self,
        hidden_states,
        forward_batch: ForwardBatch,
        use_reduce_scatter: bool = False,
    ):
        shared_out, routed_out = self._forward_core(
            hidden_states, forward_batch.forward_mode
        )

        out_aD = routed_out + shared_out

        if self.tp_size > 1 and not use_reduce_scatter:
            out_aD = tensor_model_parallel_all_reduce(out_aD)

        return out_aD

    def _forward_core(self, hidden_states, forward_mode: ForwardMode):
        if _is_cuda:
            return self._forward_core_shared_routed_overlap(hidden_states)
        else:
            return self._forward_core_normal(hidden_states)

    def _forward_core_normal(self, hidden_states):
        # router_scores: [num_tokens, num_experts]
        router_logits, _ = self.router(hidden_states)
        shared_out = self.shared_expert(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        routed_out = self.experts(hidden_states, topk_output)
        return shared_out, routed_out

    def _forward_core_shared_routed_overlap(self, hidden_states):
        alt_stream = _get_or_create_alt_stream(self.device_module)

        alt_stream.wait_stream(get_current_device_stream_fast())

        shared_out = self.shared_expert(hidden_states)

        with self.device_module.stream(alt_stream):
            # router_scores: [num_tokens, num_experts]
            router_logits, _ = self.router(hidden_states)
            topk_output = self.topk(hidden_states, router_logits)
            routed_out = self.experts(hidden_states, topk_output)
        get_current_device_stream_fast().wait_stream(alt_stream)

        return shared_out, routed_out


_alt_stream = None


def _get_or_create_alt_stream(device_module):
    global _alt_stream
    if _alt_stream is None:
        _alt_stream = device_module.Stream()
    return _alt_stream


class Llama4Attention(nn.Module):

    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.use_rope = (layer_id + 1) % 4 != 0
        self.use_qk_norm = config.use_qk_norm and self.use_rope

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_temperature_tuning = config.attn_temperature_tuning
        self.floor_scale = config.floor_scale
        self.attn_scale = config.attn_scale
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.n_rep = self.num_heads // self.num_kv_heads
        self.qk_norm = (
            RMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
            )
            if self.use_qk_norm
            else None
        )

        qkv_quant_config = quant_config
        o_quant_config = quant_config
        if quant_config and hasattr(quant_config, "ignore") and quant_config.ignore:
            if add_prefix("q_proj", prefix) in quant_config.ignore:
                qkv_quant_config = None
            if add_prefix("o_proj", prefix) in quant_config.ignore:
                o_quant_config = None

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=qkv_quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=o_quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
        )
        is_neox_style = True
        is_gguf = quant_config and quant_config.get_name() == "gguf"
        if is_gguf and config.model_type in ["llama", "llama4"]:
            is_neox_style = False

        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=int(rope_theta),
                rope_scaling=rope_scaling if rope_scaling != "default" else None,
                is_neox_style=is_neox_style,
            )
            if self.use_rope
            else None
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
        )

    def _get_attn_scale(self, positions: torch.Tensor) -> torch.Tensor:
        floor = torch.floor((positions + 1.0) / self.floor_scale)
        attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
        return attn_scale.unsqueeze(-1)

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _mul_attn_scale(self, positions, q):
        attn_scale = self._get_attn_scale(positions)
        return (q * attn_scale).to(q.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)

        qk, v = qkv.split([self.q_size + self.kv_size, self.kv_size], dim=-1)

        if self.rotary_emb is not None:
            q_view, k_view = qk.split([self.q_size, self.kv_size], dim=-1)
            q_out_unused, k_out_unused = self.rotary_emb(positions, q_view, k_view)
            del q_view, k_view, q_out_unused, k_out_unused

        if self.qk_norm is not None:
            # TODO there are still 2 redundant direct_copy_kernel_cuda for this `reshape` and (in attn backend) q.contiguous(), maybe we can fuse them later
            qk = qk.reshape(-1, self.head_dim).contiguous().bfloat16()
            qk = self.qk_norm(qk).to(torch.bfloat16)
            qk = qk.reshape(-1, self.q_size + self.kv_size)

        q, k = qk.split([self.q_size, self.kv_size], dim=-1)

        # We are applying temperature tuning (https://arxiv.org/abs/2501.19399) to NoPE layers, where
        # the inference-time temperature tuning function is customized to not affect short context
        # while working at very long context
        # https://arxiv.org/abs/2501.19399
        if self.attn_temperature_tuning and not self.use_rope:
            q = self._mul_attn_scale(positions=positions, q=q)

        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Llama4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        rope_theta = config.rope_theta
        rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        self.self_attn = Llama4Attention(
            config=config,
            layer_id=layer_id,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=False,
            bias_o_proj=False,
            prefix=add_prefix("self_attn", prefix),
        )
        self.config = config
        is_moe_layer = self._is_moe_layer(layer_id)
        is_previous_moe_layer = self._is_moe_layer(layer_id - 1)
        is_next_moe_layer = self._is_moe_layer(layer_id + 1)

        if is_moe_layer:
            self.feed_forward = Llama4MoE(
                config=config,
                layer_id=layer_id,
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size_mlp,
                hidden_act="silu",
                quant_config=quant_config,
                prefix=add_prefix("feed_forward", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_moe_layer,
            is_previous_layer_sparse=is_previous_moe_layer,
            is_next_layer_sparse=is_next_moe_layer,
        )

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def _is_moe_layer(self, layer_id: int) -> bool:
        if self.config.interleave_moe_layer_step == 0:
            return self.config.num_local_experts > 0
        return (layer_id + 1) % self.config.interleave_moe_layer_step == 0

    def get_intermediate_size(self) -> int:
        if isinstance(self.feed_forward, Llama4MoE):
            return self.config.intermediate_size
        else:
            return self.config.intermediate_size_mlp

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        # Fully Connected
        hidden_states = self.feed_forward(
            hidden_states, forward_batch, use_reduce_scatter
        )
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Llama4Model(nn.Module):
    def __init__(
        self,
        config: Llama4TextConfig,
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
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
            enable_tp=not is_dp_attention_enabled(),
        )
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Llama4DecoderLayer(
                config=config, layer_id=idx, quant_config=quant_config, prefix=prefix
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
        aux_hidden_states = []
        for i in range(len(self.layers)):
            if i in self.layers_to_capture:
                aux_hidden_states.append(hidden_states + residual)
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Llama4ForCausalLM(LlamaForCausalLM):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__(config, quant_config, prefix)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_layers(self):
        return self.model.layers

    def _init_model(
        self,
        config: Llama4TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        return Llama4Model(config, quant_config=quant_config, prefix=prefix)


EntryClass = [Llama4ForCausalLM]
