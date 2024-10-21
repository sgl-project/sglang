"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# coding=utf-8
# Adapted from
# https://github.com/THUDM/ChatGLM2-6B
"""Inference-only ChatGLM model compatible with THUDM weights."""
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import LayerNorm
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.transformers_utils.configs import ChatGLMConfig

from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

LoraConfig = None


class GLMAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int = 0,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.multi_query_attention = config.multi_query_attention
        self.total_num_kv_heads = (
            config.multi_query_group_num
            if config.multi_query_attention
            else config.num_attention_heads
        )
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.query_key_value = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quant_config,
        )
        self.dense = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        # https://huggingface.co/THUDM/chatglm3-6b-32k/blob/e210410255278dd9d74463cf396ba559c0ef801c/modeling_chatglm.py#L141
        rope_ratio = getattr(config, "rope_ratio", 1.0)
        max_positions = getattr(config, "seq_length", 8192)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim // 2,
            max_position=max_positions,
            base=10000 * rope_ratio,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(
            q,
            k,
            v,
            forward_batch,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output


class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.add_bias = config.add_bias_linear

        # Project to 4h.
        self.dense_h_to_4h = MergedColumnParallelLinear(
            config.hidden_size,
            [config.ffn_hidden_size] * 2,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

        self.activation_func = SiluAndMul()

        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            bias=config.add_bias_linear,
            quant_config=quant_config,
        )

    def forward(self, hidden_states):
        # [s, b, 4hp]
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output


class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

        self.fp32_residual_connection = config.fp32_residual_connection

        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        # Layernorm on the input data.
        self.input_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon
        )

        # Self attention.
        self.self_attention = GLMAttention(config, layer_id, cache_config, quant_config)
        self.hidden_dropout = config.hidden_dropout

        # Layernorm on the attention output
        self.post_attention_layernorm = layer_norm_func(
            config.hidden_size, eps=config.layernorm_epsilon
        )

        # MLP
        self.mlp = GLMMLP(config, quant_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            forward_batch=forward_batch,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = residual + attention_output

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = self.mlp(layernorm_output) + residual

        return output


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        config,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm

        # Number of layers.
        self.num_layers = config.num_layers

        # Transformer layers.
        self.layers = nn.ModuleList(
            [
                GLMBlock(config, i, cache_config, quant_config)
                for i in range(self.num_layers)
            ]
        )

        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            # Final layer norm before output.
            self.final_layernorm = layer_norm_func(
                config.hidden_size, eps=config.layernorm_epsilon
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                forward_batch=forward_batch,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


class ChatGLMM(nn.Module):
    def __init__(
        self,
        config,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()

        self.embedding = VocabParallelEmbedding(
            config.padded_vocab_size, config.hidden_size
        )

        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config, cache_config, quant_config)

        self.output_layer = ParallelLMHead(config.padded_vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids)

        # Run encoder.
        hidden_states = self.encoder(
            hidden_states=inputs_embeds,
            position_ids=position_ids,
            forward_batch=forward_batch,
        )
        return hidden_states


class ChatGLMForCausalLM(nn.Module):
    packed_modules_mapping = {
        "query_key_value": ["query_key_value"],
        "dense_h_to_4h": ["dense_h_to_4h"],
    }
    # LoRA specific attributes
    supported_lora_modules = [
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config: ChatGLMConfig,
        cache_config=None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__()
        self.config: ChatGLMConfig = config
        self.quant_config = quant_config
        self.max_position_embeddings = getattr(config, "max_sequence_length", 8192)
        self.transformer = ChatGLMM(config, cache_config, quant_config)
        self.lm_head = self.transformer.output_layer
        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head.weight, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_pos_emb.inv_freq" in name:
                continue
            if "word_embeddings" in name:
                name = name.replace(".word_embeddings", "")
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


class ChatGLMModel(ChatGLMForCausalLM):
    pass


EntryClass = [ChatGLMModel]
