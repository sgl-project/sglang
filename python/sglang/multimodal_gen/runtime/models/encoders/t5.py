# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from transformers: https://github.com/huggingface/transformers/blob/v4.39.0/src/transformers/models/t5/modeling_t5.py

# Derived from T5 implementation posted on HuggingFace; license below:
#
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
"""PyTorch T5 & UMT5 model."""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput, T5Config
from sglang.multimodal_gen.runtime.distributed import get_tp_rank, get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.platforms import current_platform


class AttentionType:
    """
    Attention type.
    Use string to be compatible with `torch.compile`.
    """

    # Decoder attention between previous layer Q/K/V
    DECODER = "decoder"
    # Encoder attention between previous layer Q/K/V for encoder-decoder
    ENCODER = "encoder"
    # Encoder attention between previous layer Q/K/V
    ENCODER_ONLY = "encoder_only"
    # Attention between dec. Q and enc. K/V for encoder-decoder
    ENCODER_DECODER = "encoder_decoder"


_seen_keys = set()  # 用集合记录已经出现过的 key


@dataclass
class AttentionMetadata:
    attn_bias: torch.Tensor


class T5DenseActDense(nn.Module):

    def __init__(
        self, config: T5Config, quant_config: QuantizationConfig | None = None
    ):
        super().__init__()
        self.wi = MergedColumnParallelLinear(config.d_model, [config.d_ff], bias=False)
        self.wo = RowParallelLinear(
            config.d_ff, config.d_model, bias=False, quant_config=quant_config
        )
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_states, _ = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5DenseGatedActDense(nn.Module):

    def __init__(
        self, config: T5Config, quant_config: QuantizationConfig | None = None
    ):
        super().__init__()
        self.wi_0 = MergedColumnParallelLinear(
            config.d_model, [config.d_ff], bias=False, quant_config=quant_config
        )
        self.wi_1 = MergedColumnParallelLinear(
            config.d_model, [config.d_ff], bias=False, quant_config=quant_config
        )
        # Should not run in fp16 unless mixed-precision is used,
        # see https://github.com/huggingface/transformers/issues/20287.
        self.wo = RowParallelLinear(
            config.d_ff, config.d_model, bias=False, quant_config=quant_config
        )
        self.act = get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states) -> torch.Tensor:
        hidden_gelu = self.act(self.wi_0(hidden_states)[0])
        hidden_linear, _ = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states, _ = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):

    def __init__(
        self, config: T5Config, quant_config: QuantizationConfig | None = None
    ):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense = T5DenseGatedActDense(
                config, quant_config=quant_config
            )
        else:
            self.DenseReluDense = T5DenseActDense(config, quant_config=quant_config)

        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states) -> torch.Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + forwarded_states
        return hidden_states


# T5 has attn_bias and does not use softmax scaling
class T5MultiHeadAttention(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, q, k, v, attn_bias=None):
        b, _, n, c = q.shape
        attn = torch.einsum("binc,bjnc->bnij", q, k)
        if attn_bias is not None:
            attn += attn_bias

        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        x = torch.einsum("bnij,bjnc->binc", attn, v)
        x = x.reshape(b, -1, n * c)
        return x


class T5Attention(nn.Module):

    def __init__(
        self,
        config: T5Config,
        attn_type: str,
        has_relative_attention_bias=False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attn_type = attn_type
        # Cross-attention has no relative pos encoding anyway
        self.is_decoder = attn_type == AttentionType.DECODER
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.total_num_heads = self.total_num_kv_heads = config.num_heads

        # Partition heads across multiple tensor parallel GPUs.
        tp_world_size = get_tp_world_size()
        assert config.num_heads % tp_world_size == 0
        self.n_heads = config.num_heads // tp_world_size

        self.inner_dim = self.n_heads * self.key_value_proj_dim
        # No GQA in t5.
        # self.n_kv_heads = self.n_heads

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.attn = T5MultiHeadAttention()

        if self.has_relative_attention_bias:
            self.relative_attention_bias = VocabParallelEmbedding(
                self.relative_attention_num_buckets,
                self.total_num_heads,
                org_num_embeddings=self.relative_attention_num_buckets,
                padding_size=self.relative_attention_num_buckets,
                quant_config=quant_config,
            )
        self.o = RowParallelLinear(
            self.d_model,
            self.d_model,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ) -> torch.Tensor:
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position,
        i.e. the distance in tokens from the attending position to the
        attended-to position. If bidirectional=False, then positive relative
        positions are invalid. We use smaller buckets for small absolute
        relative_position and larger buckets for larger absolute
        relative_positions. All relative positions >=max_distance map to the
        same bucket. All relative positions <=-max_distance map to the same
        bucket. This should allow for more graceful generalization to longer
        sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """  # noqa: E501
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins
        # in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None) -> torch.Tensor:
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        # max_seq_len, nh
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        x = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,  # (num_tokens, d_model)
        attention_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        bs, seq_len, _ = hidden_states.shape
        num_seqs = bs
        n, c = self.n_heads, self.d_model // self.total_num_heads
        qkv, _ = self.qkv_proj(hidden_states)
        # Projection of 'own' hidden state (self-attention). No GQA here.
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.reshape(bs, seq_len, n, c)
        k = k.reshape(bs, seq_len, n, c)
        v = v.reshape(bs, seq_len, n, c)

        assert attn_metadata is not None
        attn_bias = attn_metadata.attn_bias
        # Not compatible with CP here (as all encoder-decoder models),
        # as it assumes homogeneous batch (prefills or decodes).
        if self.has_relative_attention_bias:
            # Self-attention. Compute T5 relative positional encoding.
            # The bias term is computed on longest sequence in batch. Biases
            # for shorter sequences are slices of the longest.
            assert self.attn_type == AttentionType.ENCODER
            attn_bias = self.compute_bias(seq_len, seq_len).repeat(num_seqs, 1, 1, 1)
            attn_metadata.attn_bias = attn_bias
        else:
            # Encoder/Decoder Self-Attention Layer, attn bias already cached.
            assert attn_bias is not None

        if attention_mask is not None:
            attention_mask = (
                attention_mask.view(bs, 1, 1, -1)
                if attention_mask.ndim == 2
                else attention_mask.unsqueeze(1)
            )
            mask_val = -1e4 if current_platform.is_mps() else torch.finfo(q.dtype).min
            attn_bias.masked_fill_(attention_mask == 0, mask_val)

        if get_tp_world_size() > 1:
            rank = get_tp_rank()
            attn_bias = attn_bias[
                :, rank * self.n_heads : (rank + 1) * self.n_heads, :, :
            ]

        attn_output = self.attn(q, k, v, attn_bias)
        output, _ = self.o(attn_output)
        return output


class T5LayerSelfAttention(nn.Module):

    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.SelfAttention = T5Attention(
            config,
            AttentionType.DECODER if "decoder" in prefix else AttentionType.ENCODER,
            has_relative_attention_bias=has_relative_attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.SelfAttention",
        )
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)

        attention_output = self.SelfAttention(
            hidden_states=normed_hidden_states,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )

        hidden_states = hidden_states + attention_output

        return hidden_states


class T5LayerCrossAttention(nn.Module):

    def __init__(
        self, config, quant_config: QuantizationConfig | None = None, prefix: str = ""
    ):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config,
            AttentionType.ENCODER_DECODER,
            has_relative_attention_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.EncDecAttention",
        )
        self.layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            hidden_states=normed_hidden_states,
            attn_metadata=attn_metadata,
        )
        hidden_states = hidden_states + attention_output
        return hidden_states


class T5Block(nn.Module):

    def __init__(
        self,
        config: T5Config,
        is_decoder: bool,
        has_relative_attention_bias=False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config,
                has_relative_attention_bias=has_relative_attention_bias,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        )

        if self.is_decoder:
            self.layer.append(
                T5LayerCrossAttention(
                    config, quant_config=quant_config, prefix=f"{prefix}.cross_attn"
                )
            )

        self.layer.append(T5LayerFF(config, quant_config=quant_config))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:

        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_states.shape[:2], device=hidden_states.device
            )

        hidden_states = self.layer[0](
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )

        if self.is_decoder:
            hidden_states = self.layer[1](
                hidden_states=hidden_states, attn_metadata=attn_metadata
            )

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        return hidden_states


class T5Stack(nn.Module):

    def __init__(
        self,
        config: T5Config,
        is_decoder: bool,
        n_layers: int,
        embed_tokens=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        is_umt5: bool = False,
    ):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.is_umt5 = is_umt5
        if is_umt5:
            self.block = nn.ModuleList(
                [
                    T5Block(
                        config,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=True,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{i}",
                    )
                    for i in range(n_layers)
                ]
            )
        else:
            # Only the first block has relative positional encoding.
            self.block = nn.ModuleList(
                [
                    T5Block(
                        config,
                        is_decoder=is_decoder,
                        has_relative_attention_bias=i == 0,
                        quant_config=quant_config,
                        prefix=f"{prefix}.blocks.{i}",
                    )
                    for i in range(n_layers)
                ]
            )
        self.final_layer_norm = RMSNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)

        for idx, block in enumerate(self.block):
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attn_metadata=attn_metadata,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class T5EncoderModel(TextEncoder):

    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__(config)

        quant_config = None

        self.shared = VocabParallelEmbedding(
            config.vocab_size, config.d_model, org_num_embeddings=config.vocab_size
        )

        self.encoder = T5Stack(
            config,
            False,
            config.num_layers,
            self.shared,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            is_umt5=False,
        )

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        attn_metadata = AttentionMetadata(None)
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )

        return BaseEncoderOutput(last_hidden_state=hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q", "q"),
            (".qkv_proj", ".k", "k"),
            (".qkv_proj", ".v", "v"),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            loaded = False
            if "decoder" in name or "lm_head" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
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
                loaded = True
                break
            if not loaded:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class UMT5EncoderModel(TextEncoder):

    def __init__(self, config: T5Config, prefix: str = ""):
        super().__init__(config)

        quant_config = None

        self.shared = VocabParallelEmbedding(
            config.vocab_size, config.d_model, org_num_embeddings=config.vocab_size
        )

        self.encoder = T5Stack(
            config,
            False,
            config.num_layers,
            self.shared,
            quant_config=quant_config,
            prefix=f"{prefix}.encoder",
            is_umt5=True,
        )

    def get_input_embeddings(self):
        return self.shared

    def forward(
        self,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        attn_metadata = AttentionMetadata(None)
        hidden_states = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attn_metadata=attn_metadata,
        )

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            attention_mask=attention_mask,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            loaded = False
            if "decoder" in name or "lm_head" in name:
                continue
            for (
                param_name,
                weight_name,
                shard_id,
            ) in self.config.arch_config.stacked_params_mapping:
                if weight_name not in name:
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
                loaded = True
                break
            if not loaded:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = [T5EncoderModel, UMT5EncoderModel]
