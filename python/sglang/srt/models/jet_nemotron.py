# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py

import math
from typing import Any, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from transformers.utils import logging

from sglang.srt.configs.jet_nemotron import JetBlockConfig, JetNemotronConfig
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.attention.fla.fused_norm_gate import FusedRMSNormGated
from sglang.srt.layers.attention.jet_nemotron.dynamic_conv import (
    DynamicShortConvolution,
)
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.get_logger(__name__)


class JetBlock(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        jet_block_config = JetBlockConfig(**config.efficient_attention_config["jet"])
        self.mode = jet_block_config.mode

        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.expand_v = jet_block_config.expand_v
        self.conv_size = jet_block_config.conv_size
        self.head_dim = jet_block_config.head_dim
        self.num_heads = jet_block_config.num_heads
        self.key_dim = int(self.num_heads * self.head_dim)
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_k_dim = jet_block_config.head_dim
        self.head_v_dim = int(jet_block_config.head_dim * self.expand_v)
        self.layer_id = layer_id
        self.autotune_interval = (
            32 * 16 * 1024
        )  # 32 batch size * 16 num head * 1024 sequence length

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(self.key_dim * self.expand_v, self.value_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.key_dim * self.expand_v}, which is invalid for nn.Linear."
            )
        if not math.isclose(
            self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={self.expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * self.expand_v}, which is invalid for FusedRMSNormGated."
            )
        assert self.mode in [
            "chunk",
            "fused_recurrent",
        ], f"Not supported mode `{jet_block_config.mode}`."

        self.q_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.a_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        A = torch.empty(
            self.num_heads // self.attn_tp_size, dtype=torch.float32
        ).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        self.dt_bias = nn.Parameter(torch.ones(self.num_heads // self.attn_tp_size))
        self.dt_bias._no_weight_decay = True

        self.dynamic_conv1d = DynamicShortConvolution(
            hidden_size=self.value_dim,
            kernel_size=self.conv_size,
            generator_input_size=self.hidden_size,
            generator_reduction=jet_block_config.dconv_generator_reduction,
            static_conv_init=None,
            implementation=jet_block_config.dconv_implementation,
        )

        self.g_proj = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=float(jet_block_config.norm_eps),
            autotune_interval=self.autotune_interval,
        )
        self.o_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        q_len, _ = (
            hidden_states.shape
        )  # q_len is some combo of batch size and sequence length

        q = F.silu(self.q_proj(hidden_states))
        k = F.silu(self.k_proj(hidden_states))
        v = self.v_proj(hidden_states)

        kwargs = {
            "dynamic_conv": self.dynamic_conv1d,
            "autotune_interval": self.autotune_interval,
            "head_v_dim": self.head_v_dim,
            "head_k_dim": self.head_k_dim,
            "a": self.a_proj(hidden_states),
            "b": self.b_proj(hidden_states),
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "layer_id": self.layer_id,
            "hidden_states": hidden_states,
            "seq_len": q_len,
        }

        o = forward_batch.attn_backend.forward(
            q=q, k=k, v=v, layer=None, forward_batch=forward_batch, **kwargs
        ).squeeze(0)
        g = self.g_proj(hidden_states)
        g = g.reshape(-1, g.shape[-1] // self.head_v_dim, self.head_v_dim)
        o = self.o_norm(o, g)
        o = rearrange(o, "t h d -> t (h d)")
        o = self.o_proj(o)

        return o


class JetNemotronMLP(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {config.hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_state):
        gate_up, _ = self.gate_up_proj(hidden_state)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class JetNemotronAttention(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        sliding_window: int = -1,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = self.hidden_size // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.layer_id = layer_id
        self.sliding_window = sliding_window

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window,
            prefix=f"{prefix}.attn",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            rope_scaling=config.rope_scaling,
            dtype=torch.get_default_dtype(),
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


class JetNemotronRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        JetNemotronRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


EFFICIENT_ATTENTION_CLASSES = {
    "jet": JetBlock,
}


class JetNemotronDecoderLayer(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.layer_types[layer_id] == "attn":
            self.self_attn = JetNemotronAttention(
                config, layer_id, quant_config, prefix
            )
        elif config.layer_types[layer_id] == "swa":
            assert (
                config.efficient_attention_config is not None
            ), "Efficient attention config must be provided in JetNemotronConfig."
            assert (
                "swa" in config.efficient_attention_config
            ), "Sliding Window Attention is enabled but no `swa` configuration found in `efficient_attention_config`."
            self.self_attn = JetNemotronAttention(
                config,
                layer_id,
                quant_config,
                prefix,
                sliding_window=config.efficient_attention_config["swa"]["window_size"],
            )
        else:
            assert config.layer_types[layer_id] in EFFICIENT_ATTENTION_CLASSES, (
                f"Layer type {config.layer_types[layer_id]} not supported. Supported types are: "
                f"{['attn', 'swa'] + list(EFFICIENT_ATTENTION_CLASSES.keys())}"
            )
            self.self_attn = EFFICIENT_ATTENTION_CLASSES[config.layer_types[layer_id]](
                config, layer_id, quant_config, prefix
            )

        self.mlp = JetNemotronMLP(config, quant_config, prefix)
        self.input_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = JetNemotronRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_id = layer_id

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class JetNemotronModel(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                JetNemotronDecoderLayer(config, layer_id, quant_config, prefix)
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = JetNemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers_to_capture = []

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
    ):
        aux_hidden_states = []

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        for idx, decoder_layer in enumerate(self.layers):
            if idx in self.layers_to_capture:
                aux_hidden_states.append(hidden_states)
            hidden_states = decoder_layer(positions, hidden_states, forward_batch)

        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class JetNemotronForCausalLM(nn.Module):
    def __init__(
        self,
        config: JetNemotronConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = JetNemotronModel(config, quant_config, prefix)
        self.vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            org_num_embeddings=config.vocab_size,
            prefix=add_prefix("lm_head", prefix),
            bias=False,
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        hidden_states = self.model(input_ids, positions, forward_batch, inputs_embeds)
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> Set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if name.startswith("model.layers."):
                layer_id = int(name.split(".")[2])
                layer_type = self.config.layer_types[layer_id]
            else:
                layer_type = None
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # jet attention q_proj, k_proj, v_proj shouldn't be merged
                if weight_name not in name or (
                    layer_type == "jet" and "self_attn" in name
                ):
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def get_embed(self):
        return self.model.embed_tokens.weight

    def set_embed(self, embed):
        if (
            hasattr(self.config, "target_hidden_size")
            and self.config.target_hidden_size != self.config.hidden_size
        ):
            return
        del self.model.embed_tokens.weight
        self.model.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if layer_ids is None:
            self.capture_aux_hidden_states = True
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.capture_aux_hidden_states = True
            self.model.layers_to_capture = layer_ids


EntryClass = JetNemotronForCausalLM
