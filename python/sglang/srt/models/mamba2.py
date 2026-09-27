# SPDX-License-Identifier: Apache-2.0
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
"""Inference-only Mamba2 model (e.g. Mamba-Codestral-7B) for SGLang.

Mamba2 is a pure state-space model (SSM) architecture that replaces attention
with selective state spaces. Each decoder layer is a pre-norm followed by a
Mamba2 mixer; there is no MLP sublayer.

Reference: https://huggingface.co/mistralai/Mamba-Codestral-7B-v0.1
"""

import logging
from typing import Iterable, Optional, Set, Tuple

import torch
from torch import nn

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


class Mamba2DecoderLayer(nn.Module):
    """
    Mamba2 decoder layer with SSM mixer instead of attention.
    """

    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size

        # Pre-normalization (checkpoint key: backbone.layers.N.norm)
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

        state_shape = Mamba2StateShape.create(
            tp_world_size=get_parallel().tp_size,
            intermediate_size=config.intermediate_size,
            n_groups=config.n_groups,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            state_size=config.state_size,
            conv_kernel=config.conv_kernel,
        )
        cache_params = Mamba2CacheParams(
            shape=state_shape,
            layers=list(range(config.num_hidden_layers)),
        )

        self.mixer = MambaMixer2(
            cache_params=cache_params,
            hidden_size=config.hidden_size,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.hidden_act,
            use_rms_norm=config.rms_norm,
            quant_config=quant_config,
            prefix=add_prefix("mixer", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-norm (fused residual add) -> SSM mixer; no MLP sublayer.
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        # Run the mixer through the Mamba2 attention backend (owns the conv/ssm
        # state cache). use_triton_causal_conv keeps the conv portable off-CUDA.
        attn_backend = get_attn_backend()
        output = torch.empty_like(hidden_states)
        attn_backend.forward(
            self.mixer,
            hidden_states,
            output,
            layer_id=self.layer_id,
            forward_batch=forward_batch,
            use_triton_causal_conv=True,
        )
        hidden_states = output

        return hidden_states, residual


class Mamba2Model(nn.Module):
    """Mamba2 model without the language modeling head."""

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Mamba2DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.layer_norm_epsilon,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, forward_batch, residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Mamba2ForCausalLM(nn.Module):
    """
    Mamba2 model with a language modeling head.

    This is the main model class for Mamba2-based models like Mamba-Codestral-7B.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        cache_config=None,
    ):
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        self.model = Mamba2Model(
            config=config,
            quant_config=quant_config,
            prefix="model",
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix="lm_head",
        )

        self.logits_processor = LogitsProcessor(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            # Skip rotary embedding inverse frequencies
            if "rotary_emb.inv_freq" in name or "inv_freq" in name:
                continue

            # Remap checkpoint names to SGLang modules: backbone.* -> model.*,
            # embeddings -> embed_tokens, norm_f -> norm, A_log -> A.
            if name.startswith("backbone."):
                name = "model." + name[len("backbone.") :]
            name = name.replace("embeddings.", "embed_tokens.")
            name = name.replace("norm_f.", "norm.")
            if "A_log" in name:
                name = name.replace("A_log", "A")

            if name not in params_dict:
                logger.warning(f"Skipping parameter {name} - not found in model")
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        unloaded_params = set(params_dict.keys()) - loaded_params
        if unloaded_params:
            logger.warning(
                f"The following parameters were not loaded: {unloaded_params}"
            )

        return loaded_params


EntryClass = Mamba2ForCausalLM
