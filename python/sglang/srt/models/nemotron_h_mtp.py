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

"""Inference-only NemotronH MTP (Multi-Token Prediction) model."""

import logging
from collections.abc import Iterable
from typing import Optional

import torch
from torch import nn

from sglang.srt.configs import NemotronHConfig
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.nemotron_h import (
    NemotronHAttentionDecoderLayer,
    NemotronHForCausalLM,
    NemotronHMoEDecoderLayer,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class NemotronHMTPAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    """MTP Attention decoder layer with start projections and end norm."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (Attention)
        # Parent forward expects: hidden_states, residual, forward_batch
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


class NemotronHMTPMoEDecoderLayer(NemotronHMoEDecoderLayer):
    """MTP MoE decoder layer with start projections and end norm."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        has_start_projections: bool = False,
        has_end_norm: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
            self.hnorm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

            # Fusion layer to combine embeddings with target hidden states
            self.eh_proj = ColumnParallelLinear(
                input_size=config.hidden_size * 2,
                output_size=config.hidden_size,
                bias=False,
                gather_output=True,
                params_dtype=config.dtype
                if hasattr(config, "dtype")
                else torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.eh_proj",
            )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                config.hidden_size,
                eps=getattr(config, "layer_norm_epsilon", 1e-5),
            )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Start projections (Fusion)
        if self.has_start_projections:
            # Normalize both inputs before fusion
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1
            )
            hidden_states, _ = self.eh_proj(fused)

        # Call parent forward (MoE)
        hidden_states, residual = super().forward(
            hidden_states=hidden_states,
            residual=residual,
            forward_batch=forward_batch,
        )

        # End norm
        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None  # Consumed residual

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


class NemotronHMultiTokenPredictor(nn.Module):
    """MTP predictor with NemotronH layers."""

    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)
        assert self.num_mtp_layers == 1, (
            "Only one MTP layer is supported for NemotronH-MTP"
        )

        self.pattern_str = config.mtp_hybrid_override_pattern
        self.pattern_len = len(self.pattern_str)
        assert self.pattern_len > 0

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Build flat list of layers
        self.layers = nn.ModuleDict()

        # Total number of physical layers = num_steps * pattern_len
        total_layers = self.num_mtp_layers * self.pattern_len
        for i in range(total_layers):
            step_rel_idx = i % self.pattern_len

            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            layer_prefix = f"{prefix}.layers.{i}"

            common_kwargs = dict(
                config=config,
                layer_idx=i,
                quant_config=quant_config,
                prefix=layer_prefix,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
            )

            if char == "*":
                self.layers[str(i)] = NemotronHMTPAttentionDecoderLayer(**common_kwargs)
            elif char == "E":
                self.layers[str(i)] = NemotronHMTPMoEDecoderLayer(**common_kwargs)
            else:
                raise NotImplementedError(
                    f"Pattern char '{char}' in {self.pattern_str} not implemented"
                )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None, (
            "embed_tokens not initialized - must be shared from target model"
        )
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)

        residual = None

        for i in range(self.pattern_len):
            hidden_states, residual = self.layers[str(i)](
                inputs_embeds=inputs_embeds,
                hidden_states=hidden_states,
                residual=residual,
                forward_batch=forward_batch,
            )
        return hidden_states


class NemotronHForCausalLMMTP(NemotronHForCausalLM):
    """NemotronH MTP model."""

    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config
        # Required for parent's load_weights
        self.pp_group = get_pp_group()

        # Override config for MTP pattern (which has no Mamba layers)
        config.num_hidden_layers = len(config.mtp_hybrid_override_pattern)
        # Set hybrid_override_pattern to MTP pattern so attention backend
        # doesn't use Mamba2AttnBackend (MTP has no Mamba layers)
        config.hybrid_override_pattern = config.mtp_hybrid_override_pattern

        # MTP predictor
        self.model = NemotronHMultiTokenPredictor(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        # LM head for generating logits
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )

        self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward - applies attention-based MTP."""
        # TODO smor- check if this is correct
        hidden_states = forward_batch.spec_info.hidden_states

        hidden_states = self.model(
            input_ids,
            positions,
            hidden_states,
            forward_batch,
            input_embeds,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        super().load_weights(weights, is_mtp=True)


EntryClass = [NemotronHForCausalLMMTP]
