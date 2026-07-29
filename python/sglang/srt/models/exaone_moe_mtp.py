# Copyright 2025 The LG AI Research Team
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

# Adapted from the vLLM version of EXAONE-MoE MTP
"""Inference-only ExaoneMoE MTP Speculative Decoding."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.exaone_moe import ExaoneMoEForCausalLM, ExaoneMoEModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class ExaoneMoEForCausalLMMTP(ExaoneMoEForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        config.num_hidden_layers = 1
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.pre_fc_norm_embedding = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model = ExaoneMoEModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
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
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)

        hidden_states = forward_batch.spec_info.hidden_states

        if not forward_batch.forward_mode.is_idle():
            input_embeds = self.pre_fc_norm_embedding(input_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat((input_embeds, hidden_states), dim=-1))

        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            hidden_states,
        )

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]], is_mtp: bool = False
    ):
        super().load_weights(weights, is_mtp=True)


EntryClass = ExaoneMoEForCausalLMMTP
