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

"""Inference-only Qwen3Next MTP Speculative Decoding."""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_next import Qwen3NextForCausalLM, Qwen3NextModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3NextForCausalLMMTP(Qwen3NextForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        # if not set, model load will be broken in Qwen3NextForCausalLM load_weights()
        self.pp_group = get_pp_group()
        # self.determine_num_fused_shared_experts("Qwen3NextForCausalLMMTP")

        # currently based on the provided ckpt, we:
        # (1) do not use_dedicated_mtp_embeddings provided in ckpt since not provided and directly use the target model embeddings
        # (2) hardcode bias=False since not provided
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3NextModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
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
        # Some idle batch has 0 batch size. GemmaRMSNorm.forward would fail due to bs=0.
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


EntryClass = [Qwen3NextForCausalLMMTP]
