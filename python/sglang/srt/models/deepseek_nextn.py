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

"""Inference-only DeepSeek NextN Speculative Decoding."""
import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.ep_moe.layer import EPMoE
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.fp8_utils import (
    block_quant_to_tensor_quant,
    normalize_e4m3fn_to_e4m3fnuz,
)
from sglang.srt.layers.quantization.int8_utils import (
    block_dequant as int8_block_dequant,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.deepseek_v2 import DeepseekV2DecoderLayer, DeepseekV3ForCausalLM
from sglang.srt.utils import BumpAllocator, add_prefix, is_cuda, is_hip

_is_hip = is_hip()
_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import awq_dequantize
else:
    from vllm._custom_ops import awq_dequantize


logger = logging.getLogger(__name__)


class DeepseekModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not global_server_args_dict["enable_dp_attention"],
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.decoder = DeepseekV2DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            is_nextn=True,
            prefix=add_prefix("decoder", prefix),
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        zero_allocator = BumpAllocator(
            buffer_size=2,
            dtype=torch.float32,
            device=(
                input_embeds.device if input_embeds is not None else input_ids.device
            ),
        )

        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        hidden_states = self.eh_proj(
            torch.cat(
                (
                    self.enorm(hidden_states),
                    self.hnorm(forward_batch.spec_info.hidden_states),
                ),
                dim=-1,
            )
        )

        residual = None
        hidden_states, residual = self.decoder(
            positions, hidden_states, forward_batch, residual, zero_allocator
        )

        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class DeepseekV3ForCausalLMNextN(DeepseekV3ForCausalLM):

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
        self.determine_n_share_experts_fusion("DeepseekV3ForCausalLMNextN")

        self.model = DeepseekModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )

        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("model.shared_head.head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("model.shared_head.head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = [DeepseekV3ForCausalLMNextN]
