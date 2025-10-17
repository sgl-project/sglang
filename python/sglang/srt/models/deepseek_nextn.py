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
from typing import Iterable, Optional, Tuple, Dict
import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import (is_dp_attention_enabled,
                                            get_attention_tp_rank,
                                            get_attention_tp_size,
                                            attn_tp_all_gather_reorgan_into_tensor,
                                            )
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV3ForCausalLM,
    enable_nextn_moe_bf16_cast_to_fp8,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (BumpAllocator, 
                              add_prefix, 
                              is_cuda, 
                              get_bool_env_var,
                              prepare_input_dp_with_cp_dsa,)

from sglang.srt.configs.model_config import is_deepseek_nsa 

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()


class DeepseekModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.cp_size = get_attention_tp_size()
        if enable_nextn_moe_bf16_cast_to_fp8(quant_config):
            # refer to real DeepSeek V3 quant config
            moe_quant_config = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            moe_quant_config = None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.warning(
                "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
            )
            quant_config = None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.decoder = DeepseekV2DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            moe_quant_config=moe_quant_config,
            is_nextn=True,
            prefix=add_prefix("decoder", prefix),
            alt_stream=self.alt_stream,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        cp_input_dict = None
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

        if hidden_states.shape[0] > 0:
            hidden_states = self.eh_proj(
                torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(forward_batch.spec_info.hidden_states),
                    ),
                    dim=-1,
                )
            )
        if cp_input_dict is not None and forward_batch.forward_mode.is_context_parallel_extend():
            hidden_states_list = list(torch.split(hidden_states, cp_input_dict["split_list"], dim=0))
            hidden_states = torch.cat(
                [hidden_states_list[i] for i in cp_input_dict["zigzag_index"]], dim=0
            ).view(-1, hidden_states.shape[-1])
        residual = None
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.decoder(
                positions, hidden_states, forward_batch, residual, zero_allocator,
                cp_input_dict=cp_input_dict
            )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)
            
            if cp_input_dict is not None and forward_batch.forward_mode.is_context_parallel_extend():
                # allgather + rerrange
                bs_seq_len, hidden_size = hidden_states.shape
                hidden_states = attn_tp_all_gather_reorgan_into_tensor(hidden_states, 
                                                                cp_input_dict["toatl_seq_lens"], 
                                                                self.cp_size, 
                                                                cp_input_dict,
                                                                torch.cuda.current_stream())
                outputs_list = list(torch.split(hidden_states, cp_input_dict["reverse_split_len"], dim=0))
                hidden_states = torch.cat([outputs_list[i] for i in cp_input_dict["cp_reverse_index"]], dim=0)
                hidden_states = hidden_states.view(-1, hidden_size)

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
        # if not set, model load will be broken in DeepseekV3ForCausalLM load_weights()
        self.pp_group = get_pp_group()
        self.determine_num_fused_shared_experts("DeepseekV3ForCausalLMNextN")
        self.cp_rank = get_attention_tp_rank()
        self.cp_size = get_attention_tp_size()
        self.use_nsa = is_deepseek_nsa(config)
        self.use_dp_cp_ag_after_qlora = get_bool_env_var("SGLANG_USE_DP_CP_AG_AFTER_DSA")

        self.model = DeepseekModelNextN(
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
    ) -> torch.Tensor:
        cp_input_dict = None
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        cur_cp_seq_len = len(input_ids) // (self.cp_size * 2)
        if (cur_cp_seq_len != 0) and self.cp_size > 1 and \
            self.use_nsa and self.use_dp_cp_ag_after_qlora and \
            forward_batch.forward_mode.is_context_parallel_extend():
            cp_input_dict = prepare_input_dp_with_cp_dsa(torch.tensor(len(input_ids)), 
                                                         self.cp_rank, 
                                                         self.cp_size,
                                                         forward_batch.seq_lens_cpu.tolist())
        hidden_states = self.model(input_ids, positions, forward_batch, cp_input_dict=cp_input_dict)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = [DeepseekV3ForCausalLMNextN]