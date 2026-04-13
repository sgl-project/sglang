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
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.model_config import is_deepseek_nsa
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization import Fp8Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v4 import (
    DeepseekV4DecoderLayer,
    DeepseekV4ForCausalLM,
    enable_nextn_moe_bf16_cast_to_fp8,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    add_prefix,
    get_bool_env_var,
    is_cuda,
    is_npu,
)

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()
_is_npu = is_npu()


class DeepseekModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if enable_nextn_moe_bf16_cast_to_fp8(quant_config):
            # refer to real DeepSeek V3 quant config
            moe_quant_config_override = Fp8Config(
                is_checkpoint_fp8_serialized=True,
                weight_block_size=[128, 128],
            )
        else:
            moe_quant_config_override = None

        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.warning(
                "Overriding DeepseekV4ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
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

        self.eh_proj = None
        ignore_list = quant_config.ignore
        if ignore_list and "mtp.0.h_proj" not in ignore_list:  # int8 quant
            self.e_proj = ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("e_proj", prefix),
            )
            self.h_proj = ReplicatedLinear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("h_proj", prefix),
            )
        else:
            self.eh_proj = ReplicatedLinear(
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("eh_proj", prefix),
            )

        self.alt_stream = (
            torch.cuda.Stream()
            if _is_cuda  # or envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            else None
        )

        self.decoder = DeepseekV4DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,
            prefix="mtp.0",  # add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
        )

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_size = None
        self.hc_eps: float = 1e-6
        self.hc_mult = hc_mult = config.hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if get_bool_env_var("FORCE_DRAFT_MODEL_NON_QUANT"):
            os.environ["SGLANG_DEEPEP_BF16_DISPATCH"] = "1"
            os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "0"
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
            if self.eh_proj is not None:
                hidden_states = self.eh_proj(
                    torch.cat(
                        (
                            self.enorm(hidden_states),
                            self.hnorm(forward_batch.spec_info.hidden_states),
                        ),
                        dim=-1,
                    )
                )[0]
            else:
                hidden_states = (
                    self.e_proj(self.enorm(hidden_states))[0]
                    + self.h_proj(self.hnorm(forward_batch.spec_info.hidden_states))[0]
                )

        if nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
        residual = None

        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states, residual = self.decoder(
                positions,
                hidden_states,
                forward_batch,
                residual,
                zero_allocator,
            )

        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.hc_head(
                hidden_states,
                self.hc_head_fn,
                self.hc_head_scale,
                self.hc_head_base,
            )
            hidden_states = self.shared_head.norm(hidden_states)
            if nsa_use_prefill_cp(forward_batch, self.nsa_enable_prefill_cp):
                # allgather + rerrange
                hidden_states = cp_all_gather_rerange_output(
                    hidden_states,
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
        else:
            hidden_states = hidden_states.view(0, hidden_states.shape[-1])
        if get_bool_env_var("FORCE_DRAFT_MODEL_NON_QUANT"):
            os.environ["SGLANG_DEEPEP_BF16_DISPATCH"] = "0"
            os.environ["DEEP_NORMAL_MODE_USE_INT8_QUANT"] = "1"
        return hidden_states

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.hc_eps)
        mixes = F.linear(x, hc_fn) * rsqrt  # [16, 4]
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps  # [16, 4]
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)


class DeepseekV4ForCausalLMNextN(DeepseekV4ForCausalLM):

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
        self.determine_num_fused_shared_experts("DeepseekV4ForCausalLMNextN")
        self.use_nsa = is_deepseek_nsa(config)
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()
        else:
            self.cp_rank = None
            self.cp_size = None

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
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, self.use_nsa, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )
        hidden_states = self.model(input_ids, positions, forward_batch)
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        super().load_weights(weights, is_nextn=True)


EntryClass = [DeepseekV4ForCausalLMNextN]
