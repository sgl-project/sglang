# coding=utf-8
# Copyright 2026 The HunYuan team.
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

"""Inference-only HunyuanV3 NextN (MTP) Speculative Decoding."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.schedule_batch import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.hunyuan_v3 import HYV3DecoderLayer
from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)


class HYV3ModelNextN(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.alt_stream = torch.cuda.Stream() if is_cuda() else None

        # Force MoE for the MTP layer: first_k_dense_replace=1 would make
        # layer_id=0 pick a dense MLP instead of MoE, so override it.
        orig_first_k = getattr(config, "first_k_dense_replace", 0)
        config.first_k_dense_replace = 0
        self.decoder = HYV3DecoderLayer(
            config=config,
            layer_id=0,
            quant_config=quant_config,
            prefix=f"{prefix}.decoder",
            alt_stream=self.alt_stream,
        )
        config.first_k_dense_replace = orig_first_k

        self.shared_head = nn.Module()
        self.shared_head.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
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

        residual = None
        hidden_states, residual = self.decoder(
            positions, hidden_states, forward_batch, residual
        )

        if not forward_batch.forward_mode.is_idle():
            if residual is not None:
                hidden_states, _ = self.shared_head.norm(hidden_states, residual)
            else:
                hidden_states = self.shared_head.norm(hidden_states)

        return hidden_states


class HYV3ForCausalLMNextN(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.quant_config = quant_config

        self.model = HYV3ModelNextN(config, quant_config, prefix="model")
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix="lm_head",
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

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        nextn_layer_id = self.config.num_hidden_layers
        nextn_prefix = f"model.layers.{nextn_layer_id}."
        spec_weight_names = ("enorm", "hnorm", "eh_proj")

        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.startswith(nextn_prefix):
                subname = name[len(nextn_prefix) :]
                if any(subname.startswith(s) for s in spec_weight_names):
                    name = f"model.{subname}"
                else:
                    name = f"model.decoder.{subname}"
            elif name == "model.shared_head.norm.weight":
                pass
            elif (
                "embed_tokens" in name
                or "shared_head.head" in name
                or "lm_head" in name
            ):
                continue
            else:
                continue

            if "rotary_emb.inv_freq" in name:
                continue

            if "router.gate." in name:
                name = name.replace("router.", "")

            is_found = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                is_found = True
                break
            if is_found:
                continue

            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                is_expert_weight = True
                name_mapped = name.replace(weight_name, param_name)
                if name_mapped not in params_dict:
                    continue
                param = params_dict[name_mapped]
                weight_loader = param.weight_loader
                weight_loader(
                    param,
                    loaded_weight,
                    name_mapped,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            if is_expert_weight:
                continue

            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)


EntryClass = [HYV3ForCausalLMNextN]
