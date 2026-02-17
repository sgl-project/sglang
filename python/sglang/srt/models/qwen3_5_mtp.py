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

"""Inference-only Qwen3_5 MTP model."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class Qwen3_5ForCausalLMMTP(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.is_multimodal = hasattr(config, "text_config")
        if self.is_multimodal:
            config = config.text_config

        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.pp_group = get_pp_group()

        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        RMSNorm_cls = GemmaRMSNorm
        self.pre_fc_norm_embedding = RMSNorm_cls(
            config.hidden_size, config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = RMSNorm_cls(config.hidden_size, config.rms_norm_eps)
        config.num_hidden_layers = 1
        config.full_attention_interval = 1
        self.model = Qwen3_5ForCausalLM(
            config,
            quant_config,
            prefix=add_prefix("mtp", prefix),
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )

        self.logits_processor = LogitsProcessor(config)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight

        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert input_embeds is None
        input_embeds = forward_batch.mm_input_embeds
        if (
            forward_batch.forward_mode.is_extend()
            and forward_batch.contains_mm_inputs()
            and not forward_batch.forward_mode.is_draft_extend()
        ):
            assert input_embeds is not None
            input_embeds = torch.cat(
                [input_embeds[:-1], self.model.embed_tokens(input_ids[-1].unsqueeze(0))]
            )

        if input_embeds is None:
            input_embeds = self.model.embed_tokens(input_ids)

        hidden_states = forward_batch.spec_info.hidden_states

        if not forward_batch.forward_mode.is_idle():
            input_embeds = self.pre_fc_norm_embedding(input_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)

        hidden_states = self.fc(hidden_states)

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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for MoE experts (non-fused/fused)
        num_experts = getattr(self.config, "num_experts", None)
        if num_experts is not None:
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                ckpt_gate_proj_name="gate_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="up_proj",
                num_experts=num_experts,
            )
        else:
            expert_params_mapping = []

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        # fused experts: experts.w13_weight / experts.w2_weight
        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            param = params_dict[name]
            weight_loader = param.weight_loader
            # Let EP MoE layer handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            # Only process MTP branch weights
            if "mtp" not in name:
                continue

            if name.startswith("mtp."):
                # Remove the mtp. prefix for processing
                name = name.replace("mtp.", "model.")

                name = name.replace("model.fc", "fc")
                name = name.replace("model.pre_fc", "pre_fc")

            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            # 1) Process stacked parameters (q_proj/k_proj/v_proj & gate_proj/up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Check if this is a fused expert weight
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-matching weights
                if weight_name not in name:
                    continue

                # Skip MoE experts.* here, handled separately below
                if "mlp.experts" in name:
                    continue

                name_mapped = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if (
                    name_mapped.endswith(ignore_suffixes)
                    and name_mapped not in params_dict
                ):
                    continue

                if name_mapped not in params_dict:
                    continue

                param = params_dict[name_mapped]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                name = name_mapped
                break
            else:
                # 2) Process MoE expert weights (including fused experts)
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)

                    # Fused experts: single checkpoint weight contains multiple experts
                    if is_fused_expert and num_experts is not None:
                        if "experts.gate_up_proj" in name:
                            # gate_up_proj fused: split into w1 / w3
                            loaded_w1, loaded_w3 = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w1,
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_w3,
                                "w3",
                                num_experts,
                            )
                        else:
                            # down_proj fused: distribute entire weight
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Non-fused expert, load by expert_id/shard
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        if name_mapped not in params_dict:
                            break
                        param = params_dict[name_mapped]
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    # Skip expert weight if not handled by current rank
                    if is_expert_weight:
                        continue

                    # 3) Regular non-stacked / non-expert parameters, use default loader
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict:
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )

            loaded_params.add(name)
        return loaded_params


EntryClass = [Qwen3_5ForCausalLMMTP]
