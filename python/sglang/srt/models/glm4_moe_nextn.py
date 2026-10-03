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

"""Inference-only GLM-4.5, GLM-4.6 and GLM-4.7 Speculative Decoding."""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.glm4_moe import Glm4MoeDecoderLayer, Glm4MoeForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.utils import add_prefix, is_npu

logger = logging.getLogger(__name__)


class Glm4MoeModelNextN(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
            logger.debug(
                "Overriding Glm4MoeForCausalLMNextN quant config for modelopt_fp4 GLM-4.5 / GLM-4.6 / GLM-4.7 model."
            )
            quant_config = None

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            use_attn_tp_group=is_dp_attention_enabled(),
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        self.decoder = Glm4MoeDecoderLayer(
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

        residual_batch.start(forward_batch)
        with get_global_expert_distribution_recorder().disable_this_region():
            hidden_states = self.decoder(positions, hidden_states, forward_batch)

        hidden_states = residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = residual_batch.norm(
                hidden_states, forward_batch, self.shared_head.norm
            )

        return hidden_states


class Glm4MoeForCausalLMNextN(Glm4MoeForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.tp_size = get_parallel().tp_size
        if is_npu() and get_spec().speculative_draft_model_quantization is None:
            quant_config = None
        self.quant_config = quant_config

        # The draft's own gate: its quantization can differ from the
        # target's, and the decoder below reads the ACTIVE decision while it
        # builds. Also sets num_fused_shared_experts, which drives the
        # inherited loader's shared-expert remap.
        self.num_fused_shared_experts = 0
        self.determine_num_fused_shared_experts()

        self.model = Glm4MoeModelNextN(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("model.shared_head.head", prefix),
            use_attn_tp_group=get_parallel().enable_dp_lm_head,
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


class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    # Mapping from fused module names to their component weight names.
    # Required for quantization configs to correctly identify
    # which layers should be skipped based on the exclude_modules/ignore list.
    packed_modules_mapping = {
        "fused_qkv_a_proj_with_mqa": ["q_a_proj", "kv_a_proj_with_mqa"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    # GLM-5.2's MTP layer index differs from DeepSeek's (61), so the inherited
    # substr mapping would wrongly rewrite GLM's real layer-61 weights.
    # exclude_layers remapping for the MTP layer is handled explicitly in
    # _resolve_nextn_quant_config below instead.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "model.layers.78": "model.decoder",
        },
    )

    _NEXTN_SPEC_WEIGHT_NAMES = ("shared_head.norm", "eh_proj", "enorm", "hnorm")

    @classmethod
    def _map_mtp_ckpt_name(cls, name: str, layer_prefix: str) -> str:
        # Keep this mapping in sync with DeepseekV2WeightLoaderMixin's
        # NextN rule: MTP-specific weights live under model.*, while the
        # decoder block weights live under model.decoder.*.
        if any(part in name for part in cls._NEXTN_SPEC_WEIGHT_NAMES):
            return name.replace(layer_prefix, "model", 1)
        return name.replace(layer_prefix, "model.decoder", 1)

    def _resolve_nextn_quant_config(self, config, quant_config):
        if quant_config is None or quant_config.get_name() != "quark":
            return quant_config

        layer_prefix = f"model.layers.{config.num_hidden_layers}"

        # Quark's per-module scheme selection (e.g. MTP self_attn in PTPC-FP8
        # while MTP MoE is MXFP4) is keyed by "layer_quant_config" patterns
        # using the checkpoint's "model.layers.<N>.*" naming. SGLang queries
        # schemes by the runtime "model.*"/"model.decoder.*" prefix, so those
        # keys need the same remap as exclude_layers below, or they silently
        # fall back to the wrong (layer-type/global) scheme.
        layer_quant_config = quant_config.quant_config.get("layer_quant_config")
        if layer_quant_config:
            quant_config.quant_config["layer_quant_config"] = {
                (
                    self._map_mtp_ckpt_name(pattern, layer_prefix)
                    if pattern.startswith(layer_prefix + ".")
                    else pattern
                ): pattern_config
                for pattern, pattern_config in layer_quant_config.items()
            }

        mtp_excluded = [
            name
            for name in quant_config.exclude_layers
            if name.startswith(layer_prefix + ".")
        ]
        if not mtp_excluded:
            return quant_config

        names = set(quant_config.exclude_layers)
        for name in mtp_excluded:
            names.add(self._map_mtp_ckpt_name(name, layer_prefix))

        # Fused routed experts are queried by the coarse module prefix
        # "model.decoder.mlp.experts". Expanded per-expert leaf excludes do not
        # match that prefix, so add the coarse prefix when any routed expert in
        # the MTP layer is excluded. This keeps only that fused MoE module bf16
        # while allowing the remaining draft modules to use their quant config.
        if any(".mlp.experts." in name for name in mtp_excluded):
            names.add("model.decoder.mlp.experts")

        import copy

        quant_config = copy.copy(quant_config)
        quant_config.exclude_layers = list(names)
        return quant_config


EntryClass = [Glm4MoeForCausalLMNextN, GlmMoeDsaForCausalLMNextN]
