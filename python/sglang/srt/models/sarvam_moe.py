"""Inference-only Sarvam MoE models for SGLang.
- SarvamMLAForCausalLM (105B)
- SarvamMoEForCausalLM (30B)
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.bailing_moe import BailingMoEForCausalLM
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM

logger = logging.getLogger(__name__)


class SarvamMLAForCausalLM(DeepseekV2ForCausalLM):

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig = None,
        prefix: str = "",
    ) -> None:
        self._remap_config(config)
        super().__init__(config, quant_config, prefix)

    @staticmethod
    def _remap_config(config: PretrainedConfig) -> None:
        if not hasattr(config, "n_routed_experts"):
            config.n_routed_experts = config.num_experts
        if not hasattr(config, "n_shared_experts"):
            config.n_shared_experts = getattr(config, "num_shared_experts", None)

        if not hasattr(config, "num_experts"):
            config.num_experts = config.n_routed_experts

        if not hasattr(config, "norm_topk_prob"):
            config.norm_topk_prob = True
        if not hasattr(config, "topk_method"):
            config.topk_method = "noaux_tc"

        _defaults = {
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
            "n_group": 1,
            "topk_group": 1,
        }
        for attr, default in _defaults.items():
            if not hasattr(config, attr):
                setattr(config, attr, default)

    def determine_num_fused_shared_experts(
        self, architecture: str = "SarvamMLAForCausalLM"
    ):
        super().determine_num_fused_shared_experts(architecture)

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ):
        def _center_bias(
            ws: Iterable[Tuple[str, torch.Tensor]],
        ) -> Iterable[Tuple[str, torch.Tensor]]:
            for name, w in ws:
                if "e_score_correction_bias" in name and w.numel() > 0:
                    w = w - w.mean()
                yield name, w

        super().load_weights(_center_bias(weights), is_nextn)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=getattr(config, "n_group", None),
        )


class SarvamMoEForCausalLM(BailingMoEForCausalLM):

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],
        input_embeds: torch.Tensor = None,
    ) -> Optional[LogitsProcessorOutput]:
        start, end = split_interval

        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.word_embeddings(input_ids)
            else:
                forward_batch.hidden_states = input_embeds
            forward_batch.residual = None

        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            if forward_batch.residual is None:
                hidden_states = self.model.norm(forward_batch.hidden_states)
            else:
                hidden_states, _ = self.model.norm(
                    forward_batch.hidden_states, forward_batch.residual
                )
            forward_batch.hidden_states = hidden_states

            return self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )

        return None


EntryClass = [SarvamMLAForCausalLM, SarvamMoEForCausalLM]
