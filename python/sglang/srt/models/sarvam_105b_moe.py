"""Inference-only Sarvam MoE 105B model with MLA (Multi-head Latent Attention) for SGLang."""

import logging
from typing import Iterable, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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


EntryClass = [SarvamMLAForCausalLM]
