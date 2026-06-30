# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Inference-only GFusion model.

GFusion uses the DeepSeek V3/MLA runtime path, but dLLM denoising needs
bidirectional attention over each block and full per-position logits.
"""

from typing import Optional

from transformers import PretrainedConfig

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class GFusionModelLM(DeepseekV2ForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self._enable_bidirectional_attention()
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)

    def _enable_bidirectional_attention(self) -> None:
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            self_attn = layer.self_attn
            self_attn.attn_mqa.attn_type = AttentionType.ENCODER_ONLY
            self_attn.attn_mha.attn_type = AttentionType.ENCODER_ONLY

    def determine_num_fused_shared_experts(
        self, architecture: str = "GFusionModelLM"
    ):
        # GFusion checkpoints are DeepSeek-shaped but use their own architecture
        # names. Treat either runtime name as the expected architecture.
        archs = getattr(self.config, "architectures", None) or []
        if archs and archs[0] == "GFusionForDiffusionLM":
            architecture = "GFusionForDiffusionLM"
        return super().determine_num_fused_shared_experts(architecture=architecture)


class GFusionForDiffusionLM(GFusionModelLM):
    pass


EntryClass = [GFusionModelLM, GFusionForDiffusionLM]
