"""Fast-dVLM: a Qwen2.5-VL checkpoint converted to block diffusion.

Weights and layout match Qwen2.5-VL. The differences are full logits for the
diffusion step and a switch between bidirectional attention inside the block
being denoised and causal attention for the forward that writes its final KV.
"""

from typing import Optional, Tuple

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration


class Fast_dVLMForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    def __init__(
        self,
        config: Qwen2_5_VLConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.logits_processor = LogitsProcessor(config, return_full_logits=True)
        self._block_attention_layers: Tuple[RadixAttention, ...] = (
            ()
            if self.config.encoder_only
            else tuple(
                module
                for module in self.model.modules()
                if isinstance(module, RadixAttention)
            )
        )

    def set_block_attention_causal(self, causal: bool) -> None:
        # Backends read attn_type at forward time, so this only holds for
        # eager forwards; HierarchyBlock disables DLLM_EXTEND CUDA graphs.
        attn_type = AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
        for layer in self._block_attention_layers:
            layer.attn_type = attn_type


EntryClass = [Fast_dVLMForConditionalGeneration]
