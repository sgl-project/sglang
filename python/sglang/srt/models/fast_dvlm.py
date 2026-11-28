"""Fast-dVLM: a Qwen2.5-VL checkpoint converted to block diffusion.

Weights and layout match Qwen2.5-VL. The differences are full logits for the
diffusion step and a switch between bidirectional attention inside the block
being denoised and causal attention for the forward that writes its final KV.
"""

from typing import Iterable, Optional, Tuple

import torch
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
        # Bidirectional by default: denoising forwards replay CUDA graphs
        # captured in this state, and HierarchyBlock runs causal ones eagerly.
        self.set_block_attention_causal(False)

    def set_block_attention_causal(self, causal: bool) -> None:
        # Backends read attn_type when a forward runs or is captured, so a
        # replayed graph keeps the type it was captured with.
        attn_type = AttentionType.DECODER if causal else AttentionType.ENCODER_ONLY
        for layer in self._block_attention_layers:
            layer.attn_type = attn_type

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(
            (_flatten_weight_name(name), tensor) for name, tensor in weights
        )


def _flatten_weight_name(name: str) -> str:
    # The release is saved in the transformers >= 4.52 layout, which nests the
    # language model and vision tower under "model."; Qwen2.5-VL loads flat names.
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    if name.startswith("model.visual."):
        return name[len("model.") :]
    return name


EntryClass = [Fast_dVLMForConditionalGeneration]
