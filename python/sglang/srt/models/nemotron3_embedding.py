"""NVIDIA Nemotron-3-Embed: a bidirectional Ministral3 with normalized mean pooling."""

from typing import Iterable, Optional, Tuple

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.ministral3 import Ministral3ForCausalLM


class _FP32Pooler(Pooler):
    """Pools and normalizes in FP32, so embeddings are returned in FP32.

    Mean pooling over a packed batch subtracts one running sum; in BF16 that
    sum loses short requests packed after long ones to cancellation.
    """

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        return super().forward(hidden_states.float(), forward_batch)


class EmbeddingNemotron3Model(Ministral3ForCausalLM):
    """Nemotron-3-Embed: bidirectional Ministral3 with normalized mean pooling."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # The checkpoint ties word embeddings, so the inherited lm_head is
        # only an alias of embed_tokens and adds no parameters.
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.pooler = _FP32Pooler(pooling_type=PoolingType.MEAN, normalize=True)
        for layer in self.model.layers[self.model.start_layer : self.model.end_layer]:
            layer.self_attn.attn.attn_type = AttentionType.ENCODER_ONLY

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        kwargs["get_embedding"] = True
        return super().forward(
            input_ids, positions, forward_batch, input_embeds, **kwargs
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The checkpoint stores the bare backbone without the "model." prefix.
        backbone_prefixes = ("embed_tokens.", "layers.", "norm.")
        return super().load_weights(
            (f"model.{name}" if name.startswith(backbone_prefixes) else name, weight)
            for name, weight in weights
        )


EntryClass = [EmbeddingNemotron3Model]
