"""Giga-Embeddings: a bidirectional DeepSeek-V3 (MLA + MoE) with mean pooling."""

from typing import Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM


class _FP32Pooler(Pooler):
    """Pools and normalizes in FP32, so embeddings are returned in FP32.

    Mean pooling over a packed batch subtracts one running sum; in BF16 that
    sum loses short requests packed after long ones to cancellation.
    """

    def forward(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        return super().forward(hidden_states.float(), forward_batch)


class DeepseekV3BidirectionalModel(DeepseekV2ForCausalLM):
    """ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826.

    The DeepSeek-V3 backbone with bidirectional self-attention and no LM head,
    pooled by normalized mean over the tokens of each request.
    """

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
            layer.self_attn.attn_mha.attn_type = AttentionType.ENCODER_ONLY
            layer.self_attn.attn_mqa.attn_type = AttentionType.ENCODER_ONLY

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = True,
    ) -> EmbeddingPoolerOutput:
        assert get_embedding, f"{self.__class__.__name__} is only used for embedding"
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model(
                input_ids, positions, forward_batch, input_embeds
            )
        return self.pooler(hidden_states, forward_batch)


EntryClass = [DeepseekV3BidirectionalModel]
