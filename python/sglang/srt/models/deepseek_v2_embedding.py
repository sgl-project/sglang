# Adapted from deepseek_v2.py
from typing import Optional

import torch
from transformers import PretrainedConfig

from sglang.srt.layers.pooler import EmbeddingPoolerOutput, Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.deepseek_v2 import DeepseekV3ForCausalLM


class DeepseekV3BidirectionalModel(DeepseekV3ForCausalLM):
    """Bidirectional DeepSeek-V3 encoder for dense embedding models.

    Checkpoints exported as architectures=["DeepseekV3BidirectionalModel"], e.g.
    ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826, run the DeepSeek-V3 MoE
    backbone with encoder-style (bidirectional) attention -- enabled by
    is_causal=False on the HF config, threaded into the MHA prefill path as
    AttentionType.ENCODER_ONLY -- and produce a sentence embedding via mean
    pooling over non-padding tokens + L2 normalization.

    The absorbed-MLA attention kernels are causal-only, so these models are
    served prefill-only through the MHA path (attn_mha). CUDA-graph capture is
    disabled for them in ServerArgs so prefill never falls back to the absorbed
    (causal) kernel.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=True)

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
        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)
        return self.pooler(hidden_states, forward_batch)


EntryClass = [DeepseekV3BidirectionalModel]
