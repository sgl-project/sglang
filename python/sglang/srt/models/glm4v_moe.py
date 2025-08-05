import logging
from functools import lru_cache
from typing import Optional
import torch.nn as nn
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4v_moeConfig

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.glm4_moe import Glm4MoeModel, Glm4vForConditionalGeneration
from sglang.srt.models.glm4v import Glm4vVisionModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)

class Glm4v_moeForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(
        self,
        config: Glm4v_moeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.config = config

        self.model = Glm4MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.visual = Glm4vVisionModel(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

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
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

EntryClass = [Glm4v_moeForConditionalGeneration]
