import logging
from functools import lru_cache
from typing import Optional
import torch.nn as nn
from transformers.models.glm4v_moe.configuration_glm4v_moe import Glm4vMoeConfig

from sglang.srt.distributed import (
    get_moe_expert_parallel_world_size,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.glm4_moe import Glm4MoeModel
from sglang.srt.models.glm4v import Glm4vForConditionalGeneration, Glm4vVisionModel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix, is_cuda, log_info_on_rank0
from sglang.srt.utils.hf_transformers_utils import get_processor

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Glm4vMoeForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(
        self,
        config: Glm4vMoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)

        self.config = config
        vision_utils.update_vit_attn_dummy_heads_config(self.config)
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts("Glm4MoeForCausalLM")
        self.num_fused_shared_experts = (
            0
            if get_global_server_args().disable_shared_experts_fusion
            else config.n_shared_experts
        )

        self.model = Glm4MoeModel(
            config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.visual = Glm4vVisionModel(
            config.vision_config,
            norm_eps=getattr(config, "rms_norm_eps", 1e-5),
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        # For EAGLE3 support
        self.capture_aux_hidden_states = False


EntryClass = [Glm4vMoeForConditionalGeneration]
