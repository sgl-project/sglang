import copy
from typing import Optional

from sglang.srt.configs.points_gui import POINTSGUIConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2_vl import Qwen2VisionPatchMerger
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.utils import add_prefix

from .points_v15_chat import POINTSV15ChatModel, Qwen2VisionTransformerForNavitPOINTS


class POINTSGUIModel(POINTSV15ChatModel):
    config_class = POINTSGUIConfig
    """GUI Model from WePOINTS series.

    The architecture is similar to POINTSV15ChatModel,
    but uses Qwen3ForCausalLM instead of Qwen2ForCausalLM.
    Thus, we can reuse other code and configurations from
    POINTSV15ChatModel, such as image_processor and
    chat template, etc.
    """

    def __init__(
        self,
        config: POINTSGUIConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        **kwargs
    ) -> None:
        super().__init__()
        config.llm_config._attn_implementation = "flash_attention_2"
        config._attn_implementation_autoset = False
        self.config = config
        self.quant_config = quant_config

        llm_config = copy.deepcopy(config.llm_config)
        llm_config.architectures = ["Qwen3ForCausalLM"]
        self.llm = Qwen3ForCausalLM(
            config=llm_config,
            quant_config=quant_config,
            prefix=add_prefix("llm", prefix),
        )

        self.vision_encoder = Qwen2VisionTransformerForNavitPOINTS(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_encoder", prefix),
        )

        self.vision_projector = Qwen2VisionPatchMerger(
            d_model=config.llm_config.hidden_size,
            context_dim=1280,
            quant_config=quant_config,
            prefix=add_prefix("vision_projector", prefix),
        )
