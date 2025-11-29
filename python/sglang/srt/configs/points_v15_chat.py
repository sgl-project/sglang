from typing import Optional, Union

from transformers import PretrainedConfig, Qwen2Config
from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig


class POINTSV15ChatConfig(PretrainedConfig):
    model_type = "pointsv1.5_chat"

    def __init__(
        self,
        vision_config: Optional[Union[dict, Qwen2VLVisionConfig]] = None,
        llm_config: Optional[Union[dict, Qwen2Config]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = Qwen2VLVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Qwen2VLVisionConfig(**vision_config)
        self.vision_config = vision_config

        if llm_config is None:
            llm_config = Qwen2Config()
        elif isinstance(llm_config, dict):
            llm_config = Qwen2Config(**llm_config)

        self.llm_config = llm_config
        self.hidden_size = self.llm_config.hidden_size
