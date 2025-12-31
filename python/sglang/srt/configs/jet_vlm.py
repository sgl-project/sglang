from typing import Any

from transformers.configuration_utils import PretrainedConfig
from transformers.models.siglip import SiglipVisionConfig

from sglang.srt.configs.jet_nemotron import JetNemotronConfig
from sglang.srt.configs.mamba_utils import Mamba2CacheParams


class JetVLMConfig(PretrainedConfig):
    model_type = "jet_vlm"
    sub_configs = {
        "text_config": JetNemotronConfig,
        "vision_config": SiglipVisionConfig,
    }
    _auto_class = "AutoConfig"

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        **kwargs,
    ):
        self.text_config = (
            JetNemotronConfig(**text_config)
            if text_config is not None
            else JetNemotronConfig()
        )
        self.vision_config = (
            SiglipVisionConfig(**vision_config)
            if vision_config is not None
            else SiglipVisionConfig()
        )

        self.image_token_id = image_token_id if image_token_id is not None else -1
        self.video_token_id = video_token_id if video_token_id is not None else -1

        super().__init__(**kwargs)

    @property
    def full_attention_layer_ids(self) -> list[int]:
        return self.text_config.full_attention_layer_ids

    @property
    def linear_layer_ids(self) -> list[int]:
        return self.text_config.linear_layer_ids

    @property
    def mamba2_cache_params(self) -> Mamba2CacheParams:
        return self.text_config.mamba2_cache_params
