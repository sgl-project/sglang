# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


def _coerce_sub_config(
    sub_config: Optional[dict], default_model_type: str
) -> Optional[PretrainedConfig]:
    """Convert a config dict to a ``PretrainedConfig``.

    Unknown ``model_type`` (e.g. M3's ``minimax_m2``, absent from
    ``CONFIG_MAPPING``) falls back to ``PretrainedConfig`` so dict keys
    still become real attributes.
    """
    if not isinstance(sub_config, dict):
        return sub_config
    model_type = sub_config.get("model_type", default_model_type)
    cls = CONFIG_MAPPING.get(model_type, PretrainedConfig)
    return cls(**sub_config)


class MiniMaxVLBaseConfig(PretrainedConfig):
    def __init__(
        self,
        vision_config: Optional[dict] = None,
        text_config: Optional[dict] = None,
        image_token_index: int = 200025,
        video_token_index: int = 200026,
        image_seq_length: int = 576,
        process_image_mode: str = "dynamic_res",
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
        vision_feature_layer: int = -1,
        vision_feature_select_strategy: str = "full",
        img_token_compression_config: Optional[dict] = None,
        image_grid_pinpoints: Optional[str] = None,
        **kwargs,
    ):
        self.vision_config = _coerce_sub_config(vision_config, "clip_vision_model")
        self.text_config = _coerce_sub_config(text_config, "mixtral")

        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.image_seq_length = image_seq_length
        self.process_image_mode = process_image_mode
        self.projector_hidden_act = projector_hidden_act
        self.multimodal_projector_bias = multimodal_projector_bias
        self.vision_feature_layer = vision_feature_layer
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.img_token_compression_config = img_token_compression_config or {}
        self.image_grid_pinpoints = image_grid_pinpoints

        super().__init__(**kwargs)


class MiniMaxM3VLConfig(MiniMaxVLBaseConfig):
    model_type = "minimax_m3_vl"
