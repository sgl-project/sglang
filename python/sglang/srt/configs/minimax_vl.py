# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config for the MiniMax M3 VL family."""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


def _coerce_sub_config(
    sub_config: Optional[dict], default_model_type: str
) -> Optional[PretrainedConfig]:
    """Convert a config dict to a ``PretrainedConfig`` instance.

    If ``model_type`` is registered in HF ``CONFIG_MAPPING`` the corresponding
    config class is used; otherwise we fall back to a generic
    ``PretrainedConfig`` so all dict keys still become real attributes (M3's
    text backbone uses ``model_type="minimax_m2"`` which is not in
    ``CONFIG_MAPPING``).
    """
    if not isinstance(sub_config, dict):
        return sub_config
    model_type = sub_config.get("model_type", default_model_type)
    cls = CONFIG_MAPPING.get(model_type, PretrainedConfig)
    return cls(**sub_config)


class MiniMaxVLBaseConfig(PretrainedConfig):
    """Base config for the MiniMax VL family.

    Handles vision/text sub-config coercion. Concrete subclasses only need to
    declare a unique ``model_type`` string.
    """

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
    """MiniMax M3 VL: vision tower + M3 (mixed sparse/dense MoE) text backbone."""

    model_type = "minimax_m3_vl"
