# SPDX-License-Identifier: Apache-2.0
# Adapted from https://huggingface.co/nvidia/LocateAnything-3B/blob/main/configuration_locateanything.py
"""Config for nvidia/LocateAnything-3B.

LocateAnything is a multimodal grounding/detection model composed of a MoonViT
vision encoder, an InternVL-style ``mlp1`` projector, and a Qwen2 language model
backbone. The config is a composite that wraps a ``MoonViTConfig`` (vision) and a
``Qwen2Config`` (text) plus the special token ids used for the grounding grammar
(``<box>``/``<ref>``/coordinate tokens).
"""

from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config

from sglang.srt.configs.kimi_vl_moonvit import MoonViTConfig


class LocateAnythingConfig(PretrainedConfig):
    model_type = "locateanything"

    def __init__(
        self,
        vision_config: Optional[Union[dict, MoonViTConfig]] = None,
        text_config: Optional[Union[dict, Qwen2Config]] = None,
        image_token_index: int = 151665,
        box_start_token_id: int = 151668,
        box_end_token_id: int = 151669,
        ref_start_token_id: int = 151672,
        ref_end_token_id: int = 151673,
        coord_start_token_id: int = 151677,
        coord_end_token_id: int = 152677,
        none_token_id: int = 4064,
        mlp_connector_layers: int = 2,
        **kwargs,
    ):
        if vision_config is None:
            vision_config = MoonViTConfig()
        elif isinstance(vision_config, dict):
            vision_config = MoonViTConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Qwen2Config()
        elif isinstance(text_config, dict):
            text_config = Qwen2Config(**text_config)
        self.text_config = text_config

        self.image_token_index = image_token_index
        self.box_start_token_id = box_start_token_id
        self.box_end_token_id = box_end_token_id
        # ref_*_token_id and mlp_connector_layers are kept for round-trip
        # fidelity with the HF config; the box-grammar processor reads the box /
        # coord / none ids, and the projector hardcodes its 2-layer structure.
        self.ref_start_token_id = ref_start_token_id
        self.ref_end_token_id = ref_end_token_id
        self.coord_start_token_id = coord_start_token_id
        self.coord_end_token_id = coord_end_token_id
        self.none_token_id = none_token_id
        self.mlp_connector_layers = mlp_connector_layers

        super().__init__(**kwargs)
