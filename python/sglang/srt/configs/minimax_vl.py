# SPDX-License-Identifier: Apache-2.0

import copy
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


class MiniMaxM3TinyConfig(MiniMaxM3VLConfig):
    model_type = "minimax_m3_tiny"

    def __init__(self, **kwargs):
        # Transformers 5 generates an initializer for subclasses without one.
        MiniMaxVLBaseConfig.__init__(self, **kwargs)
        sparse = getattr(
            getattr(self, "text_config", None), "sparse_attention_config", None
        )
        if sparse:
            # M3.1 checkpoints contain index Q/K projections only.
            sparse.setdefault(
                "sparse_disable_index_value", sparse["sparse_attention_freq"].copy()
            )

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        config_dict = copy.deepcopy(config_dict)
        text = config_dict.get("text_config", {})
        layer_types = text.get("layer_types")
        draft = text.get("dflash_config", {})
        if layer_types and len(layer_types) != text.get("num_hidden_layers"):
            if len(layer_types) != draft.get("num_hidden_layers"):
                raise ValueError(
                    "MiniMax layer_types does not match target or draft depth"
                )
            # Bundled checkpoints attach draft windows to the target text config.
            draft["layer_types"] = text.pop("layer_types")
        return super().from_dict(config_dict, **kwargs)


def configure_minimax_m3_dspark(config):
    """Translate M3.1 DSpark checkpoints to the public draft config contract."""
    standalone = config.architectures == ["DSparkMiniMaxDraftModel"]
    text = copy.deepcopy(getattr(config, "text_config", config))
    if isinstance(text, dict):
        text = PretrainedConfig(**text)
    for key, value in config.to_dict().items():
        if key.startswith("dspark_"):
            setattr(text, key, value)
    draft = dict(getattr(text, "dspark_config", {}))
    if not draft.get("use_dense_ffn", getattr(text, "dspark_use_dense_ffn", False)):
        raise ValueError("M3.1 DSpark requires the dense draft FFN checkpoint")
    target_depth = (
        int(text.dflash_config["num_target_layers"])
        if standalone
        else text.num_hidden_layers
    )
    text.num_hidden_layers = int(text.dflash_config["num_hidden_layers"])
    text.num_nextn_predict_layers = 0
    text.num_mtp_modules = 0
    text.architectures = ["DSparkMiniMaxDraftModel"]
    text.sparse_attention_config = None
    text.moe_layer_freq = [0] * text.num_hidden_layers
    text.dense_intermediate_size = int(
        draft.get("ffn_hidden_size", getattr(text, "dspark_ffn_hidden_size", None))
    )
    text.block_size = int(text.dspark_block_size)
    text.markov_rank = int(text.dspark_markov_rank)
    text.markov_head_type = getattr(text, "dspark_markov_head_type", "vanilla")
    text.mask_token_id = int(text.dspark_noise_token_id)
    draft_layer_types = text.dflash_config.get("layer_types")
    if draft_layer_types:
        text.layer_types = draft_layer_types
    if standalone:
        target_ids = [int(i) for i in text.dflash_config["target_layer_ids"]]
    else:
        # Bundled checkpoints use one-based output-layer IDs.
        target_ids = sorted(int(i) - 1 for i in text.dspark_target_layer_ids)
    if (
        not target_ids
        or len(set(target_ids)) != len(target_ids)
        or any(i < 0 or i >= target_depth for i in target_ids)
    ):
        raise ValueError(
            "M3.1 DSpark capture layer IDs must be unique and in [0, target depth)"
        )
    text.dspark_target_layer_ids = target_ids
    text.dflash_config = dict(text.dflash_config)
    text.dflash_config.update(
        num_hidden_layers=text.num_hidden_layers,
        num_target_layers=target_depth,
        block_size=text.block_size,
        target_layer_ids=target_ids,
        mask_token_id=text.mask_token_id,
    )
    text.quantization_config = getattr(config, "quantization_config", None)
    return text
