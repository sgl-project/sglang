from typing import Optional, Union

from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen3 import Qwen3Config

from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.srt.multimodal.molmo2 import Molmo2Processor


def _qwen3_text_config(text_config: Union[dict, Qwen3Config]) -> Qwen3Config:
    if isinstance(text_config, Qwen3Config):
        return text_config
    if text_config.get("model_type") == Qwen3Config.model_type:
        return Qwen3Config(**text_config)
    if (
        text_config["norm_after"]
        or text_config["qk_norm_type"] != "qwen3"
        or text_config["rope_scaling"] is not None
        or text_config["rope_scaling_layers"] is not None
    ):
        raise ValueError(
            "Molmo2 support covers the Qwen3-backbone checkpoints (Molmo2-4B, "
            "Molmo2-8B); this checkpoint uses a different text backbone."
        )
    return Qwen3Config(
        vocab_size=text_config["vocab_size"],
        hidden_size=text_config["hidden_size"],
        intermediate_size=text_config["intermediate_size"],
        num_hidden_layers=text_config["num_hidden_layers"],
        num_attention_heads=text_config["num_attention_heads"],
        num_key_value_heads=text_config["num_key_value_heads"],
        head_dim=text_config["head_dim"],
        hidden_act=text_config["hidden_act"],
        max_position_embeddings=text_config["max_position_embeddings"],
        rms_norm_eps=text_config["layer_norm_eps"],
        attention_bias=text_config["qkv_bias"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": text_config["rope_theta"],
        },
        tie_word_embeddings=False,
    )


@register_customized_processor(Molmo2Processor)
class Molmo2Config(PretrainedConfig):
    model_type = "molmo2"

    def __init__(
        self,
        vit_config: Optional[dict] = None,
        adapter_config: Optional[dict] = None,
        text_config: Optional[Union[dict, Qwen3Config]] = None,
        additional_vocab_size: Optional[int] = None,
        image_patch_id: int = 151938,
        **kwargs,
    ):
        self.vit_config = vit_config or {}
        self.adapter_config = adapter_config or {}
        if additional_vocab_size is None and isinstance(text_config, dict):
            additional_vocab_size = text_config.get("additional_vocab_size")
        self.additional_vocab_size = additional_vocab_size or 0
        self.text_config = (
            _qwen3_text_config(text_config) if text_config is not None else None
        )
        self.image_patch_id = image_patch_id
        super().__init__(**kwargs)
