from typing import ClassVar

from sglang.srt.configs.qwen3_5 import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)


class InternS2MobiusVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "interns2_mobius"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InternS2MobiusTextConfig(Qwen3_5MoeTextConfig):
    model_type = "interns2_mobius_text"

    def __init__(self, **kwargs):
        layer_types = kwargs.get("layer_types")
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = "bfloat16"
        super().__init__(**kwargs)
        # The local Qwen3NextConfig currently consumes layer_types without
        # retaining it. Mobius checkpoints provide the authoritative per-layer
        # schedule, so keep it verbatim rather than regenerating it.
        if layer_types is not None:
            self.layer_types = layer_types
        if not hasattr(self, "num_blocks"):
            self.num_blocks = 4


class InternS2MobiusConfig(Qwen3_5MoeConfig):
    model_type = "interns2_mobius"
    sub_configs: ClassVar = {
        "vision_config": InternS2MobiusVisionConfig,
        "text_config": InternS2MobiusTextConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = "bfloat16"

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            if "dtype" not in text_config and "torch_dtype" not in text_config:
                text_config["dtype"] = "bfloat16"

        super().__init__(
            text_config=text_config,
            vision_config=vision_config,
            **kwargs,
        )
