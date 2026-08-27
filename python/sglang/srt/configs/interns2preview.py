from sglang.srt.configs.qwen3_5 import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)


class InternS2PreviewVisionConfig(Qwen3_5MoeVisionConfig):
    model_type = "intern_s2_preview"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class InternS2PreviewConfig(Qwen3_5MoeConfig):
    model_type = "intern_s2_preview"
    sub_configs = {
        "vision_config": InternS2PreviewVisionConfig,
        "text_config": Qwen3_5MoeTextConfig,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
