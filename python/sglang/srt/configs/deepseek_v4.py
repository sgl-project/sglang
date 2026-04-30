from typing import Optional

_fp4_experts: Optional[bool] = None


def set_fp4_experts(value: bool) -> None:
    global _fp4_experts
    _fp4_experts = value


def get_fp4_experts() -> bool:
    return bool(_fp4_experts)


class DeepSeekV4Config:
    """Configuration holder for DeepSeek V4 model parameters.

    This is not a transformers PretrainedConfig subclass to avoid Python 3.10
    dataclass inheritance issues.  Values are populated from the HuggingFace
    config.json via _load_deepseek_v4_model() which rewrites architectures /
    model_type so that AutoConfig produces a plain PretrainedConfig.
    """

    pass
