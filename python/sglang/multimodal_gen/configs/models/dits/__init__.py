# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.dits.hunyuan3d import (
    Hunyuan3DDiTConfig,
    Hunyuan3DPlainDiTConfig,
)
from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import HunyuanVideoConfig
from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig

__all__ = [
    "HunyuanVideoConfig",
    "WanVideoConfig",
    "Hunyuan3DDiTConfig",
    "Hunyuan3DPlainDiTConfig",
]
