# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

__all__ = [
    "HunyuanVAEConfig",
    "WanVAEConfig",
    "Hunyuan3DVAEConfig",
]
