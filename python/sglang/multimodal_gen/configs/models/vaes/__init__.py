# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.configs.models.vaes.dac import DacVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuan3d import Hunyuan3DVAEConfig
from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import HunyuanVAEConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

__all__ = [
    "DacVAEConfig",
    "HunyuanVAEConfig",
    "MiniMaxH3AudioVAEConfig",
    "MiniMaxH3VideoVAEConfig",
    "StableDiffusion3VAEConfig",
    "WanVAEConfig",
    "Hunyuan3DVAEConfig",
]
