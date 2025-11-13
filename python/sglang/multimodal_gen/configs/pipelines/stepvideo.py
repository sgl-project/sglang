# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import StepVideoConfig
from sglang.multimodal_gen.configs.models.vaes import StepVideoVAEConfig
from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig


@dataclass
class StepVideoT2VConfig(PipelineConfig):
    """Base configuration for StepVideo pipeline architecture."""

    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=StepVideoConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=StepVideoVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: int = 13
    timesteps_scale: bool = False
    pos_magic: str = (
        "超高清、HDR 视频、环境光、杜比全景声、画面稳定、流畅动作、逼真的细节、专业级构图、超现实主义、自然、生动、超细节、清晰。"
    )
    neg_magic: str = (
        "画面暗、低分辨率、不良手、文本、缺少手指、多余的手指、裁剪、低质量、颗粒状、签名、水印、用户名、模糊。"
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "bf16"
