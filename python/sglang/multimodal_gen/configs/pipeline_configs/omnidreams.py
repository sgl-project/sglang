# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for NVIDIA OmniDreams.

Phase 0 wires the static structure (DiT config, VAE reuse, task type) and the
2-step flow-match sigma schedule. The denoising/decoding callbacks used at GPU
time are added in later phases.
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
from sglang.multimodal_gen.configs.models.vaes.wanvae import OmniDreamsVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)


def warp_flow_match_sigmas(
    denoising_timesteps: tuple[int, ...] = (1000, 450),
    flow_shift: float = 5.0,
    sigma_min: float = 0.0,
) -> list[float]:
    """OmniDreams 2-step flow-match sigma schedule.

    Each raw timestep ``t`` maps to ``s = t / 1000`` then is warped by
    ``shift*s / (1 + (shift-1)*s)``; ``sigma_min`` is appended as the final
    target. With the distilled defaults this yields ``[1.0, 0.8036, 0.0]``.
    """
    sigmas = [
        flow_shift * (t / 1000.0) / (1.0 + (flow_shift - 1.0) * (t / 1000.0))
        for t in denoising_timesteps
    ]
    sigmas.append(sigma_min)
    return sigmas


@dataclass
class OmniDreamsPipelineConfig(PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.I2V
    # CFG disabled for the distilled checkpoint.
    should_use_guidance: bool = False
    # Native bf16 DiT; VAE in fp32 for numerical stability.
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    # Flow-match warp shift (also drives warp_flow_match_sigmas).
    flow_shift: float | None = 5.0

    dit_config: OmniDreamsDiTConfig = field(default_factory=OmniDreamsDiTConfig)
    # A.5: OmniDreams uses a Cosmos-Predict2.5-based latent space; the
    # latents_mean/std defaults match Wan 2.1 (safe fallback). Override in
    # OmniDreamsVAEArchConfig once GPU validation confirms the correct values.
    vae_config: OmniDreamsVAEConfig = field(default_factory=OmniDreamsVAEConfig)

    # 2-step distilled flow-match schedule.
    denoising_timesteps: tuple[int, ...] = (1000, 450)
    sigma_min: float = 0.0

    def denoising_sigmas(self) -> list[float]:
        return warp_flow_match_sigmas(
            self.denoising_timesteps,
            self.flow_shift if self.flow_shift is not None else 5.0,
            self.sigma_min,
        )
