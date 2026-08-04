# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_contract import (
    validate_minimax_h3_vae_latent_stats,
)


@dataclass
class MiniMaxH3AudioVAEArchConfig(VAEArchConfig):
    sample_rate: int = 32000
    latent_channels: int = 32
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    output_channel: int = 2


@dataclass
class MiniMaxH3AudioVAEConfig(VAEConfig):
    arch_config: MiniMaxH3AudioVAEArchConfig = field(
        default_factory=MiniMaxH3AudioVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True

    def post_init(self) -> None:
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="audio_vae",
            expected_channels=32,
        )


__all__ = ["MiniMaxH3AudioVAEArchConfig", "MiniMaxH3AudioVAEConfig"]
