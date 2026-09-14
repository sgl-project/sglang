# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_contract import (
    validate_minimax_h3_vae_latent_stats,
)


@dataclass
class MiniMaxH3VideoVAEArchConfig(VAEArchConfig):
    latent_channels: int = 24
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 16
    vae_clip_length: int = 17
    vae_token_drop: int = 3
    vae_encoder_tiling: int = 1
    vae_decoder_tiling: int = 1
    vae_parallel_tiling: int = 1
    vae_tile_size: int = 256
    vae_tile_overlap_min: int = 64
    vae_chunk_dim: int = -1


@dataclass
class MiniMaxH3VideoVAEConfig(VAEConfig):
    arch_config: MiniMaxH3VideoVAEArchConfig = field(
        default_factory=MiniMaxH3VideoVAEArchConfig
    )
    load_encoder: bool = True
    load_decoder: bool = True
    use_tiling: bool = True
    use_parallel_tiling: bool = True
    # The released checkpoint's quality contract uses overlapping latent
    # tiles. Parallel tiling distributes whole tiles without changing that
    # recipe. Spatial-shard decode is rejected because validation found output
    # mismatches on H3.
    parallel_decode_mode: str = "tiled"

    def resolved_parallel_decode_mode(self) -> str:
        if self.parallel_decode_mode == "auto":
            return "tiled"
        if self.parallel_decode_mode in ("spatial", "spatial_shard"):
            raise ValueError(
                "MiniMax H3 rejects spatial-shard VAE decode because it failed "
                "the released quality contract; use tiled"
            )
        if self.parallel_decode_mode == "tiled":
            return "tiled"
        if self.parallel_decode_mode == "patch":
            raise ValueError("MiniMax H3 does not support patch VAE decode; use tiled")
        raise ValueError(
            f"unsupported MiniMax H3 VAE parallel decode mode "
            f"{self.parallel_decode_mode!r}"
        )

    def update_model_arch(self, source_model_dict: dict) -> None:
        # Native-Diffusers AutoencoderKLMiniMaxH3 config field names.
        aliases = {
            "clip_length": "vae_clip_length",
            "token_drop": "vae_token_drop",
        }
        model_dict = {
            aliases.get(key, key): value for key, value in source_model_dict.items()
        }
        super().update_model_arch(model_dict)

    def post_init(self) -> None:
        self.resolved_parallel_decode_mode()
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )


__all__ = ["MiniMaxH3VideoVAEArchConfig", "MiniMaxH3VideoVAEConfig"]
