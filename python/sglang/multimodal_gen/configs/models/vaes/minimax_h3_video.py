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
    # Decoder tiles and temporal windows that share one decoder forward while a
    # request's quality tier enables the VAE fast path ("lossless" and "high");
    # "exact" keeps the released recipe of one tile and one window per forward.
    # 0 means all of them. Batching changes which cuBLAS/cuDNN algorithms run,
    # so it moves rounding, not the math. --performance-mode memory resets both
    # to 1 (MiniMaxH3PipelineConfig.validate_server_args).
    decoder_tile_batch: int = 64
    window_batch: int = 1
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
        for name, value in (
            ("decoder_tile_batch", self.decoder_tile_batch),
            ("window_batch", self.window_batch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"MiniMax H3 vae_config.{name} must be an integer >= 0 "
                    f"(0 = all), got {value!r}"
                )
        if (
            self.arch_config.latents_mean is None
            and self.arch_config.latents_std is None
        ):
            # ComfyUI DiT-forward never loads this VAE; stats stay unset.
            return
        validate_minimax_h3_vae_latent_stats(
            self.arch_config,
            component_name="video_vae",
            expected_channels=24,
        )


__all__ = ["MiniMaxH3VideoVAEArchConfig", "MiniMaxH3VideoVAEConfig"]
