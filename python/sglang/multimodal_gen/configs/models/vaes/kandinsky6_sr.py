# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 SR causal video KVAE configuration.

Two ``vae/config.json`` layouts have been observed from official Diffusers
``Kandinsky6SRPipeline`` repos: an older one nesting the KVAE architecture under
``encoder_config`` / ``decoder_config`` dicts (the config stores the *string* ``"None"`` for
their unused channel fields)::

    {"vae_type": "video-kvae", "encoder_config": {...}, "decoder_config": {...},
     "scaling_factor": 0.9103, "spatial_factor": 16, "temporal_factor": 4}

and the current one (confirmed against ``kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-
Diffusers`` by an actual strict ``load_state_dict`` on GPU hardware) with those same knobs
flattened to the top level instead -- ``decoder_ch`` / ``decoder_ch_mult`` are the only
decoder-specific overrides, everything else (``num_res_blocks``, ``z_channels``, the
temporal-compression / norm / padding knobs) is shared between encoder and decoder.
``update_model_arch`` copies whichever layout the checkpoint has; ``__post_init__`` synthesizes
``encoder_config`` / ``decoder_config`` from the flat fields when the nested ones weren't
populated, so ``Kandinsky6SRVAE.__init__`` (``runtime/models/vaes/kandinsky6_sr_vae.py``) only
ever has to handle the nested shape. Unknown *top-level* keys beyond the ones declared here are
rejected because ``update_model_arch`` would otherwise store them silently in ``extra_attrs``.
"""

from dataclasses import dataclass, field
from typing import Any

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig

_TOLERATED_EXTRA_KEYS = frozenset({"_class_name", "_diffusers_version"})


@dataclass
class Kandinsky6SRVAEArchConfig(VAEArchConfig):
    vae_type: str = "video-kvae"
    encoder_config: dict[str, Any] = field(default_factory=dict)
    decoder_config: dict[str, Any] = field(default_factory=dict)
    scaling_factor: float = 1.0
    spatial_factor: int = 16
    temporal_factor: int = 4

    # Flat top-level KVAE architecture fields (current official vae/config.json layout -- see
    # module docstring). Left at their None default (and ignored) when the checkpoint instead
    # uses the older nested encoder_config/decoder_config layout.
    in_channels: int | None = None
    out_channels: int | None = None
    z_channels: int | None = None
    ch: int | None = None
    ch_mult: list[float] | None = None
    decoder_ch: int | None = None
    decoder_ch_mult: list[float] | None = None
    num_res_blocks: int | None = None
    padding_mode: str | None = None
    temporal_compress_times: int | None = None
    temporal_compress_start_level: int | None = None
    norm_type: str | None = None
    double_z: bool | None = None
    downsample_version: int | None = None
    fix_pxs: bool | None = None
    # Present in the flat layout but unused by Kandinsky6SRVAE (the KVAE is resolution-agnostic);
    # declared only so it doesn't trip the unknown-top-level-key check below.
    resolution: int | None = None

    def __post_init__(self) -> None:
        unknown = sorted(set(self.extra_attrs) - _TOLERATED_EXTRA_KEYS)
        if unknown:
            raise ValueError(f"Kandinsky6SR VAE config has unknown keys {unknown}")
        if self.vae_type != "video-kvae":
            raise ValueError(
                f"Kandinsky6SRVAE supports only 'video-kvae', got {self.vae_type!r}"
            )
        self.spatial_compression_ratio = self.spatial_factor
        self.temporal_compression_ratio = self.temporal_factor

        if not self.encoder_config and not self.decoder_config and self.ch is not None:
            shared = dict(
                num_res_blocks=self.num_res_blocks,
                z_channels=self.z_channels,
                temporal_compress_times=self.temporal_compress_times,
                temporal_compress_start_level=self.temporal_compress_start_level,
                norm_type=self.norm_type,
                padding_mode=self.padding_mode,
            )
            self.encoder_config = {
                **shared,
                "ch": self.ch,
                "ch_mult": self.ch_mult,
                "in_channels": self.in_channels,
                "double_z": self.double_z,
                "downsample_version": self.downsample_version,
                "fix_pxs": self.fix_pxs,
            }
            self.decoder_config = {
                **shared,
                "ch": self.decoder_ch,
                "ch_mult": self.decoder_ch_mult,
                "out_ch": self.out_channels,
            }


@dataclass
class Kandinsky6SRVAEConfig(VAEConfig):
    arch_config: VAEArchConfig = field(default_factory=Kandinsky6SRVAEArchConfig)
    # SR stages own tile geometry; parallel tiling distributes these existing tiles
    # without enabling the generic VAE tiling or spatial-shard decode paths
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_parallel_decode: bool = False

    def get_vae_scale_factor(self) -> int:
        return self.arch_config.spatial_factor


__all__ = ["Kandinsky6SRVAEArchConfig", "Kandinsky6SRVAEConfig"]
