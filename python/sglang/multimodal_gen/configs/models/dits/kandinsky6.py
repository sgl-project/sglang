# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 joint video/audio checkpoint architecture.

Distilled checkpoints have n_grid velocity outputs per latent channel."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig


@dataclass
class Kandinsky6ArchConfig(DiTArchConfig):
    """Checkpoint fields and derived dimensions for Kandinsky6Transformer3DModel."""

    # Chained layer renames apply to checkpoint tensors and LoRA A/B/alpha keys.
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^(visual_transformer_blocks\.\d+)\.video_dec_block\.(.*)$": r"\1.videoT.\2",
            r"^(visual_transformer_blocks\.\d+)\.audio_dec_block\.(.*)$": r"\1.audioT.\2",
            r"^(.*feed_forward)\.net\.0\.proj\.(.*)$": r"\1.mlp.fc_in.\2",
            r"^(.*feed_forward)\.net\.2\.(.*)$": r"\1.mlp.fc_out.\2",
            r"^(video_time_embeddings|audio_time_embeddings)\.timestep_embedder\.linear_1\.(.*)$": r"\1.in_layer.\2",
            r"^(video_time_embeddings|audio_time_embeddings)\.timestep_embedder\.linear_2\.(.*)$": r"\1.out_layer.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn_norm\.(.*)$": r"\1.self_attention_norm.\2",
            r"^((?:video|audio)_text_transformer_blocks\.\d+)\.attn\.(.*)$": r"\1.self_attention.\2",
        }
    )
    lora_param_names_mapping: dict = field(
        default_factory=lambda: {r"^transformer\.(.*)$": r"\1"}
    )

    # Diffusers Kandinsky6Transformer3DModel config fields (mirror
    # transformer/config.json 1:1).
    in_visual_dim: int = 16
    out_visual_dim: int = 16
    in_text_dim: int = 3584  # Qwen2.5-7B hidden size (Reason1 text encoder)
    in_text_dim2: int = 768  # CLIP pooled dim
    time_dim: int = 1024
    patch_size: tuple[int, int, int] = (1, 2, 2)
    # checkpoint RoPE scaling is fixed across request resolutions
    scale_factor: tuple[float, float, float] = (1.0, 2.0, 2.0)
    model_dim: int = 4096
    ff_dim: int = 16384
    num_text_blocks: int = 4
    num_visual_blocks: int = 60
    axes_dims: tuple[int, int, int] = (32, 48, 48)  # 3D RoPE T/H/W split
    visual_cond: bool = True

    # False selects the video-only architecture; TI2VA pipelines use True
    is_multimodal: bool = True
    out_audio_dim: int | None = None
    in_audio_dim: int = 20
    model_dim_a: int | None = None
    time_dim_a: int | None = None
    ff_dim_a: int | None = None
    axes_dims_a: tuple[int, int, int] | None = None
    audio_freqs_scaling: float = 1.0

    # checkpoint metadata, not a runtime backend override; NABLA is rejected
    attention_engine: str = "auto"
    attention_causal: bool | None = None
    attention_local: bool | None = None
    attention_glob: bool | None = None
    attention_window: int | None = None
    attention_P: float | None = None
    attention_wT: int | None = None
    attention_wW: int | None = None
    attention_wH: int | None = None
    attention_add_sta: bool | None = None
    attention_method: str | None = None

    # checkpoint metadata; the pipeline passes unpadded text rather than masks
    text_token_padding: bool = False

    # Video<->audio fused block knobs (Kandinsky6FusedTransformerDecoderBlock).
    ca_rope: bool = False
    cross_gates: bool = False
    fix_modulation: bool = False
    # generated/reference token embeddings for tail-frame image conditioning
    visual_token_type_num_embeddings: int = 2

    # Derived, DiTArchConfig-contract fields (set in __post_init__ below;
    # not part of the diffusers config.json).
    in_channels: int = 0
    out_channels: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}.")
        if isinstance(self.scale_factor, list):
            self.scale_factor = tuple(self.scale_factor)
        if len(self.scale_factor) != 3:
            raise ValueError(
                f"scale_factor must have 3 values, got {self.scale_factor}."
            )
        self.scale_factor = tuple(float(value) for value in self.scale_factor)

        head_dim = sum(self.axes_dims)
        if self.model_dim % head_dim != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by head_dim "
                f"({head_dim})"
            )
        self.hidden_size = self.model_dim
        self.num_attention_heads = self.model_dim // head_dim
        self.in_channels = self.in_visual_dim
        self.out_channels = self.out_visual_dim
        self.num_channels_latents = self.in_visual_dim

        # resolve omitted audio dimensions from the video tower
        self.model_dim_a = self.model_dim_a or self.model_dim
        self.time_dim_a = self.time_dim_a or self.time_dim
        self.ff_dim_a = self.ff_dim_a or self.ff_dim
        self.axes_dims_a = self.axes_dims_a or self.axes_dims
        head_dim_a = sum(self.axes_dims_a)
        if self.model_dim_a % head_dim_a != 0:
            raise ValueError(
                f"model_dim_a ({self.model_dim_a}) must be divisible by "
                f"head_dim_a ({head_dim_a})"
            )

        # without fix_modulation, cross-modal modulation uses the other tower's time
        # embedding, so video and audio time widths must match
        if (
            self.is_multimodal
            and not self.fix_modulation
            and self.time_dim_a != self.time_dim
        ):
            raise ValueError(
                f"time_dim_a ({self.time_dim_a}) must equal time_dim "
                f"({self.time_dim}) unless fix_modulation=True "
                "(Kandinsky6FusedTransformerDecoderBlock's cross-modal "
                "modulation is driven by the other modality's time "
                "embedding by default)."
            )


@dataclass
class Kandinsky6VideoAudioConfig(DiTConfig):
    arch_config: DiTArchConfig = field(default_factory=Kandinsky6ArchConfig)
    prefix: str = "Kandinsky6"


__all__ = ["Kandinsky6ArchConfig", "Kandinsky6VideoAudioConfig"]
