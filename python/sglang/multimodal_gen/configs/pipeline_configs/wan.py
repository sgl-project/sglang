# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import WanS2VConfig, WanVideoConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPVisionConfig,
    T5Config,
    WanS2VAudioEncoderConfig,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

try:
    import ftfy
except ImportError:  # pragma: no cover
    ftfy = None


def _wan_basic_clean(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _wan_whitespace_clean(text: str) -> str:
    text = _wan_basic_clean(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _make_wan_s2v_text_encoder_config() -> T5Config:
    config = T5Config()
    arch = config.arch_config
    arch.vocab_size = 256384
    arch.architectures = ["UMT5EncoderModel"]
    arch.d_model = 4096
    arch.hidden_size = 4096
    arch.d_kv = 64
    arch.d_ff = 10240
    arch.num_layers = 24
    arch.num_attention_heads = 64
    arch.num_heads = 64
    arch.relative_attention_num_buckets = 32
    arch.feed_forward_proj = "gated-gelu"
    arch.text_len = 512
    arch.__post_init__()
    return config


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class WanI2VCommonConfig(PipelineConfig):
    # for all wan i2v pipelines
    def adjust_num_frames(self, num_frames):
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        if num_frames % vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            num_frames = (
                num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
            )
            return num_frames
        return num_frames


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    # WanConfig-specific added parameters

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True


@dataclass
class TurboWanT2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [988, 932, 852, 608]
    )


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig, WanI2VCommonConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    max_area: int = 480 * 832
    # WanConfig-specific parameters with defaults
    task_type: ModelTaskType = ModelTaskType.I2V
    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        # Official Wan S2V VAE encode/decode uses temporal cache with 1,4,4... chunks.
        # Keeping this on avoids the huge full-video conv path for motion latents.
        self.vae_config.use_feature_cache = True
        self.vae_config.use_parallel_encode = False
        self.vae_config.use_parallel_decode = False
        self.vae_config.use_feature_cache = False
        self.vae_config.use_parallel_encode = False
        self.vae_config.use_parallel_decode = False


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    max_area: int = 720 * 1280
    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class TurboWanI2V720Config(WanI2V720PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [996, 932, 852, 608]
    )
    boundary_ratio: float | None = 0.9

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class Wan2_2_S2V_14B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    dit_config: DiTConfig = field(default_factory=WanS2VConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (_make_wan_s2v_text_encoder_config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16",)
    )
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (_wan_whitespace_clean,)
    )
    audio_encoder_config: EncoderConfig = field(
        default_factory=WanS2VAudioEncoderConfig
    )
    audio_encoder_precision: str = "fp32"
    vae_precision: str = "bf16"
    flow_shift: float | None = 3.0
    task_type: ModelTaskType = ModelTaskType.S2V
    vae_stride = (4, 8, 8)
    max_area: int = 704 * 1024

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        z_dim = self.vae_config.arch_config.z_dim
        oh = batch.height
        ow = batch.width
        return (
            batch_size,
            z_dim,
            num_frames,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
        )

    def slice_noise_pred(self, noise, latents):
        if isinstance(noise, (list, tuple)):
            if len(noise) != 1:
                raise ValueError(
                    f"Wan S2V expected a single noise tensor, got {len(noise)} outputs"
                )
            noise = noise[0]
        return noise

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        del rotary_emb
        extra = batch.extra.get("wan_s2v", {})
        return {
            "ref_latents": extra["ref_latents"].to(device=device, dtype=dtype),
            "motion_latents": extra["motion_latents"].to(device=device, dtype=dtype),
            "cond_states": extra["cond_states"].to(device=device, dtype=dtype),
            "audio_input": extra["audio_input"].to(device=device, dtype=dtype),
            "motion_frames": extra["motion_frames"],
            "drop_motion_frames": extra["drop_motion_frames"],
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        pos_kwargs = self.prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        pos_kwargs["audio_input"] = torch.zeros_like(pos_kwargs["audio_input"])
        return pos_kwargs

    def skip_decode_scale_and_shift(self, vae) -> bool:
        return hasattr(vae, "decode_video")

    def prepare_decoding_latents(self, batch, server_args=None, vae=None):
        del server_args, vae
        extra = batch.extra["wan_s2v"]
        prefix_latents = (
            extra["ref_latents"]
            if extra["drop_motion_frames"]
            else extra["motion_latents"]
        )
        prefix_latents = prefix_latents.to(
            device=batch.latents.device,
            dtype=batch.latents.dtype,
        )
        return torch.cat([prefix_latents, batch.latents], dim=2)

    def postprocess_decoded_batch(self, frames, batch, server_args):
        del server_args
        extra = batch.extra["wan_s2v"]
        frames = frames[:, :, -extra["infer_frames"] :]
        if extra["drop_motion_frames"]:
            frames = frames[:, :, 3:]
        return frames

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.vae_config.use_feature_cache = True
        self.vae_config.use_parallel_encode = False
        self.vae_config.use_parallel_decode = False


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
