# SPDX-License-Identifier: Apache-2.0
"""Pipeline config for minWM main-branch Wan2.2-5B causal DMD."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import MinWMVideoConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config
from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_TI2V_5B_Config

MINWM_ACTION_LABELS_CONDITION = "minwm_action_labels"
MINWM_ACTION_WEIGHTS_CONDITION = "minwm_action_weights"
MINWM_PROMPT_UPDATED_CONDITION = "minwm_prompt_updated"
MINWM_TOTAL_CHUNKS_CONDITION = "minwm_total_chunks"


def _minwm_native_component_names() -> tuple[str, ...]:
    value = os.environ.get("MINWM_NATIVE_COMPONENTS", "text_encoder,vae")
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = set(names) - {"text_encoder", "vae"}
    if unknown:
        raise ValueError(f"unknown MINWM_NATIVE_COMPONENTS entries: {sorted(unknown)}")
    return names


def minwm_t5_postprocess_text(outputs, _text_inputs) -> TextConditioningOutput:
    attention_mask = getattr(outputs, "attention_mask", None)
    if attention_mask is None:
        if _text_inputs is None or "attention_mask" not in _text_inputs:
            raise ValueError("MinWM text encoding requires an attention mask")
        attention_mask = _text_inputs["attention_mask"]
    token_mask = attention_mask.to(dtype=torch.bool)
    hidden_state = outputs.last_hidden_state.masked_fill(~token_mask.unsqueeze(-1), 0.0)
    # Current minWM main keeps at least 512 positions in the packed text
    # context. Positions after the true token length are explicit zero vectors,
    # but they still participate as K/V entries in cross attention. This is a
    # model contract, not ordinary attention-mask trimming.
    seq_lens = token_mask.sum(dim=1).long().clamp_min(512)
    positions = torch.arange(hidden_state.shape[1], device=hidden_state.device)
    context_mask = positions.unsqueeze(0) < seq_lens.unsqueeze(1)
    if torch.isnan(hidden_state).any():
        raise ValueError("MinWM text encoder produced NaN embeddings")
    return TextConditioningOutput(
        prompt_embeds=hidden_state,
        prompt_embeds_mask=context_mask,
        prompt_seq_lens=[int(length) for length in seq_lens.tolist()],
    )


def _minwm_t5_config() -> T5Config:
    return T5Config(arch_config=T5ArchConfig(text_len=1024))


@dataclass
class MinWMWan22VAEArchConfig(WanVAEArchConfig):
    base_dim: int = 160
    decoder_base_dim: int | None = 256
    z_dim: int = 48
    in_channels: int = 12
    out_channels: int = 12
    patch_size: int | None = 2
    is_residual: bool = True
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    latents_mean: tuple[float, ...] = (
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.157,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.123,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.052,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    )
    latents_std: tuple[float, ...] = (
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.499,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.06,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    )


@dataclass
class MinWMWan22VAEConfig(WanVAEConfig):
    arch_config: MinWMWan22VAEArchConfig = field(
        default_factory=MinWMWan22VAEArchConfig
    )


@dataclass
class MinWMCausalDMDConfig(Wan2_2_TI2V_5B_Config):
    """Exact structural/runtime defaults of the requested 5B DMD student."""

    dit_config: DiTConfig = field(default_factory=MinWMVideoConfig)
    vae_config: VAEConfig = field(default_factory=MinWMWan22VAEConfig)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (_minwm_t5_config(),)
    )
    postprocess_text_funcs: tuple = field(
        default_factory=lambda: (minwm_t5_postprocess_text,)
    )
    # minWM main uses diffusers.AutoencoderKLWan and the HF UMT5 encoder
    # directly. Their tiny BF16 kernel differences are amplified by causal
    # rollout, so parity takes precedence over SGLang's optimized variants.
    native_component_names: tuple[str, ...] = field(
        default_factory=_minwm_native_component_names
    )
    # minWM main executes BF16 modules directly.  An additional autocast scope
    # changes Wan VAE and DiT kernel promotion/rounding, then causal KV reuse
    # amplifies the first-step drift across chunks.
    enable_autocast: bool = False
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True
    context_noise: int = 0
    vae_precision: str = "bf16"
    preprocess_vae_encode_before_dtype_cast: bool = True
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    realtime_causal_sink_size: int | None = None
    realtime_causal_kv_cache_num_frames: int | None = None

    @staticmethod
    def _native_vae_stats(latents: torch.Tensor, vae):
        mean = torch.tensor(
            vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        std = torch.tensor(
            vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, -1, 1, 1, 1)
        return mean, std

    def preprocess_vae_encode(self, image, _vae):
        # The minWM processor preserves uint8 pixels until the GPU, casts them
        # to BF16, then performs div(127.5)-1 in BF16. The generic image stage
        # normalizes in FP32 on CPU first; reconstruct the lossless uint8 grid
        # so rounding and VAE inputs match main exactly.
        pixels = ((image + 1.0) * 127.5).round_().clamp_(0, 255)
        normalized = pixels.to(torch.bfloat16).div_(127.5).sub_(1.0)
        dump_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
        if dump_root:
            dump_dir = Path(dump_root) / "sglang"
            dump_dir.mkdir(parents=True, exist_ok=True)
            torch.save(normalized.detach().cpu(), dump_dir / "vae_input.pt")
        return normalized

    def normalize_vae_encode(self, image_latents, vae):
        mean, std = self._native_vae_stats(image_latents, vae)
        normalized = (image_latents - mean) / std
        # WanVAEWrapper returns FP32, then WanPackedProcessor serializes the
        # reference latent as FP16 before V3 moves it back to BF16.  This wire
        # boundary is numerically visible and must be reproduced in-process.
        return normalized.float().to(torch.float16).to(torch.bfloat16)

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # MinWM decoding multiplies by std directly. Returning identity here
        # avoids the generic algebraically-equivalent `latent / (1 / std)`,
        # whose BF16 rounding differs.
        del device, dtype, vae
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        del server_args
        if vae is None:
            raise ValueError("MinWM decoding requires the native VAE")
        mean, std = self._native_vae_stats(latents, vae)
        return latents * std + mean

    def preprocess_realtime_condition_image(self, batch, _vae_image_processor) -> bool:
        if batch.condition_image is None:
            return False
        width = int(batch.width or 832)
        height = int(batch.height or 480)
        batch.condition_image = batch.condition_image.resize((width, height))
        batch.width = width
        batch.height = height
        return True

    def postprocess_image_latent(self, latent_condition, _batch):
        # PipelineConfig's generic I2V hook prepends a four-channel temporal
        # mask. MinWM V3 commits the clean 48-channel VAE latent directly.
        expected_channels = self.vae_config.arch_config.z_dim
        if latent_condition.shape[1] != expected_channels:
            raise ValueError(
                "MinWM reference latent must have "
                f"{expected_channels} channels, got {latent_condition.shape[1]}"
            )
        return latent_condition

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
