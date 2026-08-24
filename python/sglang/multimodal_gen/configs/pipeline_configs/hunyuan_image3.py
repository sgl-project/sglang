from dataclasses import dataclass, field

import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.hunyuan_image3 import (
    HunyuanImage3DitConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.t5 import T5ArchConfig, T5Config
from sglang.multimodal_gen.configs.models.vaes.hunyuan_image3 import (
    HunyuanImage3VAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    SpatialImagePipelineConfig,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import get_global_server_args


@dataclass
class HunyuanImage3PipelineConfig(SpatialImagePipelineConfig):
    """Configuration for the HunyuanImage-3 pipeline."""

    vae_precision: str = "fp32"  # VAE uses float32 per HF config

    should_use_guidance: bool = True
    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False
    vae_sp: bool = False

    # DiT config
    dit_config: DiTConfig = field(default_factory=HunyuanImage3DitConfig)

    # VAE config
    vae_config: VAEConfig = field(default_factory=HunyuanImage3VAEConfig)

    # Text encoder configs - HunyuanImage-3 uses built-in tokenizer
    # For now, use T5 as placeholder (actual implementation uses custom tokenizer)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(T5ArchConfig(num_heads=6)),)
    )

    enable_autocast: bool = False

    def __post_init__(self):
        self.vae_scale_factor = self.vae_config.get_vae_scale_factor()
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def supports_dynamic_batching(self):
        server_args = get_global_server_args()
        return server_args.srt_encoder_url is not None

    def supports_native_grouped_requests(self):
        return True

    def supports_sequential_dit_inference(self):
        return True

    def supports_sequential_multi_output_inference(self):
        return current_platform.is_npu()

    def get_freqs_cis(self, batch, device, rotary_emb, dtype):
        """Get 2D RoPE frequencies for image generation."""
        height = batch.height // self.vae_scale_factor
        width = batch.width // self.vae_scale_factor
        hidden_states = torch.empty(1, 1, height, width, device=device, dtype=dtype)
        cos, sin = rotary_emb(hidden_states)
        cos = shard_rotary_emb_for_sp(cos)
        sin = shard_rotary_emb_for_sp(sin)
        return cos, sin

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare positive conditioning kwargs for denoising."""
        kwargs = {
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }
        return kwargs

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Prepare negative conditioning kwargs for denoising."""
        kwargs = {
            "freqs_cis": self.get_freqs_cis(batch, device, rotary_emb, dtype),
        }
        return kwargs

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """Get scale and shift for latent decoding."""
        scaling_factor = self.vae_config.arch_config.scaling_factor
        shift_factor = getattr(self.vae_config.arch_config, "shift_factor", None)
        shift = shift_factor if shift_factor else 0.0
        return scaling_factor, shift

    def post_denoising_loop(self, latents, batch):
        """Post-process latents after denoising."""
        return latents.bfloat16()

    def post_decoding(self, frames, server_args):
        """Post-process decoded frames."""
        return self.image_processor.postprocess(frames, output_type="latent")
