import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import PIL
import torch
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.flux import FluxConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPTextConfig,
    T5Config,
    TextEncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.base import TextEncoderArchConfig
from sglang.multimodal_gen.configs.models.encoders.qwen_image import (
    _is_transformer_layer,
)
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig, FluxVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
    preprocess_text,
    shard_rotary_emb_for_sp,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    clip_postprocess_text,
    clip_preprocess_text,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import _pack_latents
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    return outputs.last_hidden_state


@dataclass
class FluxPipelineConfig(ImagePipelineConfig):
    """Configuration for the FLUX pipeline."""

    embedded_cfg_scale: float = 3.5

    task_type: ModelTaskType = ModelTaskType.T2I

    vae_tiling: bool = False

    vae_sp: bool = False

    dit_config: DiTConfig = field(default_factory=FluxConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=FluxVAEConfig)

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (CLIPTextConfig(), T5Config())
    )

    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_preprocess_text, preprocess_text),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (clip_postprocess_text, t5_postprocess_text)
    )

    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            dict(
                max_length=77,
                padding="max_length",
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
            ),
            None,
        ]
    )

    def prepare_sigmas(self, sigmas, num_inference_steps):
        return self._prepare_sigmas(sigmas, num_inference_steps)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        shape = (batch_size, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels // 4
        # pack latents
        return _pack_latents(latents, batch_size, num_channels_latents, height, width)

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[1]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[1]

    def _prepare_latent_image_ids(self, original_height, original_width, device):
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = int(original_height) // (vae_scale_factor * 2)
        width = int(original_width) // (vae_scale_factor * 2)
        latent_image_ids = torch.zeros(height, width, 3, device=device)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
        )

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)
        img_ids = self._prepare_latent_image_ids(
            original_height=height,
            original_width=width,
            device=device,
        )

        # NOTE(mick): prepare it here, to avoid unnecessary computations
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def post_denoising_loop(self, latents, batch):
        # unpack latents for flux
        (
            latents,
            batch_size,
            channels,
            height,
            width,
        ) = self._unpad_and_unpack_latents(latents, batch)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[1],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
            "pooled_projections": (
                batch.pooled_embeds[0] if batch.pooled_embeds else None
            ),
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.negative_prompt_embeds[1],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            ),
            "pooled_projections": (
                batch.neg_pooled_embeds[0] if batch.neg_pooled_embeds else None
            ),
        }


def _prepare_latent_ids(
    latents: torch.Tensor,  # (B, C, H, W)
):
    r"""
    Generates 4D position coordinates (T, H, W, L) for latent tensors.

    Args:
        latents (torch.Tensor):
            Latent tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor:
            Position IDs tensor of shape (B, H*W, 4) All batches share the same coordinate structure: T=0,
            H=[0..H-1], W=[0..W-1], L=0
    """

    batch_size, _, height, width = latents.shape

    t = torch.arange(1)  # [0] - time dimension
    h = torch.arange(height)
    w = torch.arange(width)
    layer = torch.arange(1)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    latent_ids = torch.cartesian_prod(t, h, w, layer)

    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
    return latent_ids


def _unpack_latents_with_ids(
    x: torch.Tensor, x_ids: torch.Tensor
) -> list[torch.Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    x_ids = x_ids.to(device=x.device)
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)


def _patchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.view(
        batch_size, num_channels_latents, height // 2, 2, width // 2, 2
    )
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(
        batch_size, num_channels_latents * 4, height // 2, width // 2
    )
    return latents


def _unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), 2, 2, height, width
    )
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(
        batch_size, num_channels_latents // (2 * 2), height * 2, width * 2
    )
    return latents


def _prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: Optional[torch.Tensor] = None,
):
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        layer = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, layer)
        out_ids.append(coords)

    return torch.stack(out_ids)


def _prepare_image_ids(
    image_latents: List[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
    scale: int = 10,
):
    if not isinstance(image_latents, list):
        raise ValueError(
            f"Expected `image_latents` to be a list, got {type(image_latents)}."
        )

    # create time offset for each reference image
    t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
    t_coords = [t.view(-1) for t in t_coords]

    image_latent_ids = []
    for x, t in zip(image_latents, t_coords):
        x = x.squeeze(0)
        _, height, width = x.shape

        x_ids = torch.cartesian_prod(
            t, torch.arange(height), torch.arange(width), torch.arange(1)
        )
        image_latent_ids.append(x_ids)

    image_latent_ids = torch.cat(image_latent_ids, dim=0)
    image_latent_ids = image_latent_ids.unsqueeze(0)

    return image_latent_ids


def flux2_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    hidden_states_layers: list[int] = [10, 20, 30]

    out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(
        batch_size, seq_len, num_channels * hidden_dim
    )

    return prompt_embeds


@dataclass
class Flux2MistralTextArchConfig(TextEncoderArchConfig):
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
    )
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [_is_transformer_layer]
    )

    def __post_init__(self):
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": 512,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class Flux2MistralTextConfig(TextEncoderConfig):
    arch_config: TextEncoderArchConfig = field(
        default_factory=Flux2MistralTextArchConfig
    )


def format_text_input(prompts: List[str], system_message: str = None):
    # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
    # when truncation is enabled. The processor counts [IMG] tokens and fails
    # if the count changes after truncation.
    cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

    return [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        for prompt in cleaned_txt
    ]


def flux_2_preprocess_text(prompt: str):
    system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."
    return format_text_input([prompt], system_message=system_message)


# Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._pack_latents
def flux2_pack_latents(latents):
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)

    return latents


@dataclass
class Flux2PipelineConfig(FluxPipelineConfig):
    embedded_cfg_scale: float = 4.0

    task_type: ModelTaskType = ModelTaskType.I2I

    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Flux2MistralTextConfig(),)
    )
    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (flux_2_preprocess_text,),
    )

    postprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (flux2_postprocess_text,)
    )
    vae_config: VAEConfig = field(default_factory=Flux2VAEConfig)

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        # flatten to 1-d list
        prompts = [p for prompt in prompts for p in prompt]
        inputs = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            # 2048 from official github repo, 512 from diffusers
            max_length=512,
        )

        return inputs

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        height = 2 * (
            batch.height // (self.vae_config.arch_config.vae_scale_factor * 2)
        )
        width = 2 * (batch.width // (self.vae_config.arch_config.vae_scale_factor * 2))
        num_channels_latents = self.dit_config.arch_config.in_channels
        shape = (batch_size, num_channels_latents, height // 2, width // 2)
        return shape

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def calculate_condition_image_size(
        self, image, width, height
    ) -> Optional[tuple[int, int]]:
        target_area: int = 1024 * 1024
        if width is not None and height is not None:
            if width * height > target_area:
                scale = math.sqrt(target_area / (width * height))
                width = int(width * scale)
                height = int(height * scale)
                return width, height

        return None

    def preprocess_condition_image(
        self, image, target_width, target_height, vae_image_processor: VaeImageProcessor
    ):
        img = image.resize((target_width, target_height), PIL.Image.Resampling.LANCZOS)
        image_width, image_height = img.size
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        multiple_of = vae_scale_factor * 2
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of
        img = vae_image_processor.preprocess(
            img, height=image_height, width=image_width, resize_mode="crop"
        )
        return img, (image_width, image_height)

    def postprocess_image_latent(self, latent_condition, batch):
        batch_size = batch.batch_size
        # get image_latent_ids right after scale & shift
        image_latent_ids = _prepare_image_ids([latent_condition])
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(get_local_torch_device())
        batch.condition_image_latent_ids = image_latent_ids

        # latent: (1, 128, 32, 32)
        packed = self.maybe_pack_latents(
            latent_condition, None, batch
        )  # (1, 1024, 128)
        packed = packed.squeeze(0)  # (1024, 128) - remove batch dim

        # Concatenate all reference tokens along sequence dimension
        image_latents = packed.unsqueeze(0)  # (1, N*1024, 128)
        image_latents = image_latents.repeat(batch_size, 1, 1)
        return image_latents

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        txt_ids = _prepare_text_ids(prompt_embeds).to(device=device)

        img_ids = batch.latent_ids
        if batch.image_latent is not None:
            image_latent_ids = batch.condition_image_latent_ids
            img_ids = torch.cat([img_ids, image_latent_ids], dim=1).to(device=device)

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # NOTE(mick): prepare it here, to avoid unnecessary computations
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {
            "freqs_cis": self.get_freqs_cis(
                batch.prompt_embeds[0],
                batch.width,
                batch.height,
                device,
                rotary_emb,
                batch,
            )
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {}

    def maybe_pack_latents(self, latents, batch_size, batch):
        return flux2_pack_latents(latents)

    def maybe_prepare_latent_ids(self, latents):
        return _prepare_latent_ids(latents)

    def postprocess_vae_encode(self, image_latents, vae):
        # patchify
        image_latents = _patchify_latents(image_latents)
        return image_latents

    def _check_vae_has_bn(self, vae):
        """Check if VAE has bn attribute (cached check to avoid repeated hasattr calls)."""
        if not hasattr(self, "_vae_has_bn_cache"):
            self._vae_has_bn_cache = hasattr(vae, "bn") and vae.bn is not None
        return self._vae_has_bn_cache

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        """Preprocess latents before decoding.

        Dynamically adapts based on VAE type:
        - Standard Flux2 VAE (has bn): needs unpatchify (128 channels -> 32 channels)
        - Distilled VAE (no bn): keeps patchified latents (128 channels)
        """
        if vae is not None and self._check_vae_has_bn(vae):
            return _unpatchify_latents(latents)
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """Get scale and shift for decoding.

        Dynamically adapts based on VAE type:
        - Standard Flux2 VAE (has bn): uses BatchNorm statistics
        - Distilled VAE (no bn): uses scaling_factor from config
        """
        vae_arch_config = self.vae_config.arch_config

        if self._check_vae_has_bn(vae):
            # Standard Flux2 VAE: use BatchNorm statistics
            latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
            latents_bn_std = torch.sqrt(
                vae.bn.running_var.view(1, -1, 1, 1) + vae_arch_config.batch_norm_eps
            ).to(device, dtype)
            return 1 / latents_bn_std, latents_bn_mean

        # Distilled VAE or unknown: use scaling_factor
        scaling_factor = (
            getattr(vae.config, "scaling_factor", None)
            if hasattr(vae, "config")
            else getattr(vae, "scaling_factor", None)
        ) or getattr(vae_arch_config, "scaling_factor", 0.13025)

        scale = torch.tensor(scaling_factor, device=device, dtype=dtype).view(
            1, 1, 1, 1
        )
        return 1 / scale, None

    def post_denoising_loop(self, latents, batch):
        latent_ids = batch.latent_ids
        latents = _unpack_latents_with_ids(latents, latent_ids)

        return latents

    def slice_noise_pred(self, noise, latents):
        # remove noise over input image
        noise = noise[:, : latents.size(1) :]
        return noise
