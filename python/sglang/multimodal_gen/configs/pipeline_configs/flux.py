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

    enable_autocast: bool = False

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

    task_type: ModelTaskType = ModelTaskType.TI2I

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


def _prepare_kontext_latent_ids(
    latents: torch.Tensor,  # (B, C, H, W)
    t_coord: int = 0,
):
    """
    Generates position coordinates for Flux Kontext latent tensors.

    Args:
        latents: Latent tensor of shape (B, C, H, W) or (B, C, T, H, W)
        t_coord: Time coordinate (0 for noise latents, 1 for image latents)

    Returns:
        torch.Tensor: Position IDs tensor of shape (H//2 * W//2, 3)
    """
    # Handle both 4D and 5D tensors
    if latents.ndim == 5:
        latents = latents.squeeze(2)

    _, _, height, width = latents.shape
    # Flux uses packed latents, so height and width are halved
    height = height // 2
    width = width // 2

    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 0] = t_coord  # time coordinate
    latent_image_ids[..., 1] = torch.arange(height)[:, None]
    latent_image_ids[..., 2] = torch.arange(width)[None, :]

    latent_image_ids = latent_image_ids.reshape(height * width, 3)
    return latent_image_ids


@dataclass
class FluxKontextPipelineConfig(FluxPipelineConfig):
    """Configuration for the FLUX Kontext pipeline (Image-to-Image editing)."""

    task_type: ModelTaskType = ModelTaskType.I2I

    # Preferred resolutions for Flux Kontext (from official HuggingFace implementation)
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]

    def calculate_condition_image_size(
        self, image, width, height
    ) -> Optional[tuple[int, int]]:
        """Calculate target size for condition image.

        For Kontext, the input image will be resized to match the output dimensions.
        This method returns the image's original size - the actual target size
        will be determined based on batch.width/batch.height in the pipeline.
        """
        # Return original size; actual resizing happens in preprocess_condition_image
        # using the output dimensions from batch
        if width is not None and height is not None:
            return width, height
        return None

    def prepare_calculated_size(self, image):
        """Prepare calculated size based on input image aspect ratio.

        Following official Flux Kontext implementation:
        Find the best matching resolution from PREFERRED_KONTEXT_RESOLUTIONS
        based on input image aspect ratio, then align to multiple of 16.
        """
        if image is None:
            return None

        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        multiple_of = vae_scale_factor * 2  # 16 for Flux

        image_width, image_height = image.width, image.height
        aspect_ratio = image_width / image_height

        # Find the closest matching resolution (official implementation)
        _, best_width, best_height = min(
            (abs(aspect_ratio - w / h), w, h)
            for w, h in self.PREFERRED_KONTEXT_RESOLUTIONS
        )

        # Align to multiple of 16
        best_width = best_width // multiple_of * multiple_of
        best_height = best_height // multiple_of * multiple_of

        return best_width, best_height

    def preprocess_condition_image(
        self, image, target_width, target_height, vae_image_processor
    ):
        """Preprocess condition image for VAE encoding.

        Following official Flux Kontext implementation:
        1. Get input image aspect ratio (using get_default_height_width for consistency)
        2. Find the closest matching resolution from PREFERRED_KONTEXT_RESOLUTIONS
        3. Align to multiple of 16
        4. Resize and preprocess

        Note: target_width and target_height are ignored for Kontext pipeline,
        as it uses its own resolution selection based on PREFERRED_KONTEXT_RESOLUTIONS.
        """
        if vae_image_processor is None:
            raise ValueError("vae_image_processor cannot be None for Kontext pipeline")

        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        multiple_of = vae_scale_factor * 2  # 16 for Flux

        # Get input image dimensions using get_default_height_width for consistency with official impl
        # This aligns dimensions to vae_scale_factor multiples before calculating aspect ratio
        image_height, image_width = vae_image_processor.get_default_height_width(image)
        aspect_ratio = image_width / image_height

        # Find the closest matching resolution from PREFERRED_KONTEXT_RESOLUTIONS
        # Official: (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h)
            for w, h in self.PREFERRED_KONTEXT_RESOLUTIONS
        )

        # Align to multiple of 16
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of

        # Resize image using vae_image_processor (official uses self.image_processor.resize)
        # Note: resize takes (height, width) order
        img = vae_image_processor.resize(image, height=image_height, width=image_width)

        # Preprocess image (official uses self.image_processor.preprocess)
        # Note: preprocess takes (height, width) order, same as resize
        img = vae_image_processor.preprocess(
            img, height=image_height, width=image_width
        )

        return img, (image_width, image_height)

    def postprocess_image_latent(self, latent_condition, batch):
        """Process encoded image latent for Kontext pipeline.

        For Kontext, we pack the image latents the same way as noise latents
        and prepare position IDs with t=1 coordinate.

        Note: Input image and output image can have different sizes.
        They are concatenated along the sequence dimension (L).
        """
        batch_size = batch.batch_size

        # Handle both 4D (B, C, H, W) and 5D (B, C, T, H, W) tensors
        if latent_condition.ndim == 5:
            # Squeeze the time dimension for image task (T=1)
            latent_condition = latent_condition.squeeze(2)

        _, num_channels, latent_height, latent_width = latent_condition.shape

        # Prepare image latent position IDs (with t=1 coordinate)
        image_latent_ids = _prepare_kontext_latent_ids(latent_condition, t_coord=1)
        image_latent_ids = image_latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        image_latent_ids = image_latent_ids.to(get_local_torch_device())
        batch.condition_image_latent_ids = image_latent_ids

        # Pack latents using the actual input image latent dimensions, NOT batch.height/width
        # Input: (B, C, H, W) -> Output: (B, H*W/4, C*4)
        # Use _pack_latents directly with the correct dimensions from latent_condition
        packed = _pack_latents(
            latent_condition,
            batch_size,
            num_channels,  # Use actual channels from latent
            latent_height,  # Use actual height from latent (NOT batch.height)
            latent_width,  # Use actual width from latent (NOT batch.width)
        )

        # Expand for batch size if needed
        if batch_size > packed.shape[0]:
            if batch_size % packed.shape[0] == 0:
                additional_per_prompt = batch_size // packed.shape[0]
                packed = packed.repeat(additional_per_prompt, 1, 1)
            else:
                raise ValueError(
                    f"Cannot duplicate image latents of batch size {packed.shape[0]} "
                    f"to {batch_size} prompts."
                )

        return packed

    def _prepare_latent_image_ids(self, original_height, original_width, device):
        """Prepare latent image IDs with t=0 coordinate for noise latents."""
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = int(original_height) // (vae_scale_factor * 2)
        width = int(original_width) // (vae_scale_factor * 2)
        latent_image_ids = torch.zeros(height, width, 3, device=device)
        latent_image_ids[..., 0] = 0  # t=0 for noise latents
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]
        )

        latent_image_ids = latent_image_ids.reshape(height * width, 3)
        return latent_image_ids

    def get_freqs_cis(self, prompt_embeds, width, height, device, rotary_emb, batch):
        """Get frequency embeddings for RoPE, including image latent positions."""
        txt_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device)

        # Noise latent position IDs (t=0)
        img_ids = self._prepare_latent_image_ids(
            original_height=height,
            original_width=width,
            device=device,
        )

        # If we have image latents, concatenate their position IDs
        if batch.image_latent is not None and hasattr(
            batch, "condition_image_latent_ids"
        ):
            image_latent_ids = batch.condition_image_latent_ids
            if image_latent_ids.ndim == 3:
                image_latent_ids = image_latent_ids[0]  # Take first batch
            img_ids = torch.cat([img_ids, image_latent_ids.to(device)], dim=0)

        # Compute RoPE embeddings
        img_cos, img_sin = rotary_emb.forward(img_ids)
        img_cos = shard_rotary_emb_for_sp(img_cos)
        img_sin = shard_rotary_emb_for_sp(img_sin)

        txt_cos, txt_sin = rotary_emb.forward(txt_ids)

        cos = torch.cat([txt_cos, img_cos], dim=0).to(device=device)
        sin = torch.cat([txt_sin, img_sin], dim=0).to(device=device)
        return cos, sin

    def slice_noise_pred(self, noise, latents):
        """Remove noise prediction over input image tokens.

        In Kontext, the transformer outputs predictions for both noise and image tokens.
        We only want the predictions for the noise tokens.
        """
        noise = noise[:, : latents.size(1)]
        return noise
