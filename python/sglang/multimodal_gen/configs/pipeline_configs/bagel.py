# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for BAGEL generation, planning, and image editing."""

import math
from dataclasses import dataclass, field

import torch
from PIL import Image

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.bagel import BagelDiTConfig
from sglang.multimodal_gen.configs.models.encoders.bagel_vit import (
    BagelImageEncoderConfig,
)
from sglang.multimodal_gen.configs.models.vaes.bagel import BagelVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

_BAGEL_CONTEXT_KEY = "bagel_context"


def calculate_bagel_resize_dimensions(
    width: int,
    height: int,
    *,
    max_size: int,
    min_size: int,
    stride: int,
    max_pixels: int = 14 * 14 * 9 * 1024,
) -> tuple[int, int]:
    """Calculate the official BAGEL aspect-preserving image dimensions.

    Args:
        width: Source image width in pixels.
        height: Source image height in pixels.
        max_size: Maximum long-edge size.
        min_size: Minimum short-edge size.
        stride: Required divisibility for both dimensions.
        max_pixels: Upper bound used by BAGEL's ``ImageTransform``.

    Returns:
        ``(width, height)`` rounded to the requested stride.

    Raises:
        ValueError: If an input or transform constraint is not positive.
    """
    values = (width, height, max_size, min_size, stride, max_pixels)
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError(
            "BAGEL image dimensions and resize constraints must be positive"
        )

    def apply_scale(
        current_width: int, current_height: int, scale: float
    ) -> tuple[int, int]:
        scaled_width = round(current_width * scale)
        scaled_height = round(current_height * scale)
        scaled_width = max(stride, int(round(scaled_width / stride) * stride))
        scaled_height = max(stride, int(round(scaled_height / stride) * stride))
        return scaled_width, scaled_height

    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    new_width, new_height = apply_scale(width, height, scale)

    # Match the pinned official transform, including its direct area ratio.
    if new_width * new_height > max_pixels:
        scale = max_pixels / (new_width * new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale)
    if max(new_width, new_height) > max_size:
        scale = max_size / max(new_width, new_height)
        new_width, new_height = apply_scale(new_width, new_height, scale)

    if not math.isfinite(float(new_width * new_height)):
        raise ValueError("BAGEL calculated a non-finite image size")
    return new_width, new_height


@dataclass
class BagelPipelineConfig(ImagePipelineConfig):
    """Configure request-local BAGEL text-to-image generation."""

    task_type: ModelTaskType = ModelTaskType.T2I
    should_use_guidance: bool = False
    flow_shift: float | None = 3.0

    dit_config: DiTConfig = field(default_factory=BagelDiTConfig)
    vae_config: VAEConfig = field(default_factory=BagelVAEConfig)
    dit_precision: str = "bf16"
    vae_precision: str = "bf16"
    # Official BAGEL creates the initial latent with the CPU RNG before moving
    # it to CUDA. Keep that default for fixed-seed reference parity.
    generator_device: str = "cpu"

    # BAGEL's VAE does not implement the generic tiled/parallel decode lifecycle.
    vae_tiling: bool = False
    vae_sp: bool = False

    cfg_interval: tuple[float, float] = (0.4, 1.0)
    cfg_renorm_type: str = "global"
    cfg_renorm_min: float = 0.0

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        """Disable automatic external CFG because BAGEL performs CFG internally."""
        return ModelDeploymentConfig(
            auto_enable_cfg_parallel=False,
            keep_resident_components=("dit", "vae"),
            implicit_auxiliary_layerwise_offload_components=(),
        )

    def supports_dynamic_batching(self) -> bool:
        """Return false until BAGEL gains a batched request-owned KV implementation."""
        return False

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Pass request-owned KV state and internal-CFG controls to the DiT.

        Args:
            batch: Current request carrying ``bagel_context`` in ``extra``.
            device: Denoising device (unused; context already resides there).
            rotary_emb: Transformer rotary embedding (unused).
            dtype: Denoising dtype (unused).

        Returns:
            Keyword arguments for ``BagelTransformer.forward``.

        Raises:
            RuntimeError: If the request-local BAGEL context is missing.
        """
        del device, rotary_emb, dtype
        context = batch.extra.get(_BAGEL_CONTEXT_KEY)
        if context is None:
            raise RuntimeError(
                "BAGEL request context is missing; the prefill stage must run before denoising"
            )
        return {
            "bagel_context": context,
            "guidance_scale": float(batch.guidance_scale),
            "image_guidance_scale": 1.0,
            "cfg_interval": self.cfg_interval,
            "cfg_renorm_min": self.cfg_renorm_min,
            "cfg_renorm_type": self.cfg_renorm_type,
        }

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Return no external negative branch; BAGEL computes CFG in one forward."""
        del batch, device, rotary_emb, dtype
        return {}

    def post_denoising_loop(self, latents: torch.Tensor, batch) -> torch.Tensor:
        """Unpatchify BAGEL tokens into NCHW latents and release request KV state.

        Args:
            latents: Denoised patch tokens with shape ``[S, P*P*C]`` or
                ``[1, S, P*P*C]``.
            batch: Request containing the requested output dimensions.

        Returns:
            Unpatchified VAE latents with shape ``[1, C, H/8, W/8]``.

        Raises:
            ValueError: If token count or patch width does not match the request.
        """
        try:
            if latents.ndim == 3:
                if latents.shape[0] != 1:
                    raise ValueError(
                        "BAGEL supports exactly one latent sample per request"
                    )
                latents = latents[0]
            if latents.ndim != 2:
                raise ValueError(
                    "BAGEL latents must have shape [tokens, patch_dim] or "
                    "[1, tokens, patch_dim]"
                )

            arch = self.dit_config.arch_config
            patch_size = int(arch.latent_patch_size)
            channels = int(arch.latent_channel)
            latent_downsample = int(arch.latent_downsample)
            token_height = int(batch.height) // latent_downsample
            token_width = int(batch.width) // latent_downsample
            expected_tokens = token_height * token_width
            expected_width = patch_size * patch_size * channels
            if tuple(latents.shape) != (expected_tokens, expected_width):
                raise ValueError(
                    "BAGEL latent shape does not match the request: expected "
                    f"({expected_tokens}, {expected_width}), got {tuple(latents.shape)}"
                )

            patches = latents.reshape(
                1,
                token_height,
                token_width,
                patch_size,
                patch_size,
                channels,
            )
            return torch.einsum("nhwpqc->nchpwq", patches).reshape(
                1,
                channels,
                token_height * patch_size,
                token_width * patch_size,
            )
        finally:
            # The context owns large KV tensors. Drop the request's reference before
            # VAE decode so those tensors can be reclaimed as soon as denoising ends.
            batch.extra.pop(_BAGEL_CONTEXT_KEY, None)


def _editing_vae_config() -> BagelVAEConfig:
    """Create the full VAE required by image editing without changing T2I."""
    return BagelVAEConfig(load_encoder=True, load_decoder=True)


def _thinking_dit_config() -> BagelDiTConfig:
    """Create the transformer variant that strictly loads the language head."""
    return BagelDiTConfig(load_lm_head=True)


@dataclass
class BagelThinkingPipelineConfig(BagelPipelineConfig):
    """Configure text planning followed by official three-way T2I CFG."""

    dit_config: DiTConfig = field(default_factory=_thinking_dit_config)
    thinking_image_guidance_scale: float = 1.5

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Pass the three request-owned Thinking branches to the denoiser."""
        kwargs = super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        context = kwargs["bagel_context"]
        if not context.is_thinking:
            raise RuntimeError("BAGEL Thinking requires a three-way request context")
        kwargs["image_guidance_scale"] = self.thinking_image_guidance_scale
        return kwargs


@dataclass
class BagelEditPipelineConfig(BagelPipelineConfig):
    """Configure the explicit BAGEL image-editing pipeline."""

    task_type: ModelTaskType = ModelTaskType.I2I
    vae_config: VAEConfig = field(default_factory=_editing_vae_config)
    image_encoder_config: BagelImageEncoderConfig = field(
        default_factory=BagelImageEncoderConfig
    )
    image_encoder_precision: str = "bf16"

    editing_cfg_interval: tuple[float, float] = (0.0, 1.0)
    editing_cfg_renorm_type: str = "text_channel"
    editing_image_guidance_scale: float = 2.0

    @staticmethod
    def condition_image_convert_method(image: Image.Image) -> Image.Image:
        """Match BAGEL's white-background conversion for transparent inputs."""
        has_transparency = image.info.get("transparency") is not None
        if image.mode == "RGBA" or has_transparency:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        """Keep Editing components resident when the device budget permits."""
        return ModelDeploymentConfig(
            auto_enable_cfg_parallel=False,
            keep_resident_components=("dit", "vae", "image_encoder"),
            implicit_auxiliary_layerwise_offload_components=(),
        )

    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        """Pass three-way Editing CFG state to the request-stateless denoiser."""
        kwargs = super().prepare_pos_cond_kwargs(batch, device, rotary_emb, dtype)
        context = kwargs["bagel_context"]
        if not context.is_editing:
            raise RuntimeError("BAGEL Editing requires a three-way request context")
        kwargs.update(
            image_guidance_scale=(
                self.editing_image_guidance_scale
                if batch.true_cfg_scale is None
                else float(batch.true_cfg_scale)
            ),
            cfg_interval=self.editing_cfg_interval,
            cfg_renorm_type=self.editing_cfg_renorm_type,
        )
        return kwargs

    def calculate_condition_image_size(
        self, image: Image.Image, width: int, height: int
    ) -> tuple[int, int]:
        """Return the official VAE input size for one Editing image."""
        del image
        return calculate_bagel_resize_dimensions(
            width,
            height,
            max_size=1024,
            min_size=512,
            stride=16,
        )

    def preprocess_condition_image(
        self,
        image: Image.Image,
        target_width: int,
        target_height: int,
        _vae_image_processor,
    ) -> tuple[Image.Image, tuple[int, int]]:
        """Convert to RGB and resize once before VAE and ViT preprocessing."""
        image = image.convert("RGB").resize(
            (target_width, target_height), Image.Resampling.BICUBIC
        )
        return image, (target_width, target_height)
