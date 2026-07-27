# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for the BAGEL text-to-image path."""

from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.bagel import BagelDiTConfig
from sglang.multimodal_gen.configs.models.vaes.bagel import BagelVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)

_BAGEL_CONTEXT_KEY = "bagel_context"


@dataclass
class BagelPipelineConfig(ImagePipelineConfig):
    """Configure request-local BAGEL conditioning and standard VAE decoding."""

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
                        "BAGEL T2I supports exactly one latent sample per request"
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
