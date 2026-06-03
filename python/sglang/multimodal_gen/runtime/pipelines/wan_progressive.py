# SPDX-License-Identifier: Apache-2.0
"""
Wan video progressive-resolution denoising stage.

Extends ProgressiveDenoisingStage for the Wan T2V video model:
  - Latent format: [B, C, T, H, W] (already spatial — no pack/unpack required)
  - Upsample: spatial H×W dims only; T (temporal frames) is fixed across all stages
  - No RoPE / freqs_cis update needed (Wan T2V uses no spatial positional embeddings
    that depend on H or W in the context)

Power-law spectrum constants fitted on VChitect dataset (9050 videos), spatial
spectrum P(ω) = A * |ω|^(-β):
  A    = 219.484718
  β    = 2.422687
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Power-law spectrum constants for WAN 2.1 VAE
# Fitted on VChitect (9050 videos): P(ω) = A * |ω|^(-β)
WAN_SPECTRUM_A: float = 219.484718
WAN_SPECTRUM_BETA: float = 2.422687


class WanProgressiveDenoisingStage(ProgressiveDenoisingStage):
    """Wan T2V–specific progressive denoising stage.

    Differences from the FLUX progressive stage:
    - Wan latent is [B, C, T, H, W] — no patchify pack/unpack needed.
    - Progressive upsample grows only the spatial H×W plane; T stays fixed.
    - Wan T2V has no spatial RoPE freqs_cis that depends on H/W, so
      _on_resolution_change is a no-op.
    - Initial noise must carry the temporal dimension T_lat.
    """

    def __init__(self, transformer, scheduler, pipeline=None, vae=None) -> None:
        super().__init__(
            transformer,
            scheduler,
            pipeline=pipeline,
            vae=vae,
            spectrum_A=WAN_SPECTRUM_A,
            spectrum_beta=WAN_SPECTRUM_BETA,
        )

    # ------------------------------------------------------------------
    # Pack / Unpack overrides  (Wan latent is already [B, C, T, H, W])
    # ------------------------------------------------------------------

    def _unpack_latent(
        self, latent: torch.Tensor, h_lat: int, w_lat: int
    ) -> torch.Tensor:
        return latent

    def _repack_latent(
        self,
        x_spatial: torch.Tensor,
        h_lat: int,
        w_lat: int,
        batch: Req,
        server_args: ServerArgs,
    ) -> torch.Tensor:
        return x_spatial

    # ------------------------------------------------------------------
    # Resolution-change hook  (no-op for Wan T2V)
    # ------------------------------------------------------------------

    def _on_resolution_change(
        self,
        ctx,
        batch: Req,
        server_args: ServerArgs,
        new_h_pixel: int,
        new_w_pixel: int,
    ) -> None:
        """Wan T2V has no spatial positional embeddings that require updating."""
        pass

    # ------------------------------------------------------------------
    # Initial noise  (must include the temporal dim T_lat)
    # ------------------------------------------------------------------

    def _generate_initial_noise(
        self,
        batch: Req,
        server_args: ServerArgs,
        h_lat: int,
        w_lat: int,
        seed: int,
    ) -> torch.Tensor:
        """Generate low-res initial noise [1, C, T_lat, h_lat, w_lat].

        The base-class version generates 4-D [1, C, H, W] noise and uses
        in_channels // 4 for the channel count (FLUX patchify convention).
        Wan operates directly on the 5-D latent, so we override to:
          - Use z_dim (= 16) as the correct latent channel count.
          - Preserve T_lat from the original full-res latent in batch.latents,
            since progressive upsample only grows spatial H×W.
        """
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device

        device = get_local_torch_device()
        C = server_args.pipeline_config.vae_config.arch_config.z_dim
        # batch.latents still holds the full-res latent from LatentPreparationStage
        # at this call site, so shape[2] gives the fixed T_lat.
        T_lat = batch.latents.shape[2]
        dtype = server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype if batch.prompt_embeds else torch.bfloat16
        )
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        noise = torch.randn(1, C, T_lat, h_lat, w_lat, generator=gen, dtype=dtype).to(
            device
        )
        return noise
