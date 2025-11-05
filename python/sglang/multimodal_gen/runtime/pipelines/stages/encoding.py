# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Encoding stage for diffusion pipelines.
"""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    V,  # Import validators
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class EncodingStage(PipelineStage):
    """
    Stage for encoding pixel space representations into latent space.

    This stage handles the encoding of pixel-space video/images into latent
    representations for further processing in the diffusion pipeline.
    """

    def __init__(self, vae: ParallelTiledVAE) -> None:
        self.vae: ParallelTiledVAE = vae

    @torch.no_grad()
    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage inputs."""
        result = VerificationResult()
        # Input video/images for VAE encoding: [batch_size, channels, frames, height, width]
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify encoding stage outputs."""
        result = VerificationResult()
        # Encoded latents: [batch_size, channels, frames, height_latents, width_latents]
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        return result

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Encode pixel space representations into latent space.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with encoded latents.
        """
        assert batch.latents is not None and isinstance(batch.latents, torch.Tensor)

        self.vae = self.vae.to(get_local_torch_device())

        # Setup VAE precision
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        # Normalize input to [-1, 1] range (reverse of decoding normalization)
        latents = (batch.latents * 2.0 - 1.0).clamp(-1, 1)

        # Move to appropriate device and dtype
        latents = latents.to(get_local_torch_device())

        # Encode image to latents
        with torch.autocast(
            device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
        ):
            if server_args.pipeline_config.vae_tiling:
                self.vae.enable_tiling()
            # if server_args.vae_sp:
            #     self.vae.enable_parallel()
            if not vae_autocast_enabled:
                latents = latents.to(vae_dtype)
            latents = self.vae.encode(latents).mean

        # Update batch with encoded latents
        batch.latents = latents

        # Offload models if needed
        if hasattr(self, "maybe_free_model_hooks"):
            self.maybe_free_model_hooks()

        if server_args.vae_cpu_offload:
            self.vae.to("cpu")

        return batch
