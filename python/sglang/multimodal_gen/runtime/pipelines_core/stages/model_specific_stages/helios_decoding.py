# SPDX-License-Identifier: Apache-2.0
"""
Helios-specific decoding stage.

Decodes latent chunks one at a time (matching diffusers HeliosPipeline behavior)
to avoid temporal artifacts at chunk boundaries caused by Wan VAE's causal convolutions.
"""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    DecodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class HeliosDecodingStage(DecodingStage):
    """
    Helios-specific decoding stage that decodes latent chunks independently.

    The Wan VAE uses causal 3D convolutions with feature caching. When decoding
    the full latent sequence at once, the causal conv processes all frames with
    continuous context, producing a different number of output frames per latent
    frame compared to chunk-by-chunk decoding. This causes temporal misalignment
    and visible seams at chunk boundaries.

    This stage decodes each chunk's latents separately (matching diffusers'
    HeliosPipeline behavior) and concatenates the results in pixel space.
    """

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> OutputBatch:
        latent_chunks = getattr(batch, "latent_chunks", None)

        if latent_chunks is None or len(latent_chunks) <= 1:
            # No chunked latents or single chunk — use standard decode
            return super().forward(batch, server_args)

        # Load VAE if needed
        self.load_model()

        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        # Decode each chunk separately and concatenate in pixel space
        video_chunks = []
        with self.use_declared_component(
            component_name=self.component_name,
            module=self.vae,
        ) as vae:
            assert vae is not None
            self.vae = vae
            for chunk_latents in latent_chunks:
                chunk_video = self.decode(
                    chunk_latents, server_args, vae_dtype=vae_dtype
                )
                video_chunks.append(chunk_video)

        frames = torch.cat(video_chunks, dim=2)
        frames = server_args.pipeline_config.post_decoding(frames, server_args)

        output_batch = OutputBatch(
            output=frames,
            trajectory_timesteps=batch.trajectory_timesteps,
            trajectory_latents=batch.trajectory_latents,
            trajectory_decoded=None,
            metrics=batch.metrics,
        )

        return output_batch
