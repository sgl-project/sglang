# SPDX-License-Identifier: Apache-2.0
"""ComfyUI latent prep: restore the worker session and bind the pass-through scheduler.

Multi-rank hops move CUDA tensors with NCCL, so this stage no longer walks
every field to fix pickle/gloo device mismatches.
"""

from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)


class ComfyUILatentPreparationStage(LatentPreparationStage):
    """One DiT step: restore cached conditioning, then prepare latents."""

    def verify_input(self, batch, server_args):
        bind_comfyui_session(batch)
        return super().verify_input(batch, server_args)

    def forward(self, batch, server_args):
        # DenoisingStage reads batch.scheduler. Native pipelines attach it in
        # TimestepPreparationStage; ComfyUI already owns the timestep schedule.
        get_or_create_request_scheduler(batch, self.scheduler)

        original_latents_shape = None
        if batch.latents is not None:
            original_latents_shape = batch.latents.shape

        result = super().forward(batch, server_args)

        if original_latents_shape is not None:
            # Preserve the original shape before any packing/conversion
            # (e.g., 4D spatial -> 3D sequence) so unpadding stays correct.
            result.raw_latent_shape = original_latents_shape

        return result
