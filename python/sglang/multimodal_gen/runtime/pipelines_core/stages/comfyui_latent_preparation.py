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

        # No TextEncodingStage here, so back-fill require_text_seq_lens' input.
        # A 2-D entry is [seq, dim] text unless the batch has pooled embeds, where
        # it is the pooled projection instead and its leading dim is the batch.
        pipeline_config = server_args.pipeline_config
        if batch.prompt_embeds is not None and batch.prompt_seq_lens is None:
            has_pooled = bool(batch.pooled_embeds)
            batch.prompt_seq_lens = [
                (
                    None
                    if has_pooled and e.ndim == 2
                    else pipeline_config.seq_lens_from_prompt_embeds(e)
                )
                for e in batch.prompt_embeds
            ]
        if (
            batch.negative_prompt_embeds is not None
            and batch.negative_prompt_seq_lens is None
        ):
            has_neg_pooled = bool(batch.neg_pooled_embeds)
            batch.negative_prompt_seq_lens = [
                (
                    None
                    if has_neg_pooled and e.ndim == 2
                    else pipeline_config.seq_lens_from_prompt_embeds(e)
                )
                for e in batch.negative_prompt_embeds
            ]

        original_latents_shape = None
        if batch.latents is not None:
            original_latents_shape = batch.latents.shape

        result = super().forward(batch, server_args)

        if original_latents_shape is not None:
            # Preserve the original shape before any packing/conversion
            # (e.g., 4D spatial -> 3D sequence) so unpadding stays correct.
            result.raw_latent_shape = original_latents_shape

        return result
