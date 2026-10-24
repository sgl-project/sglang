# SPDX-License-Identifier: Apache-2.0
"""SANA-Sprint few-step text-to-image pipeline.

Reuses the SANA DiT, Gemma2 encoder and DC-AE. The TrigFlow sCM sampling lives
in ``SCMScheduler``; this module adds Sprint's prompt instruction and its
per-request embedded guidance.
"""

import torch
from diffusers.image_processor import PixArtImageProcessor

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.sana_video import (
    SanaVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

# Diffusers' SanaSprintPipeline default `complex_human_instruction`, newline-joined.
SANA_SPRINT_COMPLEX_HUMAN_INSTRUCTION = (
    "Given a user prompt, generate an 'Enhanced prompt' that provides detailed "
    "visual descriptions suitable for image generation. Evaluate the level of "
    "detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, "
    "sizes, textures, and spatial relationships to create vivid and concrete "
    "scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details "
    "slightly without overcomplicating.\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled "
    "up in a round shape, sleeping peacefully on a warm sunny windowsill, "
    "surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene "
    "at dusk, featuring glowing street lamps, a diverse crowd of people in "
    "colorful clothing, and a double-decker bus passing by towering glass "
    "skyscrapers.\n"
    "Please generate only the enhanced description for the prompt below and avoid "
    "including any additional commentary or evaluations:\n"
    "User Prompt: "
)


class SanaSprintTextEncodingStage(SanaVideoTextEncodingStage):
    complex_human_instruction = SANA_SPRINT_COMPLEX_HUMAN_INSTRUCTION


class SanaSprintDenoisingStage(DenoisingStage):
    def _prepare_denoising_loop(
        self, batch: Req, server_args: ServerArgs
    ) -> DenoisingContext:
        ctx = super()._prepare_denoising_loop(batch, server_args)
        # The guidance embedding takes the request's guidance_scale, not the static
        # embedded_cfg_scale; Diffusers rounds it to the DiT dtype before scaling.
        arch = server_args.pipeline_config.dit_config.arch_config
        ctx.guidance = (
            torch.full(
                (ctx.latents.shape[0],),
                batch.guidance_scale,
                dtype=torch.float32,
                device=ctx.latents.device,
            ).to(ctx.target_dtype)
            * arch.guidance_embeds_scale
        )
        return ctx


class SanaSprintDecodingStage(DecodingStage):
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output_batch = super().forward(batch, server_args)
        frames = output_batch.output
        if frames is None or frames.shape[-2:] == (batch.height, batch.width):
            return output_batch
        # Resolution binning generated at an aspect-ratio bin; resize and
        # center-crop back to the requested size as Diffusers does.
        output_batch.output = _resize_and_crop(frames, batch.width, batch.height)
        return output_batch


def _resize_and_crop(frames: torch.Tensor, width: int, height: int) -> torch.Tensor:
    # Bilinear resizing is per plane, so every leading dim folds into the batch.
    planes = frames.flatten(0, -3).unsqueeze(1)
    planes = PixArtImageProcessor.resize_and_crop_tensor(planes, width, height)
    return planes.squeeze(1).unflatten(0, frames.shape[:-2])


class SanaSprintPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "SanaSprintPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage(
            SanaSprintTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage_primary",
        )
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[
                lambda batch, server_args: ("max_timesteps", batch.max_timesteps),
                lambda batch, server_args: (
                    "intermediate_timesteps",
                    batch.intermediate_timesteps,
                ),
                lambda batch, server_args: ("timesteps", batch.trigflow_timesteps),
            ]
        )
        self.add_standard_latent_preparation_stage()
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: SanaSprintDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
            "denoising_stage",
        )
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: SanaSprintDecodingStage(
                vae=self.get_module("vae"), pipeline=self, component_name="vae"
            ),
            "decoding_stage",
        )


EntryClass = SanaSprintPipeline
