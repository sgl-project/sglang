# SPDX-License-Identifier: Apache-2.0
"""Joint noise preparation, adapted from FastVideo's Kandinsky6LatentPreparationStage.

Draw video then audio from the same generator. Image conditioning is appended later."""

from __future__ import annotations

import math

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE


def audio_latent_duration(
    video_latent_frames: int,
    *,
    fps: float,
    audio_sample_rate: int,
    audio_downsample_factor: int,
) -> int:
    """Audio latent length from causal video duration at the request's fps."""
    pixel_frames = (video_latent_frames - 1) * 4 + 1
    return int(
        math.ceil(pixel_frames / fps * audio_sample_rate / audio_downsample_factor)
    )


class Kandinsky6LatentPreparationStage(PipelineStage):
    """Draw initial video *and* audio noise latents."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.height is None or batch.width is None:
            raise ValueError("height and width must be provided for Kandinsky6.")
        height = int(batch.height)
        width = int(batch.width)
        num_frames = int(batch.num_frames)

        pipeline_config = server_args.pipeline_config
        arch = pipeline_config.dit_config.arch_config
        temporal_ratio = (
            pipeline_config.vae_config.arch_config.temporal_compression_ratio
        )
        spatial_ratio = pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = arch.patch_size

        if num_frames % temporal_ratio != 1:
            num_frames = num_frames // temporal_ratio * temporal_ratio + 1
            batch.num_frames = num_frames

        required_divisor_h = spatial_ratio * patch_size[1]
        required_divisor_w = spatial_ratio * patch_size[2]
        if height % required_divisor_h != 0 or width % required_divisor_w != 0:
            raise ValueError(
                f"Kandinsky6 height must be divisible by {required_divisor_h} and width by "
                f"{required_divisor_w}; got height={height}, width={width}."
            )

        batch_size = batch.batch_size
        dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]
        device = get_local_torch_device()
        num_latent_frames = (num_frames - 1) // temporal_ratio + 1
        num_channels = arch.in_visual_dim
        video_shape = (
            batch_size,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
            num_channels,
        )

        generator = batch.generator
        if batch.latents is None:
            video = randn_tensor(
                video_shape, generator=generator, device=device, dtype=dtype
            )
        else:
            video = batch.latents.to(device=device, dtype=dtype)
            if tuple(video.shape) != video_shape:
                raise ValueError(
                    f"Provided latents shape {list(video.shape)} does not match expected "
                    f"Kandinsky6 video latent shape {list(video_shape)}."
                )

        if bool(arch.visual_cond):
            cond = torch.zeros_like(video)
            mask = torch.zeros(
                (*video.shape[:-1], 1), device=video.device, dtype=video.dtype
            )
            video = torch.cat([video, cond, mask], dim=-1)

        num_audio_channels = arch.in_audio_dim
        # align audio duration with request fps, not the pipeline default
        num_audio_frames = audio_latent_duration(
            num_latent_frames,
            fps=float(batch.fps),
            audio_sample_rate=pipeline_config.audio_sample_rate,
            audio_downsample_factor=pipeline_config.audio_downsample_factor,
        )
        audio_shape = (batch_size, num_audio_frames, num_audio_channels)
        if batch.audio_latents is None:
            # Drawn from the same generator right after the video noise, so
            # it is automatically an independent sample -- no manual seed
            # offset needed.
            audio = randn_tensor(
                audio_shape, generator=generator, device=device, dtype=dtype
            )
        else:
            audio = batch.audio_latents.to(device=device, dtype=dtype)
            if tuple(audio.shape) != audio_shape:
                raise ValueError(
                    f"Provided audio_latents shape {list(audio.shape)} does not match expected "
                    f"Kandinsky6 audio latent shape {list(audio_shape)}."
                )

        batch.latents = video
        batch.audio_latents = audio
        batch.raw_latent_shape = (
            batch_size,
            num_channels,
            num_latent_frames,
            height // spatial_ratio,
            width // spatial_ratio,
        )
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(3)]
        )
        return result
