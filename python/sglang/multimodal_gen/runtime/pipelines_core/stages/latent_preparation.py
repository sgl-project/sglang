# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Latent preparation stage for diffusion pipelines.
"""

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class LatentPreparationFingerprint:
    height: int | None
    width: int | None
    num_frames: int | None
    latent_num_frames: int | None
    prompt_dtype: Any
    generator_device: str | None


class LatentPreparationStage(PipelineStage):
    """
    Stage for preparing initial latent variables for the diffusion process.

    This stage handles the preparation of the initial latent variables that will be
    denoised during the diffusion process.
    """

    def __init__(self, scheduler, transformer) -> None:
        super().__init__()
        self.scheduler = scheduler
        self.transformer = transformer

    def _get_latent_dtype(
        self,
        batch: Req,
        server_args: ServerArgs,
    ):
        return server_args.pipeline_config.get_latent_dtype(
            batch.prompt_embeds[0].dtype
        )

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare initial latent variables for the diffusion process.



        Returns:
            The batch with prepared latent variables.
        """

        # Adjust video length based on VAE version if needed
        latent_num_frames = self.adjust_video_length(batch, server_args)

        batch_size = batch.batch_size

        # Get required parameters
        dtype = self._get_latent_dtype(batch, server_args)
        device = get_local_torch_device()
        generator = batch.generator
        latents = batch.latents
        num_frames = (
            latent_num_frames if latent_num_frames is not None else batch.num_frames
        )
        height = batch.height
        width = batch.width

        # TODO(will): remove this once we add input/output validation for stages
        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        # Validate generator if it's a list
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        # Generate or use provided latents
        if latents is None:
            shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch_size, num_frames
            )
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )

            latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)

            if latent_ids is not None:
                batch.latent_ids = latent_ids.to(device=device)

            latents = server_args.pipeline_config.maybe_pack_latents(
                latents, batch_size, batch
            )
        else:
            latents = latents.to(device)

        # Scale the initial noise if needed
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma
        # Update batch with prepared latents
        batch.latents = latents
        batch.raw_latent_shape = latents.shape
        return batch

    def run_grouped_requests(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> list[Req]:
        """Group only the deterministic latent-preparation subprocess.

        Latent preparation is not a pure full-stage copy: each request still
        owns its RNG stream, so raw noise must be drawn once per request with
        that request's generator. The reusable part is the deterministic work
        after raw noise generation, such as packing latent tokens and applying
        scheduler scaling. For that reason this stage uses the common
        fingerprint grouping helper but implements its own grouped execution
        instead of ``run_deduplicated_group``.
        """
        results: list[Req | None] = [None] * len(batches)

        for _, group in self._group_requests_by_fingerprint(
            batches, lambda batch: self.build_dedup_fingerprint(batch, server_args)
        ):
            indexed_batches = group
            group_batches = [batch for _, batch in indexed_batches]
            if len(group_batches) == 1 or any(
                batch.latents is not None for batch in group_batches
            ):
                for index, batch in indexed_batches:
                    results[index] = self(batch, server_args)
                continue

            first_index, first_batch = indexed_batches[0]
            first_result = self._prepare_grouped_latents(group_batches, server_args)
            self._split_batched_latents(first_result, group_batches)
            results[first_index] = first_batch
            for index, batch in indexed_batches[1:]:
                results[index] = batch

        return [result for result in results if result is not None]

    def build_dedup_fingerprint(
        self, batch: Req, server_args: ServerArgs
    ) -> LatentPreparationFingerprint:
        prompt_dtype = (
            batch.prompt_embeds[0].dtype
            if isinstance(batch.prompt_embeds, list) and batch.prompt_embeds
            else None
        )
        latent_num_frames = self.adjust_video_length(batch, server_args)
        return LatentPreparationFingerprint(
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            latent_num_frames=latent_num_frames,
            prompt_dtype=prompt_dtype,
            generator_device=batch.generator_device,
        )

    @staticmethod
    def _single_generator(batch: Req):
        if isinstance(batch.generator, list):
            assert len(batch.generator) == 1
            return batch.generator[0]
        return batch.generator

    def _prepare_grouped_latents(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> Req:
        """Prepare grouped random latents without changing per-request RNG streams.

        ``randn_tensor`` accepts a list of generators, but its batched draw is not
        guaranteed to match drawing each request independently. For multi-output
        requests we need exact equivalence to the sequential seed path, so this
        helper draws one raw latent tensor per request and only batches the
        deterministic packing/scaling work.
        """
        first_batch = batches[0]
        latent_num_frames = self.adjust_video_length(first_batch, server_args)
        batch_size = len(batches)

        dtype = self._get_latent_dtype(first_batch, server_args)
        device = get_local_torch_device()
        num_frames = (
            latent_num_frames
            if latent_num_frames is not None
            else first_batch.num_frames
        )
        height = first_batch.height
        width = first_batch.width

        if height is None or width is None:
            raise ValueError("Height and width must be provided")

        raw_latents = []
        for batch in batches:
            shape = server_args.pipeline_config.prepare_latent_shape(
                batch, 1, num_frames
            )
            raw_latents.append(
                randn_tensor(
                    shape,
                    generator=self._single_generator(batch),
                    device=device,
                    dtype=dtype,
                )
            )

        latents = torch.cat(raw_latents, dim=0)
        latent_ids = server_args.pipeline_config.maybe_prepare_latent_ids(latents)
        if latent_ids is not None:
            first_batch.latent_ids = latent_ids.to(device=device)

        original_num_outputs = first_batch.num_outputs_per_prompt
        try:
            first_batch.num_outputs_per_prompt = batch_size
            latents = server_args.pipeline_config.maybe_pack_latents(
                latents, batch_size, first_batch
            )
        finally:
            first_batch.num_outputs_per_prompt = original_num_outputs

        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        first_batch.latents = latents
        first_batch.raw_latent_shape = latents.shape
        return first_batch

    @staticmethod
    def _slice_batch_tensor(tensor: torch.Tensor, index: int, total: int):
        if tensor.shape[0] == total:
            return tensor[index : index + 1].contiguous()
        return tensor

    def _split_batched_latents(self, src: Req, batches: list[Req]) -> None:
        total = len(batches)
        assert src.latents is not None
        latents = src.latents
        latent_ids = src.latent_ids
        for index, batch in enumerate(batches):
            batch.latents = self._slice_batch_tensor(latents, index, total)
            batch.raw_latent_shape = batch.latents.shape
            if latent_ids is not None:
                batch.latent_ids = self._slice_batch_tensor(latent_ids, index, total)

    def adjust_video_length(self, batch: Req, server_args: ServerArgs) -> int:
        """
        Adjust video length based on VAE version.
        """

        video_length = batch.num_frames
        latent_num_frames = video_length
        use_temporal_scaling_frames = (
            server_args.pipeline_config.vae_config.use_temporal_scaling_frames
        )
        if use_temporal_scaling_frames:
            temporal_scale_factor = (
                server_args.pipeline_config.vae_config.arch_config.temporal_compression_ratio
            )
            latent_num_frames = (video_length - 1) // temporal_scale_factor + 1
        return int(latent_num_frames)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage inputs."""
        result = VerificationResult()
        result.add_check(
            "prompt_or_embeds",
            None,
            lambda _: V.string_or_list_strings(batch.prompt)
            or V.list_not_empty(batch.prompt_embeds),
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_of_tensors)
        result.add_check(
            "num_videos_per_prompt", batch.num_outputs_per_prompt, V.positive_int
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int)
        result.add_check("width", batch.width, V.positive_int)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify latent preparation stage outputs."""
        result = VerificationResult()
        if batch.debug:
            logger.debug(f"{batch.raw_latent_shape=}")
        # disable temporarily for image-generation models
        # result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.is_tuple)
        return result
