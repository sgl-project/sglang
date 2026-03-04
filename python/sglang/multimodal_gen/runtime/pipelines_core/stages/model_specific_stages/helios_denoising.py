# SPDX-License-Identifier: Apache-2.0
"""
Helios-specific chunked denoising stage.

Implements Stage 1 chunked denoising with multi-term memory history
and CFG Zero Star guidance. VAE decoding is handled by the standard
DecodingStage downstream.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def optimized_scale(positive_flat, negative_flat):
    """CFG Zero Star: compute optimal guidance scale."""
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def sample_block_noise(
    batch_size, channel, num_frames, height, width, gamma, patch_size=(1, 2, 2)
):
    """Generate spatially-correlated block noise for pyramid SR."""
    _, ph, pw = patch_size
    block_size = ph * pw

    cov = (
        torch.eye(block_size) * (1 + gamma) - torch.ones(block_size, block_size) * gamma
    )
    dist = torch.distributions.MultivariateNormal(
        torch.zeros(block_size, device=cov.device), covariance_matrix=cov
    )
    block_number = batch_size * channel * num_frames * (height // ph) * (width // pw)

    noise = dist.sample((block_number,))
    noise = noise.view(
        batch_size, channel, num_frames, height // ph, width // pw, ph, pw
    )
    noise = noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(
        batch_size, channel, num_frames, height, width
    )
    return noise


class HeliosChunkedDenoisingStage(PipelineStage):
    """
    Helios chunked denoising stage implementing Stage 1 loop.

    Iterates over video chunks, manages history buffers (short/mid/long),
    runs transformer per chunk with CFG guidance, scheduler step,
    and accumulates denoised latents. VAE decoding is left to DecodingStage.
    """

    def __init__(self, transformer, scheduler):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler

    @property
    def parallelism_type(self):
        return StageParallelismType.REPLICATED

    def _denoise_one_chunk(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        timesteps,
        guidance_scale,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        target_dtype,
        device,
        is_cfg_zero_star=True,
        use_zero_init=True,
        zero_steps=1,
        batch=None,
        server_args=None,
    ):
        """Denoise a single chunk with full timestep loop."""
        batch_size = latents.shape[0]
        do_cfg = guidance_scale > 1.0

        for i, t in enumerate(timesteps):
            timestep = t.expand(batch_size)
            latent_model_input = latents.to(target_dtype)

            with set_forward_context(
                current_timestep=t,
                forward_batch=None,
                attn_metadata=None,
            ):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=(
                        latents_history_short.to(target_dtype)
                        if latents_history_short is not None
                        else None
                    ),
                    latents_history_mid=(
                        latents_history_mid.to(target_dtype)
                        if latents_history_mid is not None
                        else None
                    ),
                    latents_history_long=(
                        latents_history_long.to(target_dtype)
                        if latents_history_long is not None
                        else None
                    ),
                )

            if do_cfg:
                with set_forward_context(
                    current_timestep=t,
                    forward_batch=None,
                    attn_metadata=None,
                ):
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=(
                            latents_history_short.to(target_dtype)
                            if latents_history_short is not None
                            else None
                        ),
                        latents_history_mid=(
                            latents_history_mid.to(target_dtype)
                            if latents_history_mid is not None
                            else None
                        ),
                        latents_history_long=(
                            latents_history_long.to(target_dtype)
                            if latents_history_long is not None
                            else None
                        ),
                    )

                if is_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.reshape(batch_size, -1)
                    negative_flat = noise_uncond.reshape(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(
                        batch_size, *([1] * (len(noise_pred_text.shape) - 1))
                    )
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (
                            noise_pred_text - noise_uncond * alpha
                        )
                else:
                    noise_pred = noise_uncond + guidance_scale * (
                        noise_pred - noise_uncond
                    )

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents

    def _denoise_one_chunk_stage2(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        guidance_scale,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        target_dtype,
        device,
        pyramid_num_stages,
        pyramid_num_inference_steps_list,
        is_distilled,
        is_amplify_first_chunk,
        gamma,
        is_cfg_zero_star=True,
        use_zero_init=True,
        zero_steps=1,
        batch=None,
        server_args=None,
    ):
        """Denoise a single chunk using pyramid super-resolution (Stage 2)."""
        batch_size, num_channel, num_frames, height, width = latents.shape
        patch_size = self.transformer.patch_size

        # Downsample to lowest pyramid level
        latents = latents.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, num_channel, height, width
        )
        for _ in range(pyramid_num_stages - 1):
            height //= 2
            width //= 2
            latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
        latents = latents.reshape(
            batch_size, num_frames, num_channel, height, width
        ).permute(0, 2, 1, 3, 4)

        start_point_list = None
        if is_distilled:
            start_point_list = [latents]

        do_cfg = guidance_scale > 1.0

        for i_s in range(pyramid_num_stages):
            # Compute mu for current resolution
            image_seq_len = (
                latents.shape[-1]
                * latents.shape[-2]
                * latents.shape[-3]
                // (patch_size[0] * patch_size[1] * patch_size[2])
            )
            mu = calculate_shift(image_seq_len)

            self.scheduler.set_timesteps(
                pyramid_num_inference_steps_list[i_s],
                i_s,
                device=device,
                mu=mu,
                is_amplify_first_chunk=is_amplify_first_chunk,
            )
            timesteps = self.scheduler.timesteps

            if i_s > 0:
                # Upsample 2x nearest-neighbor
                height *= 2
                width *= 2
                latents = latents.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * num_frames,
                    num_channel,
                    height // 2,
                    width // 2,
                )
                latents = F.interpolate(latents, size=(height, width), mode="nearest")
                latents = latents.reshape(
                    batch_size, num_frames, num_channel, height, width
                ).permute(0, 2, 1, 3, 4)

                # Renoise with correlated block noise
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                bs, ch, nf, h, w = latents.shape
                noise = sample_block_noise(bs, ch, nf, h, w, gamma, patch_size)
                noise = noise.to(device=device, dtype=target_dtype)
                latents = alpha * latents + beta * noise

                if is_distilled:
                    start_point_list.append(latents)

            # Denoising loop for this pyramid stage
            for idx, t in enumerate(timesteps):
                timestep = t.expand(batch_size)
                latent_model_input = latents.to(target_dtype)

                with set_forward_context(
                    current_timestep=t,
                    forward_batch=None,
                    attn_metadata=None,
                ):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=(
                            latents_history_short.to(target_dtype)
                            if latents_history_short is not None
                            else None
                        ),
                        latents_history_mid=(
                            latents_history_mid.to(target_dtype)
                            if latents_history_mid is not None
                            else None
                        ),
                        latents_history_long=(
                            latents_history_long.to(target_dtype)
                            if latents_history_long is not None
                            else None
                        ),
                    )

                if do_cfg:
                    with set_forward_context(
                        current_timestep=t,
                        forward_batch=None,
                        attn_metadata=None,
                    ):
                        noise_uncond = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=(
                                latents_history_short.to(target_dtype)
                                if latents_history_short is not None
                                else None
                            ),
                            latents_history_mid=(
                                latents_history_mid.to(target_dtype)
                                if latents_history_mid is not None
                                else None
                            ),
                            latents_history_long=(
                                latents_history_long.to(target_dtype)
                                if latents_history_long is not None
                                else None
                            ),
                        )

                    if is_cfg_zero_star:
                        noise_pred_text = noise_pred
                        positive_flat = noise_pred_text.reshape(batch_size, -1)
                        negative_flat = noise_uncond.reshape(batch_size, -1)

                        alpha_cfg = optimized_scale(positive_flat, negative_flat)
                        alpha_cfg = alpha_cfg.view(
                            batch_size,
                            *([1] * (len(noise_pred_text.shape) - 1)),
                        )
                        alpha_cfg = alpha_cfg.to(noise_pred_text.dtype)

                        if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_uncond * alpha_cfg + guidance_scale * (
                                noise_pred_text - noise_uncond * alpha_cfg
                            )
                    else:
                        noise_pred = noise_uncond + guidance_scale * (
                            noise_pred - noise_uncond
                        )

                latents = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    cur_sampling_step=idx,
                    dmd_noisy_tensor=(
                        start_point_list[i_s] if start_point_list is not None else None
                    ),
                    dmd_sigmas=self.scheduler.sigmas,
                    dmd_timesteps=self.scheduler.timesteps,
                    all_timesteps=timesteps,
                )[0]

        return latents

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the Helios chunked denoising loop."""
        pipeline_config = server_args.pipeline_config
        device = (
            batch.latents.device
            if hasattr(batch, "latents") and batch.latents is not None
            else torch.device("cuda")
        )
        target_dtype = PRECISION_TO_TYPE.get(
            server_args.pipeline_config.precision, torch.bfloat16
        )

        # Get config params
        num_latent_frames_per_chunk = pipeline_config.num_latent_frames_per_chunk
        history_sizes = sorted(list(pipeline_config.history_sizes), reverse=True)
        is_cfg_zero_star = pipeline_config.is_cfg_zero_star
        zero_steps = pipeline_config.zero_steps
        keep_first_frame = pipeline_config.keep_first_frame
        guidance_scale = batch.guidance_scale
        num_inference_steps = batch.num_inference_steps

        # Stage 2 params
        is_enable_stage2 = pipeline_config.is_enable_stage2
        pyramid_num_stages = pipeline_config.pyramid_num_stages
        pyramid_num_inference_steps_list = (
            pipeline_config.pyramid_num_inference_steps_list
        )
        is_distilled = pipeline_config.is_distilled
        is_amplify_first_chunk = pipeline_config.is_amplify_first_chunk
        gamma = pipeline_config.gamma

        # Move transformer to GPU if CPU-offloaded
        if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
            if next(self.transformer.parameters()).device.type == "cpu":
                self.transformer.to(get_local_torch_device())

        # Get encoder outputs (prompt_embeds is a list of tensors, one per encoder)
        prompt_embeds = batch.prompt_embeds
        if isinstance(prompt_embeds, list):
            prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(target_dtype)
        negative_prompt_embeds = batch.negative_prompt_embeds
        if isinstance(negative_prompt_embeds, list):
            negative_prompt_embeds = (
                negative_prompt_embeds[0] if negative_prompt_embeds else None
            )
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(target_dtype)

        # Scale factors inherited from the Wan VAE used by Helios
        # (AutoencoderKLWan: temporal_compression_ratio=4, spatial_compression_ratio=8)
        vae_scale_factor_temporal = 4
        vae_scale_factor_spatial = 8

        # Compute chunking
        height = batch.height
        width = batch.width
        num_frames = batch.num_frames
        num_channels_latents = self.transformer.in_channels

        window_num_frames = (
            num_latent_frames_per_chunk - 1
        ) * vae_scale_factor_temporal + 1
        num_latent_chunk = max(
            1, (num_frames + window_num_frames - 1) // window_num_frames
        )
        num_history_latent_frames = sum(history_sizes)
        batch_size = 1  # Helios processes one video at a time

        # Prepare history latents
        if not keep_first_frame:
            history_sizes[-1] = history_sizes[-1] + 1
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            num_history_latent_frames,
            height // vae_scale_factor_spatial,
            width // vae_scale_factor_spatial,
            device=device,
            dtype=torch.float32,
        )

        # Build frame indices
        if keep_first_frame:
            indices = torch.arange(
                0, sum([1, *history_sizes, num_latent_frames_per_chunk])
            )
            (
                indices_prefix,
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_1x,
                indices_hidden_states,
            ) = indices.split([1, *history_sizes, num_latent_frames_per_chunk], dim=0)
            indices_latents_history_short = torch.cat(
                [indices_prefix, indices_latents_history_1x], dim=0
            )
        else:
            indices = torch.arange(
                0, sum([*history_sizes, num_latent_frames_per_chunk])
            )
            (
                indices_latents_history_long,
                indices_latents_history_mid,
                indices_latents_history_short,
                indices_hidden_states,
            ) = indices.split([*history_sizes, num_latent_frames_per_chunk], dim=0)

        indices_hidden_states = indices_hidden_states.unsqueeze(0)
        indices_latents_history_short = indices_latents_history_short.unsqueeze(0)
        indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0)
        indices_latents_history_long = indices_latents_history_long.unsqueeze(0)

        # Set up scheduler
        patch_size = self.transformer.patch_size
        image_seq_len = (
            num_latent_frames_per_chunk
            * (height // vae_scale_factor_spatial)
            * (width // vae_scale_factor_spatial)
            // (patch_size[0] * patch_size[1] * patch_size[2])
        )
        # Sigma schedule from near-1.0 (pure noise) to 0.0 (clean); 0.999 avoids singularity
        sigmas = np.linspace(0.999, 0.0, num_inference_steps + 1)[:-1]
        mu = calculate_shift(image_seq_len)

        # Chunk loop
        image_latents = None
        total_generated_latent_frames = 0

        self.log_info(
            f"Starting chunked denoising: {num_latent_chunk} chunks, "
            f"{num_inference_steps} steps each"
        )

        for k in range(num_latent_chunk):
            is_first_chunk = k == 0

            # Extract history
            if keep_first_frame:
                (
                    latents_history_long,
                    latents_history_mid,
                    latents_history_1x,
                ) = history_latents[:, :, -num_history_latent_frames:].split(
                    history_sizes, dim=2
                )
                if image_latents is None and is_first_chunk:
                    latents_prefix = torch.zeros(
                        (
                            batch_size,
                            num_channels_latents,
                            1,
                            latents_history_1x.shape[-2],
                            latents_history_1x.shape[-1],
                        ),
                        device=device,
                        dtype=latents_history_1x.dtype,
                    )
                else:
                    latents_prefix = image_latents
                latents_history_short = torch.cat(
                    [latents_prefix, latents_history_1x], dim=2
                )
            else:
                (
                    latents_history_long,
                    latents_history_mid,
                    latents_history_short,
                ) = history_latents[:, :, -num_history_latent_frames:].split(
                    history_sizes, dim=2
                )

            # Generate noise latents for this chunk
            latents = torch.randn(
                batch_size,
                num_channels_latents,
                (window_num_frames - 1) // vae_scale_factor_temporal + 1,
                height // vae_scale_factor_spatial,
                width // vae_scale_factor_spatial,
                device=device,
                dtype=torch.float32,
            )

            if is_enable_stage2:
                # Stage 2: Pyramid SR denoising (handles scheduler internally)
                latents = self._denoise_one_chunk_stage2(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    target_dtype=target_dtype,
                    device=device,
                    pyramid_num_stages=pyramid_num_stages,
                    pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                    is_distilled=is_distilled,
                    is_amplify_first_chunk=(is_amplify_first_chunk and is_first_chunk),
                    gamma=gamma,
                    is_cfg_zero_star=is_cfg_zero_star,
                    use_zero_init=True,
                    zero_steps=zero_steps,
                    batch=batch,
                    server_args=server_args,
                )
            else:
                # Stage 1: Standard flat denoising
                self.scheduler.set_timesteps(
                    num_inference_steps, device=device, sigmas=sigmas, mu=mu
                )
                timesteps = self.scheduler.timesteps

                latents = self._denoise_one_chunk(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    timesteps=timesteps,
                    guidance_scale=guidance_scale,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short,
                    latents_history_mid=latents_history_mid,
                    latents_history_long=latents_history_long,
                    target_dtype=target_dtype,
                    device=device,
                    is_cfg_zero_star=is_cfg_zero_star,
                    use_zero_init=True,
                    zero_steps=zero_steps,
                    batch=batch,
                    server_args=server_args,
                )

            # Extract first frame as image_latents for subsequent chunks
            if keep_first_frame and is_first_chunk and image_latents is None:
                image_latents = latents[:, :, 0:1, :, :]

            # Update history
            total_generated_latent_frames += latents.shape[2]
            history_latents = torch.cat([history_latents, latents], dim=2)

        # Move transformer back to CPU after denoising
        if server_args.dit_cpu_offload and not server_args.use_fsdp_inference:
            if next(self.transformer.parameters()).device.type != "cpu":
                self.transformer.to("cpu")
        torch.cuda.empty_cache()

        # Store denoised latents for the standard DecodingStage to decode
        batch.latents = history_latents[:, :, -total_generated_latent_frames:]

        return batch
