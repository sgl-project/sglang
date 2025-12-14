# SPDX-License-Identifier: Apache-2.0
"""
LongCat refinement initialization stage.

This stage prepares the latent variables for LongCat's 480p->720p refinement by:
1. Loading the stage1 (480p) video
2. Upsampling it to 720p resolution
3. Encoding it with VAE
4. Mixing with noise according to t_thresh
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.longcatvideo import (
    get_bucket_config,
)
from sglang.multimodal_gen.runtime.models.vision_utils import load_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    Req as ForwardBatch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LongCatRefineInitStage(PipelineStage):
    """
    Stage for initializing LongCat refinement from a stage1 (480p) video.

    This replicates the logic from LongCatVideoPipeline.generate_refine():
    - Load stage1_video frames
    - Upsample spatially and temporally
    - VAE encode and normalize
    - Mix with noise according to t_thresh
    """

    def __init__(self, vae) -> None:
        super().__init__()
        self.vae = vae

    def forward(
        self,
        batch: ForwardBatch,
        fastvideo_args: ServerArgs,
    ) -> ForwardBatch:
        """
        Initialize latents for refinement.

        Args:
            batch: The current batch information.
            fastvideo_args: The inference arguments.

        Returns:
            The batch with initialized latents for refinement.
        """
        refine_from = batch.refine_from
        in_memory_stage1 = getattr(batch, "stage1_video", None)

        # Only run for refinement tasks: either a path (refine_from) or in-memory video is provided
        if refine_from is None and in_memory_stage1 is None:
            # Not a refinement task, skip
            return batch

        # ------------------------------------------------------------------
        # 1. Obtain stage1 frames (either from disk or from in-memory input)
        # ------------------------------------------------------------------
        if in_memory_stage1 is not None:
            # User provided stage1 frames directly (e.g., from distilled stage output)
            if len(in_memory_stage1) == 0:
                raise ValueError(
                    "stage1_video is empty; expected a non-empty list of frames"
                )

            if isinstance(in_memory_stage1[0], Image.Image):
                pil_images = in_memory_stage1
            else:
                # Assume numpy arrays or torch tensors with shape [H, W, C]
                pil_images = [
                    Image.fromarray(np.array(frame)) for frame in in_memory_stage1
                ]

            logger.info(
                "Initializing LongCat refinement from in-memory stage1_video (%s frames)",
                len(pil_images),
            )
        else:
            # Path-based refine: load video from disk (original design)
            logger.info("Initializing LongCat refinement from file: %s", refine_from)
            stage1_video_path = Path(refine_from)
            if not stage1_video_path.exists():
                raise FileNotFoundError(f"Stage1 video not found: {refine_from}")

            # Load video frames as PIL Images
            pil_images, original_fps = load_video(
                str(stage1_video_path), return_fps=True
            )
            logger.info(
                "Loaded stage1 video: %s frames @ %s fps", len(pil_images), original_fps
            )

        # Store in batch for reference (use PIL images, same as official demo)
        batch.stage1_video = pil_images

        # Get parameters from batch
        num_frames = len(pil_images)
        spatial_refine_only = batch.spatial_refine_only
        t_thresh = batch.t_thresh
        num_cond_frames = (
            batch.num_cond_frames if hasattr(batch, "num_cond_frames") else 0
        )

        # Calculate new frame count (temporal upsampling if not spatial_refine_only)
        new_num_frames = num_frames if spatial_refine_only else 2 * num_frames
        logger.info(
            "Refine mode: %s",
            "spatial only" if spatial_refine_only else "spatial + temporal",
        )

        # Update batch.num_frames to reflect the upsampled count
        batch.num_frames = new_num_frames

        # Use bucket system to select resolution (exactly like LongCat)
        # Calculate scale_factor_spatial considering SP split
        sp_size = fastvideo_args.sp_size if fastvideo_args.sp_size > 0 else 1
        vae_scale_factor_spatial = 8  # VAE spatial downsampling
        patch_size_spatial = 2  # LongCat patch size
        bsa_latent_granularity = 4
        scale_factor_spatial = (
            vae_scale_factor_spatial * patch_size_spatial * bsa_latent_granularity
        )  # 64

        # Calculate optimal split like LongCat (cp_split_hw logic)
        # For sp_size=1: [1,1], max=1
        # For sp_size=2: [1,2], max=2
        # For sp_size=4: [2,2], max=2
        # For sp_size=8: [2,4], max=4
        if sp_size > 1:
            # Get optimal 2D split factors (mimic context_parallel_util.get_optimal_split)
            factors = []
            for i in range(1, int(sp_size**0.5) + 1):
                if sp_size % i == 0:
                    factors.append([i, sp_size // i])
            cp_split_hw = min(factors, key=lambda x: abs(x[0] - x[1]))
            scale_factor_spatial *= max(cp_split_hw)
            logger.info(
                "SP split: sp_size=%s, cp_split_hw=%s, max_split=%s",
                sp_size,
                cp_split_hw,
                max(cp_split_hw),
            )
        else:
            cp_split_hw = [1, 1]

        # Get bucket config and find closest bucket for the input aspect ratio
        bucket_config = get_bucket_config("720p", scale_factor_spatial)

        # Get input aspect ratio from stage1 video
        input_height, input_width = pil_images[0].height, pil_images[0].width
        input_ratio = input_height / input_width

        # Find closest bucket
        closest_ratio = min(
            bucket_config.keys(), key=lambda x: abs(float(x) - input_ratio)
        )
        height, width = bucket_config[closest_ratio][0]

        logger.info(
            "Input aspect ratio: %.2f (%sx%s)", input_ratio, input_width, input_height
        )
        logger.info(
            "Matched bucket ratio: %s -> resolution: %sx%s",
            closest_ratio,
            width,
            height,
        )
        logger.info(
            "Target: %sx%s @ %s frames (sp_size=%s, scale_factor=%s)",
            width,
            height,
            new_num_frames,
            sp_size,
            scale_factor_spatial,
        )

        # Override batch height/width with bucket-selected resolution
        batch.height = height
        batch.width = width

        # Convert PIL images to tensor [T, C, H, W]
        stage1_video_tensor = torch.stack(
            [
                torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC -> CHW
                for img in pil_images
            ]
        ).float()  # [T, C, H, W]

        device = batch.prompt_embeds[0].device
        dtype = batch.prompt_embeds[0].dtype
        stage1_video_tensor = stage1_video_tensor.to(device=device, dtype=dtype)

        # Replicate LongCat's exact preprocessing (lines 1227-1235 in pipeline_longcat_video.py)
        # First: spatial interpolation to target (height, width) on [T, C, H, W]
        video_down = F.interpolate(
            stage1_video_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=True,
        )

        # Rearrange to [C, T, H, W] and add batch dimension -> [1, C, T, H, W]
        video_down = video_down.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
        video_down = video_down / 255.0  # Normalize to [0, 1]

        # Then: temporal+spatial interpolation to (new_num_frames, height, width)
        video_up = F.interpolate(
            video_down,
            size=(new_num_frames, height, width),
            mode="trilinear",
            align_corners=True,
        )

        # Rescale to [-1, 1] for VAE
        video_up = video_up * 2.0 - 1.0

        logger.info("Upsampled video shape: %s", video_up.shape)

        # Padding logic (exactly like LongCat lines 1237-1255)
        # Only pad temporal dimension to ensure BSA compatibility
        vae_scale_factor_temporal = 4
        num_noise_frames = video_up.shape[2] - num_cond_frames

        num_cond_latents = 0
        num_cond_frames_added = 0
        if num_cond_frames > 0:
            num_cond_latents = 1 + math.ceil(
                (num_cond_frames - 1) / vae_scale_factor_temporal
            )
            num_cond_latents = (
                math.ceil(num_cond_latents / bsa_latent_granularity)
                * bsa_latent_granularity
            )
            num_cond_frames_added = (
                1 + (num_cond_latents - 1) * vae_scale_factor_temporal - num_cond_frames
            )
            num_cond_frames = num_cond_frames + num_cond_frames_added

        num_noise_latents = math.ceil(num_noise_frames / vae_scale_factor_temporal)
        num_noise_latents = (
            math.ceil(num_noise_latents / bsa_latent_granularity)
            * bsa_latent_granularity
        )
        num_noise_frames_added = (
            num_noise_latents * vae_scale_factor_temporal - num_noise_frames
        )

        if num_cond_frames_added > 0 or num_noise_frames_added > 0:
            logger.info(
                "Padding temporal dimension for BSA: cond_frames+=%s, noise_frames+=%s",
                num_cond_frames_added,
                num_noise_frames_added,
            )
            pad_front = video_up[:, :, 0:1].repeat(1, 1, num_cond_frames_added, 1, 1)
            pad_back = video_up[:, :, -1:].repeat(1, 1, num_noise_frames_added, 1, 1)
            video_up = torch.cat([pad_front, video_up, pad_back], dim=2)
            logger.info("Padded video shape: %s", video_up.shape)

        # Update batch with actual frame count after padding
        batch.num_frames = video_up.shape[2]

        # Store padding info for later cropping (CRITICAL for correct output!)
        batch.num_cond_frames_added = num_cond_frames_added
        batch.num_noise_frames_added = num_noise_frames_added
        batch.new_frame_size_before_padding = new_num_frames

        # Store num_cond_latents for denoising stage
        if num_cond_latents > 0:
            batch.num_cond_latents = num_cond_latents
            logger.info(
                "Will use num_cond_latents=%s during denoising", num_cond_latents
            )

        logger.info(
            "Padding info: cond+=%s, noise+=%s, original=%s",
            num_cond_frames_added,
            num_noise_frames_added,
            new_num_frames,
        )

        # VAE encode
        logger.info("Encoding stage1 video with VAE...")
        vae_dtype = next(self.vae.parameters()).dtype
        vae_device = next(self.vae.parameters()).device
        video_up = video_up.to(dtype=vae_dtype, device=vae_device)

        with torch.no_grad():
            latent_dist = self.vae.encode(video_up)
            # Extract tensor from latent distribution
            if hasattr(latent_dist, "latent_dist"):
                # Nested distribution wrapper
                latent_up = latent_dist.latent_dist.sample()
            elif hasattr(latent_dist, "sample"):
                # DiagonalGaussianDistribution or similar
                latent_up = latent_dist.sample()
            elif hasattr(latent_dist, "latents"):
                # Direct latents tensor
                latent_up = latent_dist.latents
            else:
                # Assume it's already a tensor
                latent_up = latent_dist

        # Normalize latents using VAE config (exactly like LongCat)
        if hasattr(self.vae.config, "latents_mean") and hasattr(
            self.vae.config, "latents_std"
        ):
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_up.device, latent_up.dtype)
            )
            # LongCat uses: 1.0 / latents_std (equivalent to dividing by latents_std)
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latent_up.device, latent_up.dtype)
            # LongCat: (latents - mean) * (1/std)
            latent_up = (latent_up - latents_mean) * latents_std

        logger.info("Encoded latent shape: %s", latent_up.shape)

        # Mix with noise according to t_thresh
        # latent_up = (1 - t_thresh) * latent_up + t_thresh * noise
        noise = torch.randn_like(latent_up).contiguous()
        latent_up = (1 - t_thresh) * latent_up + t_thresh * noise

        logger.info("Applied t_thresh=%s noise mixing", t_thresh)

        # Store in batch
        batch.latents = latent_up.to(dtype)
        batch.raw_latent_shape = latent_up.shape

        logger.info("LongCat refinement initialization complete")

        return batch
