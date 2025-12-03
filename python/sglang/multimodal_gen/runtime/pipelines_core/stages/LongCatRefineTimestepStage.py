# SPDX-License-Identifier: Apache-2.0
"""
LongCat refinement timestep preparation stage.

This stage prepares special timesteps for LongCat refinement that start from t_thresh.
"""

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage

logger = init_logger(__name__)


class LongCatRefineTimestepStage(PipelineStage):
    """
    Stage for preparing timesteps specific to LongCat refinement.

    For refinement, we need to start from t_thresh instead of t=1.0, so we:
    1. Generate normal timesteps for num_inference_steps
    2. Filter to only keep timesteps < t_thresh * 1000
    3. Prepend t_thresh * 1000 as the first timestep
    """

    def __init__(self, scheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Prepare refinement-specific timesteps.

        Args:
            batch: The current batch information.
            server_args: The server arguments.

        Returns:
            The batch with refinement timesteps.
        """
        # Only apply if this is a refinement task
        # Trigger when either a refine_from path or in-memory stage1_video is provided
        if batch.refine_from is None and getattr(batch, "stage1_video",
                                                 None) is None:
            return batch

        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        t_thresh = batch.t_thresh

        logger.info("Preparing LongCat refinement timesteps (t_thresh=%s)",
                    t_thresh)

        # ------------------------------------------------------------------
        # 1) Match LongCatVideoPipeline.get_timesteps_sigmas (non-distill):
        #    sigmas = linspace(1, 0.001, num_inference_steps) on CPU
        # ------------------------------------------------------------------
        base_sigmas = torch.linspace(
            1.0,
            0.001,
            num_inference_steps,
            dtype=torch.float32,
            device=
            "cpu",  # scheduler.set_timesteps expects CPU-convertible sigmas
        )
        # Let the scheduler build its internal timestep schedule from sigmas
        self.scheduler.set_timesteps(num_inference_steps,
                                     sigmas=base_sigmas,
                                     device=device)
        base_timesteps = self.scheduler.timesteps

        # ------------------------------------------------------------------
        # 2) Apply t_thresh cropping exactly like generate_refine:
        #    timesteps = [t_thresh*1000] + [t for t in base_timesteps if t < t_thresh*1000]
        #    sigmas = timesteps / 1000  (with trailing zero)
        # ------------------------------------------------------------------
        t_thresh_value = t_thresh * 1000.0
        t_thresh_tensor = torch.tensor(t_thresh_value,
                                       dtype=base_timesteps.dtype,
                                       device=device)
        filtered_timesteps = base_timesteps[base_timesteps < t_thresh_tensor]

        timesteps = torch.cat(
            [t_thresh_tensor.unsqueeze(0), filtered_timesteps])

        # Update scheduler with these custom timesteps and corresponding sigmas
        self.scheduler.timesteps = timesteps
        sigmas = torch.cat([timesteps / 1000.0, torch.zeros(1, device=device)])
        self.scheduler.sigmas = sigmas

        logger.info("Refinement timesteps: %s steps starting from t=%s",
                    len(timesteps), t_thresh)
        logger.info("First few timesteps: %s", timesteps[:5].tolist())

        # Store in batch so downstream stages (denoising) use the same schedule
        batch.timesteps = timesteps

        return batch
