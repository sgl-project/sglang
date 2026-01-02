# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import time

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import dict_to_3d_list

logger = init_logger(__name__)


class DmdDenoisingStage(DenoisingStage):
    """
    Denoising stage for DMD.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=8.0)

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        # 1. to shard latents (B, C, T, H, W) along dim 2
        super()._preprocess_sp_latents(batch, server_args)

        # 2. DMD expects (B, T, C, H, W) for the main latents in the loop
        if batch.latents is not None:
            batch.latents = batch.latents.permute(0, 2, 1, 3, 4)

        # Note: batch.image_latent is kept as (B, C, T, H, W) here

    def _postprocess_sp_latents(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_tensor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # 1. convert back from DMD's (B, T, C, H, W) to standard (B, C, T, H, W)
        # this is because base gather_latents_for_sp expects dim=2 for T
        latents = latents.permute(0, 2, 1, 3, 4)

        # 2. use base method to gather
        return super()._postprocess_sp_latents(batch, latents, trajectory_tensor)

    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Run the denoising loop.
        """
        prepared_vars = self._prepare_denoising_loop(batch, server_args)

        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        latents = prepared_vars["latents"]
        video_raw_latent_shape = latents.shape

        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps,
            dtype=torch.long,
            device=get_local_torch_device(),
        )

        # prepare image_kwargs
        image_embeds = batch.image_embeds
        if len(image_embeds) > 0:
            image_embeds = [img.to(target_dtype) for img in image_embeds]

        image_kwargs = self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_hidden_states_image": image_embeds,
                "mask_strategy": dict_to_3d_list(None, t_max=50, l_max=60, h_max=24),
            },
        )

        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        prompt_embeds = prepared_vars["prompt_embeds"]

        denoising_loop_start_time = time.time()
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, "interrupt") and self.interrupt:
                    continue

                with StageProfiler(
                    f"denoising_step_{i}", logger=logger, timings=batch.timings
                ):
                    # Expand latents for I2V
                    noise_latents = latents.clone()
                    latent_model_input = latents.to(target_dtype)

                    if batch.image_latent is not None:
                        latent_model_input = torch.cat(
                            [
                                latent_model_input,
                                batch.image_latent.permute(0, 2, 1, 3, 4),
                            ],
                            dim=2,
                        ).to(target_dtype)
                    assert not torch.isnan(
                        latent_model_input
                    ).any(), "latent_model_input contains nan"

                    # Prepare inputs for transformer
                    t_expand = t.repeat(latent_model_input.shape[0])

                    guidance_expand = self.get_or_build_guidance(
                        latent_model_input.shape[0],
                        target_dtype,
                        get_local_torch_device(),
                    )

                    # Predict noise residual
                    with torch.autocast(
                        device_type="cuda",
                        dtype=target_dtype,
                        enabled=autocast_enabled,
                    ):
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)

                        batch.is_cfg_negative = False
                        with set_forward_context(
                            current_timestep=i,
                            attn_metadata=attn_metadata,
                            forward_batch=batch,
                        ):
                            # Run transformer
                            pred_noise = self.transformer(
                                latent_model_input.permute(0, 2, 1, 3, 4),
                                prompt_embeds,
                                t_expand,
                                guidance=guidance_expand,
                                **image_kwargs,
                                **pos_cond_kwargs,
                            ).permute(0, 2, 1, 3, 4)

                        pred_video = pred_noise_to_pred_video(
                            pred_noise=pred_noise.flatten(0, 1),
                            noise_input_latent=noise_latents.flatten(0, 1),
                            timestep=t_expand,
                            scheduler=self.scheduler,
                        ).unflatten(0, pred_noise.shape[:2])

                        if i < len(timesteps) - 1:
                            next_timestep = timesteps[i + 1] * torch.ones(
                                [1], dtype=torch.long, device=pred_video.device
                            )
                            noise = torch.randn(
                                video_raw_latent_shape,
                                dtype=pred_video.dtype,
                                generator=batch.generator[0],
                                device=self.device,
                            )
                            latents = self.scheduler.add_noise(
                                pred_video.flatten(0, 1),
                                noise.flatten(0, 1),
                                next_timestep,
                            ).unflatten(0, pred_video.shape[:2])
                        else:
                            latents = pred_video

                        # Update progress bar
                        if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                    self.step_profile()

        denoising_loop_end_time = time.time()
        if len(timesteps) > 0:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_loop_end_time - denoising_loop_start_time) / len(timesteps),
            )

        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=[],
            trajectory_timesteps=[],
            server_args=server_args,
        )

        return batch
