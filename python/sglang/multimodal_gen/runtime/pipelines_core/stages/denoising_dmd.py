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
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import dict_to_3d_list

logger = init_logger(__name__)


class DmdDenoisingStage(DenoisingStage):
    """
    Denoising stage for DMD.
    """

    def __init__(self, transformer, scheduler, transformer_2=None) -> None:
        super().__init__(
            transformer=transformer, scheduler=scheduler, transformer_2=transformer_2
        )
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
        workspace = prepared_vars.get("workspace")
        video_raw_latent_shape = latents.shape

        dmd_steps = server_args.pipeline_config.dmd_denoising_steps
        if not dmd_steps:
            if batch.timesteps is None:
                raise ValueError(
                    "DMD denoising steps are not set and no timesteps were prepared. "
                    "Provide dmd_denoising_steps in the pipeline config."
                )
            timesteps = batch.timesteps
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.to(
                    device=get_local_torch_device(), dtype=torch.long
                )
            else:
                timesteps = torch.tensor(
                    timesteps, dtype=torch.long, device=get_local_torch_device()
                )
        else:
            timesteps = torch.tensor(
                dmd_steps, dtype=torch.long, device=get_local_torch_device()
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

        denoising_loop_start_time = time.time()
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # Skip if interrupted
                if hasattr(self, "interrupt") and self.interrupt:
                    continue

                with StageProfiler(
                    f"denoising_step_{i}",
                    logger=logger,
                    metrics=batch.metrics,
                    perf_dump_path_provided=batch.perf_dump_path is not None,
                    record_as_step=True,
                ):
                    t_int = int(t.item())
                    if self.transformer_2 is not None:
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=self._handle_boundary_ratio(
                                    server_args, batch
                                ),
                                server_args=server_args,
                                batch=batch,
                            )
                        )
                    else:
                        current_model = self.transformer
                        self._manage_device_placement(current_model, None, server_args)
                    # Expand latents for I2V
                    if workspace is not None:
                        noise_latents = self._get_or_create_buffer(
                            workspace,
                            "noise_latents",
                            tuple(latents.shape),
                            latents.dtype,
                            latents.device,
                        )
                        noise_latents.copy_(latents)
                        latent_model_input = self._get_or_create_buffer(
                            workspace,
                            "latent_model_input",
                            tuple(latents.shape),
                            target_dtype,
                            latents.device,
                        )
                        latent_model_input.copy_(latents)
                    else:
                        noise_latents = latents.clone()
                        latent_model_input = latents.to(target_dtype)

                    if batch.image_latent is not None:
                        image_latent = batch.image_latent.permute(0, 2, 1, 3, 4)
                        if workspace is not None:
                            cat_shape = (
                                latent_model_input.shape[0],
                                latent_model_input.shape[1],
                                latent_model_input.shape[2] + image_latent.shape[2],
                                *latent_model_input.shape[3:],
                            )
                            latent_cat = self._get_or_create_buffer(
                                workspace,
                                "latent_cat",
                                cat_shape,
                                target_dtype,
                                latent_model_input.device,
                            )
                            c = latent_model_input.shape[2]
                            latent_cat[:, :, :c].copy_(latent_model_input)
                            latent_cat[:, :, c:].copy_(image_latent)
                            latent_model_input = latent_cat
                        else:
                            latent_model_input = torch.cat(
                                [latent_model_input, image_latent],
                                dim=2,
                            ).to(target_dtype)
                    assert not torch.isnan(
                        latent_model_input
                    ).any(), "latent_model_input contains nan"

                    # Prepare inputs for transformer
                    if workspace is not None:
                        t_expand = self._get_or_create_buffer(
                            workspace,
                            "t_expand",
                            (latent_model_input.shape[0],),
                            t.dtype,
                            t.device,
                        )
                        t_expand.copy_(t.expand_as(t_expand))
                    else:
                        t_expand = t.repeat(latent_model_input.shape[0])

                    guidance_expand = self.get_or_build_guidance(
                        latent_model_input.shape[0],
                        target_dtype,
                        get_local_torch_device(),
                    )

                    # Predict noise residual
                    with torch.autocast(
                        device_type=current_platform.device_type,
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
                            pred_noise = current_model(
                                hidden_states=latent_model_input.permute(0, 2, 1, 3, 4),
                                timestep=t_expand,
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
                            if workspace is not None:
                                next_timestep = self._get_or_create_buffer(
                                    workspace,
                                    "next_timestep",
                                    (1,),
                                    timesteps.dtype,
                                    pred_video.device,
                                )
                                next_timestep.fill_(timesteps[i + 1])
                                noise = self._get_or_create_buffer(
                                    workspace,
                                    "noise",
                                    tuple(video_raw_latent_shape),
                                    pred_video.dtype,
                                    self.device,
                                )
                                noise.normal_(generator=batch.generator[0])
                            else:
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

    def _select_and_manage_model(
        self,
        t_int: int,
        boundary_timestep: float | None,
        server_args: ServerArgs,
        batch: Req,
    ):
        if boundary_timestep is None or t_int >= boundary_timestep:
            # High-noise stage
            current_model = self.transformer
            model_to_offload = self.transformer_2
            current_guidance_scale = batch.guidance_scale
        else:
            # Low-noise stage
            current_model = self.transformer_2
            model_to_offload = self.transformer
            current_guidance_scale = batch.guidance_scale_2

        self._manage_device_placement(current_model, model_to_offload, server_args)

        assert current_model is not None, "The model for the current step is not set."
        return current_model, current_guidance_scale

    def _manage_device_placement(
        self,
        model_to_use: torch.nn.Module,
        model_to_offload: torch.nn.Module | None,
        server_args: ServerArgs,
    ):
        """
        Manages the offload / load behavior of dit
        """
        if not server_args.dit_cpu_offload:
            return

        # Offload the unused model if it's on CUDA
        if (
            model_to_offload is not None
            and next(model_to_offload.parameters()).device.type == "cuda"
        ):
            model_to_offload.to("cpu")

        # Load the model to use if it's on CPU
        if (
            model_to_use is not None
            and next(model_to_use.parameters()).device.type == "cpu"
        ):
            model_to_use.to(get_local_torch_device())

    def _handle_boundary_ratio(
        self,
        server_args,
        batch,
    ):
        """
        (Wan2.2) Calculate timestep to switch from high noise expert to low noise expert
        """
        boundary_ratio = server_args.pipeline_config.dit_config.boundary_ratio
        if batch.boundary_ratio is not None:
            logger.info(
                "Overriding boundary ratio from %s to %s",
                boundary_ratio,
                batch.boundary_ratio,
            )
            boundary_ratio = batch.boundary_ratio

        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.num_train_timesteps
        else:
            boundary_timestep = None

        return boundary_timestep
