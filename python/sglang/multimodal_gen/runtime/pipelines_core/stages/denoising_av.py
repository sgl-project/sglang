import copy
import time

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = init_logger(__name__)


class LTX2AVDenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(self, transformer, scheduler, vae=None, audio_vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.audio_vae = audio_vae

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
         Run the denoising loop.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with denoised latents.
        """
        # Prepare variables for the denoising loop

        prepared_vars = self._prepare_denoising_loop(batch, server_args)
        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        image_kwargs = prepared_vars["image_kwargs"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        seq_len = prepared_vars["seq_len"]
        guidance = prepared_vars["guidance"]

        audio_latents = batch.audio_latents
        audio_scheduler = copy.deepcopy(self.scheduler)

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []
        trajectory_audio_latents: list[torch.Tensor] = []

        # Run denoising loop
        denoising_start_time = time.time()

        # to avoid device-sync caused by timestep comparison
        is_warmup = batch.is_warmup
        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)
        timesteps_cpu = timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{i}",
                        logger=logger,
                        timings=batch.timings,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                    ):
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )

                        # Expand latents for I2V
                        latent_model_input = latents.to(target_dtype)
                        if batch.image_latent is not None:
                            assert (
                                not server_args.pipeline_config.task_type
                                == ModelTaskType.TI2V
                            ), "image latents should not be provided for TI2V task"
                            latent_model_input = torch.cat(
                                [latent_model_input, batch.image_latent], dim=1
                            ).to(target_dtype)

                        # Predict noise residual
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)

                        # Custom LTX-2 prediction logic handling both video and audio
                        # We try to follow the structure of _predict_noise_with_cfg in base class
                        # but adapted for AV inputs/outputs.

                        # 1. Expand inputs for CFG
                        # NOTE: Base class _predict_noise_with_cfg does NOT concatenate inputs for CFG serial execution,
                        # it runs two forward passes (cond and uncond) sequentially to save memory.
                        # However, Diffusers LTX-2 implementation concatenates them [uncond, cond] and runs one pass.
                        # To align with Diffusers behavior for LTX-2 (which might rely on joint batch norm statistics or similar,
                        # though usually DiTs don't), we stick to concatenation if that's what Diffusers does.
                        # Diffusers LTXPipeline L882: torch.cat([latent_model_input] * 2) if do_classifier_free_guidance
                        # and then passes it to the model.

                        latent_model_input = latents.to(target_dtype)
                        audio_latent_model_input = audio_latents.to(target_dtype)

                        # Re-calculate coordinates and dimensions
                        latent_num_frames = (
                            (batch.num_frames - 1)
                            // server_args.pipeline_config.vae_temporal_compression
                            + 1
                        )
                        latent_height = (
                            batch.height // server_args.pipeline_config.vae_scale_factor
                        )
                        latent_width = (
                            batch.width // server_args.pipeline_config.vae_scale_factor
                        )

                        # Audio latent dims
                        if audio_latent_model_input.ndim == 3:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[1]
                            )
                        elif audio_latent_model_input.ndim == 4:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[2]
                            )
                        else:
                            raise ValueError(
                                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
                            )

                        # LTX-2 model handles coordinate generation internally via forward_from_grid
                        # if coords are not provided.
                        video_coords = None
                        audio_coords = None

                        timestep_expand = t_device.expand(
                            latent_model_input.shape[0]
                            * (2 if batch.do_classifier_free_guidance else 1)
                        )

                        # Prepare conditions
                        encoder_hidden_states = batch.prompt_embeds[0]
                        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                        encoder_attention_mask = batch.prompt_attention_mask

                        if batch.do_classifier_free_guidance:
                            # Concatenate for single forward pass (Diffusers style)
                            latent_model_input = torch.cat([latent_model_input] * 2)
                            latent_model_input = self.scheduler.scale_model_input(
                                latent_model_input, t_device
                            )

                            audio_latent_model_input = torch.cat(
                                [audio_latent_model_input] * 2
                            )

                            neg_encoder_hidden_states = batch.negative_prompt_embeds[0]
                            neg_audio_encoder_hidden_states = (
                                batch.negative_audio_prompt_embeds[0]
                            )
                            neg_encoder_attention_mask = batch.negative_attention_mask

                            # Order: [Negative, Positive] to match Diffusers [uncond, cond] usually?
                            # Diffusers: prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                            # So yes, Negative first.
                            encoder_hidden_states = torch.cat(
                                [neg_encoder_hidden_states, encoder_hidden_states]
                            )
                            audio_encoder_hidden_states = torch.cat(
                                [
                                    neg_audio_encoder_hidden_states,
                                    audio_encoder_hidden_states,
                                ]
                            )
                            encoder_attention_mask = torch.cat(
                                [neg_encoder_attention_mask, encoder_attention_mask]
                            )
                        else:
                            latent_model_input = self.scheduler.scale_model_input(
                                latent_model_input, t_device
                            )

                        with set_forward_context(
                            current_timestep=i, attn_metadata=attn_metadata
                        ):
                            noise_pred_video, noise_pred_audio = current_model(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                timestep=timestep_expand,
                                encoder_attention_mask=encoder_attention_mask,
                                audio_encoder_attention_mask=encoder_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=batch.frame_rate,
                                audio_num_frames=audio_num_frames_latent,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                return_latents=False,
                                return_dict=False,
                            )

                        noise_pred_video = noise_pred_video.float()
                        noise_pred_audio = noise_pred_audio.float()

                        # CFG Split and Combine
                        if batch.do_classifier_free_guidance:
                            noise_pred_video_uncond, noise_pred_video_text = (
                                noise_pred_video.chunk(2)
                            )
                            noise_pred_video = (
                                noise_pred_video_uncond
                                + batch.guidance_scale
                                * (noise_pred_video_text - noise_pred_video_uncond)
                            )

                            noise_pred_audio_uncond, noise_pred_audio_text = (
                                noise_pred_audio.chunk(2)
                            )
                            noise_pred_audio = (
                                noise_pred_audio_uncond
                                + batch.guidance_scale
                                * (noise_pred_audio_text - noise_pred_audio_uncond)
                            )

                            if getattr(batch, "guidance_rescale", 0.0) > 0.0:
                                noise_pred_video = self.rescale_noise_cfg(
                                    noise_pred_video,
                                    noise_pred_video_text,
                                    guidance_rescale=batch.guidance_rescale,
                                )
                                noise_pred_audio = self.rescale_noise_cfg(
                                    noise_pred_audio,
                                    noise_pred_audio_text,
                                    guidance_rescale=batch.guidance_rescale,
                                )

                        noise_pred = noise_pred_video  # For compatibility with step() logic below if we only update video there
                        # But we need to update both.

                        # Compute the previous noisy sample
                        latents = self.scheduler.step(
                            model_output=noise_pred_video,
                            timestep=t_device,
                            sample=latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )[0]

                        # Audio denoising step
                        # NOTE: for now duplicate scheduler for audio latents in case self.scheduler sets internal state in
                        # the step method (such as _step_index)
                        if audio_latents is not None:
                            audio_latents = audio_scheduler.step(
                                model_output=noise_pred_audio,
                                timestep=t_device,
                                sample=audio_latents,
                                **extra_step_kwargs,
                                return_dict=False,
                            )[0]

                        latents = self.post_forward_for_ti2v_task(
                            batch, server_args, reserved_frames_mask, latents, z
                        )

                        # save trajectory latents if needed
                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)
                            if audio_latents is not None:
                                trajectory_audio_latents.append(audio_latents)

                        # Update progress bar
                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        if not is_warmup:
                            self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0 and not is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )

        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_audio_latents=trajectory_audio_latents,
            server_args=server_args,
            is_warmup=is_warmup,
        )

        return batch

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        trajectory_audio_latents: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
    ):
        # 1. Handle Trajectory (Video) - Copy from base
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        # 2. Handle Trajectory (Audio) - LTX-2 specific
        if trajectory_audio_latents:
            trajectory_audio_tensor = torch.stack(trajectory_audio_latents, dim=1)
            # We don't have SP support for audio latents yet (or needed?)
            batch.trajectory_audio_latents = trajectory_audio_tensor.cpu()

        # 3. Unpack and Denormalize
        # Call pipeline_config._unpad_and_unpack_latents
        # latents is video latents.
        # batch.audio_latents is audio latents.

        audio_latents = batch.audio_latents

        # NOTE: self.vae and self.audio_vae should be populated via __init__ or manual setting
        if self.vae is None or self.audio_vae is None:
            logger.warning(
                "VAE or Audio VAE not found in DenoisingStage. Skipping unpack and denormalize."
            )
            batch.latents = latents
            batch.audio_latents = audio_latents
        else:
            latents, audio_latents = (
                server_args.pipeline_config._unpad_and_unpack_latents(
                    latents, audio_latents, batch, self.vae, self.audio_vae
                )
            )

            batch.latents = latents
            batch.audio_latents = audio_latents

        # 4. Cleanup
        offload_mgr = getattr(self.transformer, "_layerwise_offload_manager", None)
        if offload_mgr is not None and getattr(offload_mgr, "enabled", False):
            offload_mgr.release_all()

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs.

        Note: LTX-2 connector stage converts `prompt_embeds`/`negative_prompt_embeds`
        from list-of-tensors to a single tensor (video context) and stores audio
        context separately.
        """

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])

        # LTX-2 may carry prompt embeddings as either a tensor (preferred) or legacy list.
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            lambda x: V.is_tensor(x) or V.list_not_empty(x),
        )

        # Keep base expectation: image_embeds is always a list (may be empty).
        result.add_check("image_embeds", batch.image_embeds, V.is_list)

        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check("eta", batch.eta, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )

        # When CFG is enabled, negative prompt embeddings must exist (tensor or legacy list).
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x),
        )
        return result

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return batch.guidance_scale > 1.0


class LTX2RefinementStage(LTX2AVDenoisingStage):
    def __init__(
        self, transformer, scheduler, distilled_sigmas, vae=None, audio_vae=None
    ):
        super().__init__(transformer, scheduler, vae, audio_vae)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Add noise to latents
        noise_scale = self.distilled_sigmas[0].to(batch.latents.device)
        noise = torch.randn_like(batch.latents)
        batch.latents = batch.latents + noise * noise_scale

        # 2. Run denoising loop with distilled_sigmas
        # Save original sigmas
        original_sigmas = self.scheduler.sigmas
        original_timesteps = self.scheduler.timesteps
        original_num_inference_steps = self.scheduler.num_inference_steps

        # Set distilled sigmas
        self.scheduler.sigmas = self.distilled_sigmas.to(self.scheduler.sigmas.device)
        # Approximation for timesteps
        self.scheduler.timesteps = self.scheduler.sigmas * 1000
        self.scheduler.num_inference_steps = len(self.distilled_sigmas) - 1

        # Call parent forward
        try:
            batch = super().forward(batch, server_args)
        finally:
            # Restore original sigmas
            self.scheduler.sigmas = original_sigmas
            self.scheduler.timesteps = original_timesteps
            self.scheduler.num_inference_steps = original_num_inference_steps

        return batch

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return False  # Stage 2 uses simple denoising (no CFG)
