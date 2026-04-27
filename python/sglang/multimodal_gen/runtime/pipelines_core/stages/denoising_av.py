import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import is_ltx23_native_variant
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    clone_scheduler_runtime,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    LTX2DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LTX2AVDenoisingStage(LTX2DenoisingStage):
    """
    Thin AV layer that adds audio trajectory gathering and final unpacking on top of
    the LTX-2 denoising semantics.
    """

    def __init__(self, transformer, scheduler, vae=None, audio_vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.audio_vae = audio_vae

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        trajectory_audio_latents: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Finalize AV requests by gathering audio latents and unpacking both streams."""
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )
        latents = self._truncate_sp_padded_token_latents(batch, latents)

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        if trajectory_audio_latents:
            trajectory_audio_tensor = torch.stack(trajectory_audio_latents, dim=1)
            batch.trajectory_audio_latents = trajectory_audio_tensor.cpu()

        audio_latents = batch.audio_latents
        if batch.did_sp_shard_audio_latents and isinstance(audio_latents, torch.Tensor):
            audio_latents = server_args.pipeline_config.gather_audio_latents_for_sp(
                audio_latents, batch
            )
            batch.audio_latents = audio_latents

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

        pipeline = self.pipeline() if self.pipeline else None
        current_phase = (
            str(getattr(batch, "extra", {}).get("ltx2_phase", ""))
            if hasattr(batch, "extra")
            else ""
        )
        release_phase_state = (
            getattr(pipeline, "release_ltx2_phase_state", None)
            if pipeline is not None
            else None
        )
        if callable(release_phase_state):
            release_phase_state(current_phase)

        if isinstance(self.transformer, OffloadableDiTMixin):
            for manager in self.transformer.layerwise_offload_managers:
                manager.release_all()


class LTX2RefinementStage(LTX2AVDenoisingStage):
    """Stage-2 refinement wrapper that re-noises distilled LTX latents once."""

    def __init__(
        self,
        transformer,
        scheduler,
        distilled_sigmas,
        vae=None,
        audio_vae=None,
        pipeline=None,
        sampler_name: str = "euler",
    ):
        super().__init__(
            transformer,
            scheduler,
            vae,
            audio_vae,
            pipeline=pipeline,
            sampler_name=sampler_name,
        )
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    @staticmethod
    def _randn_like_with_batch_generators(
        reference_tensor: torch.Tensor, batch: Req
    ) -> torch.Tensor:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list):
            bsz = int(reference_tensor.shape[0])
            valid_generators = [g for g in generator if isinstance(g, torch.Generator)]
            if len(valid_generators) == 1:
                generator = valid_generators[0]
            elif len(valid_generators) >= bsz:
                generator = valid_generators[:bsz]
            else:
                generator = None
        elif not isinstance(generator, torch.Generator):
            generator = None

        return randn_tensor(
            reference_tensor.shape,
            generator=generator,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

    @staticmethod
    def _reset_stage2_generators(batch: Req) -> None:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list) and generator:
            generator_device = str(generator[0].device)
        elif isinstance(generator, torch.Generator):
            generator_device = str(generator.device)
        else:
            generator_device = "cpu"

        seeds = getattr(batch, "seeds", None)
        if not seeds:
            seed = getattr(batch, "seed", None)
            if seed is None:
                return
            seeds = [int(seed)]

        batch.generator = [
            torch.Generator(device=generator_device).manual_seed(int(seed))
            for seed in seeds
        ]

    @staticmethod
    def _should_reset_stage2_generators(server_args: ServerArgs) -> bool:
        arch_config = getattr(
            server_args.pipeline_config.vae_config, "arch_config", None
        )
        if arch_config is not None and is_ltx23_native_variant(arch_config):
            return False
        return "LTX-2.3" not in str(getattr(server_args, "model_path", ""))

    @staticmethod
    def _build_stage2_renoise_generator(
        batch: Req, reference_tensor: torch.Tensor
    ) -> torch.Generator:
        seeds = getattr(batch, "seeds", None)
        if seeds:
            seed = int(seeds[0])
        else:
            seed = int(getattr(batch, "seed", 10))
        device = reference_tensor.device
        dtype = reference_tensor.dtype
        generator = torch.Generator(device=device).manual_seed(seed)
        video_shape = batch.extra.get("ltx2_stage1_packed_video_shape")
        audio_shape = batch.extra.get("ltx2_stage1_packed_audio_shape")
        if video_shape is not None:
            _ = torch.randn(
                tuple(video_shape), device=device, dtype=dtype, generator=generator
            )
        if audio_shape is not None:
            _ = torch.randn(
                tuple(audio_shape), device=device, dtype=dtype, generator=generator
            )
        return generator

    @staticmethod
    def _ltx2_renoise_like(
        reference_tensor: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        return torch.randn(
            reference_tensor.shape,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
            generator=generator,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the distilled refinement schedule on top of the shared AV denoiser."""
        batch.extra["ltx2_phase"] = "stage2"
        pipeline = self.pipeline() if self.pipeline else None
        ensure_phase_ready = (
            getattr(pipeline, "ensure_ltx2_phase_ready", None)
            if pipeline is not None
            else None
        )
        if callable(ensure_phase_ready):
            ensure_phase_ready("stage2")
        original_clean_latent_background = getattr(
            batch, "ltx2_ti2v_clean_latent_background", None
        )
        is_native_ti2v = (
            is_ltx23_native_variant(server_args.pipeline_config.vae_config.arch_config)
            and batch.image_path is not None
            and isinstance(batch.latents, torch.Tensor)
        )
        if is_native_ti2v:
            # Official two-stage TI2V keeps the upsampled stage-2 latent as the
            # clean background and only overwrites the conditioned frame tokens.
            batch.ltx2_ti2v_clean_latent_background = batch.latents.detach().clone()
        else:
            batch.ltx2_ti2v_clean_latent_background = None
        if self._should_reset_stage2_generators(server_args):
            self._reset_stage2_generators(batch)
        noise_scale = float(self.distilled_sigmas[0].item())
        is_ltx23 = is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )
        if is_ltx23:
            video_reference_for_gen = (
                batch.latents if isinstance(batch.latents, torch.Tensor) else None
            )
            if video_reference_for_gen is None:
                video_reference_for_gen = batch.audio_latents
            renoise_generator = self._build_stage2_renoise_generator(
                batch, video_reference_for_gen
            )
        else:
            renoise_generator = None
        if is_native_ti2v:
            prepared_latents, denoise_mask, _ = self._prepare_ltx2_ti2v_clean_state(
                batch=batch,
                latents=batch.latents,
                image_latent=batch.image_latent,
                num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                zero_clean_latent=True,
                clean_latent_background=batch.ltx2_ti2v_clean_latent_background,
            )
            if is_ltx23:
                video_noise = self._ltx2_renoise_like(
                    prepared_latents, renoise_generator
                )
            else:
                video_noise = self._randn_like_with_batch_generators(
                    prepared_latents, batch
                )
            scaled_mask = (
                denoise_mask.to(device=prepared_latents.device, dtype=torch.float32)
                * noise_scale
            )
            if is_ltx23:
                batch.latents = (
                    video_noise.float() * scaled_mask
                    + prepared_latents.float() * (1.0 - scaled_mask)
                ).to(prepared_latents.dtype)
            else:
                batch.latents = (
                    video_noise * scaled_mask + prepared_latents * (1 - scaled_mask)
                ).to(prepared_latents.dtype)
        else:
            if is_ltx23:
                video_noise = self._ltx2_renoise_like(batch.latents, renoise_generator)
                batch.latents = (
                    video_noise.float() * noise_scale
                    + batch.latents.float() * (1.0 - noise_scale)
                ).to(batch.latents.dtype)
            else:
                video_noise = self._randn_like_with_batch_generators(
                    batch.latents, batch
                )
                batch.latents = (
                    video_noise * noise_scale + batch.latents * (1 - noise_scale)
                ).to(batch.latents.dtype)

        if isinstance(batch.audio_latents, torch.Tensor):
            if is_ltx23:
                audio_noise = self._ltx2_renoise_like(
                    batch.audio_latents, renoise_generator
                )
                batch.audio_latents = (
                    audio_noise.float() * noise_scale
                    + batch.audio_latents.float() * (1.0 - noise_scale)
                ).to(batch.audio_latents.dtype)
            else:
                audio_noise = self._randn_like_with_batch_generators(
                    batch.audio_latents, batch
                )
                audio_scaled_mask = (
                    torch.ones_like(batch.audio_latents[..., :1], dtype=torch.float32)
                    * noise_scale
                )
                batch.audio_latents = (
                    audio_noise * audio_scaled_mask
                    + batch.audio_latents * (1 - audio_scaled_mask)
                ).to(batch.audio_latents.dtype)
        if not is_ltx23:
            batch.latents = batch.latents.to(
                device=batch.latents.device, dtype=torch.float32
            )
            if isinstance(batch.audio_latents, torch.Tensor):
                batch.audio_latents = batch.audio_latents.to(
                    device=batch.audio_latents.device, dtype=torch.float32
                )

        original_batch_scheduler = batch.scheduler
        original_batch_timesteps = batch.timesteps
        original_batch_num_inference_steps = batch.num_inference_steps

        scheduler = clone_scheduler_runtime(original_batch_scheduler or self.scheduler)
        distilled_device = scheduler.sigmas.device
        # Inject `0.0011` before the terminal `0.0` to avoid the
        # `sigma_next==0` singularity in res2s' `(sample - denoised) /
        # (sigma - sigma_next)`. Official `res2s_denoising_loop` does this
        # exact injection (samplers.py:262); official `euler_denoising_loop`
        # does NOT — it uses `sigma_next` directly. So gate on the active
        # sampler, not on the model variant.
        if self.sampler_name == "res2s" and self.distilled_sigmas[-1].item() == 0.0:
            scheduler_sigmas = torch.cat(
                [
                    self.distilled_sigmas[:-1],
                    torch.tensor([0.0011, 0.0], dtype=self.distilled_sigmas.dtype),
                ],
                dim=0,
            )
        else:
            scheduler_sigmas = self.distilled_sigmas

        scheduler.sigmas = scheduler_sigmas
        num_steps = len(self.distilled_sigmas) - 1
        scheduler.num_inference_steps = num_steps
        scheduler.timesteps = (self.distilled_sigmas[:num_steps] * 1000).to(
            distilled_device
        )
        scheduler._step_index = None
        scheduler._begin_index = None

        batch.scheduler = scheduler
        batch.timesteps = scheduler.timesteps
        batch.num_inference_steps = num_steps
        original_do_cfg = batch.do_classifier_free_guidance
        batch.do_classifier_free_guidance = False

        try:
            batch = super().forward(batch, server_args)
        finally:
            batch.scheduler = original_batch_scheduler
            batch.timesteps = original_batch_timesteps
            batch.num_inference_steps = original_batch_num_inference_steps
            batch.do_classifier_free_guidance = original_do_cfg
            batch.ltx2_ti2v_clean_latent_background = original_clean_latent_background

        return batch
