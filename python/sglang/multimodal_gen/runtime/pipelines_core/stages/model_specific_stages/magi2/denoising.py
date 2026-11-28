# SPDX-License-Identifier: Apache-2.0
"""Joint audio-video denoising for MAGI-2's preview and refiner passes."""

from __future__ import annotations

import torch

from sglang.multimodal_gen.configs.sample.magi2 import MAGI2_REFINER_FPS
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.preparation import (
    build_scheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def skimming_mask(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Elements where all three sign tests agree; only those get the softened scale."""
    latent = latent.to(device=cond.device, dtype=cond.dtype)
    guided = (latent - uncond) + guidance_scale * ((latent - cond) - (latent - uncond))
    denoised = latent - guided

    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * guidance_scale - uncond * (guidance_scale - 1)).sign()
    )
    deviation_influence = denoised.sign() == (denoised - latent).sign()
    return matching_pred_signs & matching_diff_after & deviation_influence


def skimmed_uncond(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
    skimmed_scale: float,
) -> torch.Tensor:
    """Two rounds with swapped mask arguments (sampler.py:326-331)."""
    # An all-zero uncond has nothing to damp, and the sign tests would read zeros
    # as agreement.
    if guidance_scale <= 1.0 or not bool(uncond.any()):
        return uncond

    fallback_weight = (skimmed_scale - 1) / (guidance_scale - 1)
    mask = skimming_mask(
        latent=latent, cond=cond, uncond=uncond, guidance_scale=guidance_scale
    )
    uncond = torch.where(mask, torch.lerp(cond, uncond, fallback_weight), uncond)

    mask = skimming_mask(
        latent=latent, cond=uncond, uncond=cond, guidance_scale=guidance_scale
    )
    return torch.where(mask, torch.lerp(cond, uncond, fallback_weight), uncond)


def apply_guidance(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
    skimmed_scale: float | None = None,
) -> torch.Tensor:
    if skimmed_scale is not None:
        uncond = skimmed_uncond(
            latent=latent,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
            skimmed_scale=skimmed_scale,
        )
    return uncond + guidance_scale * (cond - uncond)


BETA_START = 0.00085
BETA_END = 0.0120
NUM_TRAIN_TIMESTEPS = 1000


def zero_snr_sigmas(
    *, num_timesteps: int = NUM_TRAIN_TIMESTEPS, device: torch.device | None = None
) -> torch.Tensor:
    """Must stay DESCENDING: index 220 selects 0.839, not the ascending table's 0.152 (inference_engine.py:873, :844)."""
    betas = (
        torch.linspace(
            BETA_START**0.5,
            BETA_END**0.5,
            num_timesteps,
            dtype=torch.float64,
        )
        ** 2
    )
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas = alphas_cumprod.sqrt()

    first, last = sqrt_alphas[0].clone(), sqrt_alphas[-1].clone()
    sqrt_alphas = (sqrt_alphas - last) * (first / (first - last))

    return sqrt_alphas.to(device=device, dtype=torch.float32)


def upsample_latent(latent: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    # Time goes to 2 * T - 1 on the refiner grid.
    frames = latent.shape[2]
    return torch.nn.functional.interpolate(
        latent.float(),
        size=(2 * frames - 1, height, width),
        mode="trilinear",
        align_corners=True,
    )


def renoise(
    latent: torch.Tensor,
    *,
    noise: torch.Tensor,
    sigma_index: int,
    sigmas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mix ``latent`` with ``noise``; ``sigma`` is a signal coefficient, not a noise level."""
    table = sigmas if sigmas is not None else zero_snr_sigmas(device=latent.device)
    sigma = table[sigma_index].to(latent.dtype)
    return sigma * latent + (1.0 - sigma**2).sqrt() * noise


class Magi2StageHandoffStage(PipelineStage):
    """Audio is not refined: the refiner gets zero audio tokens and the preview's audio latent goes straight to its VAE."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not batch.extra["magi2_enable_refiner"]:
            return batch

        config = server_args.pipeline_config
        device = get_local_torch_device()
        arch = config.dit_config.arch_config
        frames, height, width = batch.extra["magi2_preview_grid"]
        target_frames, target_height, target_width = batch.extra["magi2_refiner_grid"]

        grid = (
            batch.latents.reshape(frames, height, width, arch.video_in_channels)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
        )
        upsampled = upsample_latent(grid, height=target_height, width=target_width)
        if upsampled.shape[2] != target_frames:
            raise ValueError(
                f"handoff produced {upsampled.shape[2]} latent frames but the "
                f"refiner grid expects {target_frames}"
            )

        noise = torch.randn(
            upsampled.shape,
            device=device,
            dtype=torch.float32,
            generator=batch.generator,
        )
        renoised = renoise(
            upsampled, noise=noise, sigma_index=config.refiner_renoise_index
        )

        batch.latents = (
            renoised.squeeze(0).permute(1, 2, 3, 0).reshape(-1, arch.video_in_channels)
        )
        batch.raw_latent_shape = (
            1,
            arch.video_in_channels,
            target_frames,
            target_height,
            target_width,
        )

        # Parked so the refiner sees no audio tokens; decoding picks it back up.
        batch.extra["magi2_preview_audio_latents"] = batch.audio_latents
        batch.audio_latents = None

        batch.scheduler = build_scheduler(
            shift=config.refiner_flow_shift,
            steps=batch.sampling_params.refiner_num_inference_steps,
            device=device,
        )
        batch.timesteps = batch.scheduler.timesteps
        batch.sigmas = batch.scheduler.sigmas.tolist()
        batch.fps = MAGI2_REFINER_FPS
        return batch


class Magi2DenoisingStage(DenoisingStage):
    """Run one denoise loop over the packed video+audio sequence.

    ``forward`` is model-owned: the two modalities carry separate guidance scales
    and separate multistep scheduler state.
    """

    def __init__(
        self,
        *,
        transformer,
        pipeline=None,
        guidance_key: str = "",
        refiner_only: bool = False,
    ) -> None:
        # scheduler=None: the schedule is per-modality, built in the preparation stage.
        super().__init__(transformer=transformer, scheduler=None, pipeline=pipeline)
        self.guidance_key = guidance_key
        self.refiner_only = refiner_only

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # The base tests whether forward is inherited, so not overriding this would
        # silently disable offload-during-compile rather than raise.
        return True

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER

    def _scales(self, batch: Req) -> tuple[float, float, float | None]:
        params = batch.sampling_params
        if self.guidance_key == "refiner":
            return (
                params.refiner_guidance_scale,
                params.refiner_audio_guidance_scale,
                None,
            )
        return (
            params.guidance_scale,
            params.audio_guidance_scale,
            params.skimmed_guidance_scale if params.use_skimmed_guidance else None,
        )

    def _predict(
        self,
        *,
        video: torch.Tensor,
        audio: torch.Tensor | None,
        text: torch.Tensor,
        layout,
        coords: torch.Tensor,
        timestep: torch.Tensor,
        ref_patches: torch.Tensor | None = None,
        ref_special: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        extra = {}
        # Only the preview accepts ref images; the refiner's allowlist raises on them.
        if ref_patches is not None:
            extra = {"ref_patches": ref_patches, "ref_special": ref_special}
        return self.transformer(
            video_latents=video,
            audio_latents=audio,
            text_embeds=text,
            layout=layout,
            coords=coords,
            timestep=timestep,
            **extra,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        _reject_unsupported_overrides(batch)
        if self.refiner_only and not batch.extra["magi2_enable_refiner"]:
            return batch
        with self._offload_for_torch_compile_warmup(batch):
            return self._denoise(batch, server_args)

    def _denoise(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        video_scale, audio_scale, skimmed_scale = self._scales(batch)
        use_guidance = server_args.pipeline_config.should_use_guidance

        layout = batch.extra["magi2_layout"]
        coords = batch.extra["magi2_coords"]
        prompt = batch.prompt_embeds[0]
        negative = (
            batch.negative_prompt_embeds[0]
            if use_guidance and batch.negative_prompt_embeds
            else None
        )
        uncond_layout = batch.extra.get("magi2_layout_uncond", layout)
        uncond_coords = batch.extra.get("magi2_coords_uncond", coords)

        video = batch.latents
        audio = batch.audio_latents
        scheduler = batch.scheduler

        ref_patches = (
            batch.extra["magi2_ref_patches"] if layout.ref_patch_index.numel() else None
        )
        ref_special = (
            batch.extra["magi2_ref_special"] if ref_patches is not None else None
        )

        for step, timestep in enumerate(batch.timesteps):
            # Per token, not a scalar: text and ref-image rows must read zero.
            step_t = timestep.to(device)
            t = packed_sequence.build_timesteps(
                layout=layout, video_t=step_t, audio_t=step_t
            )

            cond_video, cond_audio = self._predict(
                video=video,
                audio=audio,
                text=prompt,
                layout=layout,
                coords=coords,
                timestep=t,
                ref_patches=ref_patches,
                ref_special=ref_special,
            )

            if negative is not None:
                uncond_video, uncond_audio = self._predict(
                    video=video,
                    audio=audio,
                    text=negative,
                    layout=uncond_layout,
                    coords=uncond_coords,
                    timestep=packed_sequence.build_timesteps(
                        layout=uncond_layout, video_t=step_t, audio_t=step_t
                    ),
                    ref_patches=ref_patches,
                    ref_special=ref_special,
                )
                cond_video = apply_guidance(
                    latent=video,
                    cond=cond_video,
                    uncond=uncond_video,
                    guidance_scale=video_scale,
                    skimmed_scale=skimmed_scale,
                )
                if cond_audio is not None and uncond_audio is not None:
                    cond_audio = apply_guidance(
                        latent=audio,
                        cond=cond_audio,
                        uncond=uncond_audio,
                        guidance_scale=audio_scale,
                        skimmed_scale=skimmed_scale,
                    )

            video = scheduler.step(cond_video, timestep, video, return_dict=False)[0]
            if cond_audio is not None:
                audio = batch.extra["magi2_audio_scheduler"].step(
                    cond_audio, timestep, audio, return_dict=False
                )[0]

        batch.latents = video
        batch.audio_latents = audio
        return batch


def _reject_unsupported_overrides(batch: Req) -> None:
    # Magi2SamplingParams._validate rejects these at request time (before queueing).
    # This guard is a defence-in-depth fallback for requests that bypass _validate
    # (e.g. loaded via loaded_modules without HTTP validation).
    params = batch.sampling_params
    for name, value in (
        ("attention_backend_override", params.attention_backend_override),
        ("skip_softmax_params", params.skip_softmax_params),
    ):
        if value is not None:
            raise ValueError(
                f"MAGI-2 does not support per-request {name}; got {value!r}"
            )
