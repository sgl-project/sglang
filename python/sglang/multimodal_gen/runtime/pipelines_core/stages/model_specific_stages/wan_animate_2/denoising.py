# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2 denoising and output stages.

WanAnimate2DenoisingStage orchestrates clips around the shared denoising loop: the in-context DiT
takes per-clip conditioning via its ``clip_cond`` kwarg and the clip's reference K/V via
``reference_kv`` (built once per clip with ``build_reference_kv`` and dropped with the clip),
runs cond/uncond CFG as two forwards (the unconditional branch skips block 9 via
``is_unconditional``), resets the scheduler per clip on the official sigma grid, and
VAE-decodes each clip in-loop to feed the next.
WanAnimate2OutputStage then emits the assembled frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import rearrange
from torch import nn

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import (
    CFGBranch,
    CFGPolicy,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_clip_conditioning import (
    WanAnimate2ClipConditioning,
    WanAnimate2ReferenceKV,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_dpm_solver_multistep import (
    DPMSolverMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.before_denoising import (
    WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY,
    WanAnimate2RequestState,
    build_clip_conditioning,
    get_sampling_sigmas,
    request_state_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
    WanAnimate2ImageEncoderAdapter,
    WanAnimate2VaeAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
    LetterboxInfo,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    V,
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )


def _is_request_state(value: object) -> bool:
    return isinstance(value, WanAnimate2RequestState)


def _is_decoded_frames(value: object) -> bool:
    """``[T, H, W, C]`` uint8 frames assembled by the denoising stage."""
    return isinstance(value, np.ndarray) and value.ndim == 4 and value.dtype == np.uint8


@dataclass(kw_only=True)
class WanAnimate2DenoisingContext(DenoisingContext):
    clip_condition: WanAnimate2ClipConditioning
    reference_image_embeddings: torch.Tensor
    cfg_parallel: bool
    guidance_scale: float
    reference_kv: WanAnimate2ReferenceKV | None = None
    context_by_branch: dict[str, torch.Tensor] = field(default_factory=dict)


class WanAnimate2DenoisingStage(DenoisingStage):
    """Wan-Animate-2 in-context denoising over all reference-video clips.

    Clip 0 uses the precomputed conditioning; clips > 0 are built from the previous clip's decoded
    frames, denoised, decoded in-loop, overlap-trimmed, and assembled into the full video.
    """

    def __init__(
        self,
        *,
        transformer: nn.Module,
        scheduler: DPMSolverMultistepScheduler,
        pipeline: ComposedPipelineBase,
        vae: WanAnimate2VaeAdapter,
        image_encoder: WanAnimate2ImageEncoderAdapter,
    ) -> None:
        super().__init__(
            transformer=transformer, scheduler=scheduler, pipeline=pipeline, vae=vae
        )
        self.vae: WanAnimate2VaeAdapter = vae
        self.clip_decoder = DecodingStage(vae, pipeline=pipeline)
        # Clips > 0 are conditioned inside this stage, so it holds the image encoder too.
        self.image_encoder = image_encoder

    def _vae_encoder(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        with self.use_declared_component(component_name="vae", module=self.vae.vae):
            return self.vae.encode(videos)

    def _image_embedder(self, videos: list[torch.Tensor]) -> torch.Tensor:
        with self.use_declared_component(
            component_name="image_encoder", module=self.image_encoder.model
        ):
            return self.image_encoder.visual(videos)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(4)])
        result.add_check("raw_latent_shape", batch.raw_latent_shape, V.not_none)
        result.add_check("sigmas", batch.sigmas, V.list_not_empty)
        result.add_check(
            "prompt_embeds", batch.prompt_embeds, V.list_of_tensors_dims(2)
        )
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            V.list_of_tensors_dims(2),
        )
        result.add_check("image_embeds", batch.image_embeds, V.list_of_tensors_dims(3))
        result.add_check(
            "num_inference_steps", batch.num_inference_steps, V.positive_int
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.non_negative_float)
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check(
            WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY,
            batch.extra.get(WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY),
            _is_request_state,
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        request_state = batch.extra.get(WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY)
        result.add_check(
            "decoded_frames",
            (
                request_state.decoded_frames
                if isinstance(request_state, WanAnimate2RequestState)
                else None
            ),
            _is_decoded_frames,
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        # build_clip_conditioning runs CLIP.visual inside this stage, so image_encoder must stay
        # resident (it is CPU-offloaded by default for video tasks).
        uses = super().component_uses(server_args, stage_name)
        # The VAE adapter runs the VAE in fp32 with autocast off; do not let the manager
        # cast it to the configured vae precision on the way in.
        uses = [
            replace(use, target_dtype=None) if use.component_name == "vae" else use
            for use in uses
        ]
        if not any(use.component_name == "image_encoder" for use in uses):
            uses.append(
                ComponentUse(
                    stage_name=self._component_stage_name(stage_name),
                    component_name="image_encoder",
                )
            )
        return uses

    @torch.no_grad()
    def _denoise_single_clip(
        self,
        batch: Req,
        server_args: ServerArgs,
        request_state: WanAnimate2RequestState,
        clip_condition: WanAnimate2ClipConditioning,
    ) -> torch.Tensor:
        """Denoise one clip; return the latents with the reference-image slot dropped.
        ``batch`` is handed to set_forward_context, which the attention backends read, and
        supplies the metrics object the per-step records go to."""
        device = get_local_torch_device()

        prompt_embeddings = request_state.prompt_embeddings
        negative_prompt_embeddings = request_state.negative_prompt_embeddings
        reference_image_embeddings = request_state.reference_image_embeddings

        # Official sampler grid: sigma_0 is exactly 1.0, so sigmas are set explicitly.
        # set_timesteps also resets all solver state for the new clip.
        num_inference_steps = request_state.inputs.num_inference_steps
        sample_shift = server_args.pipeline_config.flow_shift
        self.scheduler.set_timesteps(
            sigmas=get_sampling_sigmas(num_inference_steps, sample_shift), device=device
        )

        guidance_scale = request_state.inputs.guidance_scale

        # negative_prompt_embeddings is always computed (an unset prompt encodes ""), so CFG
        # depends on the scale alone, unlike Req.validate which also needs a negative prompt.
        perform_cfg = guidance_scale > 1

        # CFG parallel: cond on cfg_rank 0, uncond on cfg_rank 1, then all-gather. Only
        # with --enable-cfg-parallel and initialized model-parallel groups.
        cfg_parallel = (
            perform_cfg
            and bool(server_args.enable_cfg_parallel)
            and model_parallel_is_initialized()
        )
        if cfg_parallel and get_classifier_free_guidance_world_size() != 2:
            logger.warning_once(
                "CFG parallel enabled but the CFG group world size is "
                f"{get_classifier_free_guidance_world_size()} (Wan-Animate-2 expects 2 "
                "for its cond/uncond split); using the sequential two-pass CFG path."
            )
            cfg_parallel = False
        branches = [
            CFGBranch(
                "conditional",
                True,
                {
                    "encoder_hidden_states": prompt_embeddings,
                    "is_unconditional": False,
                },
            )
        ]
        if perform_cfg:
            branches.append(
                CFGBranch(
                    "unconditional",
                    False,
                    {
                        "encoder_hidden_states": negative_prompt_embeddings,
                        "is_unconditional": True,
                    },
                )
            )
        cfg_policy = CFGPolicy(branches=branches)

        ctx = WanAnimate2DenoisingContext(
            scheduler=self.scheduler,
            extra_step_kwargs={},
            target_dtype=torch.bfloat16,
            autocast_enabled=True,
            timesteps=self.scheduler.timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=0,
            image_kwargs={},
            pos_cond_kwargs={},
            neg_cond_kwargs={},
            latents=clip_condition.init_noise,
            boundary_timestep=None,
            z=None,
            reserved_frames_mask=None,
            seq_len=None,
            guidance=None,
            is_warmup=batch.is_warmup,
            cfg_policy=cfg_policy,
            # DPM-Solver advances once per timestep, irrespective of solver order
            extra={"progress_step_interval": 1},
            clip_condition=clip_condition,
            reference_image_embeddings=reference_image_embeddings,
            cfg_parallel=cfg_parallel,
            guidance_scale=guidance_scale,
        )
        # clip trajectories are not part of the assembled-video output contract
        self._run_denoising_loop(ctx, batch, server_args, collect_trajectory=False)
        return ctx.latents[:, 1:]

    def _before_denoising_loop(
        self, ctx: WanAnimate2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        # activate the DiT before the reference pass, not just the first denoise step
        self.begin_declared_component_use(
            component_name="transformer", module=self.transformer
        )
        with set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ):
            ctx.reference_kv = self.transformer.build_reference_kv(
                clip_cond=ctx.clip_condition
            )

    def _prepare_step_state(
        self,
        ctx: WanAnimate2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_host: torch.Tensor,
        timesteps_cpu: torch.Tensor,
    ) -> DenoisingStepState:
        return DenoisingStepState(
            step_index=step_index,
            t_host=t_host,
            t_device=ctx.timesteps[step_index],
            t_int=int(t_host.item()),
            current_model=self.transformer,
            current_guidance_scale=ctx.guidance_scale,
            attn_metadata=None,
        )

    def _run_denoising_step(
        self,
        ctx: WanAnimate2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        def predict(branch: CFGBranch) -> torch.Tensor:
            with set_forward_context(
                current_timestep=step.step_index,
                attn_metadata=None,
                forward_batch=batch,
            ):
                if branch.name not in ctx.context_by_branch:
                    ctx.context_by_branch[branch.name] = (
                        self.transformer.prepare_context(
                            branch.kwargs["encoder_hidden_states"],
                            ctx.reference_image_embeddings,
                        )
                    )
                prediction = self.transformer(
                    hidden_states=ctx.latents,
                    timestep=step.t_device.unsqueeze(0),
                    encoder_hidden_states_image=ctx.reference_image_embeddings,
                    clip_cond=ctx.clip_condition,
                    reference_kv=ctx.reference_kv,
                    projected_context=ctx.context_by_branch[branch.name],
                    **branch.kwargs,
                )
            return prediction.contiguous() if ctx.cfg_parallel else prediction

        predictions = (
            run_cfg_parallel(ctx.cfg_policy, predict)
            if ctx.cfg_parallel
            else [predict(branch) for branch in ctx.cfg_policy.branches]
        )
        noise_pred = predictions[0]
        if len(predictions) == 2:
            noise_pred = predictions[1] + step.current_guidance_scale * (
                noise_pred - predictions[1]
            )
        ctx.latents = ctx.scheduler.step(
            noise_pred,
            step.t_device,
            ctx.latents.unsqueeze(0),
            return_dict=False,
        )[0].squeeze(0)

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        request_state = request_state_from_batch(batch)

        output_frames: list[np.ndarray] = []  # [H, W, C] uint8 frames

        # [C, num_frames_conditioning, H, W] bf16 in [-1, 1]
        prev_clip_conditioning_frames: torch.Tensor | None = None

        # The DiT runs in bf16 autocast like upstream Wan (timestep and norm math in fp32).
        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=torch.bfloat16,
            enabled=True,
        ):
            for clip_denoising_metadata in request_state.schedule:
                is_clip_0 = clip_denoising_metadata.clip_index == 0

                if not is_clip_0 and prev_clip_conditioning_frames is None:
                    raise ValueError(
                        "For all clips except clip-0 `prev_clip_conditioning_frames` must be not None."
                    )
                clip_condition = build_clip_conditioning(
                    request_state,
                    clip_denoising_metadata,
                    prev_clip_conditioning_frames,
                    vae_encoder=self._vae_encoder,
                    image_embedder=self._image_embedder,
                )

                latents = self._denoise_single_clip(
                    batch,
                    server_args,
                    request_state,
                    clip_condition=clip_condition,
                )

                denoised_latents = latents.to(dtype=torch.float32)
                with self.use_declared_component(
                    component_name="vae", module=self.vae.vae
                ):
                    # [1, C, T, H, W] fp32 in [-1, 1]
                    decoded_frames = self.clip_decoder.decode_raw(
                        denoised_latents.unsqueeze(0),
                        server_args,
                        vae_dtype=torch.float32,
                    )

                # [T, H, W, C] uint8
                output_frames_for_clip = (
                    rearrange(((decoded_frames + 1) * 127.5), "1 c t h w -> t h w c")
                    .detach()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

                # Drop the overlap frames the previous clip already emitted; 0 for clip 0.
                output_frames_for_clip = output_frames_for_clip[
                    clip_denoising_metadata.num_frames_conditioning_for_clip :
                ]
                output_frames.extend(output_frames_for_clip)

                # The next clip takes the last ``num_frames_conditioning`` frames.
                prev_clip_conditioning_frames = (
                    decoded_frames[
                        0, :, -request_state.inputs.num_frames_conditioning :
                    ]
                    .detach()
                    .to(torch.bfloat16)
                )
                # only the independent overlap frames survive into the next clip
                del decoded_frames, denoised_latents, latents, clip_condition

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Already-decoded [T, H, W, C] uint8; WanAnimate2OutputStage must not re-decode.
        request_state.decoded_frames = self._post_process_frames(
            output_frames,
            request_state.num_reference_video_frames,
            request_state.letterbox_info,
        )
        return batch

    @staticmethod
    def _post_process_frames(
        frames: list[np.ndarray],
        num_output_frames: int,
        letterbox_info: LetterboxInfo,
    ) -> np.ndarray:
        """Stack ``list[np.ndarray]`` to ``[T, H, W, C]`` uint8, trim to ``num_output_frames`` and crop the
        ``resize_by_area`` letterbox."""
        if len(frames) == 0:
            raise ValueError(
                "WanAnimate2DenoisingStage produced 0 decoded frames to post-process."
            )
        frames_np = np.stack(frames, axis=0)  # [T, H, W, C]
        frames_np = frames_np[:num_output_frames]

        return letterbox_info.crop(frames_np)


class WanAnimate2OutputStage(PipelineStage):
    """Terminal stage: wrap the already-decoded, assembled frames in an OutputBatch.

    Replaces the standard DecodingStage (which would VAE-decode a single-clip latent).
    """

    @property
    def role_affinity(self) -> RoleType:
        # The terminal slot of the standard stage layout; disaggregated deployment itself
        # is rejected by Wan_Animate_2_14B_Config.supports_disaggregation.
        return RoleType.DECODER

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        request_state = batch.extra.get(WAN_ANIMATE_2_REQUEST_STATE_EXTRA_KEY)
        result.add_check(
            "decoded_frames",
            (
                request_state.decoded_frames
                if isinstance(request_state, WanAnimate2RequestState)
                else None
            ),
            _is_decoded_frames,
        )
        return result

    def verify_output(
        self, batch: OutputBatch, server_args: ServerArgs
    ) -> VerificationResult:
        result = VerificationResult()
        result.add_check("output", batch.output, V.list_not_empty)
        return result

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        request_state = request_state_from_batch(batch)
        frames = request_state.decoded_frames
        if frames is None:
            raise ValueError(
                "WanAnimate2OutputStage: WanAnimate2DenoisingStage did not assemble decoded frames."
            )
        # frames: [T, H, W, C] uint8, consumed directly by the save path. Audio, when
        # extracted, is [1, C, L] fp32 in [-1, 1] with its sample rate; None gives a silent mp4.
        return OutputBatch(
            output=[frames],
            audio=request_state.audio,
            audio_sample_rate=request_state.audio_sample_rate,
            metrics=batch.metrics,
        )
