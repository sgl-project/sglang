# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 joint video+audio flow-match Euler denoising loop.

This model jointly denoises two coupled modalities through ONE transformer
call per branch (``transformer(hidden_states=video, hidden_states_audio=audio,
...)`` returns ``(video_velocity, audio_velocity)``), so it cannot use the
framework's generic single-tensor ``_denoise()`` loop (whose
``DenoisingContext`` carries exactly one ``latents`` tensor and calls
``scheduler.step()`` against it alone).

Following this codebase's "Native-stage subclass" pattern
(docs/docs/sglang-diffusion/support_new_models.mdx, "Choose a Pipeline
Shape") -- the same shape ``MiniMaxH3DenoisingStage``
(``model_specific_stages/minimax_h3/stages/denoising.py``) uses for its own
joint video+audio dual-modality loop -- this subclasses the shared
``DenoisingStage`` purely to inherit its lifecycle hooks (component
residency, cache-DiT mounting, torch.compile/BCG, offload-for-compile-
warmup, progress-bar, profiling) rather than bypassing them with a from-
scratch ``PipelineStage``. ``forward`` still replaces the parent's
``_denoise()`` with a custom loop -- the joint video+audio state handling
below is what is genuinely model-specific -- but every transformer call goes
through the parent's BCG/cache-dit-aware ``_call_transformer`` helper, and
CFG (when enabled) is dispatched through the shared ``CFGPolicy`` /
``run_cfg_parallel`` / ``run_two_branch_cfg_parallel`` helpers
(``runtime/distributed/cfg_policy.py``, ``cfg_parallel_utils.py``) the same
way the parent's own ``_predict_noise_with_cfg`` does, so ``--enable-cfg-
parallel`` dispatches Kandinsky6's two branches across GPUs instead of
running both sequentially on every rank. ``CFGPolicy.combine``/
``run_cfg_parallel`` already support a branch prediction being a tuple of
tensors (not just one), which is what lets a 2-output joint model like this
reuse them unmodified.

Video is advanced through the shared flow-match scheduler's ``step()``
(which owns the internal step index -- called exactly ONCE per iteration,
against video only). Audio is advanced with a manual Euler update using the
SAME per-step sigma delta the scheduler just consumed (``scheduler.sigmas[i
+ 1] - scheduler.sigmas[i]``), so the scheduler's internal step index is
never double-advanced -- unless the checkpoint is a PiFlow (distilled) one,
in which case a second, independent ``PiflowScheduler`` instance advances
audio instead (CFG is also unconditionally off for PiFlow).

PiFlow's video state is tracked in fp32 across steps, separately from the
bf16 ``video`` buffer used for the DiT call: ``PiflowScheduler.step``
explicitly returns fp32 to preserve precision across steps (matching the
diffusers reference's ``pipeline_kandinsky6_ti2va.py`` ``denoise_loop``,
which rebinds its video state to that fp32 result every step rather than
writing it back into a lower-precision buffer), so this loop keeps a
separate ``video_state`` fp32 tensor and casts only the DiT input to bf16,
mirroring how the audio (manual-Euler) path already keeps its own full-
precision accumulator. Flow-match (non-PiFlow) does not need this -- the
diffusers reference downcasts that state to the parameter dtype every step
too -- so the non-PiFlow path keeps writing ``scheduler.step()``'s result
directly into the bf16 ``video`` buffer, same as before.

Ported from FastVideo's ``Kandinsky6DenoisingStage``
(fastvideo/pipelines/stages/kandinsky6.py).
"""

from __future__ import annotations

import dataclasses
import math
from copy import deepcopy
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    run_cfg_parallel,
    run_two_branch_cfg_parallel,
)
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGBranch, CFGPolicy
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.image_encoding import (
    TAIL_COND_ACTIVE_EXTRA_KEY,
    VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE


class Kandinsky6DenoisingStage(DenoisingStage):
    """Run the Kandinsky6 joint video+audio denoising loop."""

    def __init__(self, transformer, scheduler, pipeline=None) -> None:
        super().__init__(
            transformer=transformer, scheduler=scheduler, pipeline=pipeline
        )

    def _owns_compile_warmup_lifecycle(self) -> bool:
        # forward() overrides DenoisingStage.forward (the joint video+audio
        # loop cannot reuse the parent's single-tensor _denoise()), but it
        # still wraps the loop in `_offload_for_torch_compile_warmup` itself
        # below -- same opt-in as MiniMaxH3DenoisingStage, the codebase's
        # other joint-modality denoising stage.
        return True

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        return [
            ComponentUse(
                stage_name,
                "transformer",
                phase="transformer",
                preferred_ready_after_request=True,
                memory_intensive=True,
            )
        ]

    @staticmethod
    def _text_rope_pos(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
        seq_len = int(mask.sum(1).max().item())
        return torch.arange(seq_len, device=device)

    def _call_transformer(self, transformer, **call_kwargs) -> Any:
        """Route one DiT forward through the BCG runner when it applies.

        Mirrors the parent ``DenoisingStage._predict_noise``'s BCG dispatch
        (``_maybe_get_bcg_runner`` / ``_bcg_run``) so Kandinsky6 benefits
        from breakable-CUDA-graph replay the same way single-tensor models
        do; a no-op (plain ``transformer(**call_kwargs)``) whenever BCG is
        disabled, which is the default.
        """
        runner = self._maybe_get_bcg_runner(transformer)
        if runner is not None:
            return self._bcg_run(runner, call_kwargs, transformer)
        return transformer(**call_kwargs)

    def _build_cfg_policy(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        text_rope_pos: torch.Tensor,
    ) -> CFGPolicy:
        """Build the (one- or two-branch) CFG policy for this request.

        Reuses ``server_args.pipeline_config.cfg_policy`` (the same object
        the generic ``DenoisingStage`` builds branches from) rather than
        constructing a bare ``CFGPolicy``, so a future model-specific
        ``combine()`` override is still honored. Branches are built here
        (not via ``CFGPolicy.build()``) because Kandinsky6's per-branch
        kwargs (``text_rope_pos`` alongside the embeddings) don't match the
        generic ``image_kwargs``/``pos_cond_kwargs``/``neg_cond_kwargs``
        shape, and because CFG here additionally requires negative prompt
        embeddings to actually be present (PiFlow forces
        ``do_classifier_free_guidance`` off upstream, but this stays
        defensive the same way the pre-refactor loop was).
        """
        branches = [
            CFGBranch(
                "conditional",
                True,
                {
                    "encoder_hidden_states": prompt_embeds,
                    "pooled_projections": pooled,
                    "text_rope_pos": text_rope_pos,
                },
            )
        ]
        if batch.do_classifier_free_guidance and batch.negative_prompt_embeds:
            neg_prompt_embeds = server_args.pipeline_config.get_neg_prompt_embeds(
                batch
            ).to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
            if not batch.neg_pooled_embeds:
                raise ValueError(
                    "Kandinsky6 requires CLIP negative pooled projections for CFG."
                )
            neg_pooled = batch.neg_pooled_embeds[0].to(
                device=prompt_embeds.device, dtype=prompt_embeds.dtype
            )
            if not batch.negative_attention_mask:
                raise ValueError(
                    "Kandinsky6 requires Qwen (Reason1) negative attention masks for CFG."
                )
            negative_text_rope_pos = self._text_rope_pos(
                batch.negative_attention_mask[0].to(prompt_embeds.device),
                prompt_embeds.device,
            )
            branches.append(
                CFGBranch(
                    "unconditional",
                    False,
                    {
                        "encoder_hidden_states": neg_prompt_embeds,
                        "pooled_projections": neg_pooled,
                        "text_rope_pos": negative_text_rope_pos,
                    },
                )
            )
        return dataclasses.replace(
            server_args.pipeline_config.cfg_policy, branches=branches
        )

    def _predict_joint_velocity(
        self,
        *,
        transformer,
        cfg_policy: CFGPolicy,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        video_input: torch.Tensor,
        audio_input: torch.Tensor,
        t_expand: torch.Tensor,
        visual_rope_pos: list[torch.Tensor],
        scale_factor: tuple[float, ...],
        sparse_params: dict[str, Any] | None,
        visual_token_type_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the CFG branch(es) for one step and combine into (video_vel, audio_vel).

        Dispatch mirrors the parent ``DenoisingStage._predict_noise_with_cfg``
        exactly: ``run_two_branch_cfg_parallel`` for the common 2-branch /
        2-rank case, ``run_cfg_parallel`` otherwise, or a plain sequential
        loop when CFG-parallel is disabled.
        """

        def predict_fn(branch: CFGBranch) -> tuple[torch.Tensor, torch.Tensor]:
            branch.configure_batch(batch)
            with set_forward_context(
                current_timestep=step_index, attn_metadata=None, forward_batch=batch
            ):
                model_output = self._call_transformer(
                    transformer,
                    hidden_states=video_input,
                    hidden_states_audio=audio_input,
                    timestep=t_expand,
                    visual_rope_pos=visual_rope_pos,
                    scale_factor=scale_factor,
                    sparse_params=sparse_params,
                    visual_token_type_ids=visual_token_type_ids,
                    # BCG clones tuple leaves so the next CFG replay cannot
                    # overwrite the previous branch's predictions
                    return_dict=False,
                    **branch.kwargs,
                )
            return model_output

        cfg_scale = server_args.pipeline_config.get_classifier_free_guidance_scale(
            batch, batch.guidance_scale
        )

        if server_args.enable_cfg_parallel:
            if (
                len(cfg_policy.branches) == 2
                and get_classifier_free_guidance_world_size() == 2
                and not cfg_policy.parallel_uses_serial_arithmetic
            ):
                video_vel, audio_vel = run_two_branch_cfg_parallel(
                    cfg_policy,
                    predict_fn,
                    cfg_scale,
                    batch,
                    server_args.pipeline_config,
                )
                return video_vel, audio_vel
            predictions = run_cfg_parallel(cfg_policy, predict_fn)
        else:
            predictions = [predict_fn(branch) for branch in cfg_policy.branches]

        video_vel, audio_vel = cfg_policy.combine(
            predictions,
            batch,
            cfg_scale,
            server_args.pipeline_config,
            cfg_parallel=server_args.enable_cfg_parallel,
        )
        return video_vel, audio_vel

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self._offload_for_torch_compile_warmup(batch):
            return self._run_joint_denoise_loop(batch, server_args)

    def _run_joint_denoise_loop(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.timesteps is None:
            raise ValueError("timesteps must be prepared before Kandinsky6 denoising.")
        if batch.latents is None or batch.audio_latents is None:
            raise ValueError(
                "video and audio latents must be prepared before Kandinsky6 denoising."
            )
        scheduler = batch.scheduler
        if scheduler is None:
            raise ValueError("scheduler must be set for Kandinsky6 denoising.")
        use_piflow = bool(getattr(scheduler, "is_piflow", False))
        if use_piflow and (
            not math.isfinite(batch.guidance_scale)
            or abs(batch.guidance_scale - 1.0) > 1e-6
        ):
            raise ValueError("Kandinsky6 PiFlow requires guidance_scale=1.0.")
        audio_scheduler = deepcopy(scheduler) if use_piflow else None

        pipeline_config = server_args.pipeline_config
        arch = pipeline_config.dit_config.arch_config
        # The runtime transformer already rejects attention_engine="nabla"
        # at construction time (NABLA sparse attention is not yet ported),
        # so sparse_params stays None on every currently-reachable
        # configuration. A future NABLA-capable DiT would fill this in.
        if arch.attention_engine == "nabla":
            raise NotImplementedError(
                "Kandinsky6 NABLA sparse-attention metadata construction is not implemented; "
                "the runtime transformer also rejects attention_engine='nabla' at construction "
                "time. Use attention_engine='auto' or 'sdpa'."
            )
        sparse_params = None

        device = get_local_torch_device()
        target_dtype = PRECISION_TO_TYPE[pipeline_config.dit_precision]

        # .clone(): batch.latents was created by Kandinsky6LatentPreparationStage
        # under an active torch.inference_mode() context further up the
        # pipeline, which marks it as an "inference tensor" -- such tensors
        # can only be mutated in place while still inside an inference_mode
        # context. This stage runs under plain @torch.no_grad() instead, and
        # mutates `video` in place every denoising step (video[..., :num_channels]
        # = ...), so clone once here to get an ordinary, freely-mutable tensor
        # before the loop starts. audio is never mutated in place (only
        # reassigned via out-of-place `+`), so it needs no clone.
        video = batch.latents.clone()
        audio = batch.audio_latents
        num_channels = int(arch.in_visual_dim)

        tail_cond = bool(batch.extra.get(TAIL_COND_ACTIVE_EXTRA_KEY, False))
        visual_token_type_ids = batch.extra.get(VISUAL_TOKEN_TYPE_IDS_EXTRA_KEY)

        prompt_embeds = pipeline_config.get_pos_prompt_embeds(batch).to(
            device=device, dtype=target_dtype
        )
        if not batch.pooled_embeds:
            raise ValueError("Kandinsky6 requires CLIP pooled projections.")
        pooled = batch.pooled_embeds[0].to(device=device, dtype=target_dtype)
        if not batch.prompt_attention_mask:
            raise ValueError(
                "Kandinsky6 requires Qwen (Reason1) prompt attention masks."
            )
        text_rope_pos = self._text_rope_pos(
            batch.prompt_attention_mask[0].to(device), device
        )

        cfg_policy = self._build_cfg_policy(
            batch,
            server_args,
            prompt_embeds=prompt_embeds,
            pooled=pooled,
            text_rope_pos=text_rope_pos,
        )

        height = int(batch.height)
        width = int(batch.width)
        spatial_ratio = pipeline_config.vae_config.arch_config.spatial_compression_ratio
        patch_size = arch.patch_size

        # Constant per-request geometry -- computed once outside the loop,
        # not per-step.
        num_video_frames = video.shape[1] - (1 if tail_cond else 0)
        t_positions = torch.arange(num_video_frames, device=device)
        if tail_cond:
            # The appended reference frame reuses T-position 0's rope row
            # (RoPE3D is a pure position -> table lookup, so duplicating the
            # first row is equivalent to the diffusers reference's
            # ``torch.cat([rope, rope[:1]])``).
            t_positions = torch.cat([t_positions, t_positions.new_zeros(1)])
        visual_rope_pos = [
            t_positions,
            torch.arange(height // spatial_ratio // patch_size[1], device=device),
            torch.arange(width // spatial_ratio // patch_size[2], device=device),
        ]
        # Fixed per-checkpoint RoPE frequency scaling read from the DiT's
        # arch config (transformer/config.json's "scale_factor", both real
        # Pro checkpoints ship [1.0, 2.0, 2.0]) -- matches the diffusers
        # reference, which resolves this once in
        # ``Kandinsky6TI2VAPipeline.__init__`` and reuses it for every
        # request regardless of the request's own height/width. NOT a
        # function of the request's resolution.
        scale_factor = arch.scale_factor
        image_latent = batch.image_latent

        total_steps = int(batch.timesteps.shape[0])
        self._maybe_enable_cache_dit_and_torch_compile(total_steps, batch)

        # PiFlow's noisy video state is tracked in fp32 across steps here,
        # separately from the `video` buffer's bf16 storage dtype -- see the
        # module docstring. Always a Tensor (not `Tensor | None`) even on the
        # flow-Euler path, where it is simply unused: the initial cast is one
        # cheap op, and keeping it non-Optional avoids re-narrowing it after
        # every `if use_piflow:` branch re-entry in the loop below.
        video_state = video[..., :num_channels].to(torch.float32)

        with self.use_declared_component(
            component_name="transformer", module=self.transformer
        ) as transformer:
            assert transformer is not None
            self.transformer = transformer

            with self.progress_bar(
                total=total_steps, batch=batch, desc="Kandinsky6 Denoising"
            ) as progress_bar:
                for i, timestep in enumerate(batch.timesteps):
                    t_expand = (
                        timestep.unsqueeze(0)
                        .repeat(video.shape[0])
                        .to(device=device, dtype=target_dtype)
                    )

                    if use_piflow:
                        video_input = video_state.to(dtype=target_dtype)
                        if video.shape[-1] > num_channels:
                            video_input = torch.cat(
                                [
                                    video_input,
                                    video[..., num_channels:].to(dtype=target_dtype),
                                ],
                                dim=-1,
                            )
                    else:
                        video_input = video.to(dtype=target_dtype)

                    video_vel, audio_vel = self._predict_joint_velocity(
                        transformer=transformer,
                        cfg_policy=cfg_policy,
                        batch=batch,
                        server_args=server_args,
                        step_index=i,
                        video_input=video_input,
                        audio_input=audio.to(dtype=target_dtype),
                        t_expand=t_expand,
                        visual_rope_pos=visual_rope_pos,
                        scale_factor=scale_factor,
                        sparse_params=sparse_params,
                        visual_token_type_ids=visual_token_type_ids,
                    )

                    # scheduler.step() is called exactly once per iteration,
                    # against video only -- it owns the internal step index.
                    if use_piflow:
                        video_state = scheduler.step(
                            video_vel, timestep, video_state, return_dict=False
                        )[0]
                        video[..., :num_channels] = video_state.to(video.dtype)
                    else:
                        video[..., :num_channels] = scheduler.step(
                            video_vel,
                            timestep,
                            video[..., :num_channels],
                            return_dict=False,
                        )[0]
                    if tail_cond:
                        ref_frame = image_latent.to(
                            device=video.device, dtype=video.dtype
                        )
                        video[:, -1:, :, :, :num_channels] = ref_frame
                        if use_piflow:
                            video_state[:, -1:] = ref_frame.to(torch.float32)

                    # Manual Euler update for audio, using the SAME per-step
                    # sigma delta the scheduler.step() call above just
                    # consumed for video.
                    if use_piflow:
                        audio = audio_scheduler.step(
                            audio_vel, timestep, audio, return_dict=False
                        )[0]
                    else:
                        step_size = scheduler.sigmas[i + 1] - scheduler.sigmas[i]
                        audio = (
                            audio
                            + step_size.to(device=audio.device, dtype=audio.dtype)
                            * audio_vel
                        )

                    if progress_bar is not None:
                        progress_bar.update()

        video = video[..., :num_channels]
        if tail_cond:
            video = video[:, :-1]

        batch.latents = video
        batch.audio_latents = audio
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(3)]
        )
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check(
            "audio_latents", batch.audio_latents, [V.is_tensor, V.with_dims(3)]
        )
        return result
