from __future__ import annotations

from typing import Any, Literal, overload

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs


@overload
def longcat_optimized_cfg(
    *,
    noise_cond: torch.Tensor,
    noise_uncond: torch.Tensor,
    guidance_scale: float,
    return_before_sign_flip: Literal[False] = ...,
    **_unused: Any,
) -> torch.Tensor: ...


@overload
def longcat_optimized_cfg(
    *,
    noise_cond: torch.Tensor,
    noise_uncond: torch.Tensor,
    guidance_scale: float,
    return_before_sign_flip: Literal[True],
    **_unused: Any,
) -> tuple[torch.Tensor, torch.Tensor]: ...


def longcat_optimized_cfg(
    *,
    noise_cond: torch.Tensor,
    noise_uncond: torch.Tensor,
    guidance_scale: float,
    return_before_sign_flip: bool = False,
    **_unused: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    reduce_dims = tuple(range(1, noise_uncond.ndim))
    denominator = noise_uncond.square().sum(dim=reduce_dims, keepdim=True)
    numerator = (noise_cond * noise_uncond).sum(dim=reduce_dims, keepdim=True)
    st_star = numerator / denominator.clamp_min(1e-12)
    projected_uncond = noise_uncond * st_star
    noise = projected_uncond + guidance_scale * (noise_cond - projected_uncond)
    # Negate for scheduler compatibility: FlowMatchEulerDiscreteScheduler.step()
    # adds model_output to the sample, so the model must supply -velocity.
    # This matches the prototype implementation in:
    # LongCat-Video/longcat_video/pipeline_longcat_video.py (comment: "negate for scheduler compatibility")
    noise = -noise
    if return_before_sign_flip:
        return -noise, noise
    return noise


class LongCatVideoDenoisingStage(DenoisingStage):
    @staticmethod
    def _assert_single_encoder(value: Any) -> Any:
        """Unwrap encoder output kwargs, asserting exactly one text encoder is present.

        LongCat T2V MVP only supports a single text encoder. This method extracts
        the single element from list/tuple encoder outputs and raises if multiple
        encoders are found.
        """
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError("LongCat T2V MVP expects exactly one text encoder.")
            return value[0]
        return value

    def _predict_noise(
        self,
        current_model,
        latent_model_input,
        timestep,
        target_dtype,
        guidance: torch.Tensor,
        **kwargs,
    ):
        # `guidance` is the embedded CFG scale tensor used by models like FLUX.
        # LongCatVideoTransformer3DModel does not accept a `guidance` parameter —
        # it uses an explicit CFG pass via longcat_optimized_cfg instead.
        # The parameter is accepted here to match the DenoisingStage signature
        # but is intentionally not forwarded to the model.
        kwargs = {key: self._assert_single_encoder(value) for key, value in kwargs.items()}
        return current_model(
            hidden_states=latent_model_input,
            timestep=timestep,
            **kwargs,
        )

    def _predict_noise_with_cfg(
        self,
        current_model,
        latent_model_input: torch.Tensor,
        timestep,
        batch,
        timestep_index: int,
        attn_metadata,
        target_dtype,
        current_guidance_scale,
        image_kwargs: dict[str, Any],
        pos_cond_kwargs: dict[str, Any],
        neg_cond_kwargs: dict[str, Any],
        server_args,
        guidance,
        latents,
    ) -> torch.Tensor:
        # This method intentionally does NOT call super()._predict_noise_with_cfg().
        # The parent class (DenoisingStage) supports: guidance_rescale, cfg_normalization,
        # CFG parallelism, and set_forward_context/attn_metadata wiring.
        # LongCat T2V MVP disables all of these (raises NotImplementedError for unsupported
        # features) and uses a custom batched CFG path that concatenates pos/neg inputs
        # into a single forward pass for efficiency.
        # If any parent features are needed in the future, refactor to call super() and extend.
        if image_kwargs:
            raise NotImplementedError("LongCat T2V MVP does not support image inputs.")
        if batch.guidance_rescale > 0.0:
            raise NotImplementedError(
                "LongCat T2V MVP does not support guidance_rescale."
            )
        if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
            raise NotImplementedError(
                "LongCat T2V MVP does not support cfg_normalization."
            )

        pos_cond_kwargs = {
            key: self._assert_single_encoder(value) for key, value in pos_cond_kwargs.items()
        }
        if not batch.do_classifier_free_guidance:
            noise_pred = self._predict_noise(
                current_model=current_model,
                latent_model_input=latent_model_input,
                timestep=timestep,
                target_dtype=target_dtype,
                guidance=guidance,
                **pos_cond_kwargs,
            )
            noise_pred = server_args.pipeline_config.slice_noise_pred(
                noise_pred, latents
            )
            # Negate for scheduler compatibility — same sign convention as longcat_optimized_cfg.
            return -noise_pred

        neg_cond_kwargs = {
            key: self._assert_single_encoder(value) for key, value in neg_cond_kwargs.items()
        }
        latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
        if isinstance(timestep, torch.Tensor):
            timestep = torch.cat([timestep, timestep], dim=0)

        encoder_hidden_states = torch.cat(
            [
                neg_cond_kwargs["encoder_hidden_states"],
                pos_cond_kwargs["encoder_hidden_states"],
            ],
            dim=0,
        )
        encoder_attention_mask = torch.cat(
            [
                neg_cond_kwargs["encoder_attention_mask"],
                pos_cond_kwargs["encoder_attention_mask"],
            ],
            dim=0,
        )

        noise_pred = current_model(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        noise_pred = server_args.pipeline_config.slice_noise_pred(noise_pred, latents)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        return self._combine_cfg_serial(
            batch,
            noise_pred_cond,
            noise_pred_uncond,
            current_guidance_scale,
        )

    def _combine_cfg_serial(
        self,
        batch,
        noise_pred_cond: torch.Tensor,
        noise_pred_uncond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        if batch.guidance_rescale > 0.0:
            raise NotImplementedError(
                "LongCat T2V MVP does not support guidance_rescale."
            )
        if batch.cfg_normalization and float(batch.cfg_normalization) > 0:
            raise NotImplementedError(
                "LongCat T2V MVP does not support cfg_normalization."
            )
        return longcat_optimized_cfg(
            noise_cond=noise_pred_cond,
            noise_uncond=noise_pred_uncond,
            guidance_scale=cfg_scale,
        )

    def _combine_cfg_parallel(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("LongCat T2V MVP does not support CFG parallelism.")

    def _run_denoising_step(
        self,
        ctx: DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Override to skip scale_model_input, which FlowMatchEulerDiscreteScheduler
        does not implement. For flow matching, scale_model_input is a no-op."""
        latent_model_input = ctx.latents.to(ctx.target_dtype)

        timestep = self.expand_timestep_before_forward(
            batch,
            server_args,
            step.t_device,
            ctx.target_dtype,
            ctx.seq_len,
            ctx.reserved_frames_mask,
        )

        # Skip ctx.scheduler.scale_model_input() — not implemented by
        # FlowMatchEulerDiscreteScheduler and is a no-op for flow matching.

        noise_pred = self._predict_noise_with_cfg(
            current_model=step.current_model,
            latent_model_input=latent_model_input,
            timestep=timestep,
            batch=batch,
            timestep_index=step.step_index,
            attn_metadata=step.attn_metadata,
            target_dtype=ctx.target_dtype,
            current_guidance_scale=step.current_guidance_scale,
            image_kwargs=ctx.image_kwargs,
            pos_cond_kwargs=ctx.pos_cond_kwargs,
            neg_cond_kwargs=ctx.neg_cond_kwargs,
            server_args=server_args,
            guidance=ctx.guidance,
            latents=ctx.latents,
        )

        ctx.latents = ctx.scheduler.step(
            model_output=noise_pred,
            timestep=step.t_device,
            sample=ctx.latents,
            **ctx.extra_step_kwargs,
            return_dict=False,
        )[0]

        ctx.latents = self.post_forward_for_ti2v_task(
            batch,
            server_args,
            ctx.reserved_frames_mask,
            ctx.latents,
            ctx.z,
        )
