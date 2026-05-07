import math
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.cfg_parallel_utils import (
    dispatch_branches,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_gather,
    cfg_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    clone_scheduler_runtime,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)

LTX23_RES2S_STEP_NOISE_SEED = -1
LTX23_RES2S_SUBSTEP_NOISE_SEED = 9999


@dataclass(slots=True)
class LTX2DenoisingContext(DenoisingContext):
    """Loop-scoped denoising state for joint LTX-2 video and audio generation."""

    audio_latents: torch.Tensor | None = None
    audio_scheduler: object | None = None
    is_ltx23_variant: bool = False
    use_ltx23_legacy_one_stage: bool = False
    replicate_audio_for_sp: bool = False
    stage: str = "one_stage"
    latent_num_frames_for_model: int = 0
    latent_height: int = 0
    latent_width: int = 0
    denoise_mask: torch.Tensor | None = None
    clean_latent: torch.Tensor | None = None
    last_denoised_video: torch.Tensor | None = None
    last_denoised_audio: torch.Tensor | None = None
    trajectory_audio_latents: list[torch.Tensor] = field(default_factory=list)
    use_native_hq_res2s_sde_noise: bool = False
    use_ltx23_hq_timestep_semantics: bool = False
    res2s_step_noise_generator: torch.Generator | None = None
    res2s_substep_noise_generator: torch.Generator | None = None


@dataclass(slots=True)
class LTX2ModelInputs:
    latent_model_input: torch.Tensor
    audio_latent_model_input: torch.Tensor
    audio_num_frames_latent: int
    video_coords: torch.Tensor | None
    audio_coords: torch.Tensor | None
    timestep_video: torch.Tensor
    timestep_audio: torch.Tensor
    prompt_timestep_video: torch.Tensor | None
    prompt_timestep_audio: torch.Tensor | None
    video_self_attention_mask: torch.Tensor | None
    audio_self_attention_mask: torch.Tensor | None
    a2v_cross_attention_mask: torch.Tensor | None
    v2a_cross_attention_mask: torch.Tensor | None


@dataclass(slots=True)
class LTX2GuidancePassSpec:
    name: str
    encoder_hidden_states: torch.Tensor
    audio_encoder_hidden_states: torch.Tensor
    encoder_attention_mask: torch.Tensor | None
    skip_video_self_attn_blocks: tuple[int, ...] = ()
    skip_audio_self_attn_blocks: tuple[int, ...] = ()
    disable_a2v_cross_attn: bool = False
    disable_v2a_cross_attn: bool = False


class LTX2DenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    _LTX2_BATCH_REPEATABLE_KWARG_KEYS = (
        "hidden_states",
        "audio_hidden_states",
        "timestep",
        "audio_timestep",
        "prompt_timestep",
        "audio_prompt_timestep",
        "video_coords",
        "audio_coords",
        "video_self_attention_mask",
        "audio_self_attention_mask",
        "a2v_cross_attention_mask",
        "v2a_cross_attention_mask",
        "encoder_attention_mask",
        "audio_encoder_attention_mask",
    )

    def __init__(
        self,
        transformer,
        scheduler,
        vae=None,
        *,
        sampler_name: str = "euler",
        **kwargs,
    ):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.sampler_name = sampler_name

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

    @property
    def parallelism_type(self) -> StageParallelismType:
        if self.server_args.enable_cfg_parallel:
            return StageParallelismType.CFG_PARALLEL
        return StageParallelismType.REPLICATED

    @staticmethod
    def _combine_cfg_parallel_av(
        video: torch.Tensor,
        audio: torch.Tensor,
        guidance_scale: float,
        cfg_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """All-reduce video and audio predictions across CFG ranks.

        Rank 0 (cond) contributes ``guidance_scale * pred``.
        Rank 1 (uncond) contributes ``(1 - guidance_scale) * pred``.
        Higher CFG ranks, if configured for multi-pass guidance, contribute
        zeros on the two-branch path.
        The sum reconstructs ``uncond + guidance_scale * (cond - uncond)``.
        """
        if cfg_rank == 0:
            video_partial = guidance_scale * video
            audio_partial = guidance_scale * audio
        elif cfg_rank == 1:
            video_partial = (1.0 - guidance_scale) * video
            audio_partial = (1.0 - guidance_scale) * audio
        else:
            video_partial = torch.zeros_like(video)
            audio_partial = torch.zeros_like(audio)
        return (
            cfg_model_parallel_all_reduce(video_partial),
            cfg_model_parallel_all_reduce(audio_partial),
        )

    def _run_legacy_one_stage_multi_branch_cfg_parallel(
        self,
        *,
        base_model_kwargs: dict[str, object],
        ctx: "LTX2DenoisingContext",
        step: "DenoisingStepState",
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        negative_encoder_hidden_states: torch.Tensor,
        negative_audio_encoder_hidden_states: torch.Tensor,
        negative_encoder_attention_mask: torch.Tensor | None,
        need_perturbed: bool,
        need_modality: bool,
        stage1_guider_params: dict[str, object],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Multi-branch CFG parallel for the legacy LTX-2.3 one-stage path.

        Distributes up to 4 forward passes (cond, neg, perturbed, modality)
        across CFG ranks via round-robin.  Each rank runs only its assigned
        passes, then an all-gather collects every output so all ranks can
        compute the guidance combination locally.
        """
        cfg_rank = get_classifier_free_guidance_rank()
        cfg_world_size = get_classifier_free_guidance_world_size()

        # Build kwargs for every pass in canonical order.
        all_passes: list[tuple[str, dict[str, object]]] = [
            (
                "cond",
                self._build_ltx2_model_kwargs(
                    ctx,
                    base_model_kwargs,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                ),
            ),
            (
                "neg",
                self._build_ltx2_model_kwargs(
                    ctx,
                    base_model_kwargs,
                    encoder_hidden_states=negative_encoder_hidden_states,
                    audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                    encoder_attention_mask=negative_encoder_attention_mask,
                ),
            ),
        ]
        if need_perturbed:
            all_passes.append(
                (
                    "perturbed",
                    self._build_ltx2_model_kwargs(
                        ctx,
                        base_model_kwargs,
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        skip_video_self_attn_blocks=tuple(
                            stage1_guider_params["video_stg_blocks"]
                        ),
                        skip_audio_self_attn_blocks=tuple(
                            stage1_guider_params["audio_stg_blocks"]
                        ),
                    ),
                )
            )
        if need_modality:
            all_passes.append(
                (
                    "modality",
                    self._build_ltx2_model_kwargs(
                        ctx,
                        base_model_kwargs,
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        disable_a2v_cross_attn=True,
                        disable_v2a_cross_attn=True,
                    ),
                )
            )

        pass_names = [name for name, _ in all_passes]
        n_passes = len(pass_names)
        assignments = dispatch_branches(n_passes, cfg_world_size)
        my_indices = assignments[cfg_rank]
        max_local = max(len(a) for a in assignments)

        local_videos: list[torch.Tensor] = []
        local_audios: list[torch.Tensor] = []

        indices_to_run = my_indices if my_indices else [0]
        with set_forward_context(
            current_timestep=step.step_index, attn_metadata=step.attn_metadata
        ):
            for idx in indices_to_run:
                _, kwargs = all_passes[idx]
                v, a = step.current_model(**kwargs)
                local_videos.append(v.float())
                local_audios.append(a.float())

        if not my_indices:
            # This rank has no real branch, but it still needs tensor shapes for all-gather.
            # The dummy branch above provides the shapes; zeros keep this rank from contributing.
            local_videos = [torch.zeros_like(local_videos[0])]
            local_audios = [torch.zeros_like(local_audios[0])]

        # Pad to max_local for unbalanced cases (n_passes not divisible by n_ranks).
        while len(local_videos) < max_local:
            local_videos.append(torch.zeros_like(local_videos[0]))
            local_audios.append(torch.zeros_like(local_audios[0]))

        # Stack -> [max_local, B, ...], flatten to [max_local*B, ...] for all-gather.
        local_v = torch.stack(local_videos, dim=0)
        local_a = torch.stack(local_audios, dim=0)
        B = local_v.shape[1]
        local_v_flat = local_v.reshape(max_local * B, *local_v.shape[2:])
        local_a_flat = local_a.reshape(max_local * B, *local_a.shape[2:])

        # All-gather along batch dim -> [cfg_world_size * max_local * B, ...].
        all_v_flat = cfg_model_parallel_all_gather(local_v_flat, dim=0)
        all_a_flat = cfg_model_parallel_all_gather(local_a_flat, dim=0)

        # Reshape to [cfg_world_size, max_local, B, ...].
        all_v = all_v_flat.reshape(cfg_world_size, max_local, B, *all_v_flat.shape[1:])
        all_a = all_a_flat.reshape(cfg_world_size, max_local, B, *all_a_flat.shape[1:])

        # Branch i was run by rank (i % cfg_world_size) at slot (i // cfg_world_size).
        return {
            name: (
                all_v[i % cfg_world_size, i // cfg_world_size],
                all_a[i % cfg_world_size, i // cfg_world_size],
            )
            for i, name in enumerate(pass_names)
        }

    @staticmethod
    def _get_video_latent_num_frames_for_model(
        batch: Req, server_args: ServerArgs, latents: torch.Tensor
    ) -> int:
        """Return the latent-frame length the DiT model should see.

        - If video latents were time-sharded for SP and are packed as token latents
          ([B, S, D]), the model only sees the local shard and must use the local
          latent-frame count (stored on the batch during SP sharding).
        - Otherwise, fall back to the global latent-frame count inferred from the
          requested output frames and the VAE temporal compression ratio.
        """
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        is_token_latents = isinstance(latents, torch.Tensor) and latents.ndim == 3

        if did_sp_shard and is_token_latents:
            if not hasattr(batch, "sp_video_latent_num_frames"):
                raise ValueError(
                    "SP-sharded LTX2 token latents require `batch.sp_video_latent_num_frames` "
                    "to be set by `LTX2PipelineConfig.shard_latents_for_sp()`."
                )
            return int(batch.sp_video_latent_num_frames)

        pc = server_args.pipeline_config
        return int(
            (batch.num_frames - 1)
            // int(pc.vae_config.arch_config.temporal_compression_ratio)
            + 1
        )

    @staticmethod
    def _truncate_sp_padded_token_latents(
        batch: Req, latents: torch.Tensor
    ) -> torch.Tensor:
        """Remove token padding introduced by SP time-sharding (if applicable)."""
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard or not (
            isinstance(latents, torch.Tensor) and latents.ndim == 3
        ):
            return latents

        raw_shape = getattr(batch, "raw_latent_shape", None)
        if not (isinstance(raw_shape, tuple) and len(raw_shape) == 3):
            return latents

        orig_s = int(raw_shape[1])
        cur_s = int(latents.shape[1])
        if cur_s == orig_s:
            return latents
        if cur_s < orig_s:
            raise ValueError(
                f"Unexpected gathered token-latents seq_len {cur_s} < original seq_len {orig_s}."
            )
        return latents[:, :orig_s, :].contiguous()

    def _maybe_enable_cache_dit(self, num_inference_steps: int, batch: Req) -> None:
        """Disable cache-dit for TI2V-style requests to avoid stale activations.

        NOTE: base denoising stage calls this hook with (num_inference_steps, batch).
        """
        if getattr(self, "_disable_cache_dit_for_request", False):
            return
        return super()._maybe_enable_cache_dit(num_inference_steps, batch)

    def _get_ltx2_stage1_guider_params(
        self, batch: Req, server_args: ServerArgs, stage: str
    ) -> dict[str, object] | None:
        if stage != "stage1":
            return None
        return batch.extra.get("ltx2_stage1_guider_params")

    @staticmethod
    def _ltx2_should_skip_step(step_index: int, skip_step: int) -> bool:
        if skip_step == 0:
            return False
        return step_index % (skip_step + 1) != 0

    @staticmethod
    def _ltx2_apply_rescale(
        cond: torch.Tensor, pred: torch.Tensor, rescale_scale: float
    ) -> torch.Tensor:
        if rescale_scale == 0.0:
            return pred
        factor = cond.std() / pred.std()
        factor = rescale_scale * factor + (1.0 - rescale_scale)
        return pred * factor

    @classmethod
    def _ltx2_combine_guided_x0_parallel(
        cls,
        *,
        latents: torch.Tensor,
        local_velocities: dict[str, torch.Tensor],
        sigma: float | torch.Tensor,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        modality_scale: float,
    ) -> torch.Tensor:
        """Combine stage-1 guidance passes that were split across CFG ranks.

        Each pass is one model forward with a different conditioning setup:
        positive prompt, negative prompt, attention-disabled perturbation, or
        audio/video cross-attention disabled. A rank only owns some passes, so
        it contributes weighted x0 terms for those passes and all-reduce
        reconstructs the full guided x0 on every rank.
        """
        coefficients = {
            "cond": cfg_scale + stg_scale + modality_scale - 1.0,
            "neg": 1.0 - cfg_scale,
            "perturbed": -stg_scale,
            "modality": 1.0 - modality_scale,
        }
        first_velocity = next(iter(local_velocities.values()))
        template = cls._ltx2_velocity_to_x0(latents, first_velocity, sigma)
        cond_partial = torch.zeros_like(template)
        pred_partial = torch.zeros_like(template)

        for name, velocity in local_velocities.items():
            denoised = cls._ltx2_velocity_to_x0(latents, velocity, sigma)
            if name == "cond":
                cond_partial = cond_partial + denoised
            pred_partial = pred_partial + denoised * coefficients[name]

        cond = cfg_model_parallel_all_reduce(cond_partial)
        pred = cfg_model_parallel_all_reduce(pred_partial)
        return cls._ltx2_apply_rescale(cond, pred, rescale_scale)

    @staticmethod
    def _ltx2_channelwise_normalize(noise: torch.Tensor) -> torch.Tensor:
        return noise.sub_(noise.mean(dim=(-2, -1), keepdim=True)).div_(
            noise.std(dim=(-2, -1), keepdim=True)
        )

    @classmethod
    def _ltx2_res2s_new_noise(
        cls,
        reference_tensor: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        noise = torch.randn(
            reference_tensor.shape,
            generator=generator,
            dtype=torch.float64,
            device=reference_tensor.device,
        )
        noise = (noise - noise.mean()) / noise.std()
        return cls._ltx2_channelwise_normalize(noise)

    @staticmethod
    def _ltx2_init_res2s_noise_generators(ctx: LTX2DenoisingContext) -> None:
        reference_tensor = (
            ctx.latents if isinstance(ctx.latents, torch.Tensor) else ctx.audio_latents
        )
        if reference_tensor is None:
            raise ValueError("LTX-2 res2s requires video or audio latents.")
        device = reference_tensor.device
        ctx.res2s_step_noise_generator = torch.Generator(device=device).manual_seed(
            LTX23_RES2S_STEP_NOISE_SEED
        )
        ctx.res2s_substep_noise_generator = torch.Generator(device=device).manual_seed(
            LTX23_RES2S_SUBSTEP_NOISE_SEED
        )

    @classmethod
    def _ltx2_res2s_noise_like(
        cls,
        reference_tensor: torch.Tensor,
        ctx: LTX2DenoisingContext,
        *,
        substep: bool,
        batch: Req | None = None,
        is_audio: bool = False,
    ) -> torch.Tensor:
        generator = (
            ctx.res2s_substep_noise_generator
            if substep
            else ctx.res2s_step_noise_generator
        )
        if generator is None:
            raise ValueError("LTX-2 res2s noise generator was not initialized.")
        if batch is not None and get_sp_world_size() > 1 and reference_tensor.ndim == 3:
            full_shape = (
                getattr(batch, "raw_audio_latent_shape", None)
                if is_audio
                else getattr(batch, "raw_latent_shape", None)
            )
            did_shard = (
                getattr(batch, "did_sp_shard_audio_latents", False)
                if is_audio
                else getattr(batch, "did_sp_shard_latents", False)
            )
            if full_shape is not None and did_shard:
                # HQ res2s normalizes SDE noise over the complete latent. If
                # each SP rank normalizes only its local slice, the sampler
                # follows a different trajectory. Recreate the same full noise
                # on every rank, then keep the time slice owned by this rank.
                full_noise = cls._ltx2_res2s_new_noise(
                    torch.empty(
                        tuple(int(dim) for dim in full_shape),
                        device=reference_tensor.device,
                        dtype=reference_tensor.dtype,
                    ),
                    generator,
                )
                if is_audio:
                    start = int(batch.sp_audio_start_frame)
                    end = start + int(batch.sp_audio_latent_num_frames)
                else:
                    start = int(batch.sp_video_start_frame) * int(
                        batch.sp_video_tokens_per_frame
                    )
                    end = start + int(reference_tensor.shape[1])
                sliced = full_noise[:, start : min(end, int(full_noise.shape[1])), :]
                if int(sliced.shape[1]) < int(reference_tensor.shape[1]):
                    pad_len = int(reference_tensor.shape[1]) - int(sliced.shape[1])
                    pad = torch.zeros(
                        (sliced.shape[0], pad_len, sliced.shape[2]),
                        device=sliced.device,
                        dtype=sliced.dtype,
                    )
                    sliced = torch.cat([sliced, pad], dim=1)
                return sliced.to(dtype=reference_tensor.dtype)
        return cls._ltx2_res2s_new_noise(reference_tensor, generator).to(
            dtype=reference_tensor.dtype
        )

    @staticmethod
    def _ltx2_apply_clean_latent_mask(
        latents: torch.Tensor,
        ctx: LTX2DenoisingContext,
    ) -> torch.Tensor:
        if ctx.denoise_mask is None or ctx.clean_latent is None:
            return latents
        return (
            latents.float() * ctx.denoise_mask
            + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
        ).to(dtype=latents.dtype)

    @staticmethod
    def _ltx2_phi_1(neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 1.0 + 0.5 * neg_h + (neg_h * neg_h) / 6.0
        return torch.where(small, series, torch.expm1(neg_h) / neg_h)

    @classmethod
    def _ltx2_phi_2(cls, neg_h: torch.Tensor) -> torch.Tensor:
        small = neg_h.abs() < 1e-4
        series = 0.5 + neg_h / 6.0 + (neg_h * neg_h) / 24.0
        exact = (torch.expm1(neg_h) - neg_h) / (neg_h * neg_h)
        return torch.where(small, series, exact)

    @classmethod
    def _ltx2_get_res2s_coefficients(
        cls, h: torch.Tensor, c2: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a21 = c2 * cls._ltx2_phi_1(-h * c2)
        b2 = cls._ltx2_phi_2(-h) / c2
        b1 = cls._ltx2_phi_1(-h) - b2
        return a21, b1, b2

    @staticmethod
    def _ltx2_phi_scalar(j: int, neg_h: float) -> float:
        if abs(neg_h) < 1e-10:
            return 1.0 / math.factorial(j)
        remainder = sum(neg_h**k / math.factorial(k) for k in range(j))
        return (math.exp(neg_h) - remainder) / (neg_h**j)

    @classmethod
    def _ltx2_get_res2s_coefficients_scalar(
        cls, h: float, c2: float = 0.5
    ) -> tuple[float, float, float]:
        a21 = c2 * cls._ltx2_phi_scalar(1, -h * c2)
        b2 = cls._ltx2_phi_scalar(2, -h) / c2
        b1 = cls._ltx2_phi_scalar(1, -h) - b2
        return a21, b1, b2

    @staticmethod
    def _ltx2_res2s_step_size_scalar(
        sigma: torch.Tensor, sigma_next: torch.Tensor
    ) -> float:
        return float(
            (
                -torch.log(
                    sigma_next.detach().double().cpu() / sigma.detach().double().cpu()
                )
            ).item()
        )

    @staticmethod
    def _ltx2_get_sde_coeff(
        sigma_next: torch.Tensor,
        *,
        sigma_up: torch.Tensor | None = None,
        sigma_down: torch.Tensor | None = None,
        sigma_max: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if sigma_down is not None:
            alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
            sigma_up = (sigma_next**2 - sigma_down**2 * alpha_ratio**2).clamp(
                min=0
            ) ** 0.5
        elif sigma_up is not None:
            sigma_up.clamp_(max=sigma_next * 0.9999)
            sigmax = sigma_max if sigma_max is not None else torch.ones_like(sigma_next)
            sigma_signal = sigmax - sigma_next
            sigma_residual = (sigma_next**2 - sigma_up**2).clamp(min=0) ** 0.5
            alpha_ratio = sigma_signal + sigma_residual
            sigma_down = sigma_residual / alpha_ratio
        else:
            alpha_ratio = torch.ones_like(sigma_next)
            sigma_down = sigma_next
            sigma_up = torch.zeros_like(sigma_next)

        sigma_up = torch.nan_to_num(
            sigma_up if sigma_up is not None else torch.zeros_like(sigma_next), 0.0
        )
        nan_mask = torch.isnan(sigma_down)
        sigma_down[nan_mask] = sigma_next[nan_mask].to(sigma_down.dtype)
        alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)
        return alpha_ratio, sigma_down, sigma_up

    @classmethod
    def _ltx2_res2s_sde_step(
        cls,
        *,
        sample: torch.Tensor,
        denoised_sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        noise: torch.Tensor,
        eta: float = 0.5,
        terminal: bool = False,
    ) -> torch.Tensor:
        # The caller decides terminal steps from Python scalars before entering
        # this helper. Keep that branch on host to avoid a CUDA bool sync in
        # every res2s SDE update.
        if terminal:
            return denoised_sample.to(dtype=sample.dtype)
        alpha_ratio, sigma_down, sigma_up = cls._ltx2_get_sde_coeff(
            sigma_next,
            sigma_up=sigma_next * eta,
        )
        eps_next = (sample - denoised_sample) / (sigma - sigma_next)
        denoised_next = sample - sigma * eps_next
        x_noised = (
            alpha_ratio * (denoised_next + sigma_down * eps_next) + sigma_up * noise
        )
        return x_noised.to(dtype=sample.dtype)

    def _ltx2_stage2_res2s_step(
        self,
        *,
        ctx: "LTX2DenoisingContext",
        batch: Req,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        model_video_velocity: torch.Tensor,
        model_audio_velocity: torch.Tensor,
        model_video_timestep: torch.Tensor | None,
        model_audio_timestep: torch.Tensor | None,
        midpoint_model_call,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """res2s RK2 step for unguided stage-2 refinement (HQ pipeline).

        Converts velocity -> x_0 denoised estimates, runs the official res2s
        update (midpoint SDE, bongmath anchor refinement, midpoint re-eval,
        final RK2 combination with SDE noise). Mirrors the guided stage-1 res2s
        math but without CFG/STG (stage-2 HQ uses the simple CFG path).
        """
        sigma_val = float(sigma.item())
        sigma_next_val = float(sigma_next.item())

        if sigma_val == 0.0:
            denoised_video = ctx.latents.float()
            denoised_audio = ctx.audio_latents.float()
        else:
            video_sigma_for_x0 = (
                model_video_timestep
                if ctx.use_ltx23_hq_timestep_semantics
                and model_video_timestep is not None
                else sigma
            )
            audio_sigma_for_x0 = (
                model_audio_timestep
                if ctx.use_ltx23_hq_timestep_semantics
                and model_audio_timestep is not None
                else sigma
            )
            denoised_video = self._ltx2_velocity_to_x0(
                ctx.latents, model_video_velocity, video_sigma_for_x0
            ).float()
            denoised_audio = self._ltx2_velocity_to_x0(
                ctx.audio_latents, model_audio_velocity, audio_sigma_for_x0
            ).float()

        if sigma_val == 0.0 or sigma_next_val == 0.0:
            next_video = denoised_video.to(dtype=ctx.latents.dtype)
            next_audio = denoised_audio.to(dtype=ctx.audio_latents.dtype)
            next_video = self._ltx2_apply_clean_latent_mask(next_video, ctx)
            return next_video, next_audio

        sigma_d = sigma.double()
        sigma_next_d = sigma_next.double()
        if ctx.use_ltx23_hq_timestep_semantics:
            h = self._ltx2_res2s_step_size_scalar(sigma_d, sigma_next_d)
            a21, b1, b2 = self._ltx2_get_res2s_coefficients_scalar(h)
            h_value = h
        else:
            h = -torch.log(torch.clamp(sigma_next_d / sigma_d, min=1e-12))
            a21, b1, b2 = self._ltx2_get_res2s_coefficients(h)
            h_value = float(h.item())
        sub_sigma = torch.sqrt(torch.clamp(sigma_d * sigma_next_d, min=0.0))

        anchor_video = ctx.latents.double()
        anchor_audio = ctx.audio_latents.double()
        eps1_video = denoised_video.double() - anchor_video
        eps1_audio = denoised_audio.double() - anchor_audio

        midpoint_video_det = anchor_video + h * a21 * eps1_video
        midpoint_audio_det = anchor_audio + h * a21 * eps1_audio

        sub_noise_video = (
            self._ltx2_res2s_noise_like(
                ctx.latents, ctx, substep=True, batch=batch
            ).float()
            if ctx.use_native_hq_res2s_sde_noise
            else self._randn_like_with_batch_generators(ctx.latents, batch).float()
        )
        sub_noise_audio = (
            self._ltx2_res2s_noise_like(
                ctx.audio_latents, ctx, substep=True, batch=batch, is_audio=True
            ).float()
            if ctx.use_native_hq_res2s_sde_noise
            else self._randn_like_with_batch_generators(
                ctx.audio_latents, batch
            ).float()
        )
        midpoint_video_latents = self._ltx2_res2s_sde_step(
            sample=anchor_video,
            denoised_sample=midpoint_video_det,
            sigma=sigma_d,
            sigma_next=sub_sigma,
            noise=sub_noise_video,
            terminal=False,
        )
        midpoint_audio_latents = self._ltx2_res2s_sde_step(
            sample=anchor_audio,
            denoised_sample=midpoint_audio_det,
            sigma=sigma_d,
            sigma_next=sub_sigma,
            noise=sub_noise_audio,
            terminal=False,
        )
        midpoint_video_model_latents = self._ltx2_apply_clean_latent_mask(
            midpoint_video_latents.to(dtype=ctx.latents.dtype), ctx
        )
        midpoint_audio_model_latents = midpoint_audio_latents.to(
            dtype=ctx.audio_latents.dtype
        )

        # Bongmath anchor refinement for the first stage-2 step.
        if h_value < 0.5 and sigma_val > 0.03:
            x_mid_v = midpoint_video_latents.double()
            x_mid_a = midpoint_audio_latents.double()
            for _ in range(100):
                anchor_video = x_mid_v - h * a21 * eps1_video
                eps1_video = denoised_video.double() - anchor_video
                anchor_audio = x_mid_a - h * a21 * eps1_audio
                eps1_audio = denoised_audio.double() - anchor_audio

        mid_v, mid_a, mid_video_timestep, mid_audio_timestep = midpoint_model_call(
            midpoint_video_model_latents, midpoint_audio_model_latents, sub_sigma
        )

        mid_video_sigma_for_x0 = (
            mid_video_timestep
            if ctx.use_ltx23_hq_timestep_semantics and mid_video_timestep is not None
            else sub_sigma
        )
        mid_audio_sigma_for_x0 = (
            mid_audio_timestep
            if ctx.use_ltx23_hq_timestep_semantics and mid_audio_timestep is not None
            else sub_sigma
        )
        midpoint_denoised_video = self._ltx2_velocity_to_x0(
            midpoint_video_latents, mid_v, mid_video_sigma_for_x0
        ).float()
        midpoint_denoised_audio = self._ltx2_velocity_to_x0(
            midpoint_audio_latents, mid_a, mid_audio_sigma_for_x0
        ).float()

        eps2_video = midpoint_denoised_video.double() - anchor_video
        eps2_audio = midpoint_denoised_audio.double() - anchor_audio

        next_video_det = anchor_video + h * (b1 * eps1_video + b2 * eps2_video)
        next_audio_det = anchor_audio + h * (b1 * eps1_audio + b2 * eps2_audio)

        step_noise_video = (
            self._ltx2_res2s_noise_like(
                ctx.latents, ctx, substep=False, batch=batch
            ).float()
            if ctx.use_native_hq_res2s_sde_noise
            else self._randn_like_with_batch_generators(ctx.latents, batch).float()
        )
        step_noise_audio = (
            self._ltx2_res2s_noise_like(
                ctx.audio_latents, ctx, substep=False, batch=batch, is_audio=True
            ).float()
            if ctx.use_native_hq_res2s_sde_noise
            else self._randn_like_with_batch_generators(
                ctx.audio_latents, batch
            ).float()
        )
        sde_sigma = sigma if ctx.use_ltx23_hq_timestep_semantics else sigma_d
        sde_sigma_next = (
            sigma_next if ctx.use_ltx23_hq_timestep_semantics else sigma_next_d
        )
        next_video = self._ltx2_res2s_sde_step(
            sample=anchor_video,
            denoised_sample=next_video_det,
            sigma=sde_sigma,
            sigma_next=sde_sigma_next,
            noise=step_noise_video,
            terminal=False,
        )
        next_audio = self._ltx2_res2s_sde_step(
            sample=anchor_audio,
            denoised_sample=next_audio_det,
            sigma=sde_sigma,
            sigma_next=sde_sigma_next,
            noise=step_noise_audio,
            terminal=False,
        )

        next_video = self._ltx2_apply_clean_latent_mask(
            next_video.to(dtype=ctx.latents.dtype), ctx
        )
        next_audio = next_audio.to(dtype=ctx.audio_latents.dtype)
        return next_video, next_audio

    @staticmethod
    def _normalize_ltx2_condition_latents(
        image_latent: torch.Tensor | list[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        if image_latent is None:
            return []
        return image_latent if isinstance(image_latent, list) else [image_latent]

    @classmethod
    def _get_ltx2_condition_spans(
        cls,
        batch: Req,
        latents: torch.Tensor,
        image_latent: torch.Tensor | list[torch.Tensor] | None,
        num_img_tokens: int,
    ) -> list[tuple[int, torch.Tensor]]:
        if num_img_tokens <= 0:
            return []
        if not (isinstance(latents, torch.Tensor) and latents.ndim == 3):
            raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")

        condition_latents = cls._normalize_ltx2_condition_latents(image_latent)
        if not condition_latents:
            return []
        if len(condition_latents) > 2:
            raise ValueError(
                "LTX-2 TI2V currently supports at most two conditioning images."
            )

        for cond in condition_latents:
            if not (isinstance(cond, torch.Tensor) and cond.ndim == 3):
                raise ValueError(
                    "Expected LTX-2 conditioning latents to be packed tensors [B, S, D]."
                )
            if int(cond.shape[1]) < int(num_img_tokens):
                raise ValueError(
                    "LTX-2 conditioning latent is shorter than one frame token span."
                )

        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard:
            if int(latents.shape[1]) < int(num_img_tokens):
                raise ValueError(
                    "LTX-2 latent sequence is shorter than one conditioning frame."
                )
            if len(condition_latents) == 1:
                return [(0, condition_latents[0])]
            return [
                (0, condition_latents[0]),
                (int(latents.shape[1]) - int(num_img_tokens), condition_latents[1]),
            ]

        tokens_per_frame = int(getattr(batch, "sp_video_tokens_per_frame", 0))
        if tokens_per_frame <= 0:
            raise ValueError(
                "SP-sharded LTX-2 TI2V requires batch.sp_video_tokens_per_frame."
            )
        if int(num_img_tokens) != int(tokens_per_frame):
            raise ValueError(
                "LTX-2 conditioning token count must match one latent frame when using SP."
            )

        raw_shape = getattr(batch, "raw_latent_shape", None)
        if raw_shape is None:
            raise ValueError("SP-sharded LTX-2 TI2V requires batch.raw_latent_shape.")
        global_seq_len = int(raw_shape[1])
        if global_seq_len % tokens_per_frame != 0:
            raise ValueError(
                "SP-sharded LTX-2 TI2V expected raw seq_len divisible by tokens_per_frame."
            )

        global_num_frames = global_seq_len // tokens_per_frame
        local_start_frame = int(getattr(batch, "sp_video_start_frame", 0))
        local_num_frames = int(getattr(batch, "sp_video_latent_num_frames", 0))
        local_end_frame = local_start_frame + local_num_frames

        spans: list[tuple[int, torch.Tensor]] = []
        if local_start_frame == 0:
            spans.append((0, condition_latents[0]))

        if len(condition_latents) == 2:
            last_global_frame = global_num_frames - 1
            if local_start_frame <= last_global_frame < local_end_frame:
                local_last_frame = last_global_frame - local_start_frame
                spans.append(
                    (local_last_frame * tokens_per_frame, condition_latents[1])
                )

        return spans

    @classmethod
    def _prepare_ltx2_ti2v_clean_state(
        cls,
        batch: Req,
        latents: torch.Tensor,
        image_latent: torch.Tensor | list[torch.Tensor] | None,
        num_img_tokens: int,
        zero_clean_latent: bool,
        clean_latent_background: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = latents.clone()
        denoise_mask = torch.ones(
            (latents.shape[0], latents.shape[1], 1),
            device=latents.device,
            dtype=torch.float32,
        )
        if clean_latent_background is not None:
            clean_latent = (
                clean_latent_background.detach()
                .clone()
                .to(device=latents.device, dtype=latents.dtype)
            )
        elif zero_clean_latent:
            clean_latent = torch.zeros_like(latents)
        else:
            clean_latent = latents.detach().clone()

        spans = cls._get_ltx2_condition_spans(
            batch=batch,
            latents=latents,
            image_latent=image_latent,
            num_img_tokens=num_img_tokens,
        )
        for start, cond in spans:
            stop = int(start) + int(num_img_tokens)
            conditioned = cls._repeat_batch_dim(
                cond[:, :num_img_tokens, :].to(
                    device=latents.device, dtype=latents.dtype
                ),
                int(latents.shape[0]),
            )
            latents[:, start:stop, :] = conditioned
            denoise_mask[:, start:stop, :] = 0.0
            clean_latent[:, start:stop, :] = conditioned
        return latents, denoise_mask, clean_latent

    @staticmethod
    def _ltx2_velocity_to_x0(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            while sigma.ndim < sample.ndim:
                sigma = sigma.unsqueeze(-1)
            return (sample.float() - sigma * velocity.float()).to(sample.dtype)
        return (sample.float() - float(sigma) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _repeat_batch_dim(tensor: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Repeat along batch dim while preserving any tokenwise timestep layout."""
        if tensor.shape[0] == int(target_batch_size):
            return tensor
        if tensor.shape[0] <= 0 or int(target_batch_size) % int(tensor.shape[0]) != 0:
            raise ValueError(
                "Cannot repeat tensor with batch="
                f"{tensor.shape[0]} to target_batch_size={target_batch_size}"
            )
        repeat_factor = int(target_batch_size) // int(tensor.shape[0])
        return tensor.repeat(repeat_factor, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def _build_ltx2_sp_padding_mask(
        batch: Req,
        *,
        seq_len: int,
        batch_size: int,
        key: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        valid = getattr(batch, key, None)
        if valid is None:
            return None
        valid = int(valid)
        if valid <= 0 or valid >= int(seq_len):
            return None
        mask = torch.ones(
            (batch_size, int(seq_len)), device=device, dtype=torch.float32
        )
        mask[:, valid:] = 0.0
        return mask

    @staticmethod
    def _get_ltx_prompt_attention_mask(
        batch: Req,
        *,
        is_ltx23_variant: bool,
        negative: bool = False,
    ) -> torch.Tensor | None:
        if is_ltx23_variant:
            return None
        return (
            batch.negative_attention_mask if negative else batch.prompt_attention_mask
        )

    @classmethod
    def _should_use_ltx23_legacy_one_stage(
        cls,
        server_args: ServerArgs,
    ) -> bool:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            return False
        return not is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name)

    @classmethod
    def _ltx2_calculate_guided_x0(
        cls,
        *,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        modality_scale: float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (cfg_scale - 1.0) * (cond - uncond_text)
            + stg_scale * (cond - uncond_perturbed)
            + (modality_scale - 1.0) * (cond - uncond_modality)
        )
        return cls._ltx2_apply_rescale(cond, pred, rescale_scale)

    @staticmethod
    def _should_pass_ltx2_text_attention_mask(
        ctx: LTX2DenoisingContext,
    ) -> bool:
        return not (ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage)

    @classmethod
    def _repeat_optional_batch_dim(
        cls,
        tensor: torch.Tensor | None,
        target_batch_size: int,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        return cls._repeat_batch_dim(tensor, target_batch_size)

    @staticmethod
    def _get_audio_num_frames_latent(audio_latent_model_input: torch.Tensor) -> int:
        if audio_latent_model_input.ndim == 3:
            return int(audio_latent_model_input.shape[1])
        if audio_latent_model_input.ndim == 4:
            return int(audio_latent_model_input.shape[2])
        raise ValueError(
            "Unexpected audio latents rank: "
            f"{audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
        )

    def _prepare_ltx2_model_inputs(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
        sigma: torch.Tensor,
    ) -> LTX2ModelInputs:
        latent_model_input = ctx.latents.to(ctx.target_dtype)
        audio_latent_model_input = ctx.audio_latents.to(ctx.target_dtype)
        audio_num_frames_latent = self._get_audio_num_frames_latent(
            audio_latent_model_input
        )

        video_coords = None
        audio_coords = None
        if not ctx.use_ltx23_legacy_one_stage:
            video_coords = server_args.pipeline_config.prepare_video_rope_coords_for_sp(
                step.current_model,
                batch,
                latent_model_input,
                num_frames=ctx.latent_num_frames_for_model,
                height=ctx.latent_height,
                width=ctx.latent_width,
            )
            audio_coords = server_args.pipeline_config.prepare_audio_rope_coords_for_sp(
                step.current_model,
                batch,
                audio_latent_model_input,
                num_frames=audio_num_frames_latent,
            )

        batch_size = int(latent_model_input.shape[0])
        use_raw_sigma_timestep = ctx.use_ltx23_hq_timestep_semantics
        use_ltx23_two_stage_prompt_timestep = (
            ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
        )
        timestep = (
            sigma.to(device=ctx.latents.device, dtype=torch.float32).expand(batch_size)
            if use_raw_sigma_timestep
            else step.t_device.to(
                device=ctx.latents.device, dtype=torch.float32
            ).expand(batch_size)
        )
        if ctx.denoise_mask is not None:
            if use_raw_sigma_timestep:
                timestep_video = (
                    timestep.view(batch_size, *([1] * (ctx.denoise_mask.ndim - 1)))
                    * ctx.denoise_mask
                )
            else:
                timestep_video = timestep.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
        elif use_raw_sigma_timestep:
            timestep_video = timestep.view(batch_size, 1, 1).expand(
                batch_size, int(latent_model_input.shape[1]), 1
            )
        elif use_ltx23_two_stage_prompt_timestep:
            timestep_video = timestep.view(batch_size, 1).expand(
                batch_size, int(latent_model_input.shape[1])
            )
        else:
            timestep_video = timestep

        if use_raw_sigma_timestep and audio_latent_model_input.ndim == 3:
            timestep_audio = timestep.view(batch_size, 1, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1]), 1
            )
        elif use_ltx23_two_stage_prompt_timestep and audio_latent_model_input.ndim == 3:
            timestep_audio = timestep.view(batch_size, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1])
            )
        else:
            timestep_audio = timestep

        prompt_timestep_video = None
        prompt_timestep_audio = None
        if ctx.use_ltx23_hq_timestep_semantics:
            prompt_timestep_video = sigma.to(
                device=ctx.latents.device, dtype=torch.float32
            ).expand(batch_size)
            prompt_timestep_audio = sigma.to(
                device=ctx.audio_latents.device, dtype=torch.float32
            ).expand(batch_size)
        elif use_ltx23_two_stage_prompt_timestep:
            timestep_scale_multiplier = float(
                getattr(step.current_model, "timestep_scale_multiplier", 1000)
            )
            prompt_timestep_video = (
                sigma.to(device=latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)
            prompt_timestep_audio = (
                sigma.to(device=audio_latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)

        if ctx.use_ltx23_legacy_one_stage:
            video_self_attention_mask = None
            audio_self_attention_mask = None
            a2v_cross_attention_mask = None
            v2a_cross_attention_mask = None
        else:
            video_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=int(latent_model_input.shape[1]),
                batch_size=batch_size,
                key="sp_video_valid_token_count",
                device=latent_model_input.device,
            )
            audio_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=audio_num_frames_latent,
                batch_size=batch_size,
                key="sp_audio_valid_token_count",
                device=audio_latent_model_input.device,
            )
            a2v_cross_attention_mask = audio_self_attention_mask
            v2a_cross_attention_mask = video_self_attention_mask

        return LTX2ModelInputs(
            latent_model_input=latent_model_input,
            audio_latent_model_input=audio_latent_model_input,
            audio_num_frames_latent=audio_num_frames_latent,
            video_coords=video_coords,
            audio_coords=audio_coords,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
            prompt_timestep_video=prompt_timestep_video,
            prompt_timestep_audio=prompt_timestep_audio,
            video_self_attention_mask=video_self_attention_mask,
            audio_self_attention_mask=audio_self_attention_mask,
            a2v_cross_attention_mask=a2v_cross_attention_mask,
            v2a_cross_attention_mask=v2a_cross_attention_mask,
        )

    def _build_ltx2_base_model_kwargs(
        self,
        ctx: LTX2DenoisingContext,
        batch: Req,
        model_inputs: LTX2ModelInputs,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "hidden_states": model_inputs.latent_model_input,
            "audio_hidden_states": model_inputs.audio_latent_model_input,
            "timestep": model_inputs.timestep_video,
            "audio_timestep": model_inputs.timestep_audio,
            "num_frames": ctx.latent_num_frames_for_model,
            "height": ctx.latent_height,
            "width": ctx.latent_width,
            "fps": batch.fps,
            "audio_num_frames": model_inputs.audio_num_frames_latent,
            "video_coords": model_inputs.video_coords,
            "audio_coords": model_inputs.audio_coords,
            "return_latents": False,
            "return_dict": False,
        }
        if not ctx.use_ltx23_legacy_one_stage:
            kwargs.update(
                {
                    "prompt_timestep": model_inputs.prompt_timestep_video,
                    "audio_prompt_timestep": model_inputs.prompt_timestep_audio,
                    "video_self_attention_mask": model_inputs.video_self_attention_mask,
                    "audio_self_attention_mask": model_inputs.audio_self_attention_mask,
                    "a2v_cross_attention_mask": model_inputs.a2v_cross_attention_mask,
                    "v2a_cross_attention_mask": model_inputs.v2a_cross_attention_mask,
                    "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                    "legacy_ltx23_one_stage_semantics": False,
                }
            )
        return kwargs

    def _build_ltx2_model_kwargs(
        self,
        ctx: LTX2DenoisingContext,
        base_model_kwargs: dict[str, object],
        *,
        encoder_hidden_states: torch.Tensor,
        audio_encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        skip_video_self_attn_blocks: tuple[int, ...] | None = None,
        skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
        disable_a2v_cross_attn: bool = False,
        disable_v2a_cross_attn: bool = False,
        perturbation_configs: tuple[dict[str, object], ...] | None = None,
    ) -> dict[str, object]:
        kwargs = dict(base_model_kwargs)
        kwargs["encoder_hidden_states"] = encoder_hidden_states
        kwargs["audio_encoder_hidden_states"] = audio_encoder_hidden_states
        if self._should_pass_ltx2_text_attention_mask(ctx):
            kwargs["encoder_attention_mask"] = encoder_attention_mask
            kwargs["audio_encoder_attention_mask"] = encoder_attention_mask
        else:
            kwargs["encoder_attention_mask"] = None
            kwargs["audio_encoder_attention_mask"] = None
        if skip_video_self_attn_blocks is not None:
            kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
        if skip_audio_self_attn_blocks is not None:
            kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
        if disable_a2v_cross_attn:
            kwargs["disable_a2v_cross_attn"] = True
        if disable_v2a_cross_attn:
            kwargs["disable_v2a_cross_attn"] = True
        if perturbation_configs is not None:
            kwargs["perturbation_configs"] = perturbation_configs
        if self.server_args.enable_cfg_parallel:
            device = get_local_torch_device()
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in kwargs.items()
            }
        return kwargs

    @staticmethod
    def _ltx2_guidance_perturbation_config(
        pass_spec: LTX2GuidancePassSpec,
    ) -> dict[str, object]:
        return {
            "skip_video_self_attn_blocks": pass_spec.skip_video_self_attn_blocks,
            "skip_audio_self_attn_blocks": pass_spec.skip_audio_self_attn_blocks,
            "skip_a2v_cross_attn": pass_spec.disable_a2v_cross_attn,
            "skip_v2a_cross_attn": pass_spec.disable_v2a_cross_attn,
        }

    @classmethod
    def _build_ltx2_guidance_perturbation_configs(
        cls,
        pass_specs: list[LTX2GuidancePassSpec],
        batch_size: int,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            cls._ltx2_guidance_perturbation_config(pass_spec)
            for pass_spec in pass_specs
            for _ in range(batch_size)
        )

    @staticmethod
    def _apply_ltx2_guidance_pass_kwargs(
        model_kwargs: dict[str, object],
        pass_spec: LTX2GuidancePassSpec,
    ) -> None:
        """Copy disable-attention options from pass_spec into model_kwargs."""
        if pass_spec.skip_video_self_attn_blocks:
            model_kwargs["skip_video_self_attn_blocks"] = (
                pass_spec.skip_video_self_attn_blocks
            )
        if pass_spec.skip_audio_self_attn_blocks:
            model_kwargs["skip_audio_self_attn_blocks"] = (
                pass_spec.skip_audio_self_attn_blocks
            )
        if pass_spec.disable_a2v_cross_attn:
            model_kwargs["disable_a2v_cross_attn"] = True
        if pass_spec.disable_v2a_cross_attn:
            model_kwargs["disable_v2a_cross_attn"] = True

    @classmethod
    def _repeat_ltx2_model_kwargs_batch(
        cls,
        model_kwargs: dict[str, object],
        target_batch_size: int,
    ) -> dict[str, object]:
        repeated_kwargs = dict(model_kwargs)
        for key in cls._LTX2_BATCH_REPEATABLE_KWARG_KEYS:
            repeated_kwargs[key] = cls._repeat_optional_batch_dim(
                repeated_kwargs.get(key), target_batch_size
            )
        return repeated_kwargs

    @staticmethod
    def _cat_or_none(
        items: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if not items or items[0] is None:
            return None
        return torch.cat(items, dim=0)

    @staticmethod
    def _split_ltx2_model_kwargs(
        model_kwargs: dict[str, object],
        split_sizes: list[int],
    ) -> list[dict[str, object]]:
        split_kwargs = [dict() for _ in split_sizes]
        for key, value in model_kwargs.items():
            if torch.is_tensor(value):
                values = list(value.split(split_sizes, dim=0))
            else:
                values = [value] * len(split_sizes)
            for index, item in enumerate(values):
                split_kwargs[index][key] = item
        return split_kwargs

    def _preprocess_sp_latents(self, batch: Req, server_args: ServerArgs):
        """LTX-2 TI2V applies image_latent in token space *after* SP sharding,
        so the base implementation must not shard it."""
        saved = batch.image_latent
        batch.image_latent = None
        super()._preprocess_sp_latents(batch, server_args)
        batch.image_latent = saved

    @staticmethod
    def _should_use_native_hq_res2s_sde_noise(server_args: ServerArgs) -> bool:
        return server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"

    @staticmethod
    def _should_use_ltx23_hq_timestep_semantics(server_args: ServerArgs) -> bool:
        return server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"

    @staticmethod
    @contextmanager
    def _temporary_ltx23_hq_timestep_semantics(model, enabled: bool):
        attr = "_sglang_use_ltx23_hq_timestep_semantics"
        previous = bool(getattr(model, attr, False))
        setattr(model, attr, enabled)
        try:
            yield
        finally:
            setattr(model, attr, previous)

    @contextmanager
    def _ltx2_model_forward_context(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
    ):
        with self._temporary_ltx23_hq_timestep_semantics(
            step.current_model, ctx.use_ltx23_hq_timestep_semantics
        ):
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                yield

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        """Extend the base context with LTX-2 audio, SP, and TI2V state."""
        self._disable_cache_dit_for_request = batch.image_path is not None
        base_ctx = super()._prepare_denoising_loop(batch, server_args)
        ctx = LTX2DenoisingContext(**base_ctx.to_kwargs())
        ctx.is_ltx23_variant = is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )
        phase = batch.extra.get("ltx2_phase")
        ctx.use_ltx23_legacy_one_stage = self._should_use_ltx23_legacy_one_stage(
            server_args
        )
        ctx.use_native_hq_res2s_sde_noise = (
            ctx.is_ltx23_variant
            and self._should_use_native_hq_res2s_sde_noise(server_args)
        )
        ctx.use_ltx23_hq_timestep_semantics = (
            ctx.is_ltx23_variant
            and self._should_use_ltx23_hq_timestep_semantics(server_args)
        )
        ctx.stage = (
            phase
            if phase is not None
            else ("stage1" if ctx.use_ltx23_legacy_one_stage else "one_stage")
        )
        ctx.audio_latents = batch.audio_latents
        # Video and audio keep separate scheduler state throughout the denoising loop.
        ctx.audio_scheduler = clone_scheduler_runtime(ctx.scheduler)

        if ctx.use_ltx23_legacy_one_stage:
            batch.ltx23_audio_replicated_for_sp = False
            batch.did_sp_shard_audio_latents = False
        else:
            ctx.replicate_audio_for_sp = False
            batch.ltx23_audio_replicated_for_sp = bool(ctx.replicate_audio_for_sp)
            if (
                ctx.is_ltx23_variant
                and get_sp_world_size() > 1
                and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                    batch.audio_latents
                )
                and not ctx.replicate_audio_for_sp
            ):
                (
                    batch.audio_latents,
                    batch.did_sp_shard_audio_latents,
                ) = server_args.pipeline_config.shard_audio_latents_for_sp(
                    batch, batch.audio_latents
                )
                ctx.audio_latents = batch.audio_latents
            else:
                batch.did_sp_shard_audio_latents = False

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        ctx.latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=ctx.latents
        )
        ctx.latent_height = (
            batch.height
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        ctx.latent_width = (
            batch.width
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        ti2v_spans = self._get_ltx2_condition_spans(
            batch=batch,
            latents=ctx.latents,
            image_latent=batch.image_latent,
            num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
        )
        do_ti2v = bool(ti2v_spans)
        if do_ti2v:
            if not (isinstance(ctx.latents, torch.Tensor) and ctx.latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            clean_latent_background = getattr(
                batch, "ltx2_ti2v_clean_latent_background", None
            )
            if not (
                isinstance(clean_latent_background, torch.Tensor)
                and clean_latent_background.shape == ctx.latents.shape
            ):
                clean_latent_background = None
            # Keep conditioned tokens clean and reuse the mask during every step update.
            ctx.latents, ctx.denoise_mask, ctx.clean_latent = (
                self._prepare_ltx2_ti2v_clean_state(
                    batch=batch,
                    latents=ctx.latents,
                    image_latent=batch.image_latent,
                    num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                    zero_clean_latent=ctx.is_ltx23_variant,
                    clean_latent_background=clean_latent_background,
                )
            )

        # Batch tensors are broadcast from CFG rank 0 and remain on its device.
        # Move every context tensor that will be used in model forward passes or
        # scheduler steps to the local device once here, before the loop begins.
        if server_args.enable_cfg_parallel:
            device = get_local_torch_device()
            ctx.latents = ctx.latents.to(device)
            ctx.timesteps = ctx.timesteps.to(device)
            if ctx.audio_latents is not None:
                ctx.audio_latents = ctx.audio_latents.to(device)
            if ctx.guidance is not None:
                ctx.guidance = ctx.guidance.to(device)
            if ctx.denoise_mask is not None:
                ctx.denoise_mask = ctx.denoise_mask.to(device)
            if ctx.clean_latent is not None:
                ctx.clean_latent = ctx.clean_latent.to(device)

        return ctx

    def _before_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Reset the mirrored audio scheduler before the shared loop begins."""
        if is_ltx2_two_stage_pipeline_name(
            server_args.pipeline_class_name
        ) and ctx.stage in ("stage1", "stage2"):
            pipeline = self.pipeline() if self.pipeline else None
            if pipeline is not None:
                pipeline.switch_lora_phase(ctx.stage, batch=batch)
        super()._before_denoising_loop(ctx, batch, server_args)
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")
        ctx.audio_scheduler.set_begin_index(0)
        if self.sampler_name == "res2s" and ctx.use_native_hq_res2s_sde_noise:
            self._ltx2_init_res2s_noise_generators(ctx)

    def _prepare_step_attn_metadata(
        self,
        ctx: LTX2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ):
        """Preserve the legacy LTX-2 attention-metadata contract."""
        # Legacy LTX-2 paths used the plain attention-metadata builder call here.
        return self._build_attn_metadata(step_index, batch, server_args)

    def _run_denoising_step(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Run one joint video/audio denoising step with LTX-2-specific guidance."""
        if ctx.audio_latents is None:
            raise ValueError("LTX-2 requires audio latents for denoising.")
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")

        # 1. Read the scheduler sigma pair and derive the Euler delta.
        sigmas = getattr(ctx.scheduler, "sigmas", None)
        if sigmas is None or not isinstance(sigmas, torch.Tensor):
            raise ValueError("Expected scheduler.sigmas to be a tensor for LTX-2.")
        sigma = sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_next = sigmas[step.step_index + 1].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        dt = sigma_next - sigma
        sigma_val = float(sigma.item())
        sigma_next_val = float(sigma_next.item())

        stage1_guider_params = self._get_ltx2_stage1_guider_params(
            batch, server_args, ctx.stage
        )
        model_inputs = self._prepare_ltx2_model_inputs(
            ctx, step, batch, server_args, sigma
        )
        batch_size = int(model_inputs.latent_model_input.shape[0])
        base_model_kwargs = self._build_ltx2_base_model_kwargs(ctx, batch, model_inputs)

        # 5. Run the branch-specific LTX forward path and apply CFG/guider logic.
        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
        )
        use_official_cfg_path = stage1_guider_params is None
        if use_official_cfg_path:
            cfg_parallel = (
                server_args.enable_cfg_parallel and batch.do_classifier_free_guidance
            )
            cfg_rank = get_classifier_free_guidance_rank() if cfg_parallel else 0

            if cfg_parallel:
                if cfg_rank == 0:
                    encoder_hidden_states = batch.prompt_embeds[0]
                    audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                    encoder_attention_mask = prompt_attention_mask
                elif cfg_rank == 1:
                    encoder_hidden_states = batch.negative_prompt_embeds[0]
                    audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
                    encoder_attention_mask = self._get_ltx_prompt_attention_mask(
                        batch,
                        is_ltx23_variant=(
                            ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
                        ),
                        negative=True,
                    )
                else:
                    encoder_hidden_states = batch.prompt_embeds[0]
                    audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                    encoder_attention_mask = prompt_attention_mask
                model_kwargs = self._build_ltx2_model_kwargs(
                    ctx,
                    base_model_kwargs,
                    encoder_hidden_states=encoder_hidden_states,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                model_kwargs = self._build_ltx2_model_kwargs(
                    ctx,
                    base_model_kwargs,
                    encoder_hidden_states=batch.prompt_embeds[0],
                    audio_encoder_hidden_states=batch.audio_prompt_embeds[0],
                    encoder_attention_mask=prompt_attention_mask,
                )
                if batch.do_classifier_free_guidance:
                    cfg_batch_size = batch_size * 2
                    model_kwargs = self._repeat_ltx2_model_kwargs_batch(
                        model_kwargs, cfg_batch_size
                    )
                    model_kwargs["encoder_hidden_states"] = torch.cat(
                        [batch.negative_prompt_embeds[0], batch.prompt_embeds[0]],
                        dim=0,
                    )
                    model_kwargs["audio_encoder_hidden_states"] = torch.cat(
                        [
                            batch.negative_audio_prompt_embeds[0],
                            batch.audio_prompt_embeds[0],
                        ],
                        dim=0,
                    )
                    if self._should_pass_ltx2_text_attention_mask(ctx):
                        repeated_attention_mask = self._cat_or_none(
                            [
                                self._get_ltx_prompt_attention_mask(
                                    batch,
                                    is_ltx23_variant=(
                                        ctx.is_ltx23_variant
                                        and not ctx.use_ltx23_legacy_one_stage
                                    ),
                                    negative=True,
                                ),
                                prompt_attention_mask,
                            ]
                        )
                        model_kwargs["encoder_attention_mask"] = repeated_attention_mask
                        model_kwargs["audio_encoder_attention_mask"] = (
                            repeated_attention_mask
                        )

            with self._ltx2_model_forward_context(ctx, step):
                model_video, model_audio = step.current_model(**model_kwargs)

            model_video = model_video.float()
            model_audio = model_audio.float()
            if cfg_parallel:
                model_video, model_audio = self._combine_cfg_parallel_av(
                    model_video, model_audio, float(batch.guidance_scale), cfg_rank
                )
            elif batch.do_classifier_free_guidance:
                model_video_uncond, model_video_text = model_video.chunk(2)
                model_audio_uncond, model_audio_text = model_audio.chunk(2)
                model_video = model_video_uncond + (
                    batch.guidance_scale * (model_video_text - model_video_uncond)
                )
                model_audio = model_audio_uncond + (
                    batch.guidance_scale * (model_audio_text - model_audio_uncond)
                )

            if self.sampler_name == "res2s":
                # HQ stage-2 uses RK2 res2s here to match official LTX-2.3 HQ
                # output. Without this path the scheduler falls back to Euler
                # and loses ~3.7 dB against the official canonical.
                def _stage2_midpoint_model_call(
                    video_latents: torch.Tensor,
                    audio_latents: torch.Tensor,
                    sigma_value: torch.Tensor,
                ) -> tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    torch.Tensor | None,
                ]:
                    original_video_latents = ctx.latents
                    original_audio_latents = ctx.audio_latents
                    ctx.latents = video_latents
                    ctx.audio_latents = audio_latents
                    try:
                        model_inputs_local = self._prepare_ltx2_model_inputs(
                            ctx, step, batch, server_args, sigma_value
                        )
                        batch_size_local = int(
                            model_inputs_local.latent_model_input.shape[0]
                        )
                        base_model_kwargs_local = self._build_ltx2_base_model_kwargs(
                            ctx, batch, model_inputs_local
                        )
                        model_kwargs_local = self._build_ltx2_model_kwargs(
                            ctx,
                            base_model_kwargs_local,
                            encoder_hidden_states=batch.prompt_embeds[0],
                            audio_encoder_hidden_states=batch.audio_prompt_embeds[0],
                            encoder_attention_mask=prompt_attention_mask,
                        )
                        if batch.do_classifier_free_guidance:
                            cfg_batch_size = batch_size_local * 2
                            model_kwargs_local = self._repeat_ltx2_model_kwargs_batch(
                                model_kwargs_local, cfg_batch_size
                            )
                            model_kwargs_local["encoder_hidden_states"] = torch.cat(
                                [
                                    batch.negative_prompt_embeds[0],
                                    batch.prompt_embeds[0],
                                ],
                                dim=0,
                            )
                            model_kwargs_local["audio_encoder_hidden_states"] = (
                                torch.cat(
                                    [
                                        batch.negative_audio_prompt_embeds[0],
                                        batch.audio_prompt_embeds[0],
                                    ],
                                    dim=0,
                                )
                            )
                            if self._should_pass_ltx2_text_attention_mask(ctx):
                                repeated_attention_mask = self._cat_or_none(
                                    [
                                        self._get_ltx_prompt_attention_mask(
                                            batch,
                                            is_ltx23_variant=(
                                                ctx.is_ltx23_variant
                                                and not ctx.use_ltx23_legacy_one_stage
                                            ),
                                            negative=True,
                                        ),
                                        prompt_attention_mask,
                                    ]
                                )
                                model_kwargs_local["encoder_attention_mask"] = (
                                    repeated_attention_mask
                                )
                                model_kwargs_local["audio_encoder_attention_mask"] = (
                                    repeated_attention_mask
                                )

                        with self._ltx2_model_forward_context(ctx, step):
                            mid_v, mid_a = step.current_model(**model_kwargs_local)

                        mid_v = mid_v.float()
                        mid_a = mid_a.float()
                        if batch.do_classifier_free_guidance:
                            mid_v_u, mid_v_t = mid_v.chunk(2)
                            mid_a_u, mid_a_t = mid_a.chunk(2)
                            mid_v = mid_v_u + batch.guidance_scale * (mid_v_t - mid_v_u)
                            mid_a = mid_a_u + batch.guidance_scale * (mid_a_t - mid_a_u)
                        return (
                            mid_v,
                            mid_a,
                            model_inputs_local.timestep_video,
                            model_inputs_local.timestep_audio,
                        )
                    finally:
                        ctx.latents = original_video_latents
                        ctx.audio_latents = original_audio_latents

                ctx.latents, ctx.audio_latents = self._ltx2_stage2_res2s_step(
                    ctx=ctx,
                    batch=batch,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    model_video_velocity=model_video,
                    model_audio_velocity=model_audio,
                    model_video_timestep=model_inputs.timestep_video,
                    model_audio_timestep=model_inputs.timestep_audio,
                    midpoint_model_call=_stage2_midpoint_model_call,
                )
            else:
                ctx.latents = ctx.scheduler.step(
                    model_video, step.t_device, ctx.latents, return_dict=False
                )[0]
                ctx.audio_latents = ctx.audio_scheduler.step(
                    model_audio, step.t_device, ctx.audio_latents, return_dict=False
                )[0]
                if ctx.denoise_mask is not None and ctx.clean_latent is not None:
                    ctx.latents = (
                        ctx.latents.float() * ctx.denoise_mask
                        + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
                    ).to(dtype=ctx.latents.dtype)
            ctx.latents = self.post_forward_for_ti2v_task(
                batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
            )
            return

        encoder_hidden_states = batch.prompt_embeds[0]
        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
        encoder_attention_mask = prompt_attention_mask
        negative_encoder_hidden_states = batch.negative_prompt_embeds[0]
        negative_audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
        negative_encoder_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
            negative=True,
        )

        video_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["video_skip_step"])
        )
        audio_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["audio_skip_step"])
        )
        need_perturbed = (
            float(stage1_guider_params["video_stg_scale"]) != 0.0
            or float(stage1_guider_params["audio_stg_scale"]) != 0.0
        )
        need_modality = (
            float(stage1_guider_params["video_modality_scale"]) != 1.0
            or float(stage1_guider_params["audio_modality_scale"]) != 1.0
        )
        stage1_cfg_parallel = (
            server_args.enable_cfg_parallel and not ctx.use_ltx23_legacy_one_stage
        )
        stage1_cfg_rank = (
            get_classifier_free_guidance_rank() if stage1_cfg_parallel else 0
        )
        stage1_cfg_world_size = (
            get_classifier_free_guidance_world_size() if stage1_cfg_parallel else 1
        )
        # NOTE: this flag must be identical across all SP ranks so that every
        # rank executes the same number of model-forward calls (each of which
        # contains NCCL collectives).
        use_split_stage1_guided_passes = (
            server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"
            or (
                is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name)
                and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
            )
        )
        # "Perturbation" means disabling selected attention paths
        # for that item (self-attention blocks or audio/video cross-attention)
        # to compute STG/modality guidance.
        #

        # Decide whether to use different pass kwargs for split model calls
        # 1. HQ splits the expanded batch into one-item model calls. Since each
        # call has only one perturbation setting, pass the disable options
        # directly as model arguments.
        # 2. TI2V/non-HQ may keep several expanded
        # items with different settings in one model call, so it needs
        # perturbation_configs: one config dict per expanded item.
        use_split_pass_kwargs = (
            server_args.pipeline_class_name == "LTX2TwoStageHQPipeline"
        )
        skip_v2a_cross_attn_for_video_gt = bool(
            batch.extra.get("ltx2_skip_v2a_cross_attn_for_video_gt", False)
        )

        def evaluate_stage1_guided_x0(
            *,
            video_latents: torch.Tensor,
            audio_latents: torch.Tensor,
            sigma_value: torch.Tensor,
            update_skip_cache: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            original_video_latents = ctx.latents
            original_audio_latents = ctx.audio_latents
            ctx.latents = video_latents
            ctx.audio_latents = audio_latents
            try:
                model_inputs_local = self._prepare_ltx2_model_inputs(
                    ctx, step, batch, server_args, sigma_value
                )
                batch_size_local = int(model_inputs_local.latent_model_input.shape[0])
                base_model_kwargs_local = self._build_ltx2_base_model_kwargs(
                    ctx, batch, model_inputs_local
                )

                if ctx.use_ltx23_legacy_one_stage:
                    with self._ltx2_model_forward_context(ctx, step):
                        v_pos, a_v_pos = step.current_model(
                            **self._build_ltx2_model_kwargs(
                                ctx,
                                base_model_kwargs_local,
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                disable_v2a_cross_attn=(
                                    skip_v2a_cross_attn_for_video_gt
                                ),
                            )
                        )
                        v_neg, a_v_neg = step.current_model(
                            **self._build_ltx2_model_kwargs(
                                ctx,
                                base_model_kwargs_local,
                                encoder_hidden_states=negative_encoder_hidden_states,
                                audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                                encoder_attention_mask=negative_encoder_attention_mask,
                                disable_v2a_cross_attn=(
                                    skip_v2a_cross_attn_for_video_gt
                                ),
                            )
                        )

                    v_pos = v_pos.float()
                    a_v_pos = a_v_pos.float()
                    v_neg = v_neg.float()
                    a_v_neg = a_v_neg.float()

                    v_ptb = None
                    a_v_ptb = None
                    if need_perturbed:
                        with self._ltx2_model_forward_context(ctx, step):
                            v_ptb, a_v_ptb = step.current_model(
                                **self._build_ltx2_model_kwargs(
                                    ctx,
                                    base_model_kwargs_local,
                                    encoder_hidden_states=encoder_hidden_states,
                                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    skip_video_self_attn_blocks=tuple(
                                        stage1_guider_params["video_stg_blocks"]
                                    ),
                                    skip_audio_self_attn_blocks=tuple(
                                        stage1_guider_params["audio_stg_blocks"]
                                    ),
                                    disable_v2a_cross_attn=(
                                        skip_v2a_cross_attn_for_video_gt
                                    ),
                                )
                            )
                        v_ptb = v_ptb.float()
                        a_v_ptb = a_v_ptb.float()

                    v_mod = None
                    a_v_mod = None
                    if need_modality:
                        with self._ltx2_model_forward_context(ctx, step):
                            v_mod, a_v_mod = step.current_model(
                                **self._build_ltx2_model_kwargs(
                                    ctx,
                                    base_model_kwargs_local,
                                    encoder_hidden_states=encoder_hidden_states,
                                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                                    encoder_attention_mask=encoder_attention_mask,
                                    disable_a2v_cross_attn=True,
                                    disable_v2a_cross_attn=True,
                                )
                            )
                        v_mod = v_mod.float()
                        a_v_mod = a_v_mod.float()
                else:
                    pass_specs: list[LTX2GuidancePassSpec] = [
                        LTX2GuidancePassSpec(
                            name="cond",
                            encoder_hidden_states=encoder_hidden_states,
                            audio_encoder_hidden_states=audio_encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            disable_v2a_cross_attn=skip_v2a_cross_attn_for_video_gt,
                        ),
                        LTX2GuidancePassSpec(
                            name="neg",
                            encoder_hidden_states=negative_encoder_hidden_states,
                            audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                            encoder_attention_mask=negative_encoder_attention_mask,
                            disable_v2a_cross_attn=skip_v2a_cross_attn_for_video_gt,
                        ),
                    ]
                    if need_perturbed:
                        pass_specs.append(
                            LTX2GuidancePassSpec(
                                name="perturbed",
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                skip_video_self_attn_blocks=tuple(
                                    stage1_guider_params["video_stg_blocks"]
                                ),
                                skip_audio_self_attn_blocks=tuple(
                                    stage1_guider_params["audio_stg_blocks"]
                                ),
                                disable_v2a_cross_attn=skip_v2a_cross_attn_for_video_gt,
                            )
                        )
                    if need_modality:
                        pass_specs.append(
                            LTX2GuidancePassSpec(
                                name="modality",
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                encoder_attention_mask=encoder_attention_mask,
                                disable_a2v_cross_attn=True,
                                disable_v2a_cross_attn=True,
                            )
                        )

                    execution_pass_specs = (
                        [
                            pass_spec
                            for index, pass_spec in enumerate(pass_specs)
                            if index % stage1_cfg_world_size == stage1_cfg_rank
                        ]
                        if stage1_cfg_parallel
                        else pass_specs
                    )
                    num_execution_passes = len(execution_pass_specs)
                    if num_execution_passes == 0:
                        raise ValueError(
                            "LTX2 stage-1 CFG parallel degree exceeds guidance pass count."
                        )
                    expanded_batch_size = batch_size_local * num_execution_passes
                    batched_model_kwargs = self._repeat_ltx2_model_kwargs_batch(
                        base_model_kwargs_local, expanded_batch_size
                    )
                    batched_model_kwargs = self._build_ltx2_model_kwargs(
                        ctx,
                        batched_model_kwargs,
                        encoder_hidden_states=torch.cat(
                            [
                                pass_spec.encoder_hidden_states
                                for pass_spec in execution_pass_specs
                            ],
                            dim=0,
                        ),
                        audio_encoder_hidden_states=torch.cat(
                            [
                                pass_spec.audio_encoder_hidden_states
                                for pass_spec in execution_pass_specs
                            ],
                            dim=0,
                        ),
                        encoder_attention_mask=self._cat_or_none(
                            [
                                pass_spec.encoder_attention_mask
                                for pass_spec in execution_pass_specs
                            ]
                        ),
                    )
                    if use_split_stage1_guided_passes:
                        split_sizes = [1] * expanded_batch_size
                        split_pass_specs = tuple(
                            pass_spec
                            for pass_spec in execution_pass_specs
                            for _ in range(batch_size_local)
                        )
                        split_perturbation_configs = (
                            ()
                            if use_split_pass_kwargs
                            else self._build_ltx2_guidance_perturbation_configs(
                                execution_pass_specs, batch_size_local
                            )
                        )
                        batched_video_chunks = []
                        batched_audio_chunks = []
                        with self._ltx2_model_forward_context(ctx, step):
                            for index, (model_kwargs_chunk, pass_spec) in enumerate(
                                zip(
                                    self._split_ltx2_model_kwargs(
                                        batched_model_kwargs, split_sizes
                                    ),
                                    split_pass_specs,
                                    strict=True,
                                )
                            ):
                                if use_split_pass_kwargs:
                                    self._apply_ltx2_guidance_pass_kwargs(
                                        model_kwargs_chunk, pass_spec
                                    )
                                else:
                                    model_kwargs_chunk["perturbation_configs"] = (
                                        split_perturbation_configs[index],
                                    )
                                video_chunk, audio_chunk = step.current_model(
                                    **model_kwargs_chunk
                                )
                                batched_video_chunks.append(video_chunk)
                                batched_audio_chunks.append(audio_chunk)

                        batched_video = torch.cat(batched_video_chunks, dim=0)
                        batched_audio = torch.cat(batched_audio_chunks, dim=0)
                    else:
                        perturbation_configs = (
                            self._build_ltx2_guidance_perturbation_configs(
                                execution_pass_specs, batch_size_local
                            )
                        )
                        with self._ltx2_model_forward_context(ctx, step):
                            batched_video, batched_audio = step.current_model(
                                **batched_model_kwargs,
                                perturbation_configs=perturbation_configs,
                            )

                    batched_video = batched_video.float()
                    batched_audio = batched_audio.float()
                    pass_outputs = {
                        pass_spec.name: (
                            video_chunk,
                            audio_chunk,
                        )
                        for pass_spec, video_chunk, audio_chunk in zip(
                            execution_pass_specs,
                            batched_video.chunk(num_execution_passes, dim=0),
                            batched_audio.chunk(num_execution_passes, dim=0),
                            strict=True,
                        )
                    }
                    if not stage1_cfg_parallel:
                        v_pos, a_v_pos = pass_outputs["cond"]
                        v_neg, a_v_neg = pass_outputs["neg"]
                        v_ptb, a_v_ptb = pass_outputs.get("perturbed", (None, None))
                        v_mod, a_v_mod = pass_outputs.get("modality", (None, None))

                sigma_value_float = float(sigma_value.item())
                video_sigma_for_x0: float | torch.Tensor = sigma_value_float
                audio_sigma_for_x0: float | torch.Tensor = sigma_value_float
                if ctx.use_ltx23_hq_timestep_semantics:
                    video_sigma_for_x0 = model_inputs_local.timestep_video
                    audio_sigma_for_x0 = model_inputs_local.timestep_audio
                elif ctx.denoise_mask is not None:
                    video_sigma_for_x0 = sigma_value.to(
                        device=video_latents.device, dtype=torch.float32
                    ) * ctx.denoise_mask.squeeze(-1)

                if stage1_cfg_parallel:
                    guided_video = self._ltx2_combine_guided_x0_parallel(
                        latents=video_latents,
                        local_velocities={
                            name: output[0] for name, output in pass_outputs.items()
                        },
                        sigma=video_sigma_for_x0,
                        cfg_scale=float(stage1_guider_params["video_cfg_scale"]),
                        stg_scale=float(stage1_guider_params["video_stg_scale"]),
                        rescale_scale=float(
                            stage1_guider_params["video_rescale_scale"]
                        ),
                        modality_scale=float(
                            stage1_guider_params["video_modality_scale"]
                        ),
                    )
                    if video_skip and ctx.last_denoised_video is not None:
                        denoised_video_local = ctx.last_denoised_video
                    else:
                        denoised_video_local = guided_video
                        if update_skip_cache:
                            ctx.last_denoised_video = guided_video

                    guided_audio = self._ltx2_combine_guided_x0_parallel(
                        latents=audio_latents,
                        local_velocities={
                            name: output[1] for name, output in pass_outputs.items()
                        },
                        sigma=audio_sigma_for_x0,
                        cfg_scale=float(stage1_guider_params["audio_cfg_scale"]),
                        stg_scale=float(stage1_guider_params["audio_stg_scale"]),
                        rescale_scale=float(
                            stage1_guider_params["audio_rescale_scale"]
                        ),
                        modality_scale=float(
                            stage1_guider_params["audio_modality_scale"]
                        ),
                    )
                    if audio_skip and ctx.last_denoised_audio is not None:
                        denoised_audio_local = ctx.last_denoised_audio
                    else:
                        denoised_audio_local = guided_audio
                        if update_skip_cache:
                            ctx.last_denoised_audio = guided_audio
                else:
                    denoised_video_local = self._ltx2_velocity_to_x0(
                        video_latents, v_pos, video_sigma_for_x0
                    )
                    denoised_audio_local = self._ltx2_velocity_to_x0(
                        audio_latents, a_v_pos, audio_sigma_for_x0
                    )
                    denoised_video_neg = self._ltx2_velocity_to_x0(
                        video_latents, v_neg, video_sigma_for_x0
                    )
                    denoised_audio_neg = self._ltx2_velocity_to_x0(
                        audio_latents, a_v_neg, audio_sigma_for_x0
                    )
                    denoised_video_perturbed = (
                        None
                        if v_ptb is None
                        else self._ltx2_velocity_to_x0(
                            video_latents, v_ptb, video_sigma_for_x0
                        )
                    )
                    denoised_audio_perturbed = (
                        None
                        if a_v_ptb is None
                        else self._ltx2_velocity_to_x0(
                            audio_latents, a_v_ptb, audio_sigma_for_x0
                        )
                    )
                    denoised_video_modality = (
                        None
                        if v_mod is None
                        else self._ltx2_velocity_to_x0(
                            video_latents, v_mod, video_sigma_for_x0
                        )
                    )
                    denoised_audio_modality = (
                        None
                        if a_v_mod is None
                        else self._ltx2_velocity_to_x0(
                            audio_latents, a_v_mod, audio_sigma_for_x0
                        )
                    )

                    guided_video = self._ltx2_calculate_guided_x0(
                        cond=denoised_video_local,
                        uncond_text=denoised_video_neg,
                        uncond_perturbed=(
                            denoised_video_perturbed
                            if denoised_video_perturbed is not None
                            else 0.0
                        ),
                        uncond_modality=(
                            denoised_video_modality
                            if denoised_video_modality is not None
                            else 0.0
                        ),
                        cfg_scale=float(stage1_guider_params["video_cfg_scale"]),
                        stg_scale=float(stage1_guider_params["video_stg_scale"]),
                        rescale_scale=float(
                            stage1_guider_params["video_rescale_scale"]
                        ),
                        modality_scale=float(
                            stage1_guider_params["video_modality_scale"]
                        ),
                    )
                    if video_skip and ctx.last_denoised_video is not None:
                        denoised_video_local = ctx.last_denoised_video
                    else:
                        denoised_video_local = guided_video
                        if update_skip_cache:
                            ctx.last_denoised_video = guided_video

                    guided_audio = self._ltx2_calculate_guided_x0(
                        cond=denoised_audio_local,
                        uncond_text=denoised_audio_neg,
                        uncond_perturbed=(
                            denoised_audio_perturbed
                            if denoised_audio_perturbed is not None
                            else 0.0
                        ),
                        uncond_modality=(
                            denoised_audio_modality
                            if denoised_audio_modality is not None
                            else 0.0
                        ),
                        cfg_scale=float(stage1_guider_params["audio_cfg_scale"]),
                        stg_scale=float(stage1_guider_params["audio_stg_scale"]),
                        rescale_scale=float(
                            stage1_guider_params["audio_rescale_scale"]
                        ),
                        modality_scale=float(
                            stage1_guider_params["audio_modality_scale"]
                        ),
                    )
                    if audio_skip and ctx.last_denoised_audio is not None:
                        denoised_audio_local = ctx.last_denoised_audio
                    else:
                        denoised_audio_local = guided_audio
                        if update_skip_cache:
                            ctx.last_denoised_audio = guided_audio

                denoised_video_local = self._ltx2_apply_clean_latent_mask(
                    denoised_video_local, ctx
                )
                return denoised_video_local, denoised_audio_local
            finally:
                ctx.latents = original_video_latents
                ctx.audio_latents = original_audio_latents

        denoised_video, denoised_audio = evaluate_stage1_guided_x0(
            video_latents=ctx.latents,
            audio_latents=ctx.audio_latents,
            sigma_value=sigma,
            update_skip_cache=True,
        )

        if self.sampler_name == "res2s":
            if sigma_val == 0.0 or sigma_next_val == 0.0:
                next_video_latents = denoised_video.to(dtype=ctx.latents.dtype)
                next_audio_latents = denoised_audio.to(dtype=ctx.audio_latents.dtype)
            else:
                sigma_d = sigma.double()
                sigma_next_d = sigma_next.double()
                if ctx.use_ltx23_hq_timestep_semantics:
                    h = self._ltx2_res2s_step_size_scalar(sigma_d, sigma_next_d)
                    a21, b1, b2 = self._ltx2_get_res2s_coefficients_scalar(h)
                    h_value = h
                else:
                    h = -torch.log(torch.clamp(sigma_next_d / sigma_d, min=1e-12))
                    a21, b1, b2 = self._ltx2_get_res2s_coefficients(h)
                    h_value = float(h.item())
                sub_sigma = torch.sqrt(torch.clamp(sigma_d * sigma_next_d, min=0.0))

                anchor_video = ctx.latents.double()
                anchor_audio = ctx.audio_latents.double()
                eps1_video = denoised_video.double() - anchor_video
                eps1_audio = denoised_audio.double() - anchor_audio

                midpoint_video_deterministic = anchor_video + h * a21 * eps1_video
                midpoint_audio_deterministic = anchor_audio + h * a21 * eps1_audio

                substep_video_noise = (
                    self._ltx2_res2s_noise_like(
                        ctx.latents, ctx, substep=True, batch=batch
                    ).float()
                    if ctx.use_native_hq_res2s_sde_noise
                    else self._randn_like_with_batch_generators(
                        ctx.latents, batch
                    ).float()
                )
                substep_audio_noise = (
                    self._ltx2_res2s_noise_like(
                        ctx.audio_latents,
                        ctx,
                        substep=True,
                        batch=batch,
                        is_audio=True,
                    ).float()
                    if ctx.use_native_hq_res2s_sde_noise
                    else self._randn_like_with_batch_generators(
                        ctx.audio_latents, batch
                    ).float()
                )

                midpoint_video_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_video,
                    denoised_sample=midpoint_video_deterministic,
                    sigma=sigma_d,
                    sigma_next=sub_sigma,
                    noise=substep_video_noise,
                    terminal=False,
                )
                midpoint_audio_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_audio,
                    denoised_sample=midpoint_audio_deterministic,
                    sigma=sigma_d,
                    sigma_next=sub_sigma,
                    noise=substep_audio_noise,
                    terminal=False,
                )

                midpoint_video_model_latents = self._ltx2_apply_clean_latent_mask(
                    midpoint_video_latents.to(dtype=ctx.latents.dtype),
                    ctx,
                )
                midpoint_audio_model_latents = midpoint_audio_latents.to(
                    dtype=ctx.audio_latents.dtype
                )

                if h_value < 0.5 and sigma_val > 0.03:
                    x_mid_v = midpoint_video_latents.double()
                    x_mid_a = midpoint_audio_latents.double()
                    for _ in range(100):
                        anchor_video = x_mid_v - h * a21 * eps1_video
                        eps1_video = denoised_video.double() - anchor_video
                        anchor_audio = x_mid_a - h * a21 * eps1_audio
                        eps1_audio = denoised_audio.double() - anchor_audio

                midpoint_denoised_video, midpoint_denoised_audio = (
                    evaluate_stage1_guided_x0(
                        video_latents=midpoint_video_model_latents,
                        audio_latents=midpoint_audio_model_latents,
                        sigma_value=sub_sigma,
                        update_skip_cache=False,
                    )
                )
                eps2_video = midpoint_denoised_video.double() - anchor_video
                eps2_audio = midpoint_denoised_audio.double() - anchor_audio

                next_video_deterministic = anchor_video + h * (
                    b1 * eps1_video + b2 * eps2_video
                )
                next_audio_deterministic = anchor_audio + h * (
                    b1 * eps1_audio + b2 * eps2_audio
                )

                step_video_noise = (
                    self._ltx2_res2s_noise_like(
                        ctx.latents, ctx, substep=False, batch=batch
                    ).float()
                    if ctx.use_native_hq_res2s_sde_noise
                    else self._randn_like_with_batch_generators(
                        ctx.latents, batch
                    ).float()
                )
                step_audio_noise = (
                    self._ltx2_res2s_noise_like(
                        ctx.audio_latents,
                        ctx,
                        substep=False,
                        batch=batch,
                        is_audio=True,
                    ).float()
                    if ctx.use_native_hq_res2s_sde_noise
                    else self._randn_like_with_batch_generators(
                        ctx.audio_latents, batch
                    ).float()
                )
                sde_sigma = sigma if ctx.use_ltx23_hq_timestep_semantics else sigma_d
                sde_sigma_next = (
                    sigma_next if ctx.use_ltx23_hq_timestep_semantics else sigma_next_d
                )
                next_video_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_video,
                    denoised_sample=next_video_deterministic,
                    sigma=sde_sigma,
                    sigma_next=sde_sigma_next,
                    noise=step_video_noise,
                    terminal=False,
                )
                next_audio_latents = self._ltx2_res2s_sde_step(
                    sample=anchor_audio,
                    denoised_sample=next_audio_deterministic,
                    sigma=sde_sigma,
                    sigma_next=sde_sigma_next,
                    noise=step_audio_noise,
                    terminal=False,
                )

                next_video_latents = self._ltx2_apply_clean_latent_mask(
                    next_video_latents.to(dtype=ctx.latents.dtype),
                    ctx,
                )
                next_audio_latents = next_audio_latents.to(
                    dtype=ctx.audio_latents.dtype
                )
        else:
            if sigma_val == 0.0:
                v_video = torch.zeros_like(denoised_video)
                v_audio = torch.zeros_like(denoised_audio)
            else:
                v_video = (
                    (ctx.latents.float() - denoised_video.float()) / sigma_val
                ).to(ctx.latents.dtype)
                v_audio = (
                    (ctx.audio_latents.float() - denoised_audio.float()) / sigma_val
                ).to(ctx.audio_latents.dtype)

            next_video_latents = (ctx.latents.float() + v_video.float() * dt).to(
                dtype=ctx.latents.dtype
            )
            next_audio_latents = (ctx.audio_latents.float() + v_audio.float() * dt).to(
                dtype=ctx.audio_latents.dtype
            )

        ctx.latents = next_video_latents
        ctx.audio_latents = next_audio_latents
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )

    def _record_trajectory(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Record audio trajectory alongside the base video trajectory."""
        super()._record_trajectory(ctx, step, batch, server_args)
        if batch.return_trajectory_latents and ctx.audio_latents is not None:
            ctx.trajectory_audio_latents.append(ctx.audio_latents)

    def _finalize_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Expose audio latents before delegating to AV-aware postprocessing."""
        batch.audio_latents = ctx.audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=ctx.latents,
            trajectory_latents=ctx.trajectory_latents,
            trajectory_timesteps=ctx.trajectory_timesteps,
            trajectory_audio_latents=ctx.trajectory_audio_latents,
            server_args=server_args,
            is_warmup=ctx.is_warmup,
        )

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
        trajectory_audio_latents: list | None = None,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Trim SP token padding before delegating to the base finalizer."""
        if trajectory_audio_latents:
            batch.trajectory_audio_latents = torch.stack(
                trajectory_audio_latents, dim=1
            ).cpu()
        latents = self._truncate_sp_padded_token_latents(batch, latents)
        super()._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )

    def _get_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list prompt embeddings for LTX-2 prompts."""
        return lambda x: V.is_tensor(x) or V.list_not_empty(x)

    def _get_negative_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list negative prompt embeddings for LTX-2 CFG."""
        return (
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x)
        )
