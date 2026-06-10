# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch  # type: ignore

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalSelfAttentionKVCache,
    CrossAttentionKVCache,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.utils import pred_noise_to_pred_video
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    get_or_create_request_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    RealtimeCausalDiTState,
    get_realtime_causal_dit_state,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class CausalDMDForwardContext:
    target_dtype: torch.dtype
    autocast_enabled: bool
    device: torch.device
    scheduler: Any
    timesteps: torch.Tensor
    latents: torch.Tensor
    prompt_embeds: Any
    image_kwargs: dict[str, Any]
    pos_cond_kwargs: dict[str, Any]
    batch_size: int
    channels: int
    num_frames: int
    height: int
    width: int


@dataclass(slots=True)
class CausalDMDCachePolicy:
    sequence_shard_enabled: bool
    num_attention_heads: int
    expected_cache_tokens: int
    expected_sink_tokens: int
    kv_cache_kwargs: dict[str, Any]


@dataclass(slots=True)
class CausalDMDRealtimeCacheContext:
    cache_state: RealtimeCausalDiTState
    persist_state: bool
    kv_cache: list[CausalSelfAttentionKVCache]
    crossattn_cache: list[CrossAttentionKVCache]
    current_start_frame: int
    chunk_idx: int


class CausalDMDDenoisingStage(DenoisingStage):
    """
    Denoising stage for causal diffusion.
    """

    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        # KV and cross-attention cache state (initialized on first forward)
        self.causal_kv_cache: list | None = None
        self.crossattn_cache: list | None = None
        # Model-dependent constants (aligned with causal_inference.py assumptions)
        self.num_transformer_blocks = self.transformer.config.arch_config.num_layers
        self.num_frames_per_block = (
            self.transformer.config.arch_config.num_frames_per_block
        )
        self.sliding_window_num_frames = (
            self.transformer.config.arch_config.sliding_window_num_frames
        )

        try:
            self.local_attn_size = getattr(
                self.transformer.model, "local_attn_size", -1
            )  # type: ignore
        except Exception:
            self.local_attn_size = -1
        self.sink_size = self.transformer.config.arch_config.sink_size

        self._causal_attn_metadata_builder_cls = None
        self._causal_attn_metadata_builder = None

    def _target_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _autocast_enabled(
        self,
        target_dtype: torch.dtype,
        server_args: ServerArgs,
    ) -> bool:
        return (target_dtype != torch.float32) and not server_args.disable_autocast

    def _prepare_frame_seq_length(self, h: int, w: int) -> int:
        patch_ratio = (
            self.transformer.config.arch_config.patch_size[-1]
            * self.transformer.config.arch_config.patch_size[-2]
        )
        self.num_token_per_frame = (h * w) // patch_ratio
        return self.num_token_per_frame

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        assert batch.latents is not None, "latents must be provided"
        return batch.latents

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        return get_or_create_request_scheduler(batch, self.scheduler)

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = torch.tensor(
            server_args.pipeline_config.dmd_denoising_steps, dtype=torch.long
        ).cpu()

        if server_args.pipeline_config.warp_denoising_step:
            logger.info("Warping timesteps...")
            scheduler_timesteps = torch.cat(
                (scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            timesteps = scheduler_timesteps[1000 - timesteps]
        timesteps = timesteps.to(device)
        logger.info("Using timesteps: %s", timesteps)
        return timesteps

    def _prepare_causal_dmd_image_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {}

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                # "encoder_hidden_states_2": batch.clip_embedding_pos,
                "encoder_attention_mask": batch.prompt_attention_mask,
            },
        )

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        prompt_embeds = batch.prompt_embeds
        assert torch.isnan(prompt_embeds[0]).sum() == 0
        return prompt_embeds

    def _prepare_causal_dmd_forward_context(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalDMDForwardContext:
        target_dtype = self._target_dtype()
        autocast_enabled = self._autocast_enabled(target_dtype, server_args)
        device = get_local_torch_device()
        scheduler = self._get_causal_dmd_scheduler(batch, server_args)
        latents = self._get_causal_dmd_latents(batch)
        b, c, t, h, w = latents.shape
        self._prepare_frame_seq_length(h, w)
        timesteps = self._prepare_causal_dmd_timesteps(
            batch,
            server_args,
            scheduler,
            device,
        )
        image_kwargs = self._prepare_causal_dmd_image_kwargs(
            batch,
            server_args,
            target_dtype,
        )
        pos_cond_kwargs = self._prepare_causal_dmd_pos_cond_kwargs(
            batch,
            server_args,
            target_dtype,
        )

        if self.attn_backend.get_enum() == AttentionBackendEnum.SLIDING_TILE_ATTN:
            self.prepare_sta_param(batch, server_args)

        prompt_embeds = self._prepare_causal_dmd_prompt_embeds(
            batch,
            server_args,
            target_dtype,
        )
        return CausalDMDForwardContext(
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            device=device,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            prompt_embeds=prompt_embeds,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            batch_size=b,
            channels=c,
            num_frames=t,
            height=h,
            width=w,
        )

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        pipeline_config = server_args.pipeline_config
        sink_size = getattr(batch, "realtime_causal_sink_size", None)
        if sink_size is None:
            sink_size = getattr(pipeline_config, "realtime_causal_sink_size", None)
        if sink_size is not None:
            if sink_size < 0:
                raise ValueError("realtime_causal_sink_size must be non-negative")
            self.sink_size = int(sink_size)

        kv_cache_num_frames = getattr(
            batch,
            "realtime_causal_kv_cache_num_frames",
            None,
        )
        if kv_cache_num_frames is None:
            kv_cache_num_frames = getattr(
                pipeline_config,
                "realtime_causal_kv_cache_num_frames",
                None,
            )
        if kv_cache_num_frames is not None:
            if kv_cache_num_frames <= 0:
                raise ValueError("realtime_causal_kv_cache_num_frames must be positive")
            self.sliding_window_num_frames = int(kv_cache_num_frames)

    def _causal_sequence_shard_enabled(self, batch: Req) -> bool:
        return False

    def _num_causal_cache_attention_heads(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> int:
        return self.transformer.num_attention_heads

    def _causal_kv_cache_kwargs(
        self,
        policy: CausalDMDCachePolicy,
    ) -> dict[str, Any]:
        return {}

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return False

    def _build_realtime_causal_cache_policy(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalDMDCachePolicy:
        self._apply_causal_cache_overrides(batch, server_args)
        sequence_shard_enabled = self._causal_sequence_shard_enabled(batch)
        policy = CausalDMDCachePolicy(
            sequence_shard_enabled=sequence_shard_enabled,
            num_attention_heads=self._num_causal_cache_attention_heads(
                sequence_shard_enabled=sequence_shard_enabled
            ),
            expected_cache_tokens=self._get_causal_kv_cache_size(
                sequence_shard_enabled=sequence_shard_enabled
            ),
            expected_sink_tokens=self._get_causal_sink_tokens(),
            kv_cache_kwargs={},
        )
        policy.kv_cache_kwargs = self._causal_kv_cache_kwargs(policy)
        return policy

    def _get_realtime_causal_cache_state(
        self,
        batch: Req,
    ) -> tuple[RealtimeCausalDiTState, bool]:
        if batch.session is not None:
            state = get_realtime_causal_dit_state(batch.session)
            return state, True
        return RealtimeCausalDiTState(), False

    def _clear_stage_causal_cache_refs(self) -> None:
        self.causal_kv_cache = None
        self.crossattn_cache = None

    def _should_reset_realtime_causal_caches(
        self,
        batch: Req,
        *,
        cache_state: RealtimeCausalDiTState,
        policy: CausalDMDCachePolicy,
    ) -> bool:
        causal_kv_cache = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache
        return (
            batch.block_idx == 0
            or causal_kv_cache is None
            or crossattn_cache is None
            or len(causal_kv_cache) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or causal_kv_cache[0].k.shape[1] != policy.expected_cache_tokens
            or causal_kv_cache[0].k.shape[2] != policy.num_attention_heads
            or causal_kv_cache[0].sink_tokens != policy.expected_sink_tokens
        )

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        policy = self._build_realtime_causal_cache_policy(batch, server_args)
        cache_state, persist_state = self._get_realtime_causal_cache_state(batch)

        if self._should_reset_realtime_causal_caches(
            batch,
            cache_state=cache_state,
            policy=policy,
        ):
            causal_kv_cache, crossattn_cache = self._initialize_causal_caches(
                batch_size=ctx.batch_size,
                max_text_len=self._get_max_text_len(server_args),
                dtype=ctx.target_dtype,
                device=ctx.device,
                kv_cache_kwargs=policy.kv_cache_kwargs,
            )
            cache_state.kv_cache = causal_kv_cache
            cache_state.crossattn_cache = crossattn_cache
            self._clear_stage_causal_cache_refs()
            cache_state.current_chunk_start_frame = 0
            cache_state.chunk_idx = 0
        else:
            causal_kv_cache = cache_state.kv_cache
            crossattn_cache = cache_state.crossattn_cache

        assert causal_kv_cache is not None
        assert crossattn_cache is not None
        return CausalDMDRealtimeCacheContext(
            cache_state=cache_state,
            persist_state=persist_state,
            kv_cache=causal_kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=cache_state.current_chunk_start_frame,
            chunk_idx=cache_state.chunk_idx,
        )

    def _advance_realtime_causal_cache(
        self,
        cache_ctx: CausalDMDRealtimeCacheContext,
        *,
        num_frames: int,
    ) -> None:
        cache_ctx.cache_state.current_chunk_start_frame += num_frames
        cache_ctx.cache_state.chunk_idx += 1

    def _realtime_causal_progress_bar(self, batch: Req, timesteps: torch.Tensor):
        if batch.session is not None:
            return nullcontext(None)
        return self.progress_bar(total=len(timesteps))

    def _denoise_realtime_causal_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx: CausalDMDForwardContext,
        cache_ctx: CausalDMDRealtimeCacheContext,
        chunk_latents: torch.Tensor,
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        prepare_context_input: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        current_start_frame = cache_ctx.current_start_frame
        with self._realtime_causal_progress_bar(batch, ctx.timesteps) as progress_bar:
            return self._denoise_and_update_causal_block(
                batch,
                server_args,
                chunk_latents=chunk_latents,
                scheduler=ctx.scheduler,
                timesteps=ctx.timesteps,
                prompt_embeds=ctx.prompt_embeds,
                kv_cache=cache_ctx.kv_cache,
                crossattn_cache=cache_ctx.crossattn_cache,
                current_start_tokens=current_start_frame * self.num_token_per_frame,
                start_frame=current_start_frame,
                image_kwargs=ctx.image_kwargs,
                pos_cond_kwargs=ctx.pos_cond_kwargs,
                target_dtype=ctx.target_dtype,
                autocast_enabled=ctx.autocast_enabled,
                device=ctx.device,
                attn_raw_latent_shape=(ctx.num_frames, ctx.height, ctx.width),
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_context_input,
                progress_bar=progress_bar,
            )

    def _build_causal_attn_metadata(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        current_timestep: int,
        raw_latent_shape: tuple[int, int, int],
        device: torch.device,
    ):
        if self.attn_backend.get_enum() != AttentionBackendEnum.VIDEO_SPARSE_ATTN:
            return None

        builder_cls = self.attn_backend.get_builder_cls()
        if builder_cls is None:
            return None
        if self._causal_attn_metadata_builder_cls is not builder_cls:
            self._causal_attn_metadata_builder_cls = builder_cls
            self._causal_attn_metadata_builder = builder_cls()
        attn_metadata = self._causal_attn_metadata_builder.build(
            current_timestep=current_timestep,
            raw_latent_shape=raw_latent_shape,
            patch_size=server_args.pipeline_config.dit_config.patch_size,
            STA_param=batch.STA_param,
            VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
            device=device,
        )
        assert attn_metadata is not None, "attn_metadata cannot be None"
        return attn_metadata

    @staticmethod
    def _single_generator(batch: Req):
        if isinstance(batch.generator, list):
            return batch.generator[0]
        return batch.generator

    @staticmethod
    def _expand_timestep(
        timestep: torch.Tensor, batch_size: int, device
    ) -> torch.Tensor:
        return timestep.reshape(1).to(device=device).expand(batch_size)

    def _forward_causal_transformer(
        self,
        batch: Req,
        *,
        latent_model_input: torch.Tensor,
        prompt_embeds,
        timestep: torch.Tensor,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        current_timestep: int,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> torch.Tensor:
        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=current_timestep,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ),
        ):
            return self.transformer(
                latent_model_input,
                prompt_embeds,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_tokens,
                start_frame=start_frame,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    def _predict_x0_btchw(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        latent_model_input: torch.Tensor,
        noise_latents_btchw: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_raw_latent_shape: tuple[int, int, int],
        current_timestep: int,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, object | None]:
        attn_metadata = self._build_causal_attn_metadata(
            batch,
            server_args,
            current_timestep=current_timestep,
            raw_latent_shape=attn_raw_latent_shape,
            device=device,
        )
        batch_size = latent_model_input.shape[0]
        timestep_2d = self._expand_timestep(
            timestep, batch_size, latent_model_input.device
        )
        pred_noise = self._forward_causal_transformer(
            batch,
            latent_model_input=latent_model_input,
            prompt_embeds=prompt_embeds,
            timestep=timestep_2d.unsqueeze(1),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        pred_noise_btchw = pred_noise.permute(0, 2, 1, 3, 4)
        x0_btchw = pred_noise_to_pred_video(
            pred_noise=pred_noise_btchw.flatten(0, 1),
            noise_input_latent=noise_latents_btchw.flatten(0, 1),
            timestep=timestep_2d,
            scheduler=scheduler,
        ).unflatten(0, pred_noise_btchw.shape[:2])
        return x0_btchw, attn_metadata

    def _add_noise_for_next_timestep(
        self,
        batch: Req,
        *,
        x0_btchw: torch.Tensor,
        raw_latent_shape: torch.Size,
        next_timestep: torch.Tensor,
        scheduler,
        device,
    ) -> torch.Tensor:
        noise = torch.randn(
            raw_latent_shape,
            dtype=x0_btchw.dtype,
            generator=self._single_generator(batch),
            device=device,
        )
        return scheduler.add_noise(
            x0_btchw.flatten(0, 1),
            noise.flatten(0, 1),
            next_timestep,
        ).unflatten(0, x0_btchw.shape[:2])

    def _denoise_causal_dmd_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> tuple[torch.Tensor, object | None]:
        current_latents = chunk_latents
        noise_latents_btchw = current_latents.permute(0, 2, 1, 3, 4)
        raw_latent_shape = noise_latents_btchw.shape
        attn_metadata = None

        for i, timestep in enumerate(timesteps):
            noise_latents = noise_latents_btchw
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            x0_btchw, attn_metadata = self._predict_x0_btchw(
                batch,
                server_args,
                latent_model_input=latent_model_input,
                noise_latents_btchw=noise_latents,
                timestep=timestep,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                attn_raw_latent_shape=attn_raw_latent_shape,
                current_timestep=i,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                device=device,
            )

            if i < len(timesteps) - 1:
                next_timestep = timesteps[i + 1 : i + 2]
                noise_latents_btchw = self._add_noise_for_next_timestep(
                    batch,
                    x0_btchw=x0_btchw,
                    raw_latent_shape=raw_latent_shape,
                    next_timestep=next_timestep,
                    scheduler=scheduler,
                    device=device,
                )
                current_latents = noise_latents_btchw.permute(0, 2, 1, 3, 4)
            else:
                current_latents = x0_btchw.permute(0, 2, 1, 3, 4)

            if progress_bar is not None:
                progress_bar.update()

        return current_latents, attn_metadata

    def _update_causal_context_cache(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        attn_metadata,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        """fill the self-attn KV cache by performing a one-time forward on DiT"""
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        timestep = torch.full(
            (context_input.shape[0], 1),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
        self._forward_causal_transformer(
            batch,
            latent_model_input=context_input.to(target_dtype),
            prompt_embeds=prompt_embeds,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=0,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

    def _warm_up_causal_context_cache(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        context_input: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
    ) -> None:
        self._update_causal_context_cache(
            batch,
            server_args,
            context_input=context_input,
            prompt_embeds=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_frame * self.num_token_per_frame,
            start_frame=current_start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            attn_metadata=None,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

    def _denoise_and_update_causal_block(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        kv_cache,
        crossattn_cache,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        prepare_context_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> torch.Tensor:
        current_latents, attn_metadata = self._denoise_causal_dmd_chunk(
            batch,
            server_args,
            chunk_latents=chunk_latents,
            scheduler=scheduler,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            device=device,
            attn_raw_latent_shape=attn_raw_latent_shape,
            prepare_model_input=prepare_model_input,
            progress_bar=progress_bar,
        )
        self._update_causal_context_cache(
            batch,
            server_args,
            context_input=prepare_context_input(current_latents),
            prompt_embeds=prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        return current_latents

    def _get_max_text_len(self, server_args: ServerArgs) -> int:
        return server_args.pipeline_config.text_encoder_configs[0].arch_config.text_len

    def _initialize_causal_caches(
        self,
        *,
        batch_size: int,
        max_text_len: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_cache_kwargs: dict | None = None,
    ) -> tuple[list, list]:
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            **(kv_cache_kwargs or {}),
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            max_text_len=max_text_len,
            dtype=dtype,
            device=device,
        )
        assert self.causal_kv_cache is not None
        assert self.crossattn_cache is not None
        return self.causal_kv_cache, self.crossattn_cache

    def _reset_causal_caches(
        self,
        *,
        kv_cache,
        crossattn_cache,
    ) -> None:
        self._reset_crossattn_cache(crossattn_cache)
        self._reset_kv_cache(kv_cache)

    def _reset_crossattn_cache(self, crossattn_cache) -> None:
        for cache_block in crossattn_cache:
            cache_block.reset()

    @staticmethod
    def _reset_kv_cache(kv_cache) -> None:
        for cache_block in kv_cache:
            cache_block.reset_indices()

    def _get_causal_kv_cache_size(
        self,
        *,
        sequence_shard_enabled: bool = False,
    ) -> int:
        if self.local_attn_size != -1:
            return self.local_attn_size * self.num_token_per_frame
        return self.num_token_per_frame * self.sliding_window_num_frames

    def _get_causal_sink_tokens(self) -> int:
        return self.sink_size * self.num_token_per_frame

    def _get_causal_attention_window_size(self, kv_cache_size: int) -> int:
        return kv_cache_size

    def _allocate_causal_kv_cache(
        self,
        *,
        batch_size: int,
        kv_cache_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dtype: torch.dtype,
        device,
        use_int_indices: bool = False,
        sink_tokens: int = 0,
        attention_window_size: int | None = None,
        allow_growth: bool = False,
    ) -> list[CausalSelfAttentionKVCache]:
        causal_kv_cache = []
        int_index = 0 if use_int_indices else None
        if attention_window_size is None:
            attention_window_size = kv_cache_size
        for _ in range(self.num_transformer_blocks):
            causal_kv_cache.append(
                CausalSelfAttentionKVCache(
                    k=torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    v=torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    global_end_index=torch.zeros(1, dtype=torch.long, device=device),
                    local_end_index=torch.zeros(1, dtype=torch.long, device=device),
                    global_end_index_int=int_index,
                    local_end_index_int=int_index,
                    cache_size=kv_cache_size,
                    sink_tokens=sink_tokens,
                    attention_window_size=attention_window_size,
                    allow_growth=allow_growth,
                )
            )
        return causal_kv_cache

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        target_dtype = ctx.target_dtype
        autocast_enabled = ctx.autocast_enabled
        scheduler = ctx.scheduler
        device = ctx.device
        timesteps = ctx.timesteps
        image_kwargs = ctx.image_kwargs
        pos_cond_kwargs = ctx.pos_cond_kwargs
        latents = ctx.latents
        prompt_embeds = ctx.prompt_embeds
        t, h, w = ctx.num_frames, ctx.height, ctx.width

        # TODO(will): make this a parameter once we add i2v support
        independent_first_frame = self.transformer.independent_first_frame

        # Initialize or reset caches
        if self.causal_kv_cache is None:
            self._initialize_causal_caches(
                batch_size=latents.shape[0],
                max_text_len=self._get_max_text_len(server_args),
                dtype=target_dtype,
                device=latents.device,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        # Optional: cache context features from provided image latents prior to generation
        current_start_frame = 0
        if getattr(batch, "image_latent", None) is not None:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            if independent_first_frame and input_frames >= 1:
                # warm-up with the very first frame independently
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            # process remaining input frames in blocks of num_frame_per_block
            while remaining_frames > 0:
                block = min(self.num_frames_per_block, remaining_frames)
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[
                        :, :, current_start_frame : current_start_frame + block, :, :
                    ],
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += block
                remaining_frames -= block

        # Base position offset from any cache warm-up
        pos_start_base = current_start_frame

        # Determine block sizes
        if not independent_first_frame or (
            independent_first_frame and batch.image_latent is not None
        ):
            if t % self.num_frames_per_block != 0:
                raise ValueError(
                    "num_frames must be divisible by num_frames_per_block for causal DMD denoising"
                )
            num_blocks = t // self.num_frames_per_block
            block_sizes = [self.num_frames_per_block] * num_blocks
            start_index = 0
        else:
            if (t - 1) % self.num_frames_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
                )
            num_blocks = (t - 1) // self.num_frames_per_block
            block_sizes = [1] + [self.num_frames_per_block] * num_blocks
            start_index = 0

        def prepare_context_input(current_latents):
            return current_latents

        # DMD loop in causal blocks
        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for current_num_frames in block_sizes:
                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames, :, :
                ]

                def prepare_model_input(current_latents):
                    latent_model_input = current_latents
                    if (
                        batch.image_latent is not None
                        and independent_first_frame
                        and start_index == 0
                    ):
                        latent_model_input = torch.cat(
                            [latent_model_input, batch.image_latent],
                            dim=2,
                        )
                    return latent_model_input

                current_start_tokens = (
                    pos_start_base + start_index
                ) * self.num_token_per_frame
                current_latents = self._denoise_and_update_causal_block(
                    batch,
                    server_args,
                    chunk_latents=current_latents,
                    scheduler=scheduler,
                    timesteps=timesteps,
                    prompt_embeds=prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=current_start_tokens,
                    start_frame=start_index,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    device=device,
                    attn_raw_latent_shape=(current_num_frames, h, w),
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_context_input,
                    progress_bar=progress_bar,
                )

                # Write back and advance
                latents[:, :, start_index : start_index + current_num_frames, :, :] = (
                    current_latents
                )

                start_index += current_num_frames

        batch.latents = latents
        return batch

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        *,
        sequence_shard_enabled: bool = False,
    ) -> None:
        """
        Initialize (but not fill) a Per-GPU KV cache aligned with the model assumptions.
        """
        num_attention_heads = self._num_causal_cache_attention_heads(
            sequence_shard_enabled=sequence_shard_enabled
        )
        attention_head_dim = self.transformer.attention_head_dim
        kv_cache_size = self._get_causal_kv_cache_size(
            sequence_shard_enabled=sequence_shard_enabled
        )
        self.causal_kv_cache = self._allocate_causal_kv_cache(
            batch_size=batch_size,
            kv_cache_size=kv_cache_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dtype=dtype,
            device=device,
            use_int_indices=self._use_causal_cache_int_indices(
                sequence_shard_enabled=sequence_shard_enabled
            ),
            sink_tokens=self._get_causal_sink_tokens(),
            attention_window_size=self._get_causal_attention_window_size(kv_cache_size),
        )

    def _initialize_crossattn_cache(
        self, batch_size, max_text_len, dtype, device
    ) -> None:
        """
        Initialize a Per-GPU cross-attention cache aligned with the Wan model assumptions.
        """
        crossattn_cache = []
        num_attention_heads = self.transformer.num_attention_heads
        attention_head_dim = self.transformer.attention_head_dim
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                CrossAttentionKVCache(
                    k=torch.zeros(
                        [
                            batch_size,
                            max_text_len,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    v=torch.zeros(
                        [
                            batch_size,
                            max_text_len,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                )
            )
        self.crossattn_cache = crossattn_cache

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs."""
        result = VerificationResult()
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check("image_embeds", batch.image_embeds, V.is_list)
        result.add_check(
            "image_latent", batch.image_latent, V.none_or_tensor_with_dims(5)
        )
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
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x),
        )
        return result
