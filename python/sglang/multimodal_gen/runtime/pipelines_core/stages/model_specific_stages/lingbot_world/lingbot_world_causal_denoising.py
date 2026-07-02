# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""LingBot-World causal DMD denoising stage."""

from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects
    ``[noise(16ch), condition(20ch)]`` concatenated along channel dim.
    Each call processes one chunk (num_frames_per_block frames).
    """

    def _get_causal_kv_cache_size(
        self,
        *,
        sequence_shard_enabled: bool = False,
    ) -> int:
        if self.local_attn_size != -1:
            return self.local_attn_size * self.num_token_per_frame

        return self.sliding_window_num_frames * self.num_token_per_frame

    def _causal_sequence_shard_enabled(self, batch: Req) -> bool:
        return bool(
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )

    def _num_causal_cache_attention_heads(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> int:
        num_attention_heads = self.transformer.num_attention_heads
        if not sequence_shard_enabled:
            return num_attention_heads

        ulysses_world_size = get_ulysses_parallel_world_size()
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
            )
        if ulysses_world_size <= 1:
            raise ValueError(
                "LingBot causal sequence sharding requires ulysses_degree > 1."
            )
        if num_attention_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
            )
        return num_attention_heads // ulysses_world_size

    def _causal_kv_cache_kwargs(
        self,
        policy: CausalDMDCachePolicy,
    ) -> dict[str, Any]:
        return {
            "sequence_shard_enabled": policy.sequence_shard_enabled,
            "kv_cache_size": policy.expected_cache_tokens,
        }

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return True

    @staticmethod
    def _chunk_has_camera_motion(actions) -> bool:
        if not actions:
            return False
        for frame_actions in actions:
            if frame_actions:
                return True
        return False

    def _uses_interactive_kv_window(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        if not self._interactive_kv_window_enabled(server_args):
            return False
        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        return "camera_actions" in condition_inputs

    @staticmethod
    def _interactive_kv_window_enabled(server_args: ServerArgs) -> bool:
        config_enabled = bool(
            getattr(
                server_args.pipeline_config,
                "interactive_kv_window_enable",
                False,
            )
        )
        return config_enabled or envs.SGLANG_LINGBOT_ENABLE_INTERACTIVE_KV_WINDOW

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        super()._apply_causal_cache_overrides(batch, server_args)
        self._sync_interactive_kv_cache_window(server_args)

    def _sync_interactive_kv_cache_window(self, server_args: ServerArgs) -> None:
        if not self._interactive_kv_window_enabled(server_args):
            return
        if self.local_attn_size != -1:
            return
        self.sliding_window_num_frames = (
            self._effective_interactive_kv_cache_num_frames(server_args)
        )

    def _effective_interactive_kv_cache_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int:
        cache_window = int(self.sliding_window_num_frames)
        if self.local_attn_size != -1:
            return cache_window

        pipeline_config = server_args.pipeline_config
        moving_window = self._moving_kv_sample_num_frames(server_args) or 0
        still_window = max(
            0, int(getattr(pipeline_config, "interactive_kv_still_window", 0))
        )
        return max(
            cache_window,
            int(self.sink_size)
            + max(moving_window, still_window)
            + int(self.num_frames_per_block),
        )

    def _build_realtime_causal_cache_policy(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> CausalDMDCachePolicy:
        policy = super()._build_realtime_causal_cache_policy(batch, server_args)
        if self._interactive_kv_window_enabled(server_args):
            policy.expected_cache_tokens = (
                self._effective_interactive_kv_cache_num_frames(server_args)
                * self.num_token_per_frame
            )
        return policy

    def _base_kv_sample_num_frames(self) -> int | None:
        sample_frames = (
            int(self.sliding_window_num_frames)
            - int(self.sink_size)
            - int(self.num_frames_per_block)
        )
        return sample_frames if sample_frames > 0 else None

    def _moving_kv_sample_num_frames(
        self,
        server_args: ServerArgs,
    ) -> int | None:
        moving_window = getattr(
            server_args.pipeline_config,
            "interactive_kv_moving_window",
            None,
        )
        if moving_window is None:
            return self._base_kv_sample_num_frames()
        return max(0, int(moving_window))

    def _get_interactive_kv_sample_num_frames(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        pipeline_config = server_args.pipeline_config
        if not self._interactive_kv_window_enabled(server_args):
            return None
        if not self._uses_interactive_kv_window(batch, server_args):
            return self._base_kv_sample_num_frames()

        dynamic_state = cache_state.runtime_cache.setdefault(
            "lingbot_interactive_kv_window",
            {
                "consecutive_still_chunks": 0,
                "sample_num_frames": None,
            },
        )
        if cache_state.chunk_idx == 0:
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = None

        moving_window = self._moving_kv_sample_num_frames(server_args)
        if moving_window is None:
            return None
        still_window = int(getattr(pipeline_config, "interactive_kv_still_window", 3))
        still_chunks_threshold = max(
            1, int(getattr(pipeline_config, "interactive_kv_still_chunks", 2))
        )
        if dynamic_state["sample_num_frames"] is None:
            dynamic_state["sample_num_frames"] = moving_window

        condition_inputs = getattr(batch, "condition_inputs", None) or {}
        if self._chunk_has_camera_motion(condition_inputs.get("camera_actions")):
            dynamic_state["consecutive_still_chunks"] = 0
            dynamic_state["sample_num_frames"] = moving_window
        else:
            dynamic_state["consecutive_still_chunks"] += 1
            if dynamic_state["consecutive_still_chunks"] >= still_chunks_threshold:
                dynamic_state["sample_num_frames"] = still_window

        return int(dynamic_state["sample_num_frames"])

    def _set_lingbot_kv_sample_tokens(
        self,
        cache_state,
        batch: Req,
        server_args: ServerArgs,
    ) -> int | None:
        self._sync_interactive_kv_cache_window(server_args)
        sample_frames = self._get_interactive_kv_sample_num_frames(
            cache_state,
            batch,
            server_args,
        )
        sample_tokens = (
            None
            if sample_frames is None
            else int(sample_frames) * self.num_token_per_frame
        )
        previous = getattr(batch, "realtime_causal_kv_sample_tokens", None)
        batch.realtime_causal_kv_sample_tokens = sample_tokens
        return previous

    @staticmethod
    def _clear_lingbot_dynamic_condition_cache(cache_state) -> None:
        runtime_cache = getattr(cache_state, "runtime_cache", None)
        if runtime_cache is None:
            return
        runtime_cache.pop("lingbot_c2ws_plucker_emb", None)
        runtime_cache.pop("lingbot_cam_conditioner", None)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result

    def _denoise_realtime_causal_chunk(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        ctx,
        cache_ctx,
        chunk_latents: torch.Tensor,
        prepare_model_input,
        prepare_context_input,
    ) -> torch.Tensor:
        previous_sample_tokens = self._set_lingbot_kv_sample_tokens(
            cache_ctx.cache_state,
            batch,
            server_args,
        )
        try:
            return super()._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=chunk_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_context_input,
            )
        finally:
            batch.realtime_causal_kv_sample_tokens = previous_sample_tokens

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        latents = batch.latents
        assert latents is not None, (
            "LingBot-World causal DMD requires prepared chunk latents. "
            "Ensure RealtimeChunkLatentPreparationStage runs before this stage."
        )
        return latents

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        scheduler = batch.scheduler
        assert scheduler is not None, (
            "LingBot-World causal DMD requires prepared DMD timesteps. "
            "Ensure DMDTimestepPreparationStage runs before this stage."
        )
        return scheduler

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = batch.timesteps
        assert timesteps is not None
        return timesteps.to(device)

    def _prepare_causal_dmd_image_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        image_embeds = getattr(batch, "image_embeds", [])
        if len(image_embeds) > 0:
            image_embeds = [ie.to(target_dtype) for ie in image_embeds]
        return {
            "encoder_hidden_states_image": image_embeds,
        }

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        # lingbot transformer forward uses varargs, so inspect filtering drops valid kwargs
        return server_args.pipeline_config.prepare_pos_cond_kwargs(
            batch,
            self.device,
            getattr(self.transformer, "rotary_emb", None),
            dtype=target_dtype,
        )

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ):
        return server_args.pipeline_config.get_pos_prompt_embeds(batch)

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
        context_noise = getattr(server_args.pipeline_config, "context_noise", 0)
        timestep = torch.full(
            (context_input.shape[0], 1),
            int(context_noise),
            device=context_input.device,
            dtype=torch.long,
        )
        with (
            torch.autocast(
                device_type=current_platform.device_type,
                dtype=target_dtype,
                enabled=autocast_enabled,
            ),
            set_forward_context(
                current_timestep=-1,
                attn_metadata=attn_metadata,
                forward_batch=batch,
            ),
        ):
            self.transformer(
                context_input.to(target_dtype),
                prompt_embeds,
                timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_tokens,
                start_frame=start_frame,
                skip_final_projection=True,
                **image_kwargs,
                **pos_cond_kwargs,
            )

    @staticmethod
    def _select_i2v_condition_chunk(
        condition_full: torch.Tensor,
        chunk_idx: int,
        chunk_size: int,
    ) -> torch.Tensor:
        condition_chunks = condition_full.split(chunk_size, dim=2)
        condition = condition_chunks[min(chunk_idx, len(condition_chunks) - 1)]

        if condition.shape[2] == chunk_size:
            return condition
        pad_frames = chunk_size - condition.shape[2]
        return torch.cat(
            [
                condition,
                condition.new_zeros(
                    condition.shape[0],
                    condition.shape[1],
                    pad_frames,
                    condition.shape[3],
                    condition.shape[4],
                ),
            ],
            dim=2,
        )

    @staticmethod
    def _build_i2v_model_input_writer(
        *,
        latents: torch.Tensor,
        condition: torch.Tensor,
        target_dtype: torch.dtype,
        device: torch.device,
    ):
        b, latent_channels, t, h, w = latents.shape
        condition = condition.to(device=device, dtype=target_dtype)
        model_input = torch.empty(
            (
                b,
                latent_channels + condition.shape[1],
                t,
                h,
                w,
            ),
            dtype=target_dtype,
            device=device,
        )
        model_input[:, latent_channels:].copy_(condition)

        def write(current_latents: torch.Tensor) -> torch.Tensor:
            model_input[:, :latent_channels].copy_(current_latents)
            return model_input

        return write

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # --- Condition: take current chunk's slice ---
        condition_full = batch.image_latent
        assert condition_full is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        latents = ctx.latents
        cache_ctx = self._prepare_realtime_causal_caches(batch, server_args, ctx)

        # Keep cross-attention K/V cache across realtime chunks; LingBot text/image
        # conditions are session-static and are invalidated by cache reset.

        # Slice condition to current chunk
        condition = self._select_i2v_condition_chunk(
            condition_full,
            cache_ctx.chunk_idx,
            ctx.num_frames,
        )

        # --- Denoising loop (single chunk) ---
        current_latents = latents
        prepare_model_input = self._build_i2v_model_input_writer(
            latents=current_latents,
            condition=condition,
            target_dtype=ctx.target_dtype,
            device=ctx.device,
        )

        try:
            current_latents = self._denoise_realtime_causal_chunk(
                batch,
                server_args,
                ctx=ctx,
                cache_ctx=cache_ctx,
                chunk_latents=current_latents,
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_model_input,
            )
        finally:
            self._clear_lingbot_dynamic_condition_cache(cache_ctx.cache_state)

        # Advance cumulative frame position
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch
