# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/Robbyant/lingbot-world

"""
LingBot-World causal DMD denoising stage.

Extends CausalDMDDenoisingStage with:
- I2V condition concatenation ([noise, condition] along channel dim)
- Session-persistent KV cache with cumulative frame position tracking
"""

import torch

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.realtime_states import (
    RealtimeCausalDiTState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LingBotWorldCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """Causal DMD denoising with I2V condition concatenation for LingBot-World.

    The LingBot-World transformer has ``in_channels = 36`` and expects
    ``[noise(16ch), condition(20ch)]`` concatenated along channel dim.
    Each call processes one chunk (num_frames_per_block frames).
    """

    def _get_cache_state(
        self,
        batch: Req,
    ) -> tuple[RealtimeCausalDiTState, bool]:
        if batch.session is not None:
            state = batch.session.get_or_create_state(RealtimeCausalDiTState)
            return state, True
        return RealtimeCausalDiTState(), False

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

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        *,
        sequence_shard_enabled: bool = False,
    ) -> None:
        kv_cache1 = []
        num_attention_heads = self.transformer.num_attention_heads
        if sequence_shard_enabled:
            ulysses_world_size = get_ulysses_parallel_world_size()
            if ulysses_world_size <= 1:
                raise ValueError(
                    "LingBot causal sequence sharding requires ulysses_degree > 1."
                )
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently supports ring_degree = 1 only."
                )
            if num_attention_heads % ulysses_world_size != 0:
                raise ValueError(
                    f"num_attention_heads ({num_attention_heads}) must be divisible by ulysses_degree ({ulysses_world_size})."
                )
            num_attention_heads //= ulysses_world_size
        attention_head_dim = self.transformer.attention_head_dim
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = self.frame_seq_length * self.sliding_window_num_frames
        logger.info(
            "LingBot KV cache init: batch=%s layers=%s frame_seq_length=%s "
            "num_frames_per_block=%s sliding_window_num_frames=%s local_attn_size=%s "
            "sink_size=%s kv_cache_tokens=%s heads=%s head_dim=%s sequence_shard=%s",
            batch_size,
            self.num_transformer_blocks,
            self.frame_seq_length,
            self.num_frames_per_block,
            self.sliding_window_num_frames,
            self.local_attn_size,
            self.transformer.config.arch_config.sink_size,
            kv_cache_size,
            num_attention_heads,
            attention_head_dim,
            sequence_shard_enabled,
        )

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [
                            batch_size,
                            kv_cache_size,
                            num_attention_heads,
                            attention_head_dim,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "global_end_index_int": 0,
                    "local_end_index_int": 0,
                }
            )

        self.kv_cache1 = kv_cache1

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

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # --- Condition: take current chunk's slice ---
        condition_full = batch.image_latent
        assert condition_full is not None, (
            "LingBot-World causal DMD requires image_latent as condition. "
            "Ensure ImageVAEEncodingStage runs before this stage."
        )
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        target_dtype = ctx.target_dtype
        autocast_enabled = ctx.autocast_enabled
        device = ctx.device
        scheduler = ctx.scheduler
        timesteps = ctx.timesteps
        image_kwargs = ctx.image_kwargs
        pos_cond_kwargs = ctx.pos_cond_kwargs
        latents = ctx.latents
        prompt_embeds = ctx.prompt_embeds
        b, t, h, w = ctx.batch_size, ctx.num_frames, ctx.height, ctx.width

        # --- KV cache from session state ---
        cache_state, persist_cache_state = self._get_cache_state(batch)
        kv_cache1 = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache
        sequence_shard_enabled = bool(
            getattr(batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )
        expected_cache_heads = self.transformer.num_attention_heads
        if sequence_shard_enabled:
            if get_ring_parallel_world_size() > 1:
                raise NotImplementedError(
                    "LingBot causal sequence sharding currently supports ulysses_degree > 1 with ring_degree = 1 only."
                )
            expected_cache_heads //= get_ulysses_parallel_world_size()

        should_reset_cache = (
            batch.block_idx == 0
            or kv_cache1 is None
            or crossattn_cache is None
            or len(kv_cache1) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or kv_cache1[0]["k"].shape[2] != expected_cache_heads
        )

        if should_reset_cache:
            kv_cache1, crossattn_cache = self._initialize_causal_caches(
                batch_size=b,
                max_text_len=self._get_max_text_len(server_args),
                dtype=target_dtype,
                device=device,
                kv_cache_kwargs={
                    "sequence_shard_enabled": sequence_shard_enabled,
                },
            )
            cache_state.kv_cache = kv_cache1
            cache_state.crossattn_cache = crossattn_cache
            # Reset frame position on cache reset
            cache_state.current_chunk_start_frame = 0
            cache_state.chunk_idx = 0

        # Keep cross-attention K/V cache across realtime chunks; LingBot text/image
        # conditions are session-static and are invalidated by the cache reset above.

        current_start_frame = cache_state.current_chunk_start_frame

        # Slice condition to current chunk
        condition_chunks = condition_full.split(t, dim=2)
        cond_idx = min(cache_state.chunk_idx, len(condition_chunks) - 1)
        condition = condition_chunks[cond_idx]

        # --- Denoising loop (single chunk) ---
        current_latents = latents

        def prepare_model_input(current_latents):
            return torch.cat([current_latents, condition], dim=1)

        def prepare_context_input(current_latents):
            return torch.cat([current_latents, condition], dim=1)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            current_latents = self._denoise_and_update_causal_block(
                batch,
                server_args,
                chunk_latents=current_latents,
                scheduler=scheduler,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                kv_cache=kv_cache1,
                crossattn_cache=crossattn_cache,
                current_start_tokens=current_start_frame * self.frame_seq_length,
                start_frame=current_start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=pos_cond_kwargs,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
                device=device,
                attn_raw_latent_shape=(t, h, w),
                prepare_model_input=prepare_model_input,
                prepare_context_input=prepare_context_input,
                progress_bar=progress_bar,
            )

        # Advance cumulative frame position
        cache_state.current_chunk_start_frame += t
        cache_state.chunk_idx += 1

        # Output denoised latents for decoder
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        if not persist_cache_state:
            cache_state.dispose()
        return batch
