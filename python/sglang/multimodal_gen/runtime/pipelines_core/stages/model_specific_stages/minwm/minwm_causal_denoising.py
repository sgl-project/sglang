# SPDX-License-Identifier: Apache-2.0
"""Realtime V3-equivalent reference/action/cache scheduling for MinWM 5B."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_LABELS_CONDITION,
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_PROMPT_UPDATED_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
    CausalDMDCachePolicy,
    CausalDMDForwardContext,
    CausalDMDRealtimeCacheContext,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import (
    CausalVaeDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.realtime.states import (
    get_realtime_causal_dit_state,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

MINWM_ACTION_HISTORY_CACHE = "minwm_action_history"
MINWM_INITIAL_NOISE_CACHE = "minwm_initial_noise"


def _parity_dump(name: str, value) -> None:
    """Persist opt-in tensors used to localize baseline/API parity drift."""
    dump_dir = os.environ.get("MINWM_PARITY_DUMP_DIR")
    if not dump_dir:
        return
    path = Path(dump_dir) / "sglang" / name
    path.parent.mkdir(parents=True, exist_ok=True)

    def to_cpu(item):
        return item.detach().cpu() if isinstance(item, torch.Tensor) else item

    if isinstance(value, dict):
        value = {key: to_cpu(item) for key, item in value.items()}
    else:
        value = to_cpu(value)
    torch.save(value, path)


class MinWMChunkLatentPreparationStage(PipelineStage):
    """Draw BFCHW noise before permuting, matching minWM's RNG fill order."""

    def __init__(self, transformer) -> None:
        super().__init__()
        self.transformer = transformer

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.latents is not None:
            return batch
        condition = batch.image_latent
        if condition is None:
            raise ValueError("MinWM realtime inference requires a first-frame latent")
        chunk_size = int(
            batch.realtime_chunk_size
            or self.transformer.config.arch_config.num_frames_per_block
        )
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        shape_tail = (
            self.transformer.config.arch_config.out_channels,
            condition.shape[3],
            condition.shape[4],
        )
        noise_bfchw = None
        if batch.session is not None:
            state = get_realtime_causal_dit_state(batch.session)
            condition_inputs = batch.condition_inputs or {}
            total_chunks = condition_inputs.get(MINWM_TOTAL_CHUNKS_CONDITION)
            if batch.block_idx == 0:
                if total_chunks is not None and int(total_chunks) < 1:
                    raise ValueError("MinWM total chunk count must be positive")
                generated_frames = chunk_size * int(total_chunks or 1)
                # V3 draws one BFCHW tensor for reference+generated latents, then
                # overwrites the reference with its clean VAE latent. Preserve that
                # otherwise-invisible reference RNG consumption and, for a bounded
                # session, the original single-call CUDA RNG fill order.
                full_noise = torch.randn(
                    (condition.shape[0], 1 + generated_frames, *shape_tail),
                    generator=generator,
                    device=get_local_torch_device(),
                    dtype=condition.dtype,
                )
                state.runtime_cache[MINWM_INITIAL_NOISE_CACHE] = full_noise[:, 1:]
                _parity_dump("image_latent.pt", condition)
                _parity_dump("initial_noise_bfchw.pt", full_noise)
            cached_noise = state.runtime_cache.get(MINWM_INITIAL_NOISE_CACHE)
            if cached_noise is not None:
                start = int(batch.block_idx) * chunk_size
                end = start + chunk_size
                if end <= cached_noise.shape[1]:
                    noise_bfchw = cached_noise[:, start:end]
                elif total_chunks is not None:
                    raise ValueError(
                        "MinWM realtime request exceeded its pre-sampled noise horizon"
                    )
        if noise_bfchw is None:
            noise_bfchw = torch.randn(
                (condition.shape[0], chunk_size, *shape_tail),
                generator=generator,
                device=get_local_torch_device(),
                dtype=condition.dtype,
            )
        batch.latents = noise_bfchw.permute(0, 2, 1, 3, 4).contiguous()
        batch.raw_latent_shape = batch.latents.shape
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, [V.is_tensor, V.with_dims(5)]
        )
        result.add_check("generator", batch.generator, V.generator_or_list_generators)
        result.add_check("latents", batch.latents, V.none_or_tensor)
        return result


class MinWMCausalDMDDenoisingStage(CausalDMDDenoisingStage):
    """One clean reference commit followed by four-frame action DMD chunks."""

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
                "MinWM causal sequence sharding supports Ulysses with "
                "ring_degree = 1 only."
            )
        if ulysses_world_size <= 1:
            raise ValueError(
                "MinWM causal sequence sharding requires ulysses_degree > 1."
            )
        if num_attention_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible "
                f"by ulysses_degree ({ulysses_world_size})."
            )
        return num_attention_heads // ulysses_world_size

    def _apply_causal_cache_overrides(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        explicit_window = getattr(batch, "realtime_causal_kv_cache_num_frames", None)
        if explicit_window is None:
            explicit_window = getattr(
                server_args.pipeline_config,
                "realtime_causal_kv_cache_num_frames",
                None,
            )
        super()._apply_causal_cache_overrides(batch, server_args)
        self._minwm_unbounded_cache = explicit_window is None
        if not self._minwm_unbounded_cache:
            return
        total_chunks = (batch.condition_inputs or {}).get(MINWM_TOTAL_CHUNKS_CONDITION)
        if total_chunks is not None:
            # minWM main uses local_attn_size=-1: retain the reference and every
            # generated latent. A bounded realtime request can allocate that
            # complete horizon once instead of treating 128 as a sliding window.
            self.sliding_window_num_frames = 1 + int(total_chunks) * int(
                self.num_frames_per_block
            )

    def _causal_kv_cache_kwargs(self, policy: CausalDMDCachePolicy) -> dict:
        return {
            "sequence_shard_enabled": policy.sequence_shard_enabled,
            "kv_cache_size": policy.expected_cache_tokens,
            "allow_growth": bool(getattr(self, "_minwm_unbounded_cache", True)),
        }

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        return sequence_shard_enabled

    def _should_reset_realtime_causal_caches(
        self,
        batch: Req,
        *,
        cache_state,
        policy,
    ) -> bool:
        if not policy.kv_cache_kwargs.get("allow_growth", False):
            return super()._should_reset_realtime_causal_caches(
                batch, cache_state=cache_state, policy=policy
            )
        causal_kv_cache = cache_state.kv_cache
        crossattn_cache = cache_state.crossattn_cache
        return (
            batch.block_idx == 0
            or causal_kv_cache is None
            or crossattn_cache is None
            or len(causal_kv_cache) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or causal_kv_cache[0].k.shape[1] < policy.expected_cache_tokens
            or causal_kv_cache[0].k.shape[2] != policy.num_attention_heads
            or causal_kv_cache[0].sink_tokens != policy.expected_sink_tokens
        )

    def _flow_prediction_to_x0(
        self,
        *,
        flow_prediction: torch.Tensor,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
    ) -> torch.Tensor:
        # minWM's FewStepRenoiseScheduler calls pred_x0_from_flow with its
        # default compute_dtype=torch.float32. Preserve that arithmetic instead
        # of the generic causal path's fp64 stabilization.
        original_dtype = noisy_latent.dtype
        timestep = timestep.reshape(-1).to(scheduler.timesteps.device)
        if timestep.numel() == 1:
            timestep = timestep.expand(noisy_latent.shape[0])
        elif timestep.numel() != noisy_latent.shape[0]:
            timestep = timestep.repeat_interleave(
                noisy_latent.shape[0] // timestep.numel()
            )
        timestep_id = torch.argmin(
            (
                scheduler.timesteps.float().unsqueeze(0) - timestep.float().unsqueeze(1)
            ).abs(),
            dim=1,
        )
        sigma = (
            scheduler.sigmas.to(noisy_latent.device)
            .float()[timestep_id]
            .reshape(-1, 1, 1, 1)
        )
        return (noisy_latent.float() - sigma * flow_prediction.float()).to(
            original_dtype
        )

    def _get_causal_dmd_scheduler(self, batch: Req, server_args: ServerArgs):
        if batch.scheduler is None:
            raise ValueError("MinWM requires DMDTimestepPreparationStage")
        return batch.scheduler

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        if batch.timesteps is None:
            raise ValueError("MinWM requires prepared DMD timesteps")
        return batch.timesteps.to(device)

    @staticmethod
    def _prompt_seq_len(batch: Req, prompt: torch.Tensor) -> int:
        seq_lens = getattr(batch, "prompt_seq_lens", None)
        if seq_lens and seq_lens[0]:
            return int(seq_lens[0][0])
        masks = getattr(batch, "prompt_embeds_mask", None)
        if masks and masks[0] is not None:
            return int(masks[0][0].sum().item())
        masks = getattr(batch, "prompt_attention_mask", None)
        if masks and masks[0] is not None:
            return int(masks[0][0].sum().item())
        return int(prompt.shape[1])

    def _prepare_causal_dmd_prompt_embeds(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        if len(batch.prompt_embeds) != 1:
            raise ValueError("MinWM realtime inference supports one text encoder")
        prompt = batch.prompt_embeds[0]
        if prompt.shape[0] != 1:
            raise ValueError("MinWM realtime inference currently requires batch size 1")
        seq_len = self._prompt_seq_len(batch, prompt)
        return prompt[:, :seq_len].to(dtype=target_dtype)

    def _action_cache_state(self, batch: Req):
        if batch.session is None:
            raise ValueError("MinWM realtime inference requires a session")
        return get_realtime_causal_dit_state(batch.session)

    def _prepare_causal_dmd_pos_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict:
        del target_dtype
        state = self._action_cache_state(batch)
        condition_inputs = batch.condition_inputs or {}
        weight_windows = condition_inputs.get(MINWM_ACTION_WEIGHTS_CONDITION)
        if batch.block_idx == 0:
            if weight_windows is None:
                history = torch.zeros(
                    (1, 1), dtype=torch.long, device=batch.latents.device
                )
            else:
                temporal_factor = int(
                    server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
                )
                history = torch.zeros(
                    (1, 1, temporal_factor, 8),
                    dtype=torch.float32,
                    device=batch.latents.device,
                )
            state.runtime_cache[MINWM_ACTION_HISTORY_CACHE] = history
        history = state.runtime_cache.get(MINWM_ACTION_HISTORY_CACHE)
        if history is None:
            raise ValueError("MinWM action history is missing for a continued session")
        history_frames = int(self.transformer.config.arch_config.action_history_frames)

        if weight_windows is not None:
            expected_frames = int(batch.latents.shape[2])
            temporal_factor = int(
                server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
            )
            if (
                not isinstance(weight_windows, list)
                or len(weight_windows) != expected_frames
            ):
                raise ValueError(
                    f"expected {expected_frames} MinWM latent action windows"
                )
            flat_rows = []
            for window in weight_windows:
                if not isinstance(window, list) or len(window) != temporal_factor:
                    raise ValueError(
                        f"each MinWM action window must contain {temporal_factor} rows"
                    )
                flat_rows.extend(window)
            flat_rows = validate_action_weights(
                flat_rows, expected_frames=expected_frames * temporal_factor
            )
            current = torch.tensor(
                flat_rows, dtype=torch.float32, device=batch.latents.device
            ).reshape(1, expected_frames, temporal_factor, 8)
            if history.ndim != 4:
                raise ValueError("MinWM action form cannot change within a session")
            action_window = torch.cat([history[:, -history_frames:], current], dim=1)
            return {"action": action_window}

        labels = condition_inputs.get(MINWM_ACTION_LABELS_CONDITION)
        if labels is None:
            labels = [0] * int(batch.latents.shape[2])
        labels = validate_action_labels(
            labels, expected_frames=int(batch.latents.shape[2])
        )
        current = torch.tensor(
            labels, dtype=torch.long, device=batch.latents.device
        ).unsqueeze(0)
        if history.ndim != 2:
            raise ValueError("MinWM action form cannot change within a session")
        action_window = torch.cat([history[:, -history_frames:], current], dim=1)
        return {"action": action_window}

    def _prepare_realtime_causal_caches(
        self,
        batch: Req,
        server_args: ServerArgs,
        ctx: CausalDMDForwardContext,
    ) -> CausalDMDRealtimeCacheContext:
        cache_ctx = super()._prepare_realtime_causal_caches(batch, server_args, ctx)
        if (batch.condition_inputs or {}).get(MINWM_PROMPT_UPDATED_CONDITION):
            self._reset_crossattn_cache(cache_ctx.crossattn_cache)

        if batch.block_idx == 0 and cache_ctx.current_start_frame == 0:
            if batch.image_latent is None or batch.image_latent.shape[2] != 1:
                raise ValueError(
                    "MinWM requires exactly one encoded reference latent on chunk zero"
                )
            reference_kwargs = dict(ctx.pos_cond_kwargs)
            if reference_kwargs["action"].ndim == 4:
                reference_kwargs["action"] = torch.zeros(
                    (
                        ctx.batch_size,
                        1,
                        reference_kwargs["action"].shape[2],
                        reference_kwargs["action"].shape[3],
                    ),
                    dtype=reference_kwargs["action"].dtype,
                    device=ctx.device,
                )
            else:
                reference_kwargs["action"] = torch.zeros(
                    (ctx.batch_size, 1), dtype=torch.long, device=ctx.device
                )
            self._warm_up_causal_context_cache(
                batch,
                server_args,
                context_input=batch.image_latent,
                prompt_embeds=ctx.prompt_embeds,
                kv_cache=cache_ctx.kv_cache,
                crossattn_cache=cache_ctx.crossattn_cache,
                current_start_frame=0,
                image_kwargs=ctx.image_kwargs,
                pos_cond_kwargs=reference_kwargs,
                target_dtype=ctx.target_dtype,
                autocast_enabled=ctx.autocast_enabled,
            )
            cache_ctx.current_start_frame = 1
            cache_ctx.cache_state.current_chunk_start_frame = 1
        return cache_ctx

    def _commit_current_actions(
        self,
        cache_ctx: CausalDMDRealtimeCacheContext,
        pos_cond_kwargs: dict,
        num_frames: int,
    ) -> None:
        history = cache_ctx.cache_state.runtime_cache[MINWM_ACTION_HISTORY_CACHE]
        current = pos_cond_kwargs["action"][:, -num_frames:]
        history_frames = int(self.transformer.config.arch_config.action_history_frames)
        cache_ctx.cache_state.runtime_cache[MINWM_ACTION_HISTORY_CACHE] = torch.cat(
            [history, current], dim=1
        )[:, -history_frames:]

    def _forward_causal_transformer(self, batch: Req, **kwargs) -> torch.Tensor:
        output = super()._forward_causal_transformer(batch, **kwargs)
        index = getattr(self, "_parity_forward_index", 0)
        prompt = kwargs["prompt_embeds"] if index == 0 else None
        _parity_dump(
            f"forward_{index:03d}.pt",
            {
                "block_idx": int(batch.block_idx),
                "latent_model_input": kwargs["latent_model_input"],
                "prompt_embeds": prompt,
                "timestep": kwargs["timestep"],
                "action": kwargs["pos_cond_kwargs"].get("action"),
                "output": output,
            },
        )
        self._parity_forward_index = index + 1
        return output

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.block_idx == 0:
            self._parity_forward_index = 0
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        cache_ctx = self._prepare_realtime_causal_caches(batch, server_args, ctx)
        current_latents = self._denoise_realtime_causal_chunk(
            batch,
            server_args,
            ctx=ctx,
            cache_ctx=cache_ctx,
            chunk_latents=ctx.latents,
            prepare_model_input=lambda latents: latents,
            prepare_context_input=lambda latents: latents,
        )
        self._commit_current_actions(cache_ctx, ctx.pos_cond_kwargs, ctx.num_frames)
        self._advance_realtime_causal_cache(cache_ctx, num_frames=ctx.num_frames)
        batch.latents = current_latents
        batch.raw_latent_shape = current_latents.shape
        _parity_dump(f"chunk_{int(batch.block_idx):03d}_latents.pt", current_latents)
        if not cache_ctx.persist_state:
            cache_ctx.cache_state.dispose()
        return batch

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


class MinWMCausalVaeDecodingStage(CausalVaeDecodingStage):
    """Seed residual Wan2.2 VAE state with the reference latent exactly once."""

    def _decode_wan_with_persistent_cache(
        self,
        latents: torch.Tensor,
        *,
        first_chunk: bool,
    ) -> torch.Tensor:
        # minWM's WanVAEWrapper converts the cached decoder result to FP32
        # before pixel-space scaling. Preserve that output boundary.
        return (
            super()
            ._decode_wan_with_persistent_cache(latents, first_chunk=first_chunk)
            .float()
        )

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs):
        generated_latents = batch.latents
        if batch.block_idx == 0:
            if batch.image_latent is None:
                raise ValueError("MinWM decoder requires the reference latent")
            batch.latents = torch.cat([batch.image_latent, generated_latents], dim=2)
        try:
            return super().forward(batch, server_args)
        finally:
            batch.latents = generated_latents
