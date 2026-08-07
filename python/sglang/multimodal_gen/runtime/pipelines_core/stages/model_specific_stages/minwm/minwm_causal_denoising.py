# SPDX-License-Identifier: Apache-2.0
"""Realtime V3-equivalent reference/action/cache scheduling for MinWM 5B."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import torch
from sglang.multimodal_gen.configs.pipeline_configs.minwm import (
    MINWM_ACTION_LABELS_CONDITION,
    MINWM_ACTION_WEIGHTS_CONDITION,
    MINWM_CHUNK_SEED_CONDITION,
    MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION,
    MINWM_CONDITION_SWITCH_CONDITION,
    MINWM_PROMPT_UPDATED_CONDITION,
    MINWM_TOTAL_CHUNKS_CONDITION,
    MINWM_TOTAL_LATENT_FRAMES_CONDITION,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    validate_action_labels,
    validate_action_weights,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_kv_cache import (
    MinWMCausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDCachePolicy,
    CausalDMDDenoisingStage,
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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.realtime_trace import (
    realtime_trace_span,
    tensor_trace_metadata,
)

MINWM_ACTION_HISTORY_CACHE = "minwm_action_history"
MINWM_INITIAL_NOISE_CACHE = "minwm_initial_noise"
MINWM_INITIAL_NOISE_CURSOR_CACHE = "minwm_initial_noise_cursor"
MINWM_T2V_FIRST_LATENT_CACHE = "minwm_t2v_first_latent"

logger = init_logger(__name__)


def _parity_dump(name: str, value) -> None:
    """Persist opt-in tensors used to localize baseline/API parity drift."""
    dump_dir = os.environ.get("MINWM_PARITY_DUMP_DIR")
    if not dump_dir:
        return
    path = (
        Path(dump_dir)
        / "sglang"
        / f"sp_{get_sp_world_size():02d}_rank_{get_sp_parallel_rank():02d}"
        / name
    )
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
        chunk_size = int(
            batch.realtime_chunk_size
            or self.transformer.config.arch_config.num_frames_per_block
        )
        generator = (
            batch.generator[0] if isinstance(batch.generator, list) else batch.generator
        )
        if condition is not None:
            batch_size = condition.shape[0]
            latent_height, latent_width = condition.shape[3:]
            latent_dtype = condition.dtype
        else:
            spatial_factor = int(
                server_args.pipeline_config.vae_config.arch_config.scale_factor_spatial
            )
            if batch.height is None or batch.width is None:
                raise ValueError("MinWM T2V requires height and width")
            batch_size = int(batch.batch_size)
            latent_height = int(batch.height) // spatial_factor
            latent_width = int(batch.width) // spatial_factor
            latent_dtype = batch.prompt_embeds[0].dtype
        shape_tail = (
            self.transformer.config.arch_config.out_channels,
            latent_height,
            latent_width,
        )
        noise_bfchw = None
        condition_inputs = batch.condition_inputs or {}
        chunk_seed = condition_inputs.get(MINWM_CHUNK_SEED_CONDITION)
        if chunk_seed is not None:
            prefix_frames = int(
                condition_inputs.get(MINWM_CHUNK_SEED_PREFIX_FRAMES_CONDITION, 0)
            )
            if prefix_frames < 0:
                raise ValueError("MinWM chunk seed prefix frames must be non-negative")
            chunk_generator = torch.Generator(device=get_local_torch_device())
            chunk_generator.manual_seed(int(chunk_seed))
            replay_noise_bfchw = torch.randn(
                (batch_size, prefix_frames + chunk_size, *shape_tail),
                generator=chunk_generator,
                device=get_local_torch_device(),
                dtype=latent_dtype,
            )
            noise_bfchw = replay_noise_bfchw[:, prefix_frames:]
        if batch.session is not None:
            state = get_realtime_causal_dit_state(batch.session)
            total_chunks = condition_inputs.get(MINWM_TOTAL_CHUNKS_CONDITION)
            total_latent_frames = condition_inputs.get(
                MINWM_TOTAL_LATENT_FRAMES_CONDITION
            )
            if batch.block_idx == 0 and chunk_seed is None:
                if total_chunks is not None and int(total_chunks) < 1:
                    raise ValueError("MinWM total chunk count must be positive")
                if total_latent_frames is not None:
                    generated_frames = int(total_latent_frames)
                elif condition is None:
                    first_block = int(
                        self.transformer.config.arch_config.num_frame_first_block
                    )
                    regular_block = int(
                        self.transformer.config.arch_config.num_frames_per_block
                    )
                    generated_frames = first_block + regular_block * (
                        int(total_chunks or 1) - 1
                    )
                else:
                    generated_frames = chunk_size * int(total_chunks or 1)
                # I2V consumes and overwrites one reference-noise slot. T2V
                # generates its first block, so every sampled slot is retained.
                reference_slots = int(condition is not None)
                full_noise = torch.randn(
                    (
                        batch_size,
                        reference_slots + generated_frames,
                        *shape_tail,
                    ),
                    generator=generator,
                    device=get_local_torch_device(),
                    dtype=latent_dtype,
                )
                state.runtime_cache[MINWM_INITIAL_NOISE_CACHE] = full_noise[
                    :, reference_slots:
                ]
                state.runtime_cache[MINWM_INITIAL_NOISE_CURSOR_CACHE] = 0
                if condition is not None:
                    _parity_dump("image_latent.pt", condition)
                _parity_dump("initial_noise_bfchw.pt", full_noise)
            cached_noise = state.runtime_cache.get(MINWM_INITIAL_NOISE_CACHE)
            if noise_bfchw is None and cached_noise is not None:
                start = int(
                    state.runtime_cache.get(MINWM_INITIAL_NOISE_CURSOR_CACHE, 0)
                )
                end = start + chunk_size
                if end <= cached_noise.shape[1]:
                    noise_bfchw = cached_noise[:, start:end]
                    state.runtime_cache[MINWM_INITIAL_NOISE_CURSOR_CACHE] = end
                elif total_chunks is not None or total_latent_frames is not None:
                    raise ValueError(
                        "MinWM realtime request exceeded its pre-sampled noise horizon"
                    )
        if noise_bfchw is None:
            noise_bfchw = torch.randn(
                (batch_size, chunk_size, *shape_tail),
                generator=generator,
                device=get_local_torch_device(),
                dtype=latent_dtype,
            )
        batch.latents = noise_bfchw.permute(0, 2, 1, 3, 4).contiguous()
        batch.raw_latent_shape = batch.latents.shape
        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            "image_latent", batch.image_latent, V.none_or_tensor_with_dims(5)
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
        self._reset_causal_cache_config_defaults()
        explicit_window = getattr(batch, "realtime_causal_kv_cache_num_frames", None)
        if explicit_window is None:
            explicit_window = getattr(
                server_args.pipeline_config,
                "realtime_causal_kv_cache_num_frames",
                None,
            )
        super()._apply_causal_cache_overrides(batch, server_args)
        arch_config = self.transformer.config.arch_config
        self._minwm_unbounded_cache = (
            explicit_window is None and int(arch_config.local_attn_size) == -1
        )
        if not self._minwm_unbounded_cache:
            return
        total_chunks = (batch.condition_inputs or {}).get(MINWM_TOTAL_CHUNKS_CONDITION)
        total_latent_frames = (batch.condition_inputs or {}).get(
            MINWM_TOTAL_LATENT_FRAMES_CONDITION
        )
        if total_latent_frames is not None:
            self.sliding_window_num_frames = int(total_latent_frames) + int(
                batch.image_latent is not None
            )
        elif total_chunks is not None:
            # minWM main uses local_attn_size=-1: retain the reference and every
            # generated latent. A bounded realtime request can allocate that
            # complete horizon once instead of treating 128 as a sliding window.
            if batch.image_latent is not None:
                self.sliding_window_num_frames = 1 + int(total_chunks) * int(
                    self.num_frames_per_block
                )
            else:
                first_block = int(
                    self.transformer.config.arch_config.num_frame_first_block
                )
                self.sliding_window_num_frames = first_block + (
                    int(total_chunks) - 1
                ) * int(self.num_frames_per_block)

    def _reset_causal_cache_config_defaults(self) -> None:
        """Prevent request-local sink/window overrides from leaking to the next session."""
        arch_config = getattr(
            getattr(self.transformer, "config", None), "arch_config", None
        )
        if arch_config is None:
            return
        if hasattr(arch_config, "sink_size"):
            self.sink_size = int(arch_config.sink_size)
        if hasattr(arch_config, "sliding_window_num_frames"):
            self.sliding_window_num_frames = int(arch_config.sliding_window_num_frames)

    def _causal_kv_cache_kwargs(self, policy: CausalDMDCachePolicy) -> dict:
        arch_config = getattr(
            getattr(self.transformer, "config", None), "arch_config", None
        )
        return {
            "sequence_shard_enabled": policy.sequence_shard_enabled,
            "kv_cache_size": policy.expected_cache_tokens,
            "allow_growth": bool(getattr(self, "_minwm_unbounded_cache", True)),
            "rope_position_mode": str(
                getattr(arch_config, "rope_position_mode", "absolute")
            ),
            "rope_max_frame_gap": int(getattr(arch_config, "rope_max_frame_gap", 1)),
            "prompt_first_frame_pin_enabled": bool(
                getattr(arch_config, "prompt_first_frame_pin_enabled", False)
            ),
            "scene_cut_rope_offset": int(
                getattr(arch_config, "scene_cut_rope_offset", 0)
            ),
            "scene_cut_sink_enabled": bool(
                getattr(arch_config, "scene_cut_sink_enabled", False)
            ),
        }

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        *,
        sequence_shard_enabled: bool = False,
        kv_cache_size: int | None = None,
        allow_growth: bool = False,
        rope_position_mode: str = "absolute",
        rope_max_frame_gap: int = 1,
        prompt_first_frame_pin_enabled: bool = False,
        scene_cut_rope_offset: int = 0,
        scene_cut_sink_enabled: bool = False,
    ) -> None:
        num_attention_heads = self._num_causal_cache_attention_heads(
            sequence_shard_enabled=sequence_shard_enabled
        )
        attention_head_dim = self.transformer.attention_head_dim
        if kv_cache_size is None:
            kv_cache_size = self._get_causal_kv_cache_size(
                sequence_shard_enabled=sequence_shard_enabled
            )
        use_int_indices = self._use_causal_cache_int_indices(
            sequence_shard_enabled=sequence_shard_enabled
        )
        int_index = 0 if use_int_indices else None
        sink_tokens = self._get_causal_sink_tokens()
        attention_window_size = self._get_causal_attention_window_size(kv_cache_size)
        self.causal_kv_cache = [
            MinWMCausalSelfAttentionKVCache(
                k=torch.zeros(
                    (
                        batch_size,
                        kv_cache_size,
                        num_attention_heads,
                        attention_head_dim,
                    ),
                    dtype=dtype,
                    device=device,
                ),
                v=torch.zeros(
                    (
                        batch_size,
                        kv_cache_size,
                        num_attention_heads,
                        attention_head_dim,
                    ),
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
                rope_position_mode=rope_position_mode,
                rope_max_frame_gap=rope_max_frame_gap,
                prompt_first_frame_pin_enabled=prompt_first_frame_pin_enabled,
                scene_cut_rope_offset=scene_cut_rope_offset,
                scene_cut_sink_enabled=scene_cut_sink_enabled,
            )
            for _ in range(self.num_transformer_blocks)
        ]

    def _use_causal_cache_int_indices(
        self,
        *,
        sequence_shard_enabled: bool,
    ) -> bool:
        del sequence_shard_enabled
        # MinWM cache selection runs eagerly even on the single-GPU path. Keep
        # host cursors authoritative so reading 30 layer cursors never forces a
        # device-to-host synchronization.
        return True

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
        first_cache = (
            causal_kv_cache[0]
            if causal_kv_cache and len(causal_kv_cache) == self.num_transformer_blocks
            else None
        )
        expected = policy.kv_cache_kwargs
        return (
            batch.block_idx == 0
            or causal_kv_cache is None
            or crossattn_cache is None
            or len(causal_kv_cache) != self.num_transformer_blocks
            or len(crossattn_cache) != self.num_transformer_blocks
            or not isinstance(first_cache, MinWMCausalSelfAttentionKVCache)
            or first_cache.k.shape[1] < policy.expected_cache_tokens
            or first_cache.k.shape[2] != policy.num_attention_heads
            or first_cache.sink_tokens != policy.expected_sink_tokens
            or first_cache.rope_position_mode
            != expected.get("rope_position_mode", "absolute")
            or first_cache.rope_max_frame_gap != expected.get("rope_max_frame_gap", 1)
            or first_cache.prompt_first_frame_pin_enabled
            != expected.get("prompt_first_frame_pin_enabled", False)
            or first_cache.scene_cut_rope_offset
            != expected.get("scene_cut_rope_offset", 0)
            or first_cache.scene_cut_sink_enabled
            != expected.get("scene_cut_sink_enabled", False)
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
            # Native V3 starts T2V with a cold action cache: the first action
            # convolution sees only the current first block. I2V has one
            # committed reference frame, represented by one noop history slot.
            initial_history_frames = int(
                getattr(batch, "image_latent", None) is not None
            )
            if weight_windows is None:
                history = torch.zeros(
                    (1, initial_history_frames),
                    dtype=torch.long,
                    device=batch.latents.device,
                )
            else:
                temporal_factor = int(
                    server_args.pipeline_config.vae_config.arch_config.scale_factor_temporal
                )
                history = torch.zeros(
                    (1, initial_history_frames, temporal_factor, 8),
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
            condition_switch = (batch.condition_inputs or {}).get(
                MINWM_CONDITION_SWITCH_CONDITION, "prompt"
            )
            if condition_switch not in {"prompt", "scene_cut"}:
                raise ValueError("MinWM condition switch must be prompt or scene_cut")
            for cache_block in cache_ctx.kv_cache:
                if condition_switch == "scene_cut":
                    cache_block.mark_scene_cut()
                else:
                    cache_block.mark_prompt_switch()

        if batch.block_idx == 0 and cache_ctx.current_start_frame == 0:
            if batch.image_latent is None:
                return cache_ctx
            if batch.image_latent.shape[2] != 1:
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
        if getattr(batch, "realtime_trace_id", None):
            with realtime_trace_span(
                logger,
                batch,
                "server.model_denoise_complete",
                component="minwm_denoising",
                input_tensor=batch.latents,
                chunk_index=batch.block_idx,
                event_id=getattr(batch, "realtime_event_id", None),
                stage=self.__class__.__name__,
                num_inference_steps=getattr(batch, "num_inference_steps", None),
            ) as trace_span:
                result = self._forward_impl(batch, server_args)
                trace_span.add_fields(
                    **tensor_trace_metadata(result.latents, prefix="latents"),
                )
                return result
        return self._forward_impl(batch, server_args)

    def _forward_impl(self, batch: Req, server_args: ServerArgs) -> Req:
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
            "image_latent", batch.image_latent, V.none_or_tensor_with_dims(5)
        )
        result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.with_dims(1)])
        result.add_check("scheduler", batch.scheduler, V.not_none)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result


class MinWMCausalUniPCDenoisingStage(MinWMCausalDMDDenoisingStage):
    """Run minWM V3's per-chunk UniPC loop while retaining realtime KV state."""

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
        # Native V3 applies UniPC to BFCHW tensors. Keep the same layout at the
        # scheduler boundary; the transformer still consumes BCFHW.
        latents_btchw = chunk_latents.permute(0, 2, 1, 3, 4).contiguous()
        attn_metadata = None
        for i, timestep in enumerate(timesteps):
            current_latents = latents_btchw.permute(0, 2, 1, 3, 4)
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            attn_metadata = self._build_causal_attn_metadata(
                batch,
                server_args,
                current_timestep=i,
                raw_latent_shape=attn_raw_latent_shape,
                device=device,
            )
            timestep_2d = self._expand_timestep(
                timestep, latent_model_input.shape[0], latent_model_input.device
            )
            flow_prediction = self._forward_causal_transformer(
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
                current_timestep=i,
                attn_metadata=attn_metadata,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            flow_prediction_btchw = flow_prediction.permute(0, 2, 1, 3, 4)
            latents_btchw = scheduler.step(
                flow_prediction_btchw,
                timestep,
                latents_btchw,
                return_dict=False,
            )[0]
            if progress_bar is not None:
                progress_bar.update()

        return latents_btchw.permute(0, 2, 1, 3, 4), attn_metadata


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
        original_block_idx = batch.block_idx
        session = getattr(batch, "session", None)
        causal_state = (
            get_realtime_causal_dit_state(session) if session is not None else None
        )
        if batch.block_idx == 0 and causal_state is not None:
            causal_state.runtime_cache.pop(MINWM_T2V_FIRST_LATENT_CACHE, None)

        if batch.block_idx == 0 and batch.image_latent is not None:
            batch.latents = torch.cat([batch.image_latent, generated_latents], dim=2)
        elif (
            batch.block_idx == 0
            and batch.image_latent is None
            and causal_state is not None
        ):
            # Wan2.2's residual decoder cannot continue from a one-latent first
            # call: on the next call one upsample branch produces two temporal
            # values while its shortcut produces four. Preserve T2V's immediate
            # first-frame response, then use this latent to reseed the decoder
            # together with the first regular block below.
            causal_state.runtime_cache[MINWM_T2V_FIRST_LATENT_CACHE] = (
                generated_latents.detach().clone()
            )
        elif (
            batch.block_idx == 1
            and batch.image_latent is None
            and causal_state is not None
        ):
            first_latent = causal_state.runtime_cache.pop(
                MINWM_T2V_FIRST_LATENT_CACHE, None
            )
            if first_latent is not None:
                batch.latents = torch.cat([first_latent, generated_latents], dim=2)
                # Make the base stage reset and seed the VAE as a fresh causal
                # decode. The leading pixel frame was already emitted by block
                # zero and is removed from this block's output below.
                batch.block_idx = 0
        try:
            result = super().forward(batch, server_args)
            if (
                original_block_idx == 1
                and batch.latents.shape[2] > generated_latents.shape[2]
            ):
                if isinstance(result.output, torch.Tensor):
                    result.output = result.output[:, :, 1:]
                elif result.output is not None:
                    result.output = [sample[:, 1:] for sample in result.output]
            return result
        finally:
            batch.block_idx = original_block_idx
            batch.latents = generated_latents
