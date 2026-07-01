# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from sglang.multimodal_gen.runtime.layers.kvcache.causal_attention_cache import (
    CausalAttentionKVView,
    CausalSelfAttentionKVCache,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationSpec,
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

LONG_LIVE2_DEFAULT_SCENE_CUT_PREFIX = "The scene transitions. "


def _latent_frame_count(batch: Req, server_args: ServerArgs) -> int:
    num_frames = batch.num_frames
    vae_config = server_args.pipeline_config.vae_config
    if vae_config.use_temporal_scaling_frames:
        temporal_scale_factor = vae_config.arch_config.temporal_compression_ratio
        num_frames = (num_frames - 1) // temporal_scale_factor + 1
    return int(num_frames)


def _causal_block_count(batch: Req, server_args: ServerArgs) -> int:
    latent_frames = _latent_frame_count(batch, server_args)
    block_size = server_args.pipeline_config.dit_config.arch_config.num_frames_per_block
    if latent_frames % block_size != 0:
        raise ValueError(
            "LongLive2 latent frames must be divisible by num_frames_per_block "
            f"({block_size}), got {latent_frames}"
        )
    return latent_frames // block_size


@dataclass(slots=True)
class LongLive2CausalSelfAttentionKVCache(CausalSelfAttentionKVCache):
    global_sink_tokens: int = 0
    pinned_start: int = -1
    pinned_len: int = 0

    def reset_indices(self) -> None:
        super().reset_indices()
        self.reset_pinned_sink()

    def reset_pinned_sink(self) -> None:
        self.pinned_start = -1
        self.pinned_len = 0

    def pin_current_chunk(self, current_num_tokens: int) -> None:
        if self.sink_tokens <= 0 or current_num_tokens <= 0:
            self.reset_pinned_sink()
            return
        _, local_end_index = self._read_indices()
        pin_len = min(self.sink_tokens, current_num_tokens)
        self.pinned_start = local_end_index - current_num_tokens
        self.pinned_len = pin_len

    def _has_pinned_sink(self) -> bool:
        return self.pinned_start >= 0 and self.pinned_len > 0

    def _effective_sink_tokens(self) -> int:
        if self._has_pinned_sink():
            if self.pinned_start == self.global_sink_tokens:
                return self.global_sink_tokens + self.pinned_len
            return self.global_sink_tokens
        return max(self.global_sink_tokens, self.sink_tokens)

    def update_and_get_attention_kv(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        current_chunk_start: int,
        debug_name: str = "LongLive2 causal KV cache",
    ) -> CausalAttentionKVView:
        num_new_tokens = key.shape[1]
        current_chunk_end = current_chunk_start + num_new_tokens
        kv_cache_size = self.cache_size
        effective_sink_tokens = self._effective_sink_tokens()
        global_end_index, local_end_index_prev = self._read_indices()
        window_start = global_end_index - local_end_index_prev

        if current_chunk_end <= global_end_index:
            local_start_index = current_chunk_start - window_start
            local_end_index = local_start_index + num_new_tokens
            updated_local_end = local_end_index_prev
            updated_global_end = global_end_index
        else:
            appended_tokens = current_chunk_end - global_end_index
            if self.allow_growth:
                self._grow_to_fit(local_end_index_prev + appended_tokens)
                kv_cache_size = self.cache_size
            if local_end_index_prev + appended_tokens > kv_cache_size:
                num_evicted_tokens = (
                    local_end_index_prev + appended_tokens - kv_cache_size
                )
                num_rolled_tokens = max(
                    0,
                    local_end_index_prev - num_evicted_tokens - effective_sink_tokens,
                )
                if num_rolled_tokens > 0:
                    self.k[
                        :,
                        effective_sink_tokens : effective_sink_tokens
                        + num_rolled_tokens,
                    ] = self.k[
                        :,
                        effective_sink_tokens
                        + num_evicted_tokens : effective_sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                    self.v[
                        :,
                        effective_sink_tokens : effective_sink_tokens
                        + num_rolled_tokens,
                    ] = self.v[
                        :,
                        effective_sink_tokens
                        + num_evicted_tokens : effective_sink_tokens
                        + num_evicted_tokens
                        + num_rolled_tokens,
                    ].clone()
                if (
                    self._has_pinned_sink()
                    and self.pinned_start >= effective_sink_tokens
                ):
                    self.pinned_start -= num_evicted_tokens
                local_end_index = kv_cache_size
            else:
                local_end_index = local_end_index_prev + appended_tokens
            local_start_index = local_end_index - num_new_tokens
            updated_local_end = local_end_index
            updated_global_end = current_chunk_end

        if (
            local_start_index < 0
            or local_end_index > kv_cache_size
            or local_end_index - local_start_index != num_new_tokens
        ):
            raise RuntimeError(
                f"Invalid {debug_name} write range: "
                f"local=[{local_start_index}, {local_end_index}), "
                f"global_end={global_end_index}, "
                f"prev_local_end={local_end_index_prev}, "
                f"kv_cache_size={kv_cache_size}, "
                f"num_new_tokens={num_new_tokens}, "
                f"current_start={current_chunk_start}, current_end={current_chunk_end}"
            )

        if self.k.requires_grad:
            self.k = self.k.detach()
        if self.v.requires_grad:
            self.v = self.v.detach()
        self.k[:, local_start_index:local_end_index] = key
        self.v[:, local_start_index:local_end_index] = value

        attn_start_index = max(0, updated_local_end - self.attention_window_size)
        self._write_indices(
            global_end_index=updated_global_end,
            local_end_index=updated_local_end,
        )
        view_k, view_v = self._attention_view(
            attn_start_index=attn_start_index,
            updated_local_end=updated_local_end,
        )
        return CausalAttentionKVView(
            k=view_k,
            v=view_v,
            local_start_index=local_start_index,
            local_end_index=local_end_index,
            visible_local_end=updated_local_end,
            visible_global_end=updated_global_end,
        )

    def _cat_cache_slices(
        self, cache_slices: list[slice]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.cat([self.k[:, cache_slice] for cache_slice in cache_slices], dim=1),
            torch.cat([self.v[:, cache_slice] for cache_slice in cache_slices], dim=1),
        )

    def _attention_view(
        self,
        *,
        attn_start_index: int,
        updated_local_end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        effective_sink_tokens = self._effective_sink_tokens()
        prepend_sink = effective_sink_tokens > 0 and attn_start_index > 0
        prepend_pinned = (
            self._has_pinned_sink()
            and self.pinned_start >= effective_sink_tokens
            and self.pinned_start < attn_start_index
        )

        if prepend_sink and prepend_pinned:
            extra_tokens = effective_sink_tokens + self.pinned_len
            sink = slice(0, effective_sink_tokens)
            pinned = slice(self.pinned_start, self.pinned_start + self.pinned_len)
            local_window_size = max(0, self.attention_window_size - extra_tokens)
            local_window_start = max(
                effective_sink_tokens,
                updated_local_end - local_window_size,
            )
            window = slice(local_window_start, updated_local_end)
            return self._cat_cache_slices([sink, pinned, window])

        if prepend_sink:
            sink = slice(0, effective_sink_tokens)
            local_window_size = max(
                0,
                self.attention_window_size - effective_sink_tokens,
            )
            local_window_start = max(
                effective_sink_tokens,
                updated_local_end - local_window_size,
            )
            window = slice(local_window_start, updated_local_end)
            return self._cat_cache_slices([sink, window])

        if prepend_pinned:
            pinned = slice(self.pinned_start, self.pinned_start + self.pinned_len)
            local_window_size = max(0, self.attention_window_size - self.pinned_len)
            local_window_start = max(0, updated_local_end - local_window_size)
            window = slice(local_window_start, updated_local_end)
            return self._cat_cache_slices([pinned, window])

        return (
            self.k[:, attn_start_index:updated_local_end],
            self.v[:, attn_start_index:updated_local_end],
        )


def expand_longlive2_shot_prompts(
    shot_prompts: list[str],
    *,
    num_blocks: int,
    shot_durations: list[int] | None = None,
    chunks_per_shot: int = 0,
    scene_cut_prefix: str = LONG_LIVE2_DEFAULT_SCENE_CUT_PREFIX,
) -> list[str]:
    return _expand_shot_prompts(
        shot_prompts,
        num_blocks=num_blocks,
        shot_durations=shot_durations,
        chunks_per_shot=chunks_per_shot,
        scene_cut_prefix=scene_cut_prefix,
    )[0]


def _expand_shot_prompts(
    shot_prompts: list[str],
    *,
    num_blocks: int,
    shot_durations: list[int] | None = None,
    chunks_per_shot: int = 0,
    scene_cut_prefix: str = LONG_LIVE2_DEFAULT_SCENE_CUT_PREFIX,
) -> tuple[list[str], list[bool], list[int]]:
    if not shot_prompts:
        raise ValueError("shot_prompts must be non-empty")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if shot_durations is not None and len(shot_durations) != len(shot_prompts):
        raise ValueError("shot_durations must match shot_prompts length")

    if shot_durations is not None:
        durations = shot_durations[: len(shot_prompts)]
    elif chunks_per_shot > 0:
        durations = [chunks_per_shot] * len(shot_prompts)
    else:
        base, extra = divmod(num_blocks, len(shot_prompts))
        durations = [base + (1 if i < extra else 0) for i in range(len(shot_prompts))]

    clamped: list[int] = []
    remaining = num_blocks
    for duration in durations:
        if remaining <= 0:
            break
        take = min(int(duration), remaining)
        clamped.append(take)
        remaining -= take
    if remaining > 0 and clamped:
        clamped[-1] += remaining
    if not clamped:
        clamped = [num_blocks]

    block_prompts: list[str] = []
    scene_cut_mask: list[bool] = []
    shot_indices: list[int] = []
    for shot_idx, (caption, duration) in enumerate(zip(shot_prompts, clamped)):
        for block_in_shot in range(duration):
            is_scene_cut = shot_idx > 0 and block_in_shot == 0
            if is_scene_cut and scene_cut_prefix:
                block_prompts.append(scene_cut_prefix + caption)
            else:
                block_prompts.append(caption)
            scene_cut_mask.append(is_scene_cut)
            shot_indices.append(shot_idx)
    return (
        block_prompts[:num_blocks],
        scene_cut_mask[:num_blocks],
        shot_indices[:num_blocks],
    )


class LongLive2TextEncodingStage(TextEncodingStage):
    def build_dedup_fingerprint(self, batch: Req, server_args: ServerArgs):
        base = super().build_dedup_fingerprint(batch, server_args)
        return (
            base,
            self.freeze_for_dedup(getattr(batch, "shot_prompts", None)),
            self.freeze_for_dedup(getattr(batch, "shot_durations", None)),
            int(getattr(batch, "chunks_per_shot", 0) or 0),
            getattr(batch, "scene_cut_prefix", None),
        )

    def _block_prompts(self, batch: Req, server_args: ServerArgs) -> list[str] | None:
        shot_prompts = getattr(batch, "shot_prompts", None)
        if shot_prompts is None:
            return None
        if isinstance(batch.prompt, list):
            raise ValueError("LongLive2 shot_prompts supports one video per request")

        block_prompts, scene_cut_mask, shot_indices = _expand_shot_prompts(
            shot_prompts,
            num_blocks=_causal_block_count(batch, server_args),
            shot_durations=getattr(batch, "shot_durations", None),
            chunks_per_shot=int(getattr(batch, "chunks_per_shot", 0) or 0),
            scene_cut_prefix=(
                LONG_LIVE2_DEFAULT_SCENE_CUT_PREFIX
                if getattr(batch, "scene_cut_prefix", None) is None
                else getattr(batch, "scene_cut_prefix")
            ),
        )
        batch.extra["longlive2_block_prompts"] = block_prompts
        batch.extra["longlive2_scene_cut_mask"] = scene_cut_mask
        batch.extra["longlive2_shot_indices"] = shot_indices
        return block_prompts

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        block_prompts = self._block_prompts(batch, server_args)
        if block_prompts is None:
            return super().forward(batch, server_args)

        original_prompt = batch.prompt
        batch.prompt = block_prompts
        try:
            return super().forward(batch, server_args)
        finally:
            batch.prompt = original_prompt


class LongLive2ImageVAEEncodingStage(ImageVAEEncodingStage):
    def preprocess(self, image):
        image = super().preprocess(image)
        if image.ndim == 5:
            image = image.squeeze(2)
        return image


class LongLive2LatentPreparationStage(LatentPreparationStage):
    def get_latent_preparation_spec(
        self,
        batch: Req,
        server_args: ServerArgs,
        batch_size: int,
        num_frames: int,
        device: torch.device | str,
    ) -> LatentPreparationSpec:
        b, c, t, h, w = server_args.pipeline_config.prepare_latent_shape(
            batch, batch_size, num_frames
        )
        return LatentPreparationSpec(
            shape=(b, t, c, h, w),
            dtype=self._get_latent_dtype(batch, server_args),
            device=device,
            prepare_latent_ids=False,
            pack_latents=False,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch = super().forward(batch, server_args)
        return self._normalize_latent_layout(batch, server_args)

    def _prepare_grouped_latents(
        self,
        batches: list[Req],
        server_args: ServerArgs,
    ) -> Req:
        batch = super()._prepare_grouped_latents(batches, server_args)
        return self._normalize_latent_layout(batch, server_args)

    @staticmethod
    def _expected_latent_channels(batch: Req, server_args: ServerArgs) -> int:
        shape = server_args.pipeline_config.prepare_latent_shape(
            batch,
            batch.batch_size,
            batch.latents.shape[1],
        )
        return int(shape[1])

    def _normalize_latent_layout(self, batch: Req, server_args: ServerArgs) -> Req:
        latents = batch.latents
        if latents is None or latents.ndim != 5:
            return batch
        expected_channels = self._expected_latent_channels(batch, server_args)
        if (
            latents.shape[1] != expected_channels
            and latents.shape[2] == expected_channels
        ):
            latents = latents.permute(0, 2, 1, 3, 4).contiguous()
            batch.latents = latents
            batch.raw_latent_shape = latents.shape
        return batch


class LongLive2CausalDenoisingStage(CausalDMDDenoisingStage):
    def __init__(self, transformer, scheduler) -> None:
        super().__init__(transformer, scheduler)
        self.causal_kv_cache_neg: list | None = None
        self.crossattn_cache_neg: list | None = None
        self._rope_temporal_offset = 0.0
        self._i2v_image_latent: torch.Tensor | None = None

    def _get_causal_dmd_latents(self, batch: Req) -> torch.Tensor:
        latents = super()._get_causal_dmd_latents(batch)
        if torch.is_inference(latents):
            latents = latents.clone()
            batch.latents = latents
        return latents

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if self._use_cfg(batch):
            return self._forward_one_shot_cfg(batch, server_args)
        return self._forward_one_shot(batch, server_args)

    @staticmethod
    def _i2v_clamp_active(batch: Req) -> bool:
        image_latent = getattr(batch, "image_latent", None)
        return image_latent is not None and image_latent.shape[2] == 1

    def _prepare_i2v_clamp(self, current_latents, start_frame):
        clamp_latent = self._i2v_image_latent if start_frame == 0 else None
        if clamp_latent is None:
            return None, 0
        clamp_latent = clamp_latent.to(
            device=current_latents.device, dtype=current_latents.dtype
        )
        return clamp_latent, clamp_latent.shape[2]

    @staticmethod
    def _use_cfg(batch: Req) -> bool:
        return bool(getattr(batch, "do_classifier_free_guidance", False))

    @staticmethod
    def _guidance_scale(batch: Req) -> float:
        return float(getattr(batch, "guidance_scale", 1.0))

    @staticmethod
    def _get_negative_prompt_embeds(batch: Req):
        negative_prompt_embeds = getattr(batch, "negative_prompt_embeds", None)
        if negative_prompt_embeds is None or (
            isinstance(negative_prompt_embeds, list)
            and len(negative_prompt_embeds) == 0
        ):
            raise ValueError(
                "LongLive2 classifier-free guidance requires negative_prompt_embeds"
            )
        return negative_prompt_embeds

    def _prepare_causal_dmd_neg_cond_kwargs(
        self,
        batch: Req,
        server_args: ServerArgs,
        target_dtype: torch.dtype,
    ) -> dict[str, Any]:
        return self.prepare_extra_func_kwargs(
            self.transformer.forward,
            {
                "encoder_attention_mask": batch.negative_attention_mask,
            },
        )

    @staticmethod
    def _block_prompt_count(batch: Req) -> int | None:
        block_prompts = batch.extra.get("longlive2_block_prompts")
        if block_prompts is None:
            return None
        return len(block_prompts)

    @classmethod
    def _select_block_conditioning(cls, value, block_index: int, block_count: int):
        if isinstance(value, torch.Tensor) and value.shape[:1] == (block_count,):
            return value[block_index : block_index + 1]
        if isinstance(value, list):
            return [
                cls._select_block_conditioning(item, block_index, block_count)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                cls._select_block_conditioning(item, block_index, block_count)
                for item in value
            )
        if isinstance(value, dict):
            return {
                key: cls._select_block_conditioning(item, block_index, block_count)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _select_block_prompt_embeds(
        cls,
        batch: Req,
        prompt_embeds,
        block_index: int,
    ):
        block_count = cls._block_prompt_count(batch)
        if block_count is None:
            return prompt_embeds
        return cls._select_block_conditioning(prompt_embeds, block_index, block_count)

    @classmethod
    def _select_block_cond_kwargs(
        cls,
        batch: Req,
        cond_kwargs: dict[str, Any],
        block_index: int,
    ) -> dict[str, Any]:
        block_count = cls._block_prompt_count(batch)
        if block_count is None:
            return cond_kwargs
        return {
            key: cls._select_block_conditioning(value, block_index, block_count)
            for key, value in cond_kwargs.items()
        }

    def _reset_crossattn_cache_for_block(self, batch: Req, *caches) -> None:
        if self._block_prompt_count(batch) is None:
            return
        for cache in caches:
            if cache is not None:
                self._reset_crossattn_cache(cache)

    def _validate_block_prompt_count(self, batch: Req, block_sizes: list[int]) -> None:
        block_count = self._block_prompt_count(batch)
        if block_count is None:
            return
        if block_count != len(block_sizes):
            raise ValueError(
                "LongLive2 multi-shot prompt count must match causal block count, "
                f"got {block_count} prompts and {len(block_sizes)} blocks"
            )

    def _multi_shot_sink_enabled(self, batch: Req) -> bool:
        return (
            self._block_prompt_count(batch) is not None
            and bool(getattr(batch, "multi_shot_sink", True))
            and self.sink_size > 0
        )

    def _cache_needs_reinit_for_batch(self, kv_cache, batch: Req) -> bool:
        if kv_cache is None or len(kv_cache) != self.num_transformer_blocks:
            return True
        has_shot_sink = isinstance(kv_cache[0], LongLive2CausalSelfAttentionKVCache)
        return has_shot_sink != self._multi_shot_sink_enabled(batch)

    def _is_scene_cut(self, batch: Req, block_index: int) -> bool:
        if not self._multi_shot_sink_enabled(batch):
            return False
        scene_cut_mask = batch.extra.get("longlive2_scene_cut_mask")
        if not isinstance(scene_cut_mask, list) or block_index >= len(scene_cut_mask):
            return False
        return bool(scene_cut_mask[block_index])

    @staticmethod
    def _shot_index(batch: Req, block_index: int) -> int:
        shot_indices = batch.extra.get("longlive2_shot_indices")
        if not isinstance(shot_indices, list) or block_index >= len(shot_indices):
            return 0
        return int(shot_indices[block_index])

    def _set_rope_temporal_offset(self, batch: Req, shot_index: int) -> None:
        offset = float(getattr(batch, "multi_shot_rope_offset", 8.0) or 0.0)
        self._rope_temporal_offset = shot_index * offset

    def _pin_current_chunk(self, kv_cache, current_num_frames: int) -> None:
        if kv_cache is None:
            return
        current_num_tokens = current_num_frames * self.num_token_per_frame
        for cache_block in kv_cache:
            pin_current_chunk = getattr(cache_block, "pin_current_chunk", None)
            if callable(pin_current_chunk):
                pin_current_chunk(current_num_tokens)

    def _initialize_kv_cache(
        self,
        batch_size,
        dtype,
        device,
        *,
        sequence_shard_enabled: bool = False,
        use_shot_sink: bool = False,
    ) -> None:
        if not use_shot_sink:
            super()._initialize_kv_cache(
                batch_size,
                dtype,
                device,
                sequence_shard_enabled=sequence_shard_enabled,
            )
            return

        num_attention_heads = self._num_causal_cache_attention_heads(
            sequence_shard_enabled=sequence_shard_enabled
        )
        attention_head_dim = self.transformer.attention_head_dim
        kv_cache_size = self._get_causal_kv_cache_size(
            sequence_shard_enabled=sequence_shard_enabled
        )
        self.causal_kv_cache = self._allocate_shot_kv_cache(
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
            global_sink_tokens=self._get_causal_sink_tokens(),
            attention_window_size=self._get_causal_attention_window_size(kv_cache_size),
        )

    def _allocate_shot_kv_cache(
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
        global_sink_tokens: int = 0,
        attention_window_size: int | None = None,
        allow_growth: bool = False,
    ) -> list[LongLive2CausalSelfAttentionKVCache]:
        causal_kv_cache = []
        int_index = 0 if use_int_indices else None
        if attention_window_size is None:
            attention_window_size = kv_cache_size
        for _ in range(self.num_transformer_blocks):
            causal_kv_cache.append(
                LongLive2CausalSelfAttentionKVCache(
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
                    global_sink_tokens=global_sink_tokens,
                    attention_window_size=attention_window_size,
                    allow_growth=allow_growth,
                )
            )
        return causal_kv_cache

    def _new_causal_cache_pair(
        self,
        *,
        batch_size: int,
        max_text_len: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list, list]:
        prev_kv_cache = self.causal_kv_cache
        prev_crossattn_cache = self.crossattn_cache
        try:
            return self._initialize_causal_caches(
                batch_size=batch_size,
                max_text_len=max_text_len,
                dtype=dtype,
                device=device,
                kv_cache_kwargs=kv_cache_kwargs,
            )
        finally:
            self.causal_kv_cache = prev_kv_cache
            self.crossattn_cache = prev_crossattn_cache

    def _reset_or_init_negative_caches(
        self,
        *,
        batch: Req,
        batch_size: int,
        max_text_len: int,
        dtype: torch.dtype,
        device: torch.device,
        kv_cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list, list]:
        if (
            self._cache_needs_reinit_for_batch(self.causal_kv_cache_neg, batch)
            or self.crossattn_cache_neg is None
        ):
            self.causal_kv_cache_neg, self.crossattn_cache_neg = (
                self._new_causal_cache_pair(
                    batch_size=batch_size,
                    max_text_len=max_text_len,
                    dtype=dtype,
                    device=device,
                    kv_cache_kwargs=kv_cache_kwargs,
                )
            )
        else:
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
            )
        return self.causal_kv_cache_neg, self.crossattn_cache_neg

    def _forward_one_shot(self, batch: Req, server_args: ServerArgs) -> Req:
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

        independent_first_frame = self.transformer.independent_first_frame
        use_shot_sink = self._multi_shot_sink_enabled(batch)
        kv_cache_kwargs = {"use_shot_sink": True} if use_shot_sink else None
        self._rope_temporal_offset = 0.0

        if self._cache_needs_reinit_for_batch(self.causal_kv_cache, batch):
            self._initialize_causal_caches(
                batch_size=latents.shape[0],
                max_text_len=self._get_max_text_len(server_args),
                dtype=target_dtype,
                device=latents.device,
                kv_cache_kwargs=kv_cache_kwargs,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        current_start_frame = 0
        clamp_i2v = self._i2v_clamp_active(batch)
        self._i2v_image_latent = batch.image_latent if clamp_i2v else None
        if getattr(batch, "image_latent", None) is not None and not clamp_i2v:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            warmup_prompt_embeds = self._select_block_prompt_embeds(
                batch,
                prompt_embeds,
                0,
            )
            warmup_pos_cond_kwargs = self._select_block_cond_kwargs(
                batch,
                pos_cond_kwargs,
                0,
            )
            if independent_first_frame and input_frames >= 1:
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=warmup_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            while remaining_frames > 0:
                block = min(self.num_frames_per_block, remaining_frames)
                context_input = image_latent[
                    :, :, current_start_frame : current_start_frame + block, :, :
                ]
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=context_input,
                    prompt_embeds=warmup_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += block
                remaining_frames -= block

        pos_start_base = current_start_frame

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

        self._validate_block_prompt_count(batch, block_sizes)

        def prepare_context_input(current_latents: torch.Tensor) -> torch.Tensor:
            return current_latents

        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for block_index, current_num_frames in enumerate(block_sizes):
                self._set_rope_temporal_offset(
                    batch,
                    self._shot_index(batch, block_index),
                )
                is_scene_cut = self._is_scene_cut(batch, block_index)

                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames, :, :
                ]
                current_prompt_embeds = self._select_block_prompt_embeds(
                    batch,
                    prompt_embeds,
                    block_index,
                )
                current_pos_cond_kwargs = self._select_block_cond_kwargs(
                    batch,
                    pos_cond_kwargs,
                    block_index,
                )
                self._reset_crossattn_cache_for_block(batch, self.crossattn_cache)

                def prepare_model_input(current_latents: torch.Tensor) -> torch.Tensor:
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
                    prompt_embeds=current_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_tokens=current_start_tokens,
                    start_frame=start_index,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=current_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    device=device,
                    attn_raw_latent_shape=(current_num_frames, h, w),
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_context_input,
                    progress_bar=progress_bar,
                )

                if is_scene_cut:
                    self._pin_current_chunk(self.causal_kv_cache, current_num_frames)

                latents[:, :, start_index : start_index + current_num_frames, :, :] = (
                    current_latents
                )
                start_index += current_num_frames

        self._rope_temporal_offset = 0.0
        batch.latents = latents
        return batch

    def _forward_one_shot_cfg(self, batch: Req, server_args: ServerArgs) -> Req:
        ctx = self._prepare_causal_dmd_forward_context(batch, server_args)
        target_dtype = ctx.target_dtype
        autocast_enabled = ctx.autocast_enabled
        scheduler = ctx.scheduler
        device = ctx.device
        timesteps = ctx.timesteps
        image_kwargs = ctx.image_kwargs
        pos_cond_kwargs = ctx.pos_cond_kwargs
        neg_cond_kwargs = self._prepare_causal_dmd_neg_cond_kwargs(
            batch,
            server_args,
            target_dtype,
        )
        latents = ctx.latents
        prompt_embeds = ctx.prompt_embeds
        negative_prompt_embeds = self._get_negative_prompt_embeds(batch)
        t, h, w = ctx.num_frames, ctx.height, ctx.width

        independent_first_frame = self.transformer.independent_first_frame
        max_text_len = self._get_max_text_len(server_args)
        use_shot_sink = self._multi_shot_sink_enabled(batch)
        kv_cache_kwargs = {"use_shot_sink": True} if use_shot_sink else None
        self._rope_temporal_offset = 0.0

        if self._cache_needs_reinit_for_batch(self.causal_kv_cache, batch):
            self.causal_kv_cache, self.crossattn_cache = self._new_causal_cache_pair(
                batch_size=latents.shape[0],
                max_text_len=max_text_len,
                dtype=target_dtype,
                device=latents.device,
                kv_cache_kwargs=kv_cache_kwargs,
            )
        else:
            assert self.crossattn_cache is not None
            self._reset_causal_caches(
                kv_cache=self.causal_kv_cache,
                crossattn_cache=self.crossattn_cache,
            )

        causal_kv_cache_neg, crossattn_cache_neg = self._reset_or_init_negative_caches(
            batch=batch,
            batch_size=latents.shape[0],
            max_text_len=max_text_len,
            dtype=target_dtype,
            device=latents.device,
            kv_cache_kwargs=kv_cache_kwargs,
        )

        current_start_frame = 0
        clamp_i2v = self._i2v_clamp_active(batch)
        self._i2v_image_latent = batch.image_latent if clamp_i2v else None
        if getattr(batch, "image_latent", None) is not None and not clamp_i2v:
            image_latent = batch.image_latent
            assert image_latent is not None
            input_frames = image_latent.shape[2]
            warmup_prompt_embeds = self._select_block_prompt_embeds(
                batch,
                prompt_embeds,
                0,
            )
            warmup_negative_prompt_embeds = self._select_block_prompt_embeds(
                batch,
                negative_prompt_embeds,
                0,
            )
            warmup_pos_cond_kwargs = self._select_block_cond_kwargs(
                batch,
                pos_cond_kwargs,
                0,
            )
            warmup_neg_cond_kwargs = self._select_block_cond_kwargs(
                batch,
                neg_cond_kwargs,
                0,
            )
            if independent_first_frame and input_frames >= 1:
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=warmup_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=image_latent[:, :, :1, :, :],
                    prompt_embeds=warmup_negative_prompt_embeds,
                    kv_cache=causal_kv_cache_neg,
                    crossattn_cache=crossattn_cache_neg,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_neg_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            while remaining_frames > 0:
                block = min(self.num_frames_per_block, remaining_frames)
                context_input = image_latent[
                    :, :, current_start_frame : current_start_frame + block, :, :
                ]
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=context_input,
                    prompt_embeds=warmup_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=context_input,
                    prompt_embeds=warmup_negative_prompt_embeds,
                    kv_cache=causal_kv_cache_neg,
                    crossattn_cache=crossattn_cache_neg,
                    current_start_frame=current_start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_neg_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                current_start_frame += block
                remaining_frames -= block

        pos_start_base = current_start_frame

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

        self._validate_block_prompt_count(batch, block_sizes)

        def prepare_context_input(current_latents: torch.Tensor) -> torch.Tensor:
            return current_latents

        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for block_index, current_num_frames in enumerate(block_sizes):
                self._set_rope_temporal_offset(
                    batch,
                    self._shot_index(batch, block_index),
                )
                is_scene_cut = self._is_scene_cut(batch, block_index)

                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames, :, :
                ]
                current_prompt_embeds = self._select_block_prompt_embeds(
                    batch,
                    prompt_embeds,
                    block_index,
                )
                current_negative_prompt_embeds = self._select_block_prompt_embeds(
                    batch,
                    negative_prompt_embeds,
                    block_index,
                )
                current_pos_cond_kwargs = self._select_block_cond_kwargs(
                    batch,
                    pos_cond_kwargs,
                    block_index,
                )
                current_neg_cond_kwargs = self._select_block_cond_kwargs(
                    batch,
                    neg_cond_kwargs,
                    block_index,
                )
                self._reset_crossattn_cache_for_block(
                    batch,
                    self.crossattn_cache,
                    crossattn_cache_neg,
                )

                def prepare_model_input(current_latents: torch.Tensor) -> torch.Tensor:
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
                current_latents = self._denoise_and_update_causal_block_cfg(
                    batch,
                    server_args,
                    chunk_latents=current_latents,
                    scheduler=scheduler,
                    timesteps=timesteps,
                    prompt_embeds=current_prompt_embeds,
                    negative_prompt_embeds=current_negative_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    kv_cache_neg=causal_kv_cache_neg,
                    crossattn_cache_neg=crossattn_cache_neg,
                    current_start_tokens=current_start_tokens,
                    start_frame=start_index,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=current_pos_cond_kwargs,
                    neg_cond_kwargs=current_neg_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                    device=device,
                    attn_raw_latent_shape=(current_num_frames, h, w),
                    prepare_model_input=prepare_model_input,
                    prepare_context_input=prepare_context_input,
                    progress_bar=progress_bar,
                )

                if is_scene_cut:
                    self._pin_current_chunk(self.causal_kv_cache, current_num_frames)
                    self._pin_current_chunk(causal_kv_cache_neg, current_num_frames)

                latents[:, :, start_index : start_index + current_num_frames, :, :] = (
                    current_latents
                )
                start_index += current_num_frames

        self._rope_temporal_offset = 0.0
        batch.latents = latents
        return batch

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
        self._manage_dit_use_site(self.transformer, "transformer", batch)
        rope_start_frame = start_frame
        if self._rope_temporal_offset != 0.0:
            rope_start_frame = start_frame + self._rope_temporal_offset
        return super()._forward_causal_transformer(
            batch,
            latent_model_input=latent_model_input,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_tokens=current_start_tokens,
            start_frame=rope_start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            current_timestep=current_timestep,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )

    def _prepare_causal_dmd_timesteps(
        self,
        batch: Req,
        server_args: ServerArgs,
        scheduler,
        device: torch.device,
    ) -> torch.Tensor:
        scheduler.set_timesteps(
            batch.num_inference_steps,
            device=device,
            shift=server_args.pipeline_config.flow_shift,
        )
        return scheduler.timesteps.to(device)

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
    ) -> tuple[torch.Tensor, Any | None]:
        scheduler.set_timesteps(
            len(timesteps),
            device=device,
            shift=server_args.pipeline_config.flow_shift,
        )
        timesteps = scheduler.timesteps.to(device)
        current_latents = chunk_latents
        attn_metadata = None
        clamp_latent, context_frames = self._prepare_i2v_clamp(
            current_latents, start_frame
        )
        if clamp_latent is not None:
            current_latents = current_latents.clone()

        for current_timestep, timestep in enumerate(timesteps):
            if clamp_latent is not None:
                current_latents[:, :, :context_frames] = clamp_latent
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            attn_metadata = self._build_causal_attn_metadata(
                batch,
                server_args,
                current_timestep=current_timestep,
                raw_latent_shape=attn_raw_latent_shape,
                device=device,
            )
            batch_size = latent_model_input.shape[0]
            timestep_2d = (
                timestep.reshape(1)
                .to(device=latent_model_input.device, dtype=torch.float32)
                .expand(batch_size, latent_model_input.shape[2])
            )
            if clamp_latent is not None:
                timestep_2d = timestep_2d.clone()
                timestep_2d[:, :context_frames] = 0
            flow_pred = self._forward_causal_transformer(
                batch,
                latent_model_input=latent_model_input,
                prompt_embeds=prompt_embeds,
                timestep=timestep_2d,
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

            next_latents = scheduler.step(
                flow_pred,
                timestep,
                current_latents,
                return_dict=False,
            )[0]

            current_latents = next_latents
            if clamp_latent is not None:
                current_latents[:, :, :context_frames] = clamp_latent

            if progress_bar is not None:
                progress_bar.update()

        return current_latents, attn_metadata

    def _denoise_causal_dmd_chunk_cfg(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        negative_prompt_embeds,
        kv_cache,
        crossattn_cache,
        kv_cache_neg,
        crossattn_cache_neg,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> tuple[torch.Tensor, Any | None]:
        scheduler.set_timesteps(
            len(timesteps),
            device=device,
            shift=server_args.pipeline_config.flow_shift,
        )
        timesteps = scheduler.timesteps.to(device)
        current_latents = chunk_latents
        attn_metadata = None
        guidance_scale = self._guidance_scale(batch)
        clamp_latent, context_frames = self._prepare_i2v_clamp(
            current_latents, start_frame
        )
        if clamp_latent is not None:
            current_latents = current_latents.clone()

        for current_timestep, timestep in enumerate(timesteps):
            if clamp_latent is not None:
                current_latents[:, :, :context_frames] = clamp_latent
            latent_model_input = prepare_model_input(current_latents).to(target_dtype)
            attn_metadata = self._build_causal_attn_metadata(
                batch,
                server_args,
                current_timestep=current_timestep,
                raw_latent_shape=attn_raw_latent_shape,
                device=device,
            )
            batch_size = latent_model_input.shape[0]
            timestep_2d = (
                timestep.reshape(1)
                .to(device=latent_model_input.device, dtype=torch.float32)
                .expand(batch_size, latent_model_input.shape[2])
            )
            if clamp_latent is not None:
                timestep_2d = timestep_2d.clone()
                timestep_2d[:, :context_frames] = 0
            flow_pred_cond = self._forward_causal_transformer(
                batch,
                latent_model_input=latent_model_input,
                prompt_embeds=prompt_embeds,
                timestep=timestep_2d,
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
            flow_pred_uncond = self._forward_causal_transformer(
                batch,
                latent_model_input=latent_model_input,
                prompt_embeds=negative_prompt_embeds,
                timestep=timestep_2d,
                kv_cache=kv_cache_neg,
                crossattn_cache=crossattn_cache_neg,
                current_start_tokens=current_start_tokens,
                start_frame=start_frame,
                image_kwargs=image_kwargs,
                pos_cond_kwargs=neg_cond_kwargs,
                current_timestep=current_timestep,
                attn_metadata=attn_metadata,
                target_dtype=target_dtype,
                autocast_enabled=autocast_enabled,
            )
            flow_pred = flow_pred_uncond + guidance_scale * (
                flow_pred_cond - flow_pred_uncond
            )

            next_latents = scheduler.step(
                flow_pred,
                timestep,
                current_latents,
                return_dict=False,
            )[0]
            current_latents = next_latents
            if clamp_latent is not None:
                current_latents[:, :, :context_frames] = clamp_latent

            if progress_bar is not None:
                progress_bar.update()

        return current_latents, attn_metadata

    def _denoise_and_update_causal_block_cfg(
        self,
        batch: Req,
        server_args: ServerArgs,
        *,
        chunk_latents: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        prompt_embeds,
        negative_prompt_embeds,
        kv_cache,
        crossattn_cache,
        kv_cache_neg,
        crossattn_cache_neg,
        current_start_tokens: int,
        start_frame: int,
        image_kwargs: dict,
        pos_cond_kwargs: dict,
        neg_cond_kwargs: dict,
        target_dtype: torch.dtype,
        autocast_enabled: bool,
        device: torch.device,
        attn_raw_latent_shape: tuple[int, int, int],
        prepare_model_input: Callable[[torch.Tensor], torch.Tensor],
        prepare_context_input: Callable[[torch.Tensor], torch.Tensor],
        progress_bar=None,
    ) -> torch.Tensor:
        current_latents, attn_metadata = self._denoise_causal_dmd_chunk_cfg(
            batch,
            server_args,
            chunk_latents=chunk_latents,
            scheduler=scheduler,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            kv_cache_neg=kv_cache_neg,
            crossattn_cache_neg=crossattn_cache_neg,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=pos_cond_kwargs,
            neg_cond_kwargs=neg_cond_kwargs,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
            device=device,
            attn_raw_latent_shape=attn_raw_latent_shape,
            prepare_model_input=prepare_model_input,
            progress_bar=progress_bar,
        )
        context_input = prepare_context_input(current_latents)
        self._update_causal_context_cache(
            batch,
            server_args,
            context_input=context_input,
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
        self._update_causal_context_cache(
            batch,
            server_args,
            context_input=context_input,
            prompt_embeds=negative_prompt_embeds,
            kv_cache=kv_cache_neg,
            crossattn_cache=crossattn_cache_neg,
            current_start_tokens=current_start_tokens,
            start_frame=start_frame,
            image_kwargs=image_kwargs,
            pos_cond_kwargs=neg_cond_kwargs,
            attn_metadata=attn_metadata,
            target_dtype=target_dtype,
            autocast_enabled=autocast_enabled,
        )
        return current_latents

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
            (context_input.shape[0], context_input.shape[2]),
            float(context_noise),
            device=context_input.device,
            dtype=torch.float32,
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
