# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Any

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.causal_denoising import (
    CAUSAL_BLOCK_PROMPTS_KEY,
    CAUSAL_SCENE_CUT_MASK_KEY,
    CAUSAL_SHOT_INDICES_KEY,
    CausalDMDDenoisingStage,
    expand_causal_block_prompts,
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


def expand_longlive2_shot_prompts(
    shot_prompts: list[str],
    *,
    num_blocks: int,
    shot_durations: list[int] | None = None,
    chunks_per_shot: int = 0,
    scene_cut_prefix: str = LONG_LIVE2_DEFAULT_SCENE_CUT_PREFIX,
) -> list[str]:
    return expand_causal_block_prompts(
        shot_prompts,
        num_blocks=num_blocks,
        shot_durations=shot_durations,
        chunks_per_shot=chunks_per_shot,
        scene_cut_prefix=scene_cut_prefix,
    )[0]


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

        block_prompts, scene_cut_mask, shot_indices = expand_causal_block_prompts(
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
        batch.extra[CAUSAL_BLOCK_PROMPTS_KEY] = block_prompts
        batch.extra[CAUSAL_SCENE_CUT_MASK_KEY] = scene_cut_mask
        batch.extra[CAUSAL_SHOT_INDICES_KEY] = shot_indices
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
        return self._forward_one_shot_common(
            batch, server_args, use_cfg=self._use_cfg(batch)
        )

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

    def _multi_shot_sink_enabled(self, batch: Req) -> bool:
        return (
            self._block_prompt_count(batch) is not None
            and bool(getattr(batch, "multi_shot_sink", True))
            and self.sink_size > 0
        )

    def _causal_kv_cache_global_sink_tokens_for_batch(self, batch: Req) -> int:
        if not self._multi_shot_sink_enabled(batch):
            return 0
        return self._get_causal_sink_tokens()

    def _is_scene_cut(self, batch: Req, block_index: int) -> bool:
        if not self._multi_shot_sink_enabled(batch):
            return False
        return super()._is_scene_cut(batch, block_index)

    def _set_rope_temporal_offset(self, batch: Req, shot_index: int) -> None:
        offset = float(getattr(batch, "multi_shot_rope_offset", 8.0) or 0.0)
        self._rope_temporal_offset = shot_index * offset

    def _forward_one_shot_common(
        self, batch: Req, server_args: ServerArgs, *, use_cfg: bool
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

        negative_prompt_embeds = None
        neg_cond_kwargs = None
        if use_cfg:
            neg_cond_kwargs = self._prepare_causal_dmd_neg_cond_kwargs(
                batch, server_args, target_dtype
            )
            negative_prompt_embeds = self._get_negative_prompt_embeds(batch)

        independent_first_frame = self.transformer.independent_first_frame
        max_text_len = self._get_max_text_len(server_args)
        kv_cache_kwargs = self._causal_kv_cache_kwargs_for_batch(batch)
        self._rope_temporal_offset = 0.0

        if self._cache_needs_reinit_for_batch(self.causal_kv_cache, batch):
            self._initialize_causal_caches(
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

        kv_cache_neg = None
        crossattn_cache_neg = None
        if use_cfg:
            kv_cache_neg, crossattn_cache_neg = self._reset_or_init_negative_caches(
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
                batch, prompt_embeds, 0
            )
            warmup_pos_cond_kwargs = self._select_block_cond_kwargs(
                batch, pos_cond_kwargs, 0
            )
            warmup_neg_prompt_embeds = (
                self._select_block_prompt_embeds(batch, negative_prompt_embeds, 0)
                if use_cfg
                else None
            )
            warmup_neg_cond_kwargs = (
                self._select_block_cond_kwargs(batch, neg_cond_kwargs, 0)
                if use_cfg
                else None
            )

            def warm_up(context_input, start_frame):
                self._warm_up_causal_context_cache(
                    batch,
                    server_args,
                    context_input=context_input,
                    prompt_embeds=warmup_prompt_embeds,
                    kv_cache=self.causal_kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start_frame=start_frame,
                    image_kwargs=image_kwargs,
                    pos_cond_kwargs=warmup_pos_cond_kwargs,
                    target_dtype=target_dtype,
                    autocast_enabled=autocast_enabled,
                )
                if use_cfg:
                    self._warm_up_causal_context_cache(
                        batch,
                        server_args,
                        context_input=context_input,
                        prompt_embeds=warmup_neg_prompt_embeds,
                        kv_cache=kv_cache_neg,
                        crossattn_cache=crossattn_cache_neg,
                        current_start_frame=start_frame,
                        image_kwargs=image_kwargs,
                        pos_cond_kwargs=warmup_neg_cond_kwargs,
                        target_dtype=target_dtype,
                        autocast_enabled=autocast_enabled,
                    )

            if independent_first_frame and input_frames >= 1:
                warm_up(image_latent[:, :, :1, :, :], current_start_frame)
                current_start_frame += 1
                remaining_frames = input_frames - 1
            else:
                remaining_frames = input_frames

            while remaining_frames > 0:
                block = min(self.num_frames_per_block, remaining_frames)
                warm_up(
                    image_latent[
                        :, :, current_start_frame : current_start_frame + block, :, :
                    ],
                    current_start_frame,
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
        else:
            if (t - 1) % self.num_frames_per_block != 0:
                raise ValueError(
                    "(num_frames - 1) must be divisible by num_frame_per_block when independent_first_frame=True"
                )
            num_blocks = (t - 1) // self.num_frames_per_block
            block_sizes = [1] + [self.num_frames_per_block] * num_blocks

        start_index = 0
        self._validate_block_prompt_count(batch, block_sizes)

        def prepare_context_input(current_latents):
            return current_latents

        with self.progress_bar(total=len(block_sizes) * len(timesteps)) as progress_bar:
            for block_index, current_num_frames in enumerate(block_sizes):
                self._set_rope_temporal_offset(
                    batch, self._shot_index(batch, block_index)
                )
                is_scene_cut = self._is_scene_cut(batch, block_index)

                current_latents = latents[
                    :, :, start_index : start_index + current_num_frames, :, :
                ]
                current_prompt_embeds = self._select_block_prompt_embeds(
                    batch, prompt_embeds, block_index
                )
                current_pos_cond_kwargs = self._select_block_cond_kwargs(
                    batch, pos_cond_kwargs, block_index
                )

                caches = [self.crossattn_cache]
                if use_cfg:
                    caches.append(crossattn_cache_neg)
                self._reset_crossattn_cache_for_block(batch, *caches)

                def prepare_model_input(current_latents):
                    latent_model_input = current_latents
                    if (
                        batch.image_latent is not None
                        and independent_first_frame
                        and start_index == 0
                    ):
                        latent_model_input = torch.cat(
                            [latent_model_input, batch.image_latent], dim=2
                        )
                    return latent_model_input

                current_start_tokens = (
                    pos_start_base + start_index
                ) * self.num_token_per_frame
                block_kwargs = dict(
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
                if use_cfg:
                    current_latents = self._denoise_and_update_causal_block_cfg(
                        batch,
                        server_args,
                        negative_prompt_embeds=self._select_block_prompt_embeds(
                            batch, negative_prompt_embeds, block_index
                        ),
                        kv_cache_neg=kv_cache_neg,
                        crossattn_cache_neg=crossattn_cache_neg,
                        neg_cond_kwargs=self._select_block_cond_kwargs(
                            batch, neg_cond_kwargs, block_index
                        ),
                        **block_kwargs,
                    )
                else:
                    current_latents = self._denoise_and_update_causal_block(
                        batch, server_args, **block_kwargs
                    )

                if is_scene_cut:
                    self._pin_current_chunk(self.causal_kv_cache, current_num_frames)
                    if use_cfg:
                        self._pin_current_chunk(kv_cache_neg, current_num_frames)

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
