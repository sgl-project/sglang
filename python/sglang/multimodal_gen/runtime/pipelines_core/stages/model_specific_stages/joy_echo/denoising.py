# SPDX-License-Identifier: Apache-2.0
import dataclasses

import torch

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joy_echo.memory import (
    build_memory_audio_rope_coords,
    build_memory_self_attention_block_mask,
    build_memory_video_rope_coords,
    build_paired_memory_cross_mask,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    DenoisingStepState,
    LTX2DenoisingContext,
    LTX2ModelInputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class JoyEchoDMDDenoisingStage(LTX2AVDenoisingStage):
    """JoyEcho DMD denoising with optional memory prefix and late-layer masks."""

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        ctx = super()._prepare_denoising_loop(batch, server_args)
        if get_sp_world_size() <= 1:
            return ctx
        # JoyEcho DMD: shard video on time, replicate full audio on every rank.
        # Dual-sharding audio+video adds heavy a2v/v2a all_gather per layer; shot 1
        # already uses this layout when memory is injected.
        ctx.replicate_audio_for_sp = True
        batch.ltx23_audio_replicated_for_sp = True
        batch.did_sp_shard_audio_latents = False
        return ctx

    @staticmethod
    def _zero_sp_shard_padding(
        latents: torch.Tensor,
        *,
        valid_token_count: int | None,
    ) -> torch.Tensor:
        if valid_token_count is None or int(valid_token_count) >= int(latents.shape[1]):
            return latents
        latents = latents.clone()
        latents[:, int(valid_token_count) :, :] = 0.0
        return latents

    @staticmethod
    def _expand_sp_token_timestep(
        timestep: torch.Tensor,
        *,
        batch_size: int,
        seq_len: int,
        valid_token_count: int | None,
    ) -> torch.Tensor:
        """Expand legacy [B] timesteps to [B, S] and zero SP padding tokens."""
        if timestep.ndim >= 2 and int(timestep.shape[1]) == int(seq_len):
            ts = timestep
        elif timestep.ndim == 1:
            ts = timestep.view(batch_size, 1).expand(batch_size, int(seq_len))
        else:
            ts = timestep
        if valid_token_count is not None and int(valid_token_count) < int(seq_len):
            ts = ts.clone()
            ts[:, int(valid_token_count) :] = 0.0
        return ts

    def _prepare_ltx2_model_inputs(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
        sigma: torch.Tensor,
    ) -> LTX2ModelInputs:
        model_inputs = super()._prepare_ltx2_model_inputs(
            ctx, step, batch, server_args, sigma
        )
        if not batch.did_sp_shard_latents:
            return model_inputs

        batch_size = int(model_inputs.latent_model_input.shape[0])
        seq_v = int(model_inputs.latent_model_input.shape[1])
        video_valid = batch.sp_video_valid_token_count
        video_self_attention_mask = self._build_ltx2_sp_padding_mask(
            batch,
            seq_len=seq_v,
            batch_size=batch_size,
            key="sp_video_valid_token_count",
            device=model_inputs.latent_model_input.device,
        )
        video_coords = server_args.pipeline_config.prepare_video_rope_coords_for_sp(
            step.current_model,
            batch,
            model_inputs.latent_model_input,
            num_frames=ctx.latent_num_frames_for_model,
            height=ctx.latent_height,
            width=ctx.latent_width,
        )
        timestep_video = self._expand_sp_token_timestep(
            model_inputs.timestep_video,
            batch_size=batch_size,
            seq_len=seq_v,
            valid_token_count=int(video_valid) if video_valid is not None else None,
        )

        audio_self_attention_mask = model_inputs.audio_self_attention_mask
        audio_coords = model_inputs.audio_coords
        timestep_audio = model_inputs.timestep_audio
        a2v_cross_attention_mask = model_inputs.a2v_cross_attention_mask
        v2a_cross_attention_mask = video_self_attention_mask

        if batch.did_sp_shard_audio_latents:
            seq_a = int(model_inputs.audio_num_frames_latent)
            audio_valid = batch.sp_audio_valid_token_count
            audio_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=seq_a,
                batch_size=batch_size,
                key="sp_audio_valid_token_count",
                device=model_inputs.audio_latent_model_input.device,
            )
            audio_coords = server_args.pipeline_config.prepare_audio_rope_coords_for_sp(
                step.current_model,
                batch,
                model_inputs.audio_latent_model_input,
                num_frames=model_inputs.audio_num_frames_latent,
            )
            timestep_audio = self._expand_sp_token_timestep(
                model_inputs.timestep_audio,
                batch_size=batch_size,
                seq_len=seq_a,
                valid_token_count=int(audio_valid) if audio_valid is not None else None,
            )
            a2v_cross_attention_mask = audio_self_attention_mask

        return dataclasses.replace(
            model_inputs,
            video_coords=video_coords,
            audio_coords=audio_coords,
            timestep_video=timestep_video,
            timestep_audio=timestep_audio,
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
        kwargs = super()._build_ltx2_base_model_kwargs(ctx, batch, model_inputs)
        if not batch.did_sp_shard_latents:
            return kwargs
        kwargs.update(
            {
                "video_self_attention_mask": model_inputs.video_self_attention_mask,
                "audio_self_attention_mask": model_inputs.audio_self_attention_mask,
                "a2v_cross_attention_mask": model_inputs.a2v_cross_attention_mask,
                "v2a_cross_attention_mask": model_inputs.v2a_cross_attention_mask,
                "audio_replicated_for_sp": bool(ctx.replicate_audio_for_sp),
                "legacy_ltx23_one_stage_semantics": False,
            }
        )
        return kwargs

    @staticmethod
    def _dmd_add_noise(
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma_t = sigma.to(device=original.device, dtype=original.dtype)
        if sigma_t.ndim == 1:
            sigma_t = sigma_t.reshape(-1, *[1] * (original.ndim - 1))
        elif sigma_t.ndim == 2:
            sigma_t = sigma_t.reshape(*sigma_t.shape, *[1] * (original.ndim - 2))
        return (1.0 - sigma_t) * original + sigma_t * noise

    def _sample_sp_consistent_noise(
        self,
        local_reference: torch.Tensor,
        batch: Req,
        server_args: ServerArgs,
        *,
        shard_video: bool,
        shard_audio: bool,
    ) -> torch.Tensor:
        """Sample renoise on the global latent layout, then shard for SP."""
        if shard_video:
            raw_shape = batch.raw_latent_shape
            if not (isinstance(raw_shape, tuple) and len(raw_shape) == 3):
                raise ValueError(
                    "SP DMD renoise requires packed video `batch.raw_latent_shape`."
                )
            full_reference = torch.empty(
                tuple(raw_shape),
                device=local_reference.device,
                dtype=local_reference.dtype,
            )
            full_noise = self._randn_like_with_batch_generators(full_reference, batch)
            sharded_noise, _ = server_args.pipeline_config.shard_latents_for_sp(
                batch, full_noise
            )
            return sharded_noise

        if shard_audio:
            orig_audio_len = batch.sp_audio_orig_num_frames
            if orig_audio_len <= 0:
                raise ValueError(
                    "SP DMD renoise requires `batch.sp_audio_orig_num_frames`."
                )
            full_reference = torch.empty(
                (
                    int(local_reference.shape[0]),
                    int(orig_audio_len),
                    int(local_reference.shape[2]),
                ),
                device=local_reference.device,
                dtype=local_reference.dtype,
            )
            full_noise = self._randn_like_with_batch_generators(full_reference, batch)
            sharded_noise, _ = server_args.pipeline_config.shard_audio_latents_for_sp(
                batch, full_noise
            )
            return sharded_noise

        return self._randn_like_with_batch_generators(local_reference, batch)

    @staticmethod
    def _apply_memory_prefix_to_timestep(
        timestep: torch.Tensor,
        *,
        memory_seq_len: int,
        target_seq_len: int,
    ) -> torch.Tensor:
        if memory_seq_len <= 0:
            return timestep
        batch_size = int(timestep.shape[0])
        device = timestep.device
        dtype = timestep.dtype
        if timestep.ndim == 3:
            memory_ts = torch.zeros(
                batch_size,
                memory_seq_len,
                timestep.shape[-1],
                device=device,
                dtype=dtype,
            )
            target_ts = timestep[:, :target_seq_len, :]
            return torch.cat([memory_ts, target_ts], dim=1)
        if timestep.ndim == 2:
            memory_ts = torch.zeros(
                batch_size, memory_seq_len, device=device, dtype=dtype
            )
            target_ts = timestep[:, :target_seq_len]
            return torch.cat([memory_ts, target_ts], dim=1)
        if timestep.ndim == 1:
            # JoyEcho legacy one-stage uses scalar [B] timesteps; memory tokens
            # must see sigma=0 (clean) while target tokens keep the current sigma.
            memory_ts = torch.zeros(
                batch_size, memory_seq_len, 1, device=device, dtype=dtype
            )
            target_ts = timestep.view(batch_size, 1, 1).expand(
                batch_size, target_seq_len, 1
            )
            return torch.cat([memory_ts, target_ts], dim=1)
        return timestep

    def _build_memory_model_inputs(
        self,
        model_inputs: LTX2ModelInputs,
        batch: Req,
        ctx: LTX2DenoisingContext,
        server_args: ServerArgs,
        step_model,
    ) -> tuple[LTX2ModelInputs, dict[str, int]]:
        memory_info = batch.extra.get("joy_echo_memory")
        if not memory_info:
            return model_inputs, {}

        memory_video = memory_info["memory_video_packed"].to(
            device=model_inputs.latent_model_input.device,
            dtype=model_inputs.latent_model_input.dtype,
        )
        memory_audio = memory_info["memory_audio"].to(
            device=model_inputs.audio_latent_model_input.device,
            dtype=model_inputs.audio_latent_model_input.dtype,
        )

        target_video = model_inputs.latent_model_input
        target_audio = model_inputs.audio_latent_model_input
        memory_video_len = int(memory_video.shape[1])
        memory_audio_len = int(memory_audio.shape[1])
        target_video_len = int(target_video.shape[1])
        target_audio_len = int(target_audio.shape[1])
        tokens_per_latent_frame = int(ctx.latent_height) * int(ctx.latent_width)
        if (
            tokens_per_latent_frame <= 0
            or memory_video_len % tokens_per_latent_frame != 0
        ):
            num_memory_slots = int(memory_info["num_memory_slots"])
        else:
            num_memory_slots = memory_video_len // tokens_per_latent_frame

        latent_model_input = torch.cat([memory_video, target_video], dim=1)
        audio_latent_model_input = torch.cat([memory_audio, target_audio], dim=1)

        timestep_video = self._apply_memory_prefix_to_timestep(
            model_inputs.timestep_video,
            memory_seq_len=memory_video_len,
            target_seq_len=target_video_len,
        )
        timestep_audio = self._apply_memory_prefix_to_timestep(
            model_inputs.timestep_audio,
            memory_seq_len=memory_audio_len,
            target_seq_len=target_audio_len,
        )

        device = latent_model_input.device
        batch_size = int(latent_model_input.shape[0])

        sp_world_size = get_sp_world_size()
        sp_on = sp_world_size > 1 and batch.did_sp_shard_latents
        if sp_on:
            target_video_full_len = int(target_video_len) * int(sp_world_size)
            raw_shape = batch.raw_latent_shape
            if isinstance(raw_shape, tuple) and len(raw_shape) == 3:
                target_video_valid_len = int(raw_shape[1])
            else:
                target_video_valid_len = target_video_full_len
            sp_target_start_offset = batch.sp_video_start_frame
        else:
            target_video_full_len = target_video_len
            target_video_valid_len = target_video_len
            sp_target_start_offset = 0

        a2v_mask = build_paired_memory_cross_mask(
            batch_size=batch_size,
            query_memory_seq_len=memory_video_len,
            query_target_seq_len=target_video_len,
            kv_memory_seq_len=memory_audio_len,
            kv_target_seq_len=target_audio_len,
            num_memory_slots=num_memory_slots,
            device=device,
            kv_segment_lengths=memory_info.get("memory_audio_segment_lengths"),
        )
        v2a_mask = build_paired_memory_cross_mask(
            batch_size=batch_size,
            query_memory_seq_len=memory_audio_len,
            query_target_seq_len=target_audio_len,
            kv_memory_seq_len=memory_video_len,
            kv_target_seq_len=target_video_full_len,
            num_memory_slots=num_memory_slots,
            device=device,
            query_segment_lengths=memory_info.get("memory_audio_segment_lengths"),
        )
        video_self_attention_mask = None
        if sp_on and target_video_full_len > target_video_valid_len:
            v2a_mask[:, :, memory_video_len + target_video_valid_len :] = False
        if sp_on:
            vself_len = memory_video_len + target_video_full_len
            video_self_attention_mask = torch.ones(
                (batch_size, vself_len), device=device, dtype=torch.bool
            )
            if target_video_full_len > target_video_valid_len:
                video_self_attention_mask[
                    :, memory_video_len + target_video_valid_len :
                ] = False
        audio_self_attention_mask = build_memory_self_attention_block_mask(
            batch_size=batch_size,
            memory_seq_len=memory_audio_len,
            target_seq_len=target_audio_len,
            device=device,
        )

        config = server_args.pipeline_config
        late_layer_ratio = 1.0
        memory_position_mode = "reference"
        if isinstance(config, JoyEchoPipelineConfig):
            late_layer_ratio = float(config.late_layer_ratio)
            memory_position_mode = config.memory_position_mode

        video_coords = build_memory_video_rope_coords(
            rope=step_model.rope,
            batch_size=batch_size,
            memory_video_len=memory_video_len,
            target_num_frames=int(ctx.latent_num_frames_for_model),
            latent_height=int(ctx.latent_height),
            latent_width=int(ctx.latent_width),
            device=device,
            fps=float(batch.fps),
            memory_position_mode=str(
                memory_info.get("memory_position_mode", memory_position_mode)
            ),
            memory_downscale_factor=int(memory_info.get("memory_downscale_factor", 1)),
            sp_target_start_offset=sp_target_start_offset,
        )
        audio_coords = build_memory_audio_rope_coords(
            audio_rope=step_model.audio_rope,
            batch_size=batch_size,
            memory_audio_len=memory_audio_len,
            target_audio_len=target_audio_len,
            device=device,
            memory_position_mode=str(
                memory_info.get("memory_position_mode", memory_position_mode)
            ),
        )

        return (
            LTX2ModelInputs(
                latent_model_input=latent_model_input,
                audio_latent_model_input=audio_latent_model_input,
                audio_num_frames_latent=memory_audio_len + target_audio_len,
                video_coords=video_coords,
                audio_coords=audio_coords,
                timestep_video=timestep_video,
                timestep_audio=timestep_audio,
                prompt_timestep_video=model_inputs.prompt_timestep_video,
                prompt_timestep_audio=model_inputs.prompt_timestep_audio,
                video_self_attention_mask=video_self_attention_mask,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_mask,
                v2a_cross_attention_mask=v2a_mask,
            ),
            {
                "memory_video_len": memory_video_len,
                "memory_audio_len": memory_audio_len,
                "late_layer_ratio": late_layer_ratio,
                "audio_replicated_for_sp": sp_on,
                "video_memory_prefix_len": memory_video_len if sp_on else 0,
            },
        )

    def _run_denoising_step(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        if ctx.audio_latents is None:
            raise ValueError("JoyEcho requires audio latents for denoising.")
        if ctx.audio_scheduler is None:
            raise ValueError("JoyEcho audio scheduler was not prepared.")

        sigmas = ctx.scheduler.sigmas
        if not isinstance(sigmas, torch.Tensor):
            raise ValueError("Expected scheduler.sigmas to be a tensor for JoyEcho.")

        sigma = sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_next = sigmas[step.step_index + 1].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_val = float(sigma.item())
        sigma_next_val = float(sigma_next.item())

        model_inputs = self._prepare_ltx2_model_inputs(
            ctx, step, batch, server_args, sigma
        )
        model_inputs, memory_meta = self._build_memory_model_inputs(
            model_inputs, batch, ctx, server_args, step.current_model
        )

        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=ctx.is_ltx23_variant,
        )
        base_model_kwargs = self._build_ltx2_base_model_kwargs(ctx, batch, model_inputs)
        model_kwargs = self._build_ltx2_model_kwargs(
            ctx,
            base_model_kwargs,
            encoder_hidden_states=batch.prompt_embeds[0],
            audio_encoder_hidden_states=batch.audio_prompt_embeds[0],
            encoder_attention_mask=prompt_attention_mask,
        )
        if memory_meta:
            # Legacy one-stage LTX2 skips mask kwargs in the base builder; memory
            # mode must always pass paired cross/self masks to the DiT.
            model_kwargs["late_layer_ratio"] = memory_meta["late_layer_ratio"]
            model_kwargs["late_audio_self_attention_mask"] = None
            model_kwargs["video_self_attention_mask"] = (
                model_inputs.video_self_attention_mask
            )
            model_kwargs["audio_self_attention_mask"] = (
                model_inputs.audio_self_attention_mask
            )
            model_kwargs["a2v_cross_attention_mask"] = (
                model_inputs.a2v_cross_attention_mask
            )
            model_kwargs["v2a_cross_attention_mask"] = (
                model_inputs.v2a_cross_attention_mask
            )
            model_kwargs["audio_replicated_for_sp"] = memory_meta[
                "audio_replicated_for_sp"
            ]
            model_kwargs["video_memory_prefix_len"] = memory_meta[
                "video_memory_prefix_len"
            ]

        with self._ltx2_model_forward_context(ctx, step):
            model_video, model_audio = step.current_model(**model_kwargs)

        if memory_meta:
            memory_video_len = memory_meta["memory_video_len"]
            memory_audio_len = memory_meta["memory_audio_len"]
            model_video = model_video[:, memory_video_len:, :]
            if model_audio is not None:
                model_audio = model_audio[:, memory_audio_len:, :]

        denoised_video = self._ltx2_velocity_to_x0(
            ctx.latents, model_video.float(), sigma_val
        )
        denoised_audio = self._ltx2_velocity_to_x0(
            ctx.audio_latents, model_audio.float(), sigma_val
        )
        denoised_video = self._ltx2_apply_clean_latent_mask(denoised_video, ctx)

        if sigma_next_val > 0.0:
            video_noise = self._sample_sp_consistent_noise(
                ctx.latents,
                batch,
                server_args,
                shard_video=batch.did_sp_shard_latents,
                shard_audio=False,
            ).float()
            audio_noise = self._sample_sp_consistent_noise(
                ctx.audio_latents,
                batch,
                server_args,
                shard_video=False,
                shard_audio=batch.did_sp_shard_audio_latents,
            ).float()
            next_video_latents = self._dmd_add_noise(
                denoised_video, video_noise, sigma_next
            ).to(dtype=ctx.latents.dtype)
            next_audio_latents = self._dmd_add_noise(
                denoised_audio, audio_noise, sigma_next
            ).to(dtype=ctx.audio_latents.dtype)
        else:
            next_video_latents = denoised_video.to(dtype=ctx.latents.dtype)
            next_audio_latents = denoised_audio.to(dtype=ctx.audio_latents.dtype)

        if batch.did_sp_shard_latents:
            next_video_latents = self._zero_sp_shard_padding(
                next_video_latents,
                valid_token_count=batch.sp_video_valid_token_count,
            )
        if batch.did_sp_shard_audio_latents:
            next_audio_latents = self._zero_sp_shard_padding(
                next_audio_latents,
                valid_token_count=batch.sp_audio_valid_token_count,
            )

        ctx.latents = next_video_latents
        ctx.audio_latents = next_audio_latents
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )
