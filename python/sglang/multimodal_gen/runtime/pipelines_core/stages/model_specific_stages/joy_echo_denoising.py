# SPDX-License-Identifier: Apache-2.0
import torch

from sglang.multimodal_gen.configs.pipeline_configs.joy_echo import (
    JoyEchoPipelineConfig,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.joy_echo_memory import (
    build_memory_audio_rope_coords,
    build_memory_self_attention_block_mask,
    build_memory_video_rope_coords,
    build_paired_memory_cross_mask,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.ltx_2_denoising import (
    DenoisingStepState,
    LTX2DenoisingContext,
    LTX2ModelInputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class JoyEchoDMDDenoisingStage(LTX2AVDenoisingStage):
    """JoyEcho DMD denoising with optional memory prefix and late-layer masks."""

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
                batch_size, memory_seq_len, timestep.shape[-1], device=device, dtype=dtype
            )
            target_ts = timestep[:, :target_seq_len, :]
            return torch.cat([memory_ts, target_ts], dim=1)
        if timestep.ndim == 2:
            memory_ts = torch.zeros(batch_size, memory_seq_len, device=device, dtype=dtype)
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
        if tokens_per_latent_frame <= 0 or memory_video_len % tokens_per_latent_frame != 0:
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
        # a2v: video queries attend to audio keys -> [B, V, A]
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
        # v2a: audio queries attend to video keys -> [B, A, V]
        v2a_mask = build_paired_memory_cross_mask(
            batch_size=batch_size,
            query_memory_seq_len=memory_audio_len,
            query_target_seq_len=target_audio_len,
            kv_memory_seq_len=memory_video_len,
            kv_target_seq_len=target_video_len,
            num_memory_slots=num_memory_slots,
            device=device,
            query_segment_lengths=memory_info.get("memory_audio_segment_lengths"),
        )
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
            memory_downscale_factor=int(
                memory_info.get("memory_downscale_factor", 1)
            ),
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
                video_self_attention_mask=None,
                audio_self_attention_mask=audio_self_attention_mask,
                a2v_cross_attention_mask=a2v_mask,
                v2a_cross_attention_mask=v2a_mask,
            ),
            {
                "memory_video_len": memory_video_len,
                "memory_audio_len": memory_audio_len,
                "late_layer_ratio": late_layer_ratio,
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
            video_noise = self._randn_like_with_batch_generators(
                ctx.latents, batch
            ).float()
            audio_noise = self._randn_like_with_batch_generators(
                ctx.audio_latents, batch
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

        ctx.latents = next_video_latents
        ctx.audio_latents = next_audio_latents
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )
