import copy
import time

import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class LTX2AVDenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(self, transformer, scheduler, vae=None, audio_vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self.audio_vae = audio_vae

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
        return int((batch.num_frames - 1) // int(pc.vae_temporal_compression) + 1)

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
        """Disable cache-dit for TI2V-style requests (image-conditioned), to avoid stale activations.

        NOTE: base denoising stage calls this hook with (num_inference_steps, batch).
        """
        if getattr(self, "_disable_cache_dit_for_request", False):
            return
        return super()._maybe_enable_cache_dit(num_inference_steps, batch)

    @staticmethod
    def _resize_center_crop(
        img: PIL.Image.Image, *, width: int, height: int
    ) -> PIL.Image.Image:
        return img.resize((width, height), resample=PIL.Image.Resampling.BILINEAR)

    @staticmethod
    def _pil_to_normed_tensor(img: PIL.Image.Image) -> torch.Tensor:
        # PIL -> numpy [0,1] -> torch [B,C,H,W], then [-1,1]
        arr = pil_to_numpy(img)
        t = numpy_to_pt(arr)
        return normalize(t)

    @staticmethod
    def _should_apply_ltx2_ti2v(batch: Req) -> bool:
        """True if we have an image-latent token prefix to condition with.

        SP note: when token latents are time-sharded, only the rank that owns the
        *global* first latent frame should apply TI2V conditioning (rank with start_frame==0).
        """
        if (
            batch.image_latent is None
            or int(getattr(batch, "ltx2_num_image_tokens", 0)) <= 0
        ):
            return False
        did_sp_shard = bool(getattr(batch, "did_sp_shard_latents", False))
        if not did_sp_shard:
            return True
        return int(getattr(batch, "sp_video_start_frame", 0)) == 0

    def _prepare_ltx2_image_latent(self, batch: Req, server_args: ServerArgs) -> None:
        """Encode `batch.image_path` into packed token latents for LTX-2 TI2V."""
        if (
            batch.image_latent is not None
            and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
        ):
            return
        batch.ltx2_num_image_tokens = 0
        batch.image_latent = None

        if batch.image_path is None:
            return
        if batch.width is None or batch.height is None:
            raise ValueError("width/height must be provided for LTX-2 TI2V.")
        if self.vae is None:
            raise ValueError("VAE must be provided for LTX-2 TI2V.")

        image_path = (
            batch.image_path[0]
            if isinstance(batch.image_path, list)
            else batch.image_path
        )

        img = load_image(image_path)
        img = self._resize_center_crop(
            img, width=int(batch.width), height=int(batch.height)
        )
        batch.condition_image = img

        latents_device = (
            batch.latents.device
            if isinstance(batch.latents, torch.Tensor)
            else torch.device("cpu")
        )
        image_tensor = self._pil_to_normed_tensor(img).to(
            latents_device, dtype=torch.float32
        )
        # [B, C, H, W] -> [B, C, 1, H, W]
        video_condition = image_tensor.unsqueeze(2)

        self.vae = self.vae.to(latents_device)
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not server_args.disable_autocast

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=vae_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(vae_dtype)

            latent_dist: DiagonalGaussianDistribution = self.vae.encode(video_condition)
            if isinstance(latent_dist, AutoencoderKLOutput):
                latent_dist = latent_dist.latent_dist

        mode = server_args.pipeline_config.vae_config.encode_sample_mode()
        if mode == "argmax":
            latent = latent_dist.mode()
        elif mode == "sample":
            if batch.generator is None:
                raise ValueError("Generator must be provided for VAE sampling.")
            latent = latent_dist.sample(batch.generator)
        else:
            raise ValueError(f"Unsupported encode_sample_mode: {mode}")

        # Match the normalized latent space used by this pipeline (inverse of DecodingStage.scale_and_shift).
        scaling_factor, shift_factor = (
            server_args.pipeline_config.get_decode_scale_and_shift(
                device=latent.device, dtype=latent.dtype, vae=self.vae
            )
        )
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(latent.device)
        if isinstance(scaling_factor, torch.Tensor):
            scaling_factor = scaling_factor.to(latent.device)
        if shift_factor is not None:
            latent = latent - shift_factor
        latent = latent * scaling_factor

        packed = server_args.pipeline_config.maybe_pack_latents(
            latent, latent.shape[0], batch
        )
        if not (isinstance(packed, torch.Tensor) and packed.ndim == 3):
            raise ValueError("Expected packed image latents [B, S0, D].")

        # Fail-fast token count: must match one latent frame's tokens.
        vae_sf = int(server_args.pipeline_config.vae_scale_factor)
        patch = int(server_args.pipeline_config.patch_size)
        latent_h = int(batch.height) // vae_sf
        latent_w = int(batch.width) // vae_sf
        expected_tokens = (latent_h // patch) * (latent_w // patch)
        if int(packed.shape[1]) != int(expected_tokens):
            raise ValueError(
                "LTX-2 conditioning token count mismatch: "
                f"{int(packed.shape[1])=} {int(expected_tokens)=}."
            )

        batch.image_latent = packed
        batch.ltx2_num_image_tokens = int(packed.shape[1])

        if batch.debug:
            logger.info(
                "LTX2 TI2V conditioning prepared: %d tokens (shape=%s) for %sx%s",
                batch.ltx2_num_image_tokens,
                tuple(batch.image_latent.shape),
                batch.width,
                batch.height,
            )

        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """
         Run the denoising loop.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with denoised latents.
        """
        # Disable cache-dit for image-conditioned requests (TI2V-style) for correctness/debuggability.
        self._disable_cache_dit_for_request = batch.image_path is not None

        # Prepare variables for the denoising loop

        prepared_vars = self._prepare_denoising_loop(batch, server_args)
        extra_step_kwargs = prepared_vars["extra_step_kwargs"]
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        image_kwargs = prepared_vars["image_kwargs"]
        pos_cond_kwargs = prepared_vars["pos_cond_kwargs"]
        neg_cond_kwargs = prepared_vars["neg_cond_kwargs"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        seq_len = prepared_vars["seq_len"]
        guidance = prepared_vars["guidance"]

        audio_latents = batch.audio_latents
        audio_scheduler = copy.deepcopy(self.scheduler)

        # Prepare TI2V conditioning once (encode image -> patchify tokens).
        self._prepare_ltx2_image_latent(batch, server_args)

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=latents
        )
        latent_height = batch.height // server_args.pipeline_config.vae_scale_factor
        latent_width = batch.width // server_args.pipeline_config.vae_scale_factor

        # Initialize lists for ODE trajectory
        trajectory_timesteps: list[torch.Tensor] = []
        trajectory_latents: list[torch.Tensor] = []
        trajectory_audio_latents: list[torch.Tensor] = []

        # Run denoising loop
        denoising_start_time = time.time()

        # to avoid device-sync caused by timestep comparison
        is_warmup = batch.is_warmup
        self.scheduler.set_begin_index(0)
        audio_scheduler.set_begin_index(0)
        timesteps_cpu = timesteps.cpu()
        num_timesteps = timesteps_cpu.shape[0]

        do_ti2v = self._should_apply_ltx2_ti2v(batch)
        num_img_tokens = int(getattr(batch, "ltx2_num_image_tokens", 0))
        denoise_mask = None
        clean_latent = None
        if do_ti2v:
            if not (isinstance(latents, torch.Tensor) and latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            latents[:, :num_img_tokens, :] = batch.image_latent[
                :, :num_img_tokens, :
            ].to(device=latents.device, dtype=latents.dtype)
            denoise_mask = torch.ones(
                (latents.shape[0], latents.shape[1], 1),
                device=latents.device,
                dtype=torch.float32,
            )
            denoise_mask[:, :num_img_tokens, :] = 0.0
            clean_latent = latents.detach().clone()
            clean_latent[:, :num_img_tokens, :] = batch.image_latent[
                :, :num_img_tokens, :
            ].to(device=latents.device, dtype=latents.dtype)

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=target_dtype,
            enabled=autocast_enabled,
        ):
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t_host in enumerate(timesteps_cpu):
                    with StageProfiler(
                        f"denoising_step_{i}",
                        logger=logger,
                        timings=batch.timings,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                    ):
                        t_int = int(t_host.item())
                        t_device = timesteps[i]
                        current_model, current_guidance_scale = (
                            self._select_and_manage_model(
                                t_int=t_int,
                                boundary_timestep=boundary_timestep,
                                server_args=server_args,
                                batch=batch,
                            )
                        )

                        # Predict noise residual
                        attn_metadata = self._build_attn_metadata(i, batch, server_args)

                        # === LTX-2 sigma-space Euler step (flow matching) ===
                        # Use scheduler-generated sigmas (includes terminal sigma=0).
                        sigmas = getattr(self.scheduler, "sigmas", None)
                        if sigmas is None or not isinstance(sigmas, torch.Tensor):
                            raise ValueError(
                                "Expected scheduler.sigmas to be a tensor for LTX-2."
                            )
                        sigma = sigmas[i].to(device=latents.device, dtype=torch.float32)
                        sigma_next = sigmas[i + 1].to(
                            device=latents.device, dtype=torch.float32
                        )
                        dt = sigma_next - sigma

                        latent_model_input = latents.to(target_dtype)
                        audio_latent_model_input = audio_latents.to(target_dtype)

                        latent_num_frames = latent_num_frames_for_model

                        # Audio latent dims
                        if audio_latent_model_input.ndim == 3:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[1]
                            )
                        elif audio_latent_model_input.ndim == 4:
                            audio_num_frames_latent = int(
                                audio_latent_model_input.shape[2]
                            )
                        else:
                            raise ValueError(
                                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
                            )

                        # LTX-2 model can generate coords internally.
                        video_coords = None
                        audio_coords = None

                        timestep = t_device.expand(int(latent_model_input.shape[0]))
                        if do_ti2v and denoise_mask is not None:
                            timestep_video = timestep.unsqueeze(
                                -1
                            ) * denoise_mask.squeeze(-1)
                        else:
                            timestep_video = timestep
                        timestep_audio = timestep

                        # Conditions
                        encoder_hidden_states = batch.prompt_embeds[0]
                        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                        encoder_attention_mask = batch.prompt_attention_mask

                        # Follow ltx-pipelines structure: separate pos/neg forward passes,
                        # then apply CFG on denoised (x0) predictions.
                        with set_forward_context(
                            current_timestep=i, attn_metadata=attn_metadata
                        ):
                            v_pos, a_v_pos = current_model(
                                hidden_states=latent_model_input,
                                audio_hidden_states=audio_latent_model_input,
                                encoder_hidden_states=encoder_hidden_states,
                                audio_encoder_hidden_states=audio_encoder_hidden_states,
                                timestep=timestep_video,
                                audio_timestep=timestep_audio,
                                encoder_attention_mask=encoder_attention_mask,
                                audio_encoder_attention_mask=encoder_attention_mask,
                                num_frames=latent_num_frames,
                                height=latent_height,
                                width=latent_width,
                                fps=batch.fps,
                                audio_num_frames=audio_num_frames_latent,
                                video_coords=video_coords,
                                audio_coords=audio_coords,
                                return_latents=False,
                                return_dict=False,
                            )

                            if batch.do_classifier_free_guidance:
                                neg_encoder_hidden_states = (
                                    batch.negative_prompt_embeds[0]
                                )
                                neg_audio_encoder_hidden_states = (
                                    batch.negative_audio_prompt_embeds[0]
                                )
                                neg_encoder_attention_mask = (
                                    batch.negative_attention_mask
                                )

                                v_neg, a_v_neg = current_model(
                                    hidden_states=latent_model_input,
                                    audio_hidden_states=audio_latent_model_input,
                                    encoder_hidden_states=neg_encoder_hidden_states,
                                    audio_encoder_hidden_states=neg_audio_encoder_hidden_states,
                                    timestep=timestep_video,
                                    audio_timestep=timestep_audio,
                                    encoder_attention_mask=neg_encoder_attention_mask,
                                    audio_encoder_attention_mask=neg_encoder_attention_mask,
                                    num_frames=latent_num_frames,
                                    height=latent_height,
                                    width=latent_width,
                                    fps=batch.fps,
                                    audio_num_frames=audio_num_frames_latent,
                                    video_coords=video_coords,
                                    audio_coords=audio_coords,
                                    return_latents=False,
                                    return_dict=False,
                                )
                            else:
                                v_neg = None
                                a_v_neg = None

                        v_pos = v_pos.float()
                        a_v_pos = a_v_pos.float()
                        if v_neg is not None:
                            v_neg = v_neg.float()
                        if a_v_neg is not None:
                            a_v_neg = a_v_neg.float()

                        # Velocity -> denoised (x0): x0 = x - sigma * v
                        sigma_val = float(sigma.item())
                        denoised_video = latents.float() - sigma_val * v_pos
                        denoised_audio = audio_latents.float() - sigma_val * a_v_pos

                        if (
                            batch.do_classifier_free_guidance
                            and v_neg is not None
                            and a_v_neg is not None
                        ):
                            denoised_video_neg = latents.float() - sigma_val * v_neg
                            denoised_audio_neg = (
                                audio_latents.float() - sigma_val * a_v_neg
                            )
                            denoised_video = denoised_video + (
                                batch.guidance_scale - 1.0
                            ) * (denoised_video - denoised_video_neg)
                            denoised_audio = denoised_audio + (
                                batch.guidance_scale - 1.0
                            ) * (denoised_audio - denoised_audio_neg)

                        # Apply conditioning mask (keep conditioned tokens clean).
                        if (
                            do_ti2v
                            and denoise_mask is not None
                            and clean_latent is not None
                        ):
                            denoised_video = (
                                denoised_video * denoise_mask
                                + clean_latent.float() * (1.0 - denoise_mask)
                            )

                        # Euler step in sigma space: x_next = x + (sigma_next - sigma) * v,
                        # where v = (x - x0) / sigma.
                        if sigma_val == 0.0:
                            v_video = torch.zeros_like(denoised_video)
                            v_audio = torch.zeros_like(denoised_audio)
                        else:
                            v_video = (latents.float() - denoised_video) / sigma_val
                            v_audio = (
                                audio_latents.float() - denoised_audio
                            ) / sigma_val

                        latents = (latents.float() + v_video * dt).to(
                            dtype=latents.dtype
                        )
                        audio_latents = (audio_latents.float() + v_audio * dt).to(
                            dtype=audio_latents.dtype
                        )

                        if do_ti2v:
                            latents[:, :num_img_tokens, :] = batch.image_latent[
                                :, :num_img_tokens, :
                            ].to(device=latents.device, dtype=latents.dtype)

                        latents = self.post_forward_for_ti2v_task(
                            batch, server_args, reserved_frames_mask, latents, z
                        )

                        # save trajectory latents if needed
                        if batch.return_trajectory_latents:
                            trajectory_timesteps.append(t_host)
                            trajectory_latents.append(latents)
                            if audio_latents is not None:
                                trajectory_audio_latents.append(audio_latents)

                        # Update progress bar
                        if i == num_timesteps - 1 or (
                            (i + 1) > num_warmup_steps
                            and (i + 1) % self.scheduler.order == 0
                            and progress_bar is not None
                        ):
                            progress_bar.update()

                        if not is_warmup:
                            self.step_profile()

        denoising_end_time = time.time()

        if num_timesteps > 0 and not is_warmup:
            self.log_info(
                "average time per step: %.4f seconds",
                (denoising_end_time - denoising_start_time) / len(timesteps),
            )

        batch.audio_latents = audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            trajectory_audio_latents=trajectory_audio_latents,
            server_args=server_args,
            is_warmup=is_warmup,
        )

        return batch

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        trajectory_audio_latents: list,
        server_args: ServerArgs,
        is_warmup: bool = False,
    ):
        # 1. Handle Trajectory (Video) - Copy from base
        if trajectory_latents:
            trajectory_tensor = torch.stack(trajectory_latents, dim=1)
            trajectory_timesteps_tensor = torch.stack(trajectory_timesteps, dim=0)
        else:
            trajectory_tensor = None
            trajectory_timesteps_tensor = None

        latents, trajectory_tensor = self._postprocess_sp_latents(
            batch, latents, trajectory_tensor
        )

        # If SP time-sharding padded whole frames worth of tokens, remove padding
        # after gather and before unpacking.
        latents = self._truncate_sp_padded_token_latents(batch, latents)

        if trajectory_tensor is not None and trajectory_timesteps_tensor is not None:
            batch.trajectory_timesteps = trajectory_timesteps_tensor.cpu()
            batch.trajectory_latents = trajectory_tensor.cpu()

        # 2. Handle Trajectory (Audio) - LTX-2 specific
        if trajectory_audio_latents:
            trajectory_audio_tensor = torch.stack(trajectory_audio_latents, dim=1)
            # We don't have SP support for audio latents yet (or needed?)
            batch.trajectory_audio_latents = trajectory_audio_tensor.cpu()

        # 3. Unpack and Denormalize
        # Call pipeline_config._unpad_and_unpack_latents
        # latents is video latents.
        # batch.audio_latents is audio latents.

        audio_latents = batch.audio_latents

        # NOTE: self.vae and self.audio_vae should be populated via __init__ or manual setting
        if self.vae is None or self.audio_vae is None:
            logger.warning(
                "VAE or Audio VAE not found in DenoisingStage. Skipping unpack and denormalize."
            )
            batch.latents = latents
            batch.audio_latents = audio_latents
        else:
            latents, audio_latents = (
                server_args.pipeline_config._unpad_and_unpack_latents(
                    latents, audio_latents, batch, self.vae, self.audio_vae
                )
            )

            batch.latents = latents
            batch.audio_latents = audio_latents

        # 4. Cleanup
        offload_mgr = getattr(self.transformer, "_layerwise_offload_manager", None)
        if offload_mgr is not None and getattr(offload_mgr, "enabled", False):
            offload_mgr.release_all()

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify denoising stage inputs.

        Note: LTX-2 connector stage converts `prompt_embeds`/`negative_prompt_embeds`
        from list-of-tensors to a single tensor (video context) and stores audio
        context separately.
        """

        result = VerificationResult()
        result.add_check("timesteps", batch.timesteps, [V.is_tensor, V.min_dims(1)])

        # LTX-2 may carry prompt embeddings as either a tensor (preferred) or legacy list.
        result.add_check(
            "prompt_embeds",
            batch.prompt_embeds,
            lambda x: V.is_tensor(x) or V.list_not_empty(x),
        )

        # Keep base expectation: image_embeds is always a list (may be empty).
        result.add_check("image_embeds", batch.image_embeds, V.is_list)

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

        # When CFG is enabled, negative prompt embeddings must exist (tensor or legacy list).
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x),
        )
        return result

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return batch.guidance_scale > 1.0


class LTX2RefinementStage(LTX2AVDenoisingStage):
    def __init__(
        self, transformer, scheduler, distilled_sigmas, vae=None, audio_vae=None
    ):
        super().__init__(transformer, scheduler, vae, audio_vae)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        # 1. Add noise to latents
        noise_scale = self.distilled_sigmas[0].to(batch.latents.device)
        noise = torch.randn_like(batch.latents)
        batch.latents = batch.latents + noise * noise_scale

        # 2. Run denoising loop with distilled_sigmas
        # Save original sigmas
        original_sigmas = self.scheduler.sigmas
        original_timesteps = self.scheduler.timesteps
        original_num_inference_steps = self.scheduler.num_inference_steps

        # Set distilled sigmas
        self.scheduler.sigmas = self.distilled_sigmas.to(self.scheduler.sigmas.device)
        # Approximation for timesteps
        self.scheduler.timesteps = self.scheduler.sigmas * 1000
        self.scheduler.num_inference_steps = len(self.distilled_sigmas) - 1

        # Call parent forward
        try:
            batch = super().forward(batch, server_args)
        finally:
            # Restore original sigmas
            self.scheduler.sigmas = original_sigmas
            self.scheduler.timesteps = original_timesteps
            self.scheduler.num_inference_steps = original_num_inference_steps

        return batch

    def do_classifier_free_guidance(self, batch: Req) -> bool:
        return False  # Stage 2 uses simple denoising (no CFG)
