import copy
import math
import os
import time
from io import BytesIO
from pathlib import Path

import av
import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.torch_utils import randn_tensor

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
from sglang.multimodal_gen.runtime.utils.layerwise_offload import OffloadableDiTMixin
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
        return int(
            (batch.num_frames - 1)
            // int(pc.vae_config.arch_config.temporal_compression_ratio)
            + 1
        )

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

    def _get_ltx2_stage1_guider_params(
        self, batch: Req, server_args: ServerArgs, stage: str
    ) -> dict[str, object] | None:
        if stage != "stage1":
            return None
        return batch.extra.get("ltx2_stage1_guider_params")

    @staticmethod
    def _ltx2_should_skip_step(step_index: int, skip_step: int) -> bool:
        if skip_step == 0:
            return False
        return step_index % (skip_step + 1) != 0

    @staticmethod
    def _ltx2_apply_rescale(
        cond: torch.Tensor, pred: torch.Tensor, rescale_scale: float
    ) -> torch.Tensor:
        if rescale_scale == 0.0:
            return pred
        factor = cond.std() / pred.std()
        factor = rescale_scale * factor + (1.0 - rescale_scale)
        return pred * factor

    @staticmethod
    def _ltx2_step_dump_dir(step_index: int) -> Path | None:
        dump_dir = os.environ.get("SGLANG_DIFFUSION_LTX2_STEP_DUMP_DIR")
        if not dump_dir:
            return None
        target_step = int(os.environ.get("SGLANG_DIFFUSION_LTX2_STEP_DUMP_INDEX", "-1"))
        if step_index != target_step:
            return None
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _ltx2_dump_step_tensors(
        dump_dir: Path,
        *,
        step_index: int,
        stage: str,
        sigma: float,
        latents_before: torch.Tensor,
        denoised_video: torch.Tensor,
        latents_after: torch.Tensor,
        audio_latents_before: torch.Tensor | None,
        denoised_audio: torch.Tensor | None,
        audio_latents_after: torch.Tensor | None,
        video_cond: torch.Tensor | None = None,
        video_uncond: torch.Tensor | None = None,
        video_ptb: torch.Tensor | None = None,
        video_mod: torch.Tensor | None = None,
        audio_cond: torch.Tensor | None = None,
        audio_uncond: torch.Tensor | None = None,
        audio_ptb: torch.Tensor | None = None,
        audio_mod: torch.Tensor | None = None,
        video_clean_latent: torch.Tensor | None = None,
        video_denoise_mask: torch.Tensor | None = None,
        image_latent: torch.Tensor | None = None,
    ) -> None:
        torch.save(
            {
                "stage": stage,
                "step_index": step_index,
                "sigma": sigma,
                "video_latent_before": latents_before.detach().cpu(),
                "video_denoised": denoised_video.detach().cpu(),
                "video_latent_after": latents_after.detach().cpu(),
                "audio_latent_before": (
                    None
                    if audio_latents_before is None
                    else audio_latents_before.detach().cpu()
                ),
                "audio_denoised": (
                    None if denoised_audio is None else denoised_audio.detach().cpu()
                ),
                "audio_latent_after": (
                    None
                    if audio_latents_after is None
                    else audio_latents_after.detach().cpu()
                ),
                "video_cond": None if video_cond is None else video_cond.detach().cpu(),
                "video_uncond": (
                    None if video_uncond is None else video_uncond.detach().cpu()
                ),
                "video_ptb": None if video_ptb is None else video_ptb.detach().cpu(),
                "video_mod": None if video_mod is None else video_mod.detach().cpu(),
                "audio_cond": None if audio_cond is None else audio_cond.detach().cpu(),
                "audio_uncond": (
                    None if audio_uncond is None else audio_uncond.detach().cpu()
                ),
                "audio_ptb": None if audio_ptb is None else audio_ptb.detach().cpu(),
                "audio_mod": None if audio_mod is None else audio_mod.detach().cpu(),
                "video_clean_latent": (
                    None
                    if video_clean_latent is None
                    else video_clean_latent.detach().cpu()
                ),
                "video_denoise_mask": (
                    None
                    if video_denoise_mask is None
                    else video_denoise_mask.detach().cpu()
                ),
                "image_latent": (
                    None if image_latent is None else image_latent.detach().cpu()
                ),
            },
            dump_dir / f"{stage}_step{step_index}.pt",
        )

    @classmethod
    def _ltx2_calculate_guided_x0(
        cls,
        *,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
        cfg_scale: float,
        stg_scale: float,
        rescale_scale: float,
        modality_scale: float,
    ) -> torch.Tensor:
        pred = (
            cond
            + (cfg_scale - 1.0) * (cond - uncond_text)
            + stg_scale * (cond - uncond_perturbed)
            + (modality_scale - 1.0) * (cond - uncond_modality)
        )
        return cls._ltx2_apply_rescale(cond, pred, rescale_scale)

    @staticmethod
    def _resize_center_crop(
        img: PIL.Image.Image, *, width: int, height: int
    ) -> PIL.Image.Image:
        return img.resize((width, height), resample=PIL.Image.Resampling.BILINEAR)

    @staticmethod
    def _apply_video_codec_compression(
        img_array: np.ndarray, crf: int = 33
    ) -> np.ndarray:
        """Encode as a single H.264 frame and decode back to simulate compression artifacts."""
        if crf == 0:
            return img_array
        height, width = img_array.shape[0] // 2 * 2, img_array.shape[1] // 2 * 2
        img_array = img_array[:height, :width]
        buffer = BytesIO()
        container = av.open(buffer, mode="w", format="mp4")
        stream = container.add_stream(
            "libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"}
        )
        stream.height, stream.width = height, width
        frame = av.VideoFrame.from_ndarray(img_array, format="rgb24").reformat(
            format="yuv420p"
        )
        container.mux(stream.encode(frame))
        container.mux(stream.encode())
        container.close()
        buffer.seek(0)
        container = av.open(buffer)
        decoded = next(container.decode(container.streams.video[0]))
        container.close()
        return decoded.to_ndarray(format="rgb24")

    @staticmethod
    def _resize_center_crop_tensor(
        img: PIL.Image.Image,
        *,
        width: int,
        height: int,
        device: torch.device,
        dtype: torch.dtype,
        apply_codec_compression: bool = True,
        codec_crf: int = 33,
    ) -> torch.Tensor:
        """Resize, center-crop, and normalize to [1, C, 1, H, W] tensor in [-1, 1]."""
        img_array = np.array(img).astype(np.uint8)[..., :3]
        if apply_codec_compression:
            img_array = LTX2AVDenoisingStage._apply_video_codec_compression(
                img_array, crf=codec_crf
            )
        tensor = (
            torch.from_numpy(img_array.astype(np.float32))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=device)
        )
        src_h, src_w = tensor.shape[2], tensor.shape[3]
        scale = max(height / src_h, width / src_w)
        new_h, new_w = math.ceil(src_h * scale), math.ceil(src_w * scale)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        top, left = (new_h - height) // 2, (new_w - width) // 2
        tensor = tensor[:, :, top : top + height, left : left + width]
        return ((tensor / 127.5 - 1.0).to(dtype=dtype)).unsqueeze(2)

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
        img_array = np.array(img).astype(np.uint8)[..., :3]
        img_array = self._apply_video_codec_compression(img_array, crf=33)
        conditioned_img = PIL.Image.fromarray(img_array)
        batch.condition_image = self._resize_center_crop(
            conditioned_img, width=int(batch.width), height=int(batch.height)
        )

        latents_device = (
            batch.latents.device
            if isinstance(batch.latents, torch.Tensor)
            else torch.device("cpu")
        )
        encode_dtype = batch.latents.dtype
        original_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.vae = self.vae.to(device=latents_device, dtype=encode_dtype)
        vae_autocast_enabled = (
            original_dtype != torch.float32
        ) and not server_args.disable_autocast

        video_condition = self._resize_center_crop_tensor(
            conditioned_img,
            width=int(batch.width),
            height=int(batch.height),
            device=latents_device,
            dtype=encode_dtype,
            apply_codec_compression=False,
        )

        with torch.autocast(
            device_type=current_platform.device_type,
            dtype=original_dtype,
            enabled=vae_autocast_enabled,
        ):
            try:
                if server_args.pipeline_config.vae_tiling:
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(encode_dtype)

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

        # Per-channel normalization: normalized = (x - mean) / std
        mean = self.vae.latents_mean.view(1, -1, 1, 1, 1).to(latent)
        std = self.vae.latents_std.view(1, -1, 1, 1, 1).to(latent)
        latent = (latent - mean) / std

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

        self.vae.to(original_dtype)
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
        target_dtype = prepared_vars["target_dtype"]
        autocast_enabled = prepared_vars["autocast_enabled"]
        timesteps = prepared_vars["timesteps"]
        num_inference_steps = prepared_vars["num_inference_steps"]
        num_warmup_steps = prepared_vars["num_warmup_steps"]
        latents = prepared_vars["latents"]
        boundary_timestep = prepared_vars["boundary_timestep"]
        z = prepared_vars["z"]
        reserved_frames_mask = prepared_vars["reserved_frames_mask"]
        stage = batch.extra.get("ltx2_phase", "stage1")
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
        latent_height = (
            batch.height
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        latent_width = (
            batch.width
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )

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
                        metrics=batch.metrics,
                        perf_dump_path_provided=batch.perf_dump_path is not None,
                        record_as_step=True,
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
                        step_dump_dir = self._ltx2_step_dump_dir(i)
                        latents_before_dump = (
                            latents if step_dump_dir is not None else None
                        )
                        audio_latents_before_dump = (
                            audio_latents if step_dump_dir is not None else None
                        )

                        latent_model_input = latents.to(target_dtype)
                        audio_latent_model_input = audio_latents.to(target_dtype)
                        stage1_guider_params = self._get_ltx2_stage1_guider_params(
                            batch, server_args, stage
                        )
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

                        use_official_cfg_path = stage1_guider_params is None
                        if use_official_cfg_path:
                            encoder_hidden_states = batch.prompt_embeds[0]
                            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                            encoder_attention_mask = batch.prompt_attention_mask
                            if batch.do_classifier_free_guidance:
                                latent_model_input = torch.cat(
                                    [latent_model_input] * 2, dim=0
                                )
                                audio_latent_model_input = torch.cat(
                                    [audio_latent_model_input] * 2, dim=0
                                )
                                encoder_hidden_states = torch.cat(
                                    [
                                        batch.negative_prompt_embeds[0],
                                        encoder_hidden_states,
                                    ],
                                    dim=0,
                                )
                                audio_encoder_hidden_states = torch.cat(
                                    [
                                        batch.negative_audio_prompt_embeds[0],
                                        audio_encoder_hidden_states,
                                    ],
                                    dim=0,
                                )
                                encoder_attention_mask = torch.cat(
                                    [
                                        batch.negative_attention_mask,
                                        encoder_attention_mask,
                                    ],
                                    dim=0,
                                )
                                timestep_video = timestep_video.expand(
                                    int(latent_model_input.shape[0])
                                )
                                timestep_audio = timestep_audio.expand(
                                    int(latent_model_input.shape[0])
                                )

                            with set_forward_context(
                                current_timestep=i, attn_metadata=attn_metadata
                            ):
                                model_video, model_audio = current_model(
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

                            model_video = model_video.float()
                            model_audio = model_audio.float()
                            if batch.do_classifier_free_guidance:
                                model_video_uncond, model_video_text = (
                                    model_video.chunk(2)
                                )
                                model_audio_uncond, model_audio_text = (
                                    model_audio.chunk(2)
                                )
                                model_video = model_video_uncond + (
                                    batch.guidance_scale
                                    * (model_video_text - model_video_uncond)
                                )
                                model_audio = model_audio_uncond + (
                                    batch.guidance_scale
                                    * (model_audio_text - model_audio_uncond)
                                )
                            v_pos = model_video
                            a_v_pos = model_audio
                            v_neg = None
                            a_v_neg = None

                            latents = self.scheduler.step(
                                v_pos, t_device, latents, return_dict=False
                            )[0]
                            audio_latents = audio_scheduler.step(
                                a_v_pos, t_device, audio_latents, return_dict=False
                            )[0]
                            latents = self.post_forward_for_ti2v_task(
                                batch, server_args, reserved_frames_mask, latents, z
                            )

                            if batch.return_trajectory_latents:
                                trajectory_timesteps.append(t_host)
                                trajectory_latents.append(latents)
                                if audio_latents is not None:
                                    trajectory_audio_latents.append(audio_latents)

                            if i == num_timesteps - 1 or (
                                (i + 1) > num_warmup_steps
                                and (i + 1) % self.scheduler.order == 0
                                and progress_bar is not None
                            ):
                                progress_bar.update()

                            if not is_warmup:
                                self.step_profile()
                            continue
                        else:
                            # Follow ltx-pipelines structure: separate pos/neg forward passes,
                            # then apply CFG on denoised (x0) predictions.
                            encoder_hidden_states = batch.prompt_embeds[0]
                            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
                            encoder_attention_mask = batch.prompt_attention_mask
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

                                if (
                                    stage1_guider_params is not None
                                    or batch.do_classifier_free_guidance
                                ):
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
                        denoised_video = (latents.float() - sigma_val * v_pos).to(
                            latents.dtype
                        )
                        denoised_audio = (
                            audio_latents.float() - sigma_val * a_v_pos
                        ).to(audio_latents.dtype)
                        denoised_video_cond = denoised_video
                        denoised_audio_cond = denoised_audio
                        denoised_video_neg = None
                        denoised_audio_neg = None
                        denoised_video_perturbed = None
                        denoised_audio_perturbed = None
                        denoised_video_modality = None
                        denoised_audio_modality = None

                        if (
                            (
                                stage1_guider_params is not None
                                or batch.do_classifier_free_guidance
                            )
                            and v_neg is not None
                            and a_v_neg is not None
                        ):
                            denoised_video_neg = (
                                latents.float() - sigma_val * v_neg
                            ).to(latents.dtype)
                            denoised_audio_neg = (
                                audio_latents.float() - sigma_val * a_v_neg
                            ).to(audio_latents.dtype)
                        if stage1_guider_params is not None:
                            video_skip = self._ltx2_should_skip_step(
                                i, int(stage1_guider_params["video_skip_step"])
                            )
                            audio_skip = self._ltx2_should_skip_step(
                                i, int(stage1_guider_params["audio_skip_step"])
                            )

                            need_perturbed = (
                                float(stage1_guider_params["video_stg_scale"]) != 0.0
                                or float(stage1_guider_params["audio_stg_scale"]) != 0.0
                            )
                            if need_perturbed:
                                with set_forward_context(
                                    current_timestep=i, attn_metadata=attn_metadata
                                ):
                                    v_ptb, a_v_ptb = current_model(
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
                                        skip_video_self_attn_blocks=tuple(
                                            stage1_guider_params["video_stg_blocks"]
                                        ),
                                        skip_audio_self_attn_blocks=tuple(
                                            stage1_guider_params["audio_stg_blocks"]
                                        ),
                                    )
                                denoised_video_perturbed = (
                                    latents.float() - sigma_val * v_ptb.float()
                                ).to(latents.dtype)
                                denoised_audio_perturbed = (
                                    audio_latents.float() - sigma_val * a_v_ptb.float()
                                ).to(audio_latents.dtype)

                            need_modality = (
                                float(stage1_guider_params["video_modality_scale"])
                                != 1.0
                                or float(stage1_guider_params["audio_modality_scale"])
                                != 1.0
                            )
                            if need_modality:
                                with set_forward_context(
                                    current_timestep=i, attn_metadata=attn_metadata
                                ):
                                    v_mod, a_v_mod = current_model(
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
                                        disable_a2v_cross_attn=True,
                                        disable_v2a_cross_attn=True,
                                    )
                                denoised_video_modality = (
                                    latents.float() - sigma_val * v_mod.float()
                                ).to(latents.dtype)
                                denoised_audio_modality = (
                                    audio_latents.float() - sigma_val * a_v_mod.float()
                                ).to(audio_latents.dtype)

                            if not video_skip:
                                denoised_video = self._ltx2_calculate_guided_x0(
                                    cond=denoised_video,
                                    uncond_text=(
                                        denoised_video_neg
                                        if denoised_video_neg is not None
                                        else denoised_video
                                    ),
                                    uncond_perturbed=(
                                        denoised_video_perturbed
                                        if denoised_video_perturbed is not None
                                        else 0.0
                                    ),
                                    uncond_modality=(
                                        denoised_video_modality
                                        if denoised_video_modality is not None
                                        else 0.0
                                    ),
                                    cfg_scale=float(
                                        stage1_guider_params["video_cfg_scale"]
                                    ),
                                    stg_scale=float(
                                        stage1_guider_params["video_stg_scale"]
                                    ),
                                    rescale_scale=float(
                                        stage1_guider_params["video_rescale_scale"]
                                    ),
                                    modality_scale=float(
                                        stage1_guider_params["video_modality_scale"]
                                    ),
                                )
                            if not audio_skip:
                                denoised_audio = self._ltx2_calculate_guided_x0(
                                    cond=denoised_audio,
                                    uncond_text=(
                                        denoised_audio_neg
                                        if denoised_audio_neg is not None
                                        else denoised_audio
                                    ),
                                    uncond_perturbed=(
                                        denoised_audio_perturbed
                                        if denoised_audio_perturbed is not None
                                        else 0.0
                                    ),
                                    uncond_modality=(
                                        denoised_audio_modality
                                        if denoised_audio_modality is not None
                                        else 0.0
                                    ),
                                    cfg_scale=float(
                                        stage1_guider_params["audio_cfg_scale"]
                                    ),
                                    stg_scale=float(
                                        stage1_guider_params["audio_stg_scale"]
                                    ),
                                    rescale_scale=float(
                                        stage1_guider_params["audio_rescale_scale"]
                                    ),
                                    modality_scale=float(
                                        stage1_guider_params["audio_modality_scale"]
                                    ),
                                )
                        elif (
                            batch.do_classifier_free_guidance
                            and denoised_video_neg is not None
                            and denoised_audio_neg is not None
                        ):
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
                            v_video = (
                                (latents.float() - denoised_video.float()) / sigma_val
                            ).to(latents.dtype)
                            v_audio = (
                                (audio_latents.float() - denoised_audio.float())
                                / sigma_val
                            ).to(audio_latents.dtype)

                        latents = (latents.float() + v_video.float() * dt).to(
                            dtype=latents.dtype
                        )
                        audio_latents = (
                            audio_latents.float() + v_audio.float() * dt
                        ).to(dtype=audio_latents.dtype)
                        if step_dump_dir is not None:
                            self._ltx2_dump_step_tensors(
                                step_dump_dir,
                                step_index=i,
                                stage=stage,
                                sigma=sigma_val,
                                latents_before=latents_before_dump,
                                denoised_video=denoised_video,
                                latents_after=latents,
                                audio_latents_before=audio_latents_before_dump,
                                denoised_audio=denoised_audio,
                                audio_latents_after=audio_latents,
                                video_cond=denoised_video_cond,
                                video_uncond=denoised_video_neg,
                                video_ptb=denoised_video_perturbed,
                                video_mod=denoised_video_modality,
                                audio_cond=denoised_audio_cond,
                                audio_uncond=denoised_audio_neg,
                                audio_ptb=denoised_audio_perturbed,
                                audio_mod=denoised_audio_modality,
                                video_clean_latent=clean_latent,
                                video_denoise_mask=denoise_mask,
                                image_latent=batch.image_latent,
                            )

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
        *args,
        **kwargs,
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
        # TODO: make this a general denoising-stage hook
        if isinstance(self.transformer, OffloadableDiTMixin):
            for manager in self.transformer.layerwise_offload_managers:
                manager.release_all()

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


class LTX2RefinementStage(LTX2AVDenoisingStage):
    def __init__(
        self, transformer, scheduler, distilled_sigmas, vae=None, audio_vae=None
    ):
        super().__init__(transformer, scheduler, vae, audio_vae)
        self.distilled_sigmas = torch.tensor(distilled_sigmas)

    @staticmethod
    def _randn_like_with_batch_generators(
        reference_tensor: torch.Tensor, batch: Req
    ) -> torch.Tensor:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list):
            bsz = int(reference_tensor.shape[0])
            valid_generators = [g for g in generator if isinstance(g, torch.Generator)]
            if len(valid_generators) == 1:
                generator = valid_generators[0]
            elif len(valid_generators) >= bsz:
                generator = valid_generators[:bsz]
            else:
                generator = None
        elif not isinstance(generator, torch.Generator):
            generator = None

        return randn_tensor(
            reference_tensor.shape,
            generator=generator,
            device=reference_tensor.device,
            dtype=reference_tensor.dtype,
        )

    @staticmethod
    def _reset_stage2_generators(batch: Req) -> None:
        generator = getattr(batch, "generator", None)
        if isinstance(generator, list) and generator:
            generator_device = str(generator[0].device)
        elif isinstance(generator, torch.Generator):
            generator_device = str(generator.device)
        else:
            generator_device = "cpu"

        seeds = getattr(batch, "seeds", None)
        if not seeds:
            seed = getattr(batch, "seed", None)
            if seed is None:
                return
            seeds = [int(seed)]

        batch.generator = [
            torch.Generator(device=generator_device).manual_seed(int(seed))
            for seed in seeds
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.extra["ltx2_phase"] = "stage2"
        self._reset_stage2_generators(batch)
        noise_scale = self.distilled_sigmas[0].to(batch.latents.device)
        video_noise = self._randn_like_with_batch_generators(batch.latents, batch)
        batch.latents = video_noise * noise_scale + batch.latents * (1 - noise_scale)

        if isinstance(batch.audio_latents, torch.Tensor):
            audio_noise = self._randn_like_with_batch_generators(
                batch.audio_latents, batch
            )
            audio_noise_scale = noise_scale.to(
                batch.audio_latents.device, batch.audio_latents.dtype
            )
            batch.audio_latents = (
                audio_noise * audio_noise_scale
                + batch.audio_latents * (1 - audio_noise_scale)
            )
        batch.latents = batch.latents.to(
            device=batch.latents.device, dtype=torch.float32
        )
        if isinstance(batch.audio_latents, torch.Tensor):
            batch.audio_latents = batch.audio_latents.to(
                device=batch.audio_latents.device, dtype=torch.float32
            )

        # Stage 2 runs at full resolution, so Stage 1 TI2V conditioning is invalid.
        batch.image_latent = None
        batch.ltx2_num_image_tokens = 0

        # Use a private scheduler copy to avoid mutating shared state.
        original_scheduler = self.scheduler
        original_batch_timesteps = batch.timesteps
        original_batch_num_inference_steps = batch.num_inference_steps

        self.scheduler = copy.deepcopy(original_scheduler)
        distilled_device = self.scheduler.sigmas.device
        self.scheduler.sigmas = self.distilled_sigmas.to(distilled_device)
        num_steps = len(self.distilled_sigmas) - 1
        self.scheduler.num_inference_steps = num_steps
        self.scheduler.timesteps = (self.distilled_sigmas[:num_steps] * 1000).to(
            distilled_device
        )
        self.scheduler._step_index = None
        self.scheduler._begin_index = None

        batch.timesteps = self.scheduler.timesteps
        batch.num_inference_steps = num_steps
        original_do_cfg = batch.do_classifier_free_guidance
        batch.do_classifier_free_guidance = False

        try:
            batch = super().forward(batch, server_args)
        finally:
            self.scheduler = original_scheduler
            batch.timesteps = original_batch_timesteps
            batch.num_inference_steps = original_batch_num_inference_steps
            batch.do_classifier_free_guidance = original_do_cfg

        return batch
