import copy
import json
import math
import os
from dataclasses import dataclass, field
from io import BytesIO

import av
import numpy as np
import PIL.Image
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.runtime.distributed import get_sp_world_size
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_3_condition_encoder import (
    LTX23VideoConditionEncoder,
)
from sglang.multimodal_gen.runtime.models.vision_utils import (
    load_image,
    normalize,
    numpy_to_pt,
    pil_to_numpy,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingContext,
    DenoisingStage,
    DenoisingStepState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


@dataclass(slots=True)
class LTX2DenoisingContext(DenoisingContext):
    """Loop-scoped denoising state for joint LTX-2 video and audio generation."""

    audio_latents: torch.Tensor | None = None
    audio_scheduler: object | None = None
    is_ltx23_variant: bool = False
    use_ltx23_legacy_one_stage: bool = False
    replicate_audio_for_sp: bool = False
    stage: str = "one_stage"
    latent_num_frames_for_model: int = 0
    latent_height: int = 0
    latent_width: int = 0
    denoise_mask: torch.Tensor | None = None
    clean_latent: torch.Tensor | None = None
    last_denoised_video: torch.Tensor | None = None
    last_denoised_audio: torch.Tensor | None = None
    trajectory_audio_latents: list[torch.Tensor] = field(default_factory=list)


class LTX2DenoisingStage(DenoisingStage):
    """
    LTX-2 specific denoising stage that handles joint video and audio generation.
    """

    def __init__(self, transformer, scheduler, vae=None, **kwargs):
        super().__init__(
            transformer=transformer, scheduler=scheduler, vae=vae, **kwargs
        )
        self._condition_image_encoder = None
        self._condition_image_encoder_dir = None

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
    def _prepare_ltx2_ti2v_clean_state(
        latents: torch.Tensor,
        image_latent: torch.Tensor,
        num_img_tokens: int,
        zero_clean_latent: bool,
        clean_latent_background: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = latents.clone()
        conditioned = image_latent[:, :num_img_tokens, :].to(
            device=latents.device, dtype=latents.dtype
        )
        latents[:, :num_img_tokens, :] = conditioned
        denoise_mask = torch.ones(
            (latents.shape[0], latents.shape[1], 1),
            device=latents.device,
            dtype=torch.float32,
        )
        denoise_mask[:, :num_img_tokens, :] = 0.0
        if clean_latent_background is not None:
            clean_latent = (
                clean_latent_background.detach()
                .clone()
                .to(device=latents.device, dtype=latents.dtype)
            )
        elif zero_clean_latent:
            clean_latent = torch.zeros_like(latents)
        else:
            clean_latent = latents.detach().clone()
        clean_latent[:, :num_img_tokens, :] = conditioned
        return latents, denoise_mask, clean_latent

    @staticmethod
    def _ltx2_velocity_to_x0(
        sample: torch.Tensor,
        velocity: torch.Tensor,
        sigma: float | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.to(device=sample.device, dtype=torch.float32)
            while sigma.ndim < sample.ndim:
                sigma = sigma.unsqueeze(-1)
            return (sample.float() - sigma * velocity.float()).to(sample.dtype)
        return (sample.float() - float(sigma) * velocity.float()).to(sample.dtype)

    @staticmethod
    def _repeat_batch_dim(tensor: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Repeat along batch dim while preserving any tokenwise timestep layout."""
        if tensor.shape[0] == int(target_batch_size):
            return tensor
        if tensor.shape[0] <= 0 or int(target_batch_size) % int(tensor.shape[0]) != 0:
            raise ValueError(
                f"Cannot repeat tensor with batch={tensor.shape[0]} to target_batch_size={target_batch_size}"
            )
        repeat_factor = int(target_batch_size) // int(tensor.shape[0])
        return tensor.repeat(repeat_factor, *([1] * (tensor.ndim - 1)))

    @staticmethod
    def _build_ltx2_sp_padding_mask(
        batch: Req,
        *,
        seq_len: int,
        batch_size: int,
        key: str,
        device: torch.device,
    ) -> torch.Tensor | None:
        valid = getattr(batch, key, None)
        if valid is None:
            return None
        valid = int(valid)
        if valid <= 0 or valid >= int(seq_len):
            return None
        mask = torch.ones(
            (batch_size, int(seq_len)), device=device, dtype=torch.float32
        )
        mask[:, valid:] = 0.0
        return mask

    @staticmethod
    def _get_ltx_prompt_attention_mask(
        batch: Req,
        *,
        is_ltx23_variant: bool,
        negative: bool = False,
    ) -> torch.Tensor | None:
        if is_ltx23_variant:
            return None
        return (
            batch.negative_attention_mask if negative else batch.prompt_attention_mask
        )

    @classmethod
    def _should_use_ltx23_legacy_one_stage(
        cls,
        server_args: ServerArgs,
        pipeline_name: str | None,
    ) -> bool:
        if not is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        ):
            return False
        if server_args.pipeline_class_name == "LTX2TwoStagePipeline":
            return False
        return pipeline_name != "LTX2TwoStagePipeline"

    @classmethod
    def _should_shard_ltx23_legacy_one_stage_audio_latents(
        cls,
        batch: Req,
        server_args: ServerArgs,
    ) -> bool:
        return bool(
            get_sp_world_size() > 1
            and is_ltx23_native_variant(
                server_args.pipeline_config.vae_config.arch_config
            )
            and cls._should_use_ltx23_legacy_one_stage(server_args, None)
            and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                batch.audio_latents
            )
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
            img_array = LTX2DenoisingStage._apply_video_codec_compression(
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

    @staticmethod
    def _should_replicate_ltx23_audio_for_sp(
        batch: Req,
        server_args: ServerArgs,
        *,
        is_ltx23_variant: bool,
    ) -> bool:
        return False

    def _get_condition_image_encoder(
        self,
        server_args: ServerArgs,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> LTX23VideoConditionEncoder | None:
        arch_config = server_args.pipeline_config.vae_config.arch_config
        encoder_subdir = str(getattr(arch_config, "condition_encoder_subdir", ""))
        if not encoder_subdir:
            return None

        vae_model_path = server_args.model_paths["vae"]
        encoder_dir = os.path.join(vae_model_path, encoder_subdir)
        config_path = os.path.join(encoder_dir, "config.json")
        weights_path = os.path.join(encoder_dir, "model.safetensors")
        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise ValueError(
                f"LTX-2 condition encoder files not found under {encoder_dir}"
            )

        cached_dir = self._condition_image_encoder_dir
        encoder = self._condition_image_encoder
        if encoder is None or cached_dir != encoder_dir:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            encoder = LTX23VideoConditionEncoder(config)
            encoder.load_state_dict(safetensors_load_file(weights_path), strict=True)
            self._condition_image_encoder = encoder
            self._condition_image_encoder_dir = encoder_dir

        encoder = encoder.to(device=device, dtype=dtype)
        return encoder

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
        vae_autocast_enabled = (
            original_dtype != torch.float32
        ) and not server_args.disable_autocast
        condition_image_encoder = self._get_condition_image_encoder(
            server_args, device=latents_device, dtype=encode_dtype
        )
        if condition_image_encoder is None:
            self.vae = self.vae.to(device=latents_device, dtype=encode_dtype)

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
                if (
                    condition_image_encoder is None
                    and server_args.pipeline_config.vae_tiling
                ):
                    self.vae.enable_tiling()
            except Exception:
                pass
            if not vae_autocast_enabled:
                video_condition = video_condition.to(encode_dtype)

            if condition_image_encoder is not None:
                latent = condition_image_encoder(video_condition)
            else:
                latent_dist: DiagonalGaussianDistribution = self.vae.encode(
                    video_condition
                )
                if isinstance(latent_dist, AutoencoderKLOutput):
                    latent_dist = latent_dist.latent_dist

        if condition_image_encoder is None:
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
        else:
            latent = latent.to(dtype=encode_dtype)

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

        if condition_image_encoder is None:
            self.vae.to(original_dtype)
        if server_args.vae_cpu_offload:
            self.vae = self.vae.to("cpu")
            if condition_image_encoder is not None:
                self._condition_image_encoder = condition_image_encoder.to("cpu")

    def _prepare_denoising_loop(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> LTX2DenoisingContext:
        """Extend the base context with LTX-2 audio, SP, and TI2V state."""
        self._disable_cache_dit_for_request = batch.image_path is not None
        base_ctx = super()._prepare_denoising_loop(batch, server_args)
        ctx = LTX2DenoisingContext(**base_ctx.to_kwargs())
        ctx.is_ltx23_variant = is_ltx23_native_variant(
            server_args.pipeline_config.vae_config.arch_config
        )
        phase = batch.extra.get("ltx2_phase")
        pipeline = self.pipeline() if self.pipeline else None
        pipeline_name = pipeline.pipeline_name if pipeline is not None else None
        ctx.use_ltx23_legacy_one_stage = self._should_use_ltx23_legacy_one_stage(
            server_args, pipeline_name
        )
        ctx.stage = (
            phase
            if phase is not None
            else ("stage1" if ctx.use_ltx23_legacy_one_stage else "one_stage")
        )
        ctx.audio_latents = batch.audio_latents
        # Video and audio keep separate scheduler state throughout the denoising loop.
        ctx.audio_scheduler = copy.deepcopy(self.scheduler)

        # Prepare image latents and embeddings for LTX-2 TI2V generation.
        self._prepare_ltx2_image_latent(batch, server_args)
        do_ti2v = self._should_apply_ltx2_ti2v(batch)

        if ctx.use_ltx23_legacy_one_stage:
            batch.ltx23_audio_replicated_for_sp = False
            batch.did_sp_shard_audio_latents = False
        else:
            ctx.replicate_audio_for_sp = self._should_replicate_ltx23_audio_for_sp(
                batch,
                server_args,
                is_ltx23_variant=ctx.is_ltx23_variant,
            )
            batch.ltx23_audio_replicated_for_sp = bool(ctx.replicate_audio_for_sp)
            if (
                ctx.is_ltx23_variant
                and get_sp_world_size() > 1
                and server_args.pipeline_config.can_shard_audio_latents_for_sp(
                    batch.audio_latents
                )
                and not ctx.replicate_audio_for_sp
            ):
                (
                    batch.audio_latents,
                    batch.did_sp_shard_audio_latents,
                ) = server_args.pipeline_config.shard_audio_latents_for_sp(
                    batch, batch.audio_latents
                )
                ctx.audio_latents = batch.audio_latents
            else:
                batch.did_sp_shard_audio_latents = False

        # For LTX-2 packed token latents, SP sharding happens on the time dimension
        # (frames). The model must see local latent frames (RoPE offset is applied
        # inside the model using SP rank).
        ctx.latent_num_frames_for_model = self._get_video_latent_num_frames_for_model(
            batch=batch, server_args=server_args, latents=ctx.latents
        )
        ctx.latent_height = (
            batch.height
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        ctx.latent_width = (
            batch.width
            // server_args.pipeline_config.vae_config.arch_config.spatial_compression_ratio
        )
        if do_ti2v:
            if not (isinstance(ctx.latents, torch.Tensor) and ctx.latents.ndim == 3):
                raise ValueError("LTX-2 TI2V expects packed token latents [B, S, D].")
            clean_latent_background = getattr(
                batch, "ltx2_ti2v_clean_latent_background", None
            )
            if not (
                isinstance(clean_latent_background, torch.Tensor)
                and clean_latent_background.shape == ctx.latents.shape
            ):
                clean_latent_background = None
            # Keep conditioned tokens clean and reuse the mask during every step update.
            ctx.latents, ctx.denoise_mask, ctx.clean_latent = (
                self._prepare_ltx2_ti2v_clean_state(
                    latents=ctx.latents,
                    image_latent=batch.image_latent,
                    num_img_tokens=int(getattr(batch, "ltx2_num_image_tokens", 0)),
                    zero_clean_latent=ctx.is_ltx23_variant,
                    clean_latent_background=clean_latent_background,
                )
            )
        return ctx

    def _before_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Reset the mirrored audio scheduler before the shared loop begins."""
        super()._before_denoising_loop(ctx, batch, server_args)
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")
        ctx.audio_scheduler.set_begin_index(0)

    def _prepare_step_attn_metadata(
        self,
        ctx: LTX2DenoisingContext,
        batch: Req,
        server_args: ServerArgs,
        step_index: int,
        t_int: int,
        timesteps_cpu: torch.Tensor,
    ):
        """Preserve the legacy LTX-2 attention-metadata contract."""
        # Legacy LTX-2 paths used the plain attention-metadata builder call here.
        del ctx, t_int, timesteps_cpu
        return self._build_attn_metadata(step_index, batch, server_args)

    def _run_denoising_step(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Run one joint video/audio denoising step with LTX-2-specific guidance."""
        if ctx.audio_latents is None:
            raise ValueError("LTX-2 requires audio latents for denoising.")
        if ctx.audio_scheduler is None:
            raise ValueError("LTX-2 audio scheduler was not prepared.")

        # 1. Read the scheduler sigma pair and derive the Euler delta.
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is None or not isinstance(sigmas, torch.Tensor):
            raise ValueError("Expected scheduler.sigmas to be a tensor for LTX-2.")
        sigma = sigmas[step.step_index].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        sigma_next = sigmas[step.step_index + 1].to(
            device=ctx.latents.device, dtype=torch.float32
        )
        dt = sigma_next - sigma

        # 2. Materialize the current video/audio latent inputs in the compute dtype.
        latent_model_input = ctx.latents.to(ctx.target_dtype)
        audio_latent_model_input = ctx.audio_latents.to(ctx.target_dtype)
        stage1_guider_params = self._get_ltx2_stage1_guider_params(
            batch, server_args, ctx.stage
        )

        if audio_latent_model_input.ndim == 3:
            audio_num_frames_latent = int(audio_latent_model_input.shape[1])
        elif audio_latent_model_input.ndim == 4:
            audio_num_frames_latent = int(audio_latent_model_input.shape[2])
        else:
            raise ValueError(
                f"Unexpected audio latents rank: {audio_latent_model_input.ndim}, shape={tuple(audio_latent_model_input.shape)}"
            )

        # 3. Prepare any LTX-specific RoPE coordinates and timestep layouts.
        video_coords = None
        audio_coords = None
        if not ctx.use_ltx23_legacy_one_stage:
            video_coords = server_args.pipeline_config.prepare_video_rope_coords_for_sp(
                step.current_model,
                batch,
                latent_model_input,
                num_frames=ctx.latent_num_frames_for_model,
                height=ctx.latent_height,
                width=ctx.latent_width,
            )
            audio_coords = server_args.pipeline_config.prepare_audio_rope_coords_for_sp(
                step.current_model,
                batch,
                audio_latent_model_input,
                num_frames=audio_num_frames_latent,
            )

        batch_size = int(latent_model_input.shape[0])
        timestep = step.t_device.expand(batch_size)
        if ctx.denoise_mask is not None:
            timestep_video = timestep.unsqueeze(-1) * ctx.denoise_mask.squeeze(-1)
        elif ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
            timestep_video = timestep.view(batch_size, 1).expand(
                batch_size, int(latent_model_input.shape[1])
            )
        else:
            timestep_video = timestep

        if (
            ctx.is_ltx23_variant
            and not ctx.use_ltx23_legacy_one_stage
            and audio_latent_model_input.ndim == 3
        ):
            timestep_audio = timestep.view(batch_size, 1).expand(
                batch_size, int(audio_latent_model_input.shape[1])
            )
        else:
            timestep_audio = timestep

        prompt_timestep_video = None
        prompt_timestep_audio = None
        if ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage:
            timestep_scale_multiplier = float(
                getattr(step.current_model, "timestep_scale_multiplier", 1000)
            )
            prompt_timestep_video = (
                sigma.to(device=latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)
            prompt_timestep_audio = (
                sigma.to(device=audio_latent_model_input.device, dtype=torch.float32)
                * timestep_scale_multiplier
            ).expand(batch_size)

        # 4. Build attention masks that account for SP padding and replicated audio.
        if ctx.use_ltx23_legacy_one_stage:
            video_self_attention_mask = None
            audio_self_attention_mask = None
            a2v_cross_attention_mask = None
            v2a_cross_attention_mask = None
        else:
            video_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=int(latent_model_input.shape[1]),
                batch_size=batch_size,
                key="sp_video_valid_token_count",
                device=latent_model_input.device,
            )
            audio_self_attention_mask = self._build_ltx2_sp_padding_mask(
                batch,
                seq_len=audio_num_frames_latent,
                batch_size=batch_size,
                key="sp_audio_valid_token_count",
                device=audio_latent_model_input.device,
            )
            a2v_cross_attention_mask = audio_self_attention_mask
            v2a_cross_attention_mask = video_self_attention_mask

        def build_model_kwargs(
            *,
            encoder_hidden_states: torch.Tensor,
            audio_encoder_hidden_states: torch.Tensor,
            encoder_attention_mask: torch.Tensor | None,
            skip_video_self_attn_blocks: tuple[int, ...] | None = None,
            skip_audio_self_attn_blocks: tuple[int, ...] | None = None,
            disable_a2v_cross_attn: bool = False,
            disable_v2a_cross_attn: bool = False,
        ) -> dict[str, object]:
            kwargs: dict[str, object] = {
                "hidden_states": latent_model_input,
                "audio_hidden_states": audio_latent_model_input,
                "encoder_hidden_states": encoder_hidden_states,
                "audio_encoder_hidden_states": audio_encoder_hidden_states,
                "timestep": timestep_video,
                "audio_timestep": timestep_audio,
                "encoder_attention_mask": encoder_attention_mask,
                "audio_encoder_attention_mask": encoder_attention_mask,
                "num_frames": ctx.latent_num_frames_for_model,
                "height": ctx.latent_height,
                "width": ctx.latent_width,
                "fps": batch.fps,
                "audio_num_frames": audio_num_frames_latent,
                "video_coords": video_coords,
                "audio_coords": audio_coords,
                "return_latents": False,
                "return_dict": False,
            }
            if not ctx.use_ltx23_legacy_one_stage:
                kwargs.update(
                    {
                        "prompt_timestep": prompt_timestep_video,
                        "audio_prompt_timestep": prompt_timestep_audio,
                        "video_self_attention_mask": video_self_attention_mask,
                        "audio_self_attention_mask": audio_self_attention_mask,
                        "a2v_cross_attention_mask": a2v_cross_attention_mask,
                        "v2a_cross_attention_mask": v2a_cross_attention_mask,
                        "audio_replicated_for_sp": ctx.replicate_audio_for_sp,
                        "legacy_ltx23_one_stage_semantics": False,
                    }
                )
            if skip_video_self_attn_blocks is not None:
                kwargs["skip_video_self_attn_blocks"] = skip_video_self_attn_blocks
            if skip_audio_self_attn_blocks is not None:
                kwargs["skip_audio_self_attn_blocks"] = skip_audio_self_attn_blocks
            if disable_a2v_cross_attn:
                kwargs["disable_a2v_cross_attn"] = True
            if disable_v2a_cross_attn:
                kwargs["disable_v2a_cross_attn"] = True
            return kwargs

        # 5. Run the branch-specific LTX forward path and apply CFG/guider logic.
        prompt_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
        )
        use_official_cfg_path = stage1_guider_params is None
        if use_official_cfg_path:
            encoder_hidden_states = batch.prompt_embeds[0]
            audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
            encoder_attention_mask = prompt_attention_mask
            if batch.do_classifier_free_guidance:
                latent_model_input = torch.cat([latent_model_input] * 2, dim=0)
                audio_latent_model_input = torch.cat(
                    [audio_latent_model_input] * 2, dim=0
                )
                encoder_hidden_states = torch.cat(
                    [batch.negative_prompt_embeds[0], encoder_hidden_states], dim=0
                )
                audio_encoder_hidden_states = torch.cat(
                    [
                        batch.negative_audio_prompt_embeds[0],
                        audio_encoder_hidden_states,
                    ],
                    dim=0,
                )
                if encoder_attention_mask is not None:
                    encoder_attention_mask = torch.cat(
                        [
                            self._get_ltx_prompt_attention_mask(
                                batch,
                                is_ltx23_variant=(
                                    ctx.is_ltx23_variant
                                    and not ctx.use_ltx23_legacy_one_stage
                                ),
                                negative=True,
                            ),
                            encoder_attention_mask,
                        ],
                        dim=0,
                    )
                cfg_batch_size = int(latent_model_input.shape[0])
                timestep_video = self._repeat_batch_dim(timestep_video, cfg_batch_size)
                timestep_audio = self._repeat_batch_dim(timestep_audio, cfg_batch_size)
                if prompt_timestep_video is not None:
                    prompt_timestep_video = self._repeat_batch_dim(
                        prompt_timestep_video, cfg_batch_size
                    )
                if prompt_timestep_audio is not None:
                    prompt_timestep_audio = self._repeat_batch_dim(
                        prompt_timestep_audio, cfg_batch_size
                    )
                if video_self_attention_mask is not None:
                    video_self_attention_mask = self._repeat_batch_dim(
                        video_self_attention_mask, cfg_batch_size
                    )
                if audio_self_attention_mask is not None:
                    audio_self_attention_mask = self._repeat_batch_dim(
                        audio_self_attention_mask, cfg_batch_size
                    )
                if a2v_cross_attention_mask is not None:
                    a2v_cross_attention_mask = self._repeat_batch_dim(
                        a2v_cross_attention_mask, cfg_batch_size
                    )
                if v2a_cross_attention_mask is not None:
                    v2a_cross_attention_mask = self._repeat_batch_dim(
                        v2a_cross_attention_mask, cfg_batch_size
                    )

            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                model_video, model_audio = step.current_model(
                    **build_model_kwargs(
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                )

            model_video = model_video.float()
            model_audio = model_audio.float()
            if batch.do_classifier_free_guidance:
                model_video_uncond, model_video_text = model_video.chunk(2)
                model_audio_uncond, model_audio_text = model_audio.chunk(2)
                model_video = model_video_uncond + (
                    batch.guidance_scale * (model_video_text - model_video_uncond)
                )
                model_audio = model_audio_uncond + (
                    batch.guidance_scale * (model_audio_text - model_audio_uncond)
                )

            ctx.latents = self.scheduler.step(
                model_video, step.t_device, ctx.latents, return_dict=False
            )[0]
            ctx.audio_latents = ctx.audio_scheduler.step(
                model_audio, step.t_device, ctx.audio_latents, return_dict=False
            )[0]
            if ctx.denoise_mask is not None and ctx.clean_latent is not None:
                ctx.latents = (
                    ctx.latents.float() * ctx.denoise_mask
                    + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
                ).to(dtype=ctx.latents.dtype)
            ctx.latents = self.post_forward_for_ti2v_task(
                batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
            )
            return

        encoder_hidden_states = batch.prompt_embeds[0]
        audio_encoder_hidden_states = batch.audio_prompt_embeds[0]
        encoder_attention_mask = prompt_attention_mask
        negative_encoder_hidden_states = batch.negative_prompt_embeds[0]
        negative_audio_encoder_hidden_states = batch.negative_audio_prompt_embeds[0]
        negative_encoder_attention_mask = self._get_ltx_prompt_attention_mask(
            batch,
            is_ltx23_variant=(
                ctx.is_ltx23_variant and not ctx.use_ltx23_legacy_one_stage
            ),
            negative=True,
        )

        video_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["video_skip_step"])
        )
        audio_skip = self._ltx2_should_skip_step(
            step.step_index, int(stage1_guider_params["audio_skip_step"])
        )
        need_perturbed = (
            float(stage1_guider_params["video_stg_scale"]) != 0.0
            or float(stage1_guider_params["audio_stg_scale"]) != 0.0
        )
        need_modality = (
            float(stage1_guider_params["video_modality_scale"]) != 1.0
            or float(stage1_guider_params["audio_modality_scale"]) != 1.0
        )

        if ctx.use_ltx23_legacy_one_stage:
            with set_forward_context(
                current_timestep=step.step_index, attn_metadata=step.attn_metadata
            ):
                v_pos, a_v_pos = step.current_model(
                    **build_model_kwargs(
                        encoder_hidden_states=encoder_hidden_states,
                        audio_encoder_hidden_states=audio_encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                    )
                )
                v_neg, a_v_neg = step.current_model(
                    **build_model_kwargs(
                        encoder_hidden_states=negative_encoder_hidden_states,
                        audio_encoder_hidden_states=negative_audio_encoder_hidden_states,
                        encoder_attention_mask=negative_encoder_attention_mask,
                    )
                )

            v_pos = v_pos.float()
            a_v_pos = a_v_pos.float()
            v_neg = v_neg.float()
            a_v_neg = a_v_neg.float()

            v_ptb = None
            a_v_ptb = None
            if need_perturbed:
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    v_ptb, a_v_ptb = step.current_model(
                        **build_model_kwargs(
                            encoder_hidden_states=encoder_hidden_states,
                            audio_encoder_hidden_states=audio_encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            skip_video_self_attn_blocks=tuple(
                                stage1_guider_params["video_stg_blocks"]
                            ),
                            skip_audio_self_attn_blocks=tuple(
                                stage1_guider_params["audio_stg_blocks"]
                            ),
                        )
                    )
                v_ptb = v_ptb.float()
                a_v_ptb = a_v_ptb.float()

            v_mod = None
            a_v_mod = None
            if need_modality:
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    v_mod, a_v_mod = step.current_model(
                        **build_model_kwargs(
                            encoder_hidden_states=encoder_hidden_states,
                            audio_encoder_hidden_states=audio_encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask,
                            disable_a2v_cross_attn=True,
                            disable_v2a_cross_attn=True,
                        )
                    )
                v_mod = v_mod.float()
                a_v_mod = a_v_mod.float()
        else:
            # NOTE: this flag must be identical across all SP ranks so that
            # every rank executes the same number of model-forward calls (each
            # of which contains NCCL collectives).
            # _should_apply_ltx2_ti2v() is SP-rank-dependent (only the rank owning the first latent
            # frame returns True), so we must NOT use it here.
            # Instead we check the rank-invariant attribute that is always set on every
            # rank when the request is a TI2V request.
            use_split_two_stage_ti2v_guider = (
                server_args.pipeline_class_name == "LTX2TwoStagePipeline"
                and int(getattr(batch, "ltx2_num_image_tokens", 0)) > 0
            )

            def cat_or_none(items: list[torch.Tensor | None]) -> torch.Tensor | None:
                if items[0] is None:
                    return None
                return torch.cat(items, dim=0)

            pass_specs: list[
                tuple[
                    str,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor | None,
                    dict[str, object],
                ]
            ] = [
                (
                    "cond",
                    encoder_hidden_states,
                    audio_encoder_hidden_states,
                    encoder_attention_mask,
                    {
                        "skip_video_self_attn_blocks": (),
                        "skip_audio_self_attn_blocks": (),
                        "skip_a2v_cross_attn": False,
                        "skip_v2a_cross_attn": False,
                    },
                ),
                (
                    "neg",
                    negative_encoder_hidden_states,
                    negative_audio_encoder_hidden_states,
                    negative_encoder_attention_mask,
                    {
                        "skip_video_self_attn_blocks": (),
                        "skip_audio_self_attn_blocks": (),
                        "skip_a2v_cross_attn": False,
                        "skip_v2a_cross_attn": False,
                    },
                ),
            ]
            if need_perturbed:
                pass_specs.append(
                    (
                        "perturbed",
                        encoder_hidden_states,
                        audio_encoder_hidden_states,
                        encoder_attention_mask,
                        {
                            "skip_video_self_attn_blocks": tuple(
                                stage1_guider_params["video_stg_blocks"]
                            ),
                            "skip_audio_self_attn_blocks": tuple(
                                stage1_guider_params["audio_stg_blocks"]
                            ),
                            "skip_a2v_cross_attn": False,
                            "skip_v2a_cross_attn": False,
                        },
                    )
                )
            if need_modality:
                pass_specs.append(
                    (
                        "modality",
                        encoder_hidden_states,
                        audio_encoder_hidden_states,
                        encoder_attention_mask,
                        {
                            "skip_video_self_attn_blocks": (),
                            "skip_audio_self_attn_blocks": (),
                            "skip_a2v_cross_attn": True,
                            "skip_v2a_cross_attn": True,
                        },
                    )
                )

            num_passes = len(pass_specs)
            expanded_batch_size = batch_size * num_passes
            perturbation_configs = tuple(
                perturbation_config
                for _, _, _, _, perturbation_config in pass_specs
                for _ in range(batch_size)
            )
            batched_hidden_states = self._repeat_batch_dim(
                latent_model_input, expanded_batch_size
            )
            batched_audio_hidden_states = self._repeat_batch_dim(
                audio_latent_model_input, expanded_batch_size
            )
            batched_encoder_hidden_states = torch.cat(
                [item[1] for item in pass_specs], dim=0
            )
            batched_audio_encoder_hidden_states = torch.cat(
                [item[2] for item in pass_specs], dim=0
            )
            batched_timestep_video = self._repeat_batch_dim(
                timestep_video, expanded_batch_size
            )
            batched_timestep_audio = self._repeat_batch_dim(
                timestep_audio, expanded_batch_size
            )
            batched_prompt_timestep_video = (
                None
                if prompt_timestep_video is None
                else self._repeat_batch_dim(prompt_timestep_video, expanded_batch_size)
            )
            batched_prompt_timestep_audio = (
                None
                if prompt_timestep_audio is None
                else self._repeat_batch_dim(prompt_timestep_audio, expanded_batch_size)
            )
            batched_encoder_attention_mask = cat_or_none(
                [item[3] for item in pass_specs]
            )
            batched_audio_encoder_attention_mask = cat_or_none(
                [item[3] for item in pass_specs]
            )
            batched_video_coords = (
                None
                if video_coords is None
                else self._repeat_batch_dim(video_coords, expanded_batch_size)
            )
            batched_audio_coords = (
                None
                if audio_coords is None
                else self._repeat_batch_dim(audio_coords, expanded_batch_size)
            )
            batched_video_self_attention_mask = (
                None
                if video_self_attention_mask is None
                else self._repeat_batch_dim(
                    video_self_attention_mask, expanded_batch_size
                )
            )
            batched_audio_self_attention_mask = (
                None
                if audio_self_attention_mask is None
                else self._repeat_batch_dim(
                    audio_self_attention_mask, expanded_batch_size
                )
            )
            batched_a2v_cross_attention_mask = (
                None
                if a2v_cross_attention_mask is None
                else self._repeat_batch_dim(
                    a2v_cross_attention_mask, expanded_batch_size
                )
            )
            batched_v2a_cross_attention_mask = (
                None
                if v2a_cross_attention_mask is None
                else self._repeat_batch_dim(
                    v2a_cross_attention_mask, expanded_batch_size
                )
            )
            if use_split_two_stage_ti2v_guider:
                split_sizes = [1] * expanded_batch_size

                def split_or_none(
                    tensor: torch.Tensor | None,
                ) -> list[torch.Tensor | None]:
                    if tensor is None:
                        return [None] * len(split_sizes)
                    return list(tensor.split(split_sizes, dim=0))

                batched_video_chunks = []
                batched_audio_chunks = []
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    for (
                        hidden_states_chunk,
                        audio_hidden_states_chunk,
                        encoder_hidden_states_chunk,
                        audio_encoder_hidden_states_chunk,
                        timestep_video_chunk,
                        timestep_audio_chunk,
                        prompt_timestep_video_chunk,
                        prompt_timestep_audio_chunk,
                        encoder_attention_mask_chunk,
                        audio_encoder_attention_mask_chunk,
                        video_coords_chunk,
                        audio_coords_chunk,
                        video_self_attention_mask_chunk,
                        audio_self_attention_mask_chunk,
                        a2v_cross_attention_mask_chunk,
                        v2a_cross_attention_mask_chunk,
                        perturbation_config_chunk,
                    ) in zip(
                        batched_hidden_states.split(split_sizes, dim=0),
                        batched_audio_hidden_states.split(split_sizes, dim=0),
                        batched_encoder_hidden_states.split(split_sizes, dim=0),
                        batched_audio_encoder_hidden_states.split(split_sizes, dim=0),
                        batched_timestep_video.split(split_sizes, dim=0),
                        batched_timestep_audio.split(split_sizes, dim=0),
                        split_or_none(batched_prompt_timestep_video),
                        split_or_none(batched_prompt_timestep_audio),
                        split_or_none(batched_encoder_attention_mask),
                        split_or_none(batched_audio_encoder_attention_mask),
                        split_or_none(batched_video_coords),
                        split_or_none(batched_audio_coords),
                        split_or_none(batched_video_self_attention_mask),
                        split_or_none(batched_audio_self_attention_mask),
                        split_or_none(batched_a2v_cross_attention_mask),
                        split_or_none(batched_v2a_cross_attention_mask),
                        ((cfg,) for cfg in perturbation_configs),
                        strict=True,
                    ):
                        video_chunk, audio_chunk = step.current_model(
                            hidden_states=hidden_states_chunk,
                            audio_hidden_states=audio_hidden_states_chunk,
                            encoder_hidden_states=encoder_hidden_states_chunk,
                            audio_encoder_hidden_states=audio_encoder_hidden_states_chunk,
                            timestep=timestep_video_chunk,
                            audio_timestep=timestep_audio_chunk,
                            prompt_timestep=prompt_timestep_video_chunk,
                            audio_prompt_timestep=prompt_timestep_audio_chunk,
                            encoder_attention_mask=encoder_attention_mask_chunk,
                            audio_encoder_attention_mask=audio_encoder_attention_mask_chunk,
                            num_frames=ctx.latent_num_frames_for_model,
                            height=ctx.latent_height,
                            width=ctx.latent_width,
                            fps=batch.fps,
                            audio_num_frames=audio_num_frames_latent,
                            video_coords=video_coords_chunk,
                            audio_coords=audio_coords_chunk,
                            video_self_attention_mask=video_self_attention_mask_chunk,
                            audio_self_attention_mask=audio_self_attention_mask_chunk,
                            a2v_cross_attention_mask=a2v_cross_attention_mask_chunk,
                            v2a_cross_attention_mask=v2a_cross_attention_mask_chunk,
                            audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                            perturbation_configs=perturbation_config_chunk,
                            return_latents=False,
                            return_dict=False,
                        )
                        batched_video_chunks.append(video_chunk)
                        batched_audio_chunks.append(audio_chunk)

                batched_video = torch.cat(batched_video_chunks, dim=0)
                batched_audio = torch.cat(batched_audio_chunks, dim=0)
            else:
                with set_forward_context(
                    current_timestep=step.step_index, attn_metadata=step.attn_metadata
                ):
                    batched_video, batched_audio = step.current_model(
                        hidden_states=batched_hidden_states,
                        audio_hidden_states=batched_audio_hidden_states,
                        encoder_hidden_states=batched_encoder_hidden_states,
                        audio_encoder_hidden_states=batched_audio_encoder_hidden_states,
                        timestep=batched_timestep_video,
                        audio_timestep=batched_timestep_audio,
                        prompt_timestep=batched_prompt_timestep_video,
                        audio_prompt_timestep=batched_prompt_timestep_audio,
                        encoder_attention_mask=batched_encoder_attention_mask,
                        audio_encoder_attention_mask=batched_audio_encoder_attention_mask,
                        num_frames=ctx.latent_num_frames_for_model,
                        height=ctx.latent_height,
                        width=ctx.latent_width,
                        fps=batch.fps,
                        audio_num_frames=audio_num_frames_latent,
                        video_coords=batched_video_coords,
                        audio_coords=batched_audio_coords,
                        video_self_attention_mask=batched_video_self_attention_mask,
                        audio_self_attention_mask=batched_audio_self_attention_mask,
                        a2v_cross_attention_mask=batched_a2v_cross_attention_mask,
                        v2a_cross_attention_mask=batched_v2a_cross_attention_mask,
                        audio_replicated_for_sp=ctx.replicate_audio_for_sp,
                        perturbation_configs=perturbation_configs,
                        return_latents=False,
                        return_dict=False,
                    )

            batched_video = batched_video.float()
            batched_audio = batched_audio.float()
            pass_outputs = {
                pass_name: (
                    video_chunk,
                    audio_chunk,
                )
                for (pass_name, _, _, _, _), video_chunk, audio_chunk in zip(
                    pass_specs,
                    batched_video.chunk(num_passes, dim=0),
                    batched_audio.chunk(num_passes, dim=0),
                    strict=True,
                )
            }
            v_pos, a_v_pos = pass_outputs["cond"]
            v_neg, a_v_neg = pass_outputs["neg"]
            v_ptb, a_v_ptb = pass_outputs.get("perturbed", (None, None))
            v_mod, a_v_mod = pass_outputs.get("modality", (None, None))

        sigma_val = float(sigma.item())
        video_sigma_for_x0: float | torch.Tensor = sigma_val
        if ctx.denoise_mask is not None:
            video_sigma_for_x0 = sigma.to(
                device=ctx.latents.device, dtype=torch.float32
            ) * ctx.denoise_mask.squeeze(-1)

        denoised_video = self._ltx2_velocity_to_x0(
            ctx.latents, v_pos, video_sigma_for_x0
        )
        denoised_audio = self._ltx2_velocity_to_x0(
            ctx.audio_latents, a_v_pos, sigma_val
        )
        denoised_video_neg = self._ltx2_velocity_to_x0(
            ctx.latents, v_neg, video_sigma_for_x0
        )
        denoised_audio_neg = self._ltx2_velocity_to_x0(
            ctx.audio_latents, a_v_neg, sigma_val
        )
        denoised_video_perturbed = (
            None
            if v_ptb is None
            else self._ltx2_velocity_to_x0(ctx.latents, v_ptb, video_sigma_for_x0)
        )
        denoised_audio_perturbed = (
            None
            if a_v_ptb is None
            else self._ltx2_velocity_to_x0(ctx.audio_latents, a_v_ptb, sigma_val)
        )
        denoised_video_modality = (
            None
            if v_mod is None
            else self._ltx2_velocity_to_x0(ctx.latents, v_mod, video_sigma_for_x0)
        )
        denoised_audio_modality = (
            None
            if a_v_mod is None
            else self._ltx2_velocity_to_x0(ctx.audio_latents, a_v_mod, sigma_val)
        )

        if not video_skip:
            denoised_video = self._ltx2_calculate_guided_x0(
                cond=denoised_video,
                uncond_text=denoised_video_neg,
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
                cfg_scale=float(stage1_guider_params["video_cfg_scale"]),
                stg_scale=float(stage1_guider_params["video_stg_scale"]),
                rescale_scale=float(stage1_guider_params["video_rescale_scale"]),
                modality_scale=float(stage1_guider_params["video_modality_scale"]),
            )
            ctx.last_denoised_video = denoised_video
        elif ctx.last_denoised_video is not None:
            denoised_video = ctx.last_denoised_video

        if not audio_skip:
            denoised_audio = self._ltx2_calculate_guided_x0(
                cond=denoised_audio,
                uncond_text=denoised_audio_neg,
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
                cfg_scale=float(stage1_guider_params["audio_cfg_scale"]),
                stg_scale=float(stage1_guider_params["audio_stg_scale"]),
                rescale_scale=float(stage1_guider_params["audio_rescale_scale"]),
                modality_scale=float(stage1_guider_params["audio_modality_scale"]),
            )
            ctx.last_denoised_audio = denoised_audio
        elif ctx.last_denoised_audio is not None:
            denoised_audio = ctx.last_denoised_audio

        if ctx.denoise_mask is not None and ctx.clean_latent is not None:
            denoised_video = (
                denoised_video * ctx.denoise_mask
                + ctx.clean_latent.float() * (1.0 - ctx.denoise_mask)
            ).to(denoised_video.dtype)

        # 6. Convert x0 predictions back to velocity and update both latent streams.
        if sigma_val == 0.0:
            v_video = torch.zeros_like(denoised_video)
            v_audio = torch.zeros_like(denoised_audio)
        else:
            v_video = ((ctx.latents.float() - denoised_video.float()) / sigma_val).to(
                ctx.latents.dtype
            )
            v_audio = (
                (ctx.audio_latents.float() - denoised_audio.float()) / sigma_val
            ).to(ctx.audio_latents.dtype)

        ctx.latents = (ctx.latents.float() + v_video.float() * dt).to(
            dtype=ctx.latents.dtype
        )
        ctx.audio_latents = (ctx.audio_latents.float() + v_audio.float() * dt).to(
            dtype=ctx.audio_latents.dtype
        )
        ctx.latents = self.post_forward_for_ti2v_task(
            batch, server_args, ctx.reserved_frames_mask, ctx.latents, ctx.z
        )

    def _record_trajectory(
        self,
        ctx: LTX2DenoisingContext,
        step: DenoisingStepState,
        batch: Req,
        server_args: ServerArgs,
    ) -> None:
        """Record audio trajectory alongside the base video trajectory."""
        super()._record_trajectory(ctx, step, batch, server_args)
        if batch.return_trajectory_latents and ctx.audio_latents is not None:
            ctx.trajectory_audio_latents.append(ctx.audio_latents)

    def _finalize_denoising_loop(
        self, ctx: LTX2DenoisingContext, batch: Req, server_args: ServerArgs
    ) -> None:
        """Expose audio latents before delegating to AV-aware postprocessing."""
        batch.audio_latents = ctx.audio_latents
        self._post_denoising_loop(
            batch=batch,
            latents=ctx.latents,
            trajectory_latents=ctx.trajectory_latents,
            trajectory_timesteps=ctx.trajectory_timesteps,
            trajectory_audio_latents=ctx.trajectory_audio_latents,
            server_args=server_args,
            is_warmup=ctx.is_warmup,
        )

    def _post_denoising_loop(
        self,
        batch: Req,
        latents: torch.Tensor,
        trajectory_latents: list,
        trajectory_timesteps: list,
        server_args: ServerArgs,
        trajectory_audio_latents: list | None = None,
        is_warmup: bool = False,
        *args,
        **kwargs,
    ):
        """Trim SP token padding before delegating to the base finalizer."""
        if trajectory_audio_latents:
            batch.trajectory_audio_latents = torch.stack(
                trajectory_audio_latents, dim=1
            ).cpu()
        latents = self._truncate_sp_padded_token_latents(batch, latents)
        super()._post_denoising_loop(
            batch=batch,
            latents=latents,
            trajectory_latents=trajectory_latents,
            trajectory_timesteps=trajectory_timesteps,
            server_args=server_args,
            is_warmup=is_warmup,
        )

    def _get_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list prompt embeddings for LTX-2 prompts."""
        del batch
        return lambda x: V.is_tensor(x) or V.list_not_empty(x)

    def _get_negative_prompt_embeds_validator(self, batch: Req):
        """Allow either tensor or list negative prompt embeddings for LTX-2 CFG."""
        return (
            lambda x: (not batch.do_classifier_free_guidance)
            or V.is_tensor(x)
            or V.list_not_empty(x)
        )
