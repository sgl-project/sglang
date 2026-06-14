# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline stages: image preprocess, tokenization, latent / timestep
prep, denoising, decode.

Cosmos3 has no separate text encoder — text is tokenized with Qwen2's chat
template and embedded inside the transformer's UND pathway. The same
``Cosmos3Pipeline`` serves T2V, I2V, V2V, and T2I; mode is dispatched
per-request from ``batch.data_type`` and the presence of
``batch.preprocessed_image`` / ``batch.preprocessed_video``.
"""

import copy
import json
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    cfg_model_parallel_all_reduce,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.vision_utils import load_video
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
    ACTION_MODES,
    EMBODIMENT_TO_DOMAIN_ID,
    build_action_prompt,
    denormalize_action,
    get_raw_action_dim,
    load_action_stats,
    normalize_action,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.srt.utils.common import get_compiler_backend

logger = init_logger(__name__)

COSMOS3_DEFAULT_NEGATIVE_PROMPT = ""
COSMOS3_DURATION_TEMPLATE = (
    "The video is {duration:.1f} seconds long and is of {fps} FPS."
)
COSMOS3_VIDEO_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate videos from a given prompt."
)
COSMOS3_IMAGE_SYSTEM_PROMPT = (
    "You are a helpful assistant who will generate images from a given prompt."
)

# Per-mode flow-shift defaults, applied only when the request and pipeline
# config leave flow_shift unset. T2V/I2V fall back to the checkpoint scheduler
# config instead of a hard-coded value.
COSMOS3_T2I_FLOW_SHIFT = 3.0
COSMOS3_V2V_FLOW_SHIFT = 10.0
COSMOS3_ACTION_FLOW_SHIFT = 5.0


def _resize_crop_pil(
    image: PIL.Image.Image, target_w: int, target_h: int
) -> PIL.Image.Image:
    """Aspect-preserving resize then center-crop to ``target_w x target_h``."""
    scale = max(target_w / image.width, target_h / image.height)
    resize_w = int(np.ceil(scale * image.width))
    resize_h = int(np.ceil(scale * image.height))
    image = image.resize((resize_w, resize_h), PIL.Image.Resampling.LANCZOS)
    left = (resize_w - target_w) // 2
    top = (resize_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def _pil_to_normalized_tensor(image: PIL.Image.Image) -> torch.Tensor:
    """PIL RGB → ``[3, H, W]`` float32 tensor in ``[-1, 1]``."""
    arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class Cosmos3ImagePreprocessStage(PipelineStage):
    """Load, aspect-resize, and center-crop the conditioning input.

    For I2V: writes ``[1, 3, H, W]`` to ``batch.preprocessed_image``.
    For V2V: writes ``[1, 3, T_in, H, W]`` to ``batch.preprocessed_video``.
    No-op for T2V / T2I.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image_path = batch.image_path
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None
        video_path = batch.video_path
        if isinstance(video_path, list):
            video_path = video_path[0] if video_path else None

        if image_path and video_path:
            raise ValueError(
                "Cosmos3 accepts either --image-path (I2V) or --video-path "
                "(V2V), not both"
            )

        target_h, target_w = batch.height, batch.width

        if isinstance(image_path, str) and image_path:
            image = PIL.Image.open(image_path).convert("RGB")
            image = _resize_crop_pil(image, target_w, target_h)
            batch.preprocessed_image = _pil_to_normalized_tensor(image).unsqueeze(0)
            self.log_info(f"Preprocessed conditioning image to {target_w}x{target_h}")
            return batch

        if isinstance(video_path, str) and video_path:
            frames = load_video(video_path)
            if not frames:
                raise ValueError(f"No frames decoded from video: {video_path!r}")

            keep = (
                getattr(batch.sampling_params, "condition_video_keep", "first")
                or "first"
            )
            if keep not in ("first", "last"):
                raise ValueError(
                    f"condition_video_keep must be 'first' or 'last', got {keep!r}"
                )
            cond_indexes = self._resolve_condition_indexes(batch)
            # Encode the full output-length video so that the latent positions
            # we lock match what the decoder will reconstruct at those frame
            # indices. Encoding only the first ``max_idx*4+1`` frames produces
            # an out-of-distribution latent for the locked slots and decodes
            # to noise.
            num_source_frames = max(cond_indexes) * 4 + 1
            num_target_frames = batch.num_frames
            if keep == "last":
                frames = frames[-num_source_frames:]
            else:
                frames = frames[:num_source_frames]
            if len(frames) < num_source_frames:
                frames = frames + [frames[-1]] * (num_source_frames - len(frames))
            if len(frames) < num_target_frames:
                frames = frames + [frames[-1]] * (num_target_frames - len(frames))

            processed = [
                _pil_to_normalized_tensor(
                    _resize_crop_pil(f.convert("RGB"), target_w, target_h)
                )
                for f in frames
            ]
            video_tensor = torch.stack(processed, dim=1).unsqueeze(0).contiguous()
            batch.preprocessed_video = video_tensor
            self.log_info(
                f"Preprocessed conditioning video to "
                f"{video_tensor.shape[2]}x{target_h}x{target_w} "
                f"(keep={keep}, source frames={num_source_frames}, padded to {num_target_frames})"
            )

        return batch

    @staticmethod
    def _resolve_condition_indexes(batch: Req) -> list[int]:
        """Resolve condition_frame_indexes for V2V (default ``[0, 1]``).

        Inverse-dynamics action mode conditions on the whole input video, so
        every latent frame is locked.
        """
        if (
            getattr(batch.sampling_params, "action_mode", None)
            == ACTION_MODE_INVERSE_DYNAMICS
        ):
            num_latent_frames = (batch.num_frames - 1) // 4 + 1
            return list(range(num_latent_frames))
        cond_indexes = getattr(batch.sampling_params, "condition_frame_indexes", None)
        if not cond_indexes:
            return [0, 1]
        return sorted(set(int(i) for i in cond_indexes))


class Cosmos3TokenizationStage(PipelineStage):
    """Tokenization stage for Cosmos3.

    Applies the Qwen2 chat template, appends a duration suffix, and writes
    ``text_ids`` / ``text_mask`` into ``batch.extra`` for the denoising stage.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, tokenizer):
        super().__init__()
        if tokenizer is None:
            raise ValueError(
                "Cosmos3TokenizationStage requires a tokenizer; expected the "
                "Qwen2 tokenizer loaded from the checkpoint's text_tokenizer/ "
                "subfolder."
            )
        self.tokenizer = tokenizer

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        return result

    def _tokenize_prompt(
        self,
        text: str,
        max_sequence_length: int,
        device: torch.device,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize a prompt using Qwen2 chat template.

        Returns (input_ids, attention_mask, seq_len) as [1, S] tensors.
        """
        conversations = []
        if use_system_prompt:
            conversations.append(
                {
                    "role": "system",
                    "content": system_prompt or COSMOS3_VIDEO_SYSTEM_PROMPT,
                }
            )
        conversations.append({"role": "user", "content": text})

        result = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
        )

        # Handle different return types from apply_chat_template
        # Fast tokenizer returns BatchEncoding, slow tokenizer returns list[int]
        if hasattr(result, "input_ids"):
            # BatchEncoding from fast tokenizer
            token_ids = list(result.input_ids)
        elif isinstance(result, list):
            # Already a list from slow tokenizer
            token_ids = list(result)
        else:
            raise TypeError(
                f"Unexpected return type from apply_chat_template: {type(result)}"
            )

        # Reserve room for the two special tokens (EOS + vision_start) so the
        # final length cannot exceed ``max_sequence_length``.
        token_ids = token_ids[: max_sequence_length - 2]

        # Add EOS and vision_start tokens
        token_ids.append(self.tokenizer.eos_token_id)
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        if vision_start_id is not None:
            token_ids.append(vision_start_id)

        seq_len = len(token_ids)

        # Pad to max_sequence_length
        pad_len = max_sequence_length - seq_len
        attention_mask = [1] * seq_len + [0] * pad_len
        pad_token_id = self.tokenizer.pad_token_id or 0
        token_ids = token_ids + [pad_token_id] * pad_len

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attention_mask], dtype=torch.long, device=device)
        return input_ids, attention_mask, seq_len

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Tokenize prompt and negative prompt."""
        device = get_local_torch_device()
        prompt = batch.prompt
        negative_prompt = batch.negative_prompt or COSMOS3_DEFAULT_NEGATIVE_PROMPT

        # Get parameters
        max_sequence_length = getattr(batch, "max_sequence_length", None) or 512
        use_duration_template = getattr(batch, "use_duration_template", None)
        if use_duration_template is None:
            use_duration_template = getattr(
                server_args.pipeline_config, "use_duration_template", True
            )
        use_system_prompt = getattr(batch, "use_system_prompt", None)
        if use_system_prompt is None:
            use_system_prompt = getattr(
                server_args.pipeline_config, "use_system_prompt", False
            )
        fps = batch.fps or 24.0
        num_frames = batch.num_frames
        is_image_gen = batch.data_type == DataType.IMAGE
        system_prompt = (
            COSMOS3_IMAGE_SYSTEM_PROMPT if is_image_gen else COSMOS3_VIDEO_SYSTEM_PROMPT
        )

        # Action mode uses a structured JSON caption with neither a system
        # prompt nor the duration suffix.
        if getattr(batch.sampling_params, "action_mode", None) is not None:
            prompt = build_action_prompt(
                prompt,
                getattr(batch.sampling_params, "action_view_point", "ego_view"),
                num_frames,
                fps,
                batch.height,
                batch.width,
            )
            use_system_prompt = False
            use_duration_template = False
            self.log_info(f"Action prompt: {prompt}")

        # Apply duration template if enabled (no temporal concept for T2I).
        if use_duration_template and not is_image_gen and num_frames > 1:
            duration = num_frames / fps
            suffix = COSMOS3_DURATION_TEMPLATE.format(duration=duration, fps=fps)
            prompt = f"{prompt} {suffix}"
            self.log_info(f"Prompt with duration: '{prompt}'")

        # Tokenize prompts
        cond_ids, cond_mask, cond_seq_len = self._tokenize_prompt(
            prompt, max_sequence_length, device, use_system_prompt, system_prompt
        )
        uncond_ids, uncond_mask, uncond_seq_len = self._tokenize_prompt(
            negative_prompt,
            max_sequence_length,
            device,
            use_system_prompt,
            system_prompt,
        )
        # official Cosmos3 consumes packed text; keep a shared length for CFG batching
        shared_seq_len = max(cond_seq_len, uncond_seq_len)
        cond_ids = cond_ids[:, :shared_seq_len]
        cond_mask = cond_mask[:, :shared_seq_len]
        uncond_ids = uncond_ids[:, :shared_seq_len]
        uncond_mask = uncond_mask[:, :shared_seq_len]

        # Store in batch.extra for denoising stage
        batch.extra["cond_text_ids"] = cond_ids
        batch.extra["cond_text_mask"] = cond_mask
        batch.extra["uncond_text_ids"] = uncond_ids
        batch.extra["uncond_text_mask"] = uncond_mask
        batch.extra["cond_text_seq_len"] = cond_seq_len
        batch.extra["uncond_text_seq_len"] = uncond_seq_len
        batch.extra["fps"] = fps

        # Mark as processed (even though we don't use standard embeddings)
        batch.is_prompt_processed = True

        return batch


class Cosmos3LatentPreparationStage(PipelineStage):
    """Initialize the noisy latent for Cosmos3.

    T2V / T2I produce pure Gaussian noise. I2V / V2V VAE-encode the
    conditioning input, write the resulting latents at the conditioned
    frame indexes, and stash a per-frame velocity mask plus the full
    condition latent so the denoiser can re-blend after each scheduler step.
    I2V is the special case of conditioning at frame ``[0]`` with the image
    expanded across the temporal axis; V2V conditions at ``[0, 1]`` (or a
    user-supplied list) with frames from the input video.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, vae, transformer):
        super().__init__()
        self.vae = vae
        self.transformer = transformer

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("height", batch.height, V.positive_int_divisible(16))
        result.add_check("width", batch.width, V.positive_int_divisible(16))
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def _vae_encode(self, video: torch.Tensor) -> torch.Tensor:
        """VAE-encode a [B, 3, T, H, W] pixel tensor and normalize the latent.

        WanVAE returns a ``DiagonalGaussianDistribution``; ``mode()`` keeps
        the encoding deterministic for I2V conditioning.
        """
        latent = self.vae.encode(video).mode()
        mean = (
            torch.as_tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        std = (
            torch.as_tensor(self.vae.config.latents_std)
            .view(1, -1, 1, 1, 1)
            .to(latent.device, latent.dtype)
        )
        return (latent - mean) / std

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Prepare initial latents (pure noise for T2V/T2I, image-conditioned for I2V)."""
        device = get_local_torch_device()
        dtype = torch.bfloat16

        vae_scale_factor_temporal = getattr(self.vae.config, "scale_factor_temporal", 4)
        vae_scale_factor_spatial = getattr(self.vae.config, "scale_factor_spatial", 16)

        num_channels_latents = self.transformer.latent_channel
        num_latent_frames = (batch.num_frames - 1) // vae_scale_factor_temporal + 1
        height_latent = batch.height // vae_scale_factor_spatial
        width_latent = batch.width // vae_scale_factor_spatial

        shape = (
            1,
            num_channels_latents,
            num_latent_frames,
            height_latent,
            width_latent,
        )

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)

        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        is_video_gen = batch.data_type == DataType.VIDEO
        has_image_cond = batch.preprocessed_image is not None and is_video_gen
        has_video_cond = batch.preprocessed_video is not None and is_video_gen

        if has_image_cond or has_video_cond:
            vae_dtype = next(self.vae.parameters()).dtype

            if has_video_cond:
                pixel_input = batch.preprocessed_video.to(
                    device=device, dtype=vae_dtype
                )
                cond_indexes = Cosmos3ImagePreprocessStage._resolve_condition_indexes(
                    batch
                )
            else:
                pixel_input = batch.preprocessed_image.unsqueeze(2).to(
                    device=device, dtype=vae_dtype
                )
                cond_indexes = [0]

            with torch.no_grad():
                cond_latent = self._vae_encode(pixel_input).to(dtype)

            max_idx = max(cond_indexes)
            if max_idx >= num_latent_frames:
                raise ValueError(
                    f"condition_frame_indexes={cond_indexes} exceeds the "
                    f"latent frame count {num_latent_frames} for "
                    f"num_frames={batch.num_frames}"
                )

            condition_latents = torch.zeros_like(noise)
            condition_mask = torch.zeros(
                1, 1, num_latent_frames, 1, 1, device=device, dtype=dtype
            )
            for idx in cond_indexes:
                src = min(idx, cond_latent.shape[2] - 1)
                condition_latents[:, :, idx, :, :] = cond_latent[:, :, src, :, :]
                condition_mask[:, :, idx, :, :] = 1.0

            latents = (
                condition_mask * condition_latents + (1.0 - condition_mask) * noise
            )
            batch.extra["condition_latents"] = condition_latents
            batch.extra["velocity_mask"] = 1.0 - condition_mask
            mode = "V2V" if has_video_cond else "I2V"
            self.log_info(
                f"Prepared {mode} latents with conditioning at frames {cond_indexes}"
            )
        else:
            latents = noise

        batch.latents = latents
        batch.raw_latent_shape = shape

        batch.extra["video_shape"] = (num_latent_frames, height_latent, width_latent)
        batch.extra["vae_scale_factor_temporal"] = vae_scale_factor_temporal
        batch.extra["vae_scale_factor_spatial"] = vae_scale_factor_spatial

        self.log_info(f"Prepared latents with shape {shape}")

        sound_duration = float(getattr(batch, "sound_duration", 0.0) or 0.0)
        if sound_duration > 0.0:
            if not getattr(self.transformer, "sound_gen", False):
                raise ValueError(
                    "sound generation was requested (sound_duration > 0) but the "
                    "loaded Cosmos3 checkpoint has no sound modality (sound_gen is "
                    "False)."
                )
            sound_latent_fps = self.transformer.sound_latent_fps
            sound_latent_frames = max(1, round(sound_duration * sound_latent_fps))
            sound_shape = (1, self.transformer.sound_dim, sound_latent_frames)
            batch.audio_latents = torch.randn(
                sound_shape, generator=generator, device=device, dtype=dtype
            )
            self.log_info(f"Prepared sound latents with shape {sound_shape}")

        action_mode = getattr(batch.sampling_params, "action_mode", None)
        if action_mode is not None:
            if getattr(self.transformer, "action_dim", None) is None:
                raise ValueError(
                    "action_mode is set but the loaded Cosmos3 checkpoint has no "
                    "action modality (action_gen is False)."
                )
            self._prepare_action_latents(batch, generator, device, dtype)
        return batch

    @staticmethod
    def _resolve_domain_id(batch: Req) -> int:
        """Resolve action embodiment domain ID; required for action generation."""
        domain_id = getattr(batch.sampling_params, "domain_id", None)
        if domain_id is not None:
            domain_id = int(domain_id)
            if domain_id < 0:
                raise ValueError(f"domain_id must be non-negative, got {domain_id}")
            return domain_id
        domain_name = getattr(batch.sampling_params, "domain_name", None)
        if domain_name:
            key = str(domain_name).strip().lower()
            if key not in EMBODIMENT_TO_DOMAIN_ID:
                raise ValueError(
                    f"Unknown action domain name {domain_name!r}. "
                    f"Valid names: {sorted(EMBODIMENT_TO_DOMAIN_ID)}"
                )
            return EMBODIMENT_TO_DOMAIN_ID[key]
        raise ValueError(
            "Cosmos3 action generation requires --domain-id or --domain-name."
        )

    def _prepare_action_latents(
        self,
        batch: Req,
        generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Prepare action latents and conditioning, writing them onto ``batch``.

        Action tokens run at frame rate (no temporal compression), so the chunk
        length is ``num_frames - 1`` with ``start_frame_offset=1`` so each action
        aligns with the frame it drives.

        Three modes:
        - ``forward_dynamics``: the user supplies the action; all tokens are
          clean conditioning (velocity mask 0) and the model predicts video.
        - ``policy`` / ``inverse_dynamics``: actions are denoised from noise
          (velocity mask 1); ``raw_action_dim`` is required.
        """
        sp = batch.sampling_params
        mode = str(sp.action_mode).strip().lower()
        if mode not in ACTION_MODES:
            raise ValueError(
                f"Unsupported action_mode={sp.action_mode!r}; "
                f"expected one of {sorted(ACTION_MODES)}"
            )
        action_dim = self.transformer.action_dim
        num_frames = batch.num_frames

        action_chunk_size = num_frames - 1 if num_frames > 1 else 1
        action_offset = 1 if action_chunk_size == num_frames - 1 else 0

        domain_id = self._resolve_domain_id(batch)
        raw_action_dim = getattr(sp, "raw_action_dim", None)
        if raw_action_dim is None:
            embodiment = getattr(sp, "domain_name", None)
            if embodiment:
                raw_action_dim = get_raw_action_dim(embodiment)

        if mode == ACTION_MODE_FORWARD_DYNAMICS:
            raw = getattr(sp, "action", None)
            if raw is None:
                raise ValueError(
                    "action_mode='forward_dynamics' requires an 'action' array "
                    "(list[list[float]] of shape [T, D])."
                )
            if isinstance(raw, str):
                raw = json.loads(raw)
            action = torch.as_tensor(np.asarray(raw), dtype=torch.float32)
            if action.ndim == 3 and action.shape[0] == 1:
                action = action.squeeze(0)
            if action.ndim != 2:
                raise ValueError(
                    f"action must have shape [T, D], got {tuple(action.shape)}"
                )
            if action.shape[0] < action_chunk_size:
                pad = action[-1:].repeat(action_chunk_size - action.shape[0], 1)
                action = torch.cat([action, pad], dim=0)
            elif action.shape[0] > action_chunk_size:
                action = action[:action_chunk_size]
            if raw_action_dim is None:
                raw_action_dim = int(action.shape[-1])
            stats_path = getattr(sp, "action_stats_path", None)
            if stats_path is not None:
                method = getattr(sp, "action_normalization", "quantile")
                action = normalize_action(action, method, load_action_stats(stats_path))
            if action.shape[-1] < action_dim:
                pad = torch.zeros(action.shape[0], action_dim - action.shape[-1])
                action = torch.cat([action, pad], dim=-1)
            clean_action = action.to(device=device, dtype=dtype).unsqueeze(0)
        else:
            if raw_action_dim is None:
                raise ValueError(f"action_mode={mode!r} requires --raw-action-dim.")
            clean_action = torch.zeros(
                1, action_chunk_size, action_dim, device=device, dtype=dtype
            )

        raw_action_dim = int(raw_action_dim)
        if not 0 < raw_action_dim <= action_dim:
            raise ValueError(
                f"raw_action_dim must be in [1, {action_dim}], got {raw_action_dim}"
            )

        # condition_mask marks clean (given) action tokens. forward_dynamics
        # conditions on the whole action sequence; the others denoise it fully.
        condition_mask = torch.zeros(
            1, action_chunk_size, 1, device=device, dtype=dtype
        )
        if mode == ACTION_MODE_FORWARD_DYNAMICS:
            condition_mask[:] = 1.0

        noise = torch.randn(
            1,
            action_chunk_size,
            action_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        noise[:, :, raw_action_dim:] = 0.0
        clean_action[:, :, raw_action_dim:] = 0.0
        action_latents = condition_mask * clean_action + (1.0 - condition_mask) * noise

        batch.action_latents = action_latents
        batch.extra["action_domain_ids"] = torch.tensor(
            [domain_id], dtype=torch.long, device=device
        )
        batch.extra["action_velocity_mask"] = 1.0 - condition_mask
        batch.extra["action_condition_latents"] = clean_action
        batch.extra["raw_action_dim"] = raw_action_dim
        batch.extra["action_start_frame_offset"] = action_offset
        self.log_info(
            f"Prepared action latents with shape {tuple(action_latents.shape)} "
            f"(mode={mode}, domain_id={domain_id}, raw_action_dim={raw_action_dim}, "
            f"start_frame_offset={action_offset})"
        )


class Cosmos3TimestepPreparationStage(PipelineStage):
    """
    Timestep preparation stage for Cosmos3.

    Sets up the diffusion scheduler timesteps.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        self.default_flow_shift = getattr(
            getattr(scheduler, "config", None), "flow_shift", None
        )

    def _default_flow_shift_for_mode(self, batch: Req) -> float | None:
        """Resolve the per-mode default flow_shift for the request.

        T2V and I2V keep the checkpoint scheduler's flow_shift; T2I, V2V, and
        action use their own defaults.
        """
        if getattr(batch.sampling_params, "action_mode", None) is not None:
            return COSMOS3_ACTION_FLOW_SHIFT
        if batch.data_type == DataType.IMAGE:
            return COSMOS3_T2I_FLOW_SHIFT
        if batch.preprocessed_video is not None:
            return COSMOS3_V2V_FLOW_SHIFT
        return self.default_flow_shift

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Prepare scheduler timesteps."""
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        flow_shift = getattr(batch, "flow_shift", None)
        if flow_shift is None:
            flow_shift = server_args.pipeline_config.flow_shift
        if flow_shift is None:
            flow_shift = self._default_flow_shift_for_mode(batch)
        if flow_shift is not None and hasattr(self.scheduler, "set_shift"):
            self.scheduler.set_shift(float(flow_shift))

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        batch.timesteps = self.scheduler.timesteps

        self.log_info(
            f"Prepared {len(batch.timesteps)} timesteps (flow_shift={flow_shift})"
        )
        return batch


class Cosmos3DenoisingStage(PipelineStage):
    """Cosmos3 denoise loop, including CFG and the parallelism modes.

    The UND pathway runs once and its K/V is cached per cache_key (``cond`` /
    ``uncond`` / ``cfg_batched``); the GEN pathway runs every step.

    Parallelism modes (combine freely):
    - **CFG-parallel** — splits the conditional and unconditional branches
      across CFG ranks. Each rank runs one branch, then a single all-reduce
      combines them via ``g·cond + (1−g)·uncond``. Default 2-GPU recipe.
    - **Ulysses (sequence parallel)** — shards the visual sequence across an
      SP group. The cross-attention all-gathers visual K/V inside the
      kernel; after the last GEN layer we all-gather hidden_gen back to
      full length.
    - **CFG + Ulysses** — when both are on, the SP group only contains ranks
      that share a CFG context, so each context shards independently.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, transformer, scheduler, server_args: ServerArgs | None = None):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.server_args = server_args
        self._logged_parallel_config = False

        # Apply torch.compile if enabled
        if server_args is not None:
            self._maybe_enable_torch_compile(transformer, server_args)

    def _maybe_enable_torch_compile(
        self, transformer: nn.Module, server_args: ServerArgs
    ) -> None:
        """Regional ``torch.compile`` over the GEN decoder blocks.

        Only ``gen_layers`` are compiled — they are the per-step hot path and
        all share the same module class, so a single compilation amortizes
        across them. The UND ``language_model`` runs once per prompt and is
        cached, so compiling it would only pay warmup cost.

        Caveat for Ulysses (``sp_size > 1``): the cross-attention's all-to-all
        calls into ``torch.distributed.all_to_all_single`` through a Python
        wrapper that fetches the process group at call time, which graph-breaks
        Dynamo. Compile still works but loses some speedup on that path. The
        headline 2-GPU CFG-parallel recipe (``sp_size == 1``) skips the SP
        branch entirely and compiles cleanly.
        """
        if not server_args.enable_torch_compile or not isinstance(
            transformer, nn.Module
        ):
            return

        if current_platform.is_npu():
            compile_kwargs: dict[str, Any] = {
                "backend": get_compiler_backend(),
                "fullgraph": False,
                "dynamic": False,
            }
        else:
            try:
                import torch._inductor.config as _inductor_cfg

                _inductor_cfg.reorder_for_compute_comm_overlap = True
            except ImportError:
                pass
            # Lift Dynamo's per-callable cache cap above the default (64).
            # Each gen_layer is its own compiled object, and several shape
            # specializations (cond/uncond, with/without residual carry,
            # SP on/off) can accumulate. 128 leaves headroom without
            # encouraging unbounded specialization.
            torch._dynamo.config.cache_size_limit = max(
                getattr(torch._dynamo.config, "cache_size_limit", 64), 128
            )
            compile_kwargs = {
                "mode": "default",
                "fullgraph": False,
                "dynamic": False,
            }

        gen_layers = getattr(transformer, "gen_layers", None)
        if gen_layers is not None and isinstance(gen_layers, nn.ModuleList):
            logger.info(
                "Compiling %d Cosmos3 gen_layers with %s",
                len(gen_layers),
                compile_kwargs,
            )
            for i, layer in enumerate(gen_layers):
                gen_layers[i] = torch.compile(layer, **compile_kwargs)
        else:
            logger.warning("Cosmos3 gen_layers not found, skipping torch.compile")

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        result.add_check("timesteps", batch.timesteps, V.is_tensor)
        return result

    def step_profile(self):
        profiler = SGLDiffusionProfiler.get_instance()
        if profiler:
            profiler.step_denoising_step()

    def _run_transformer(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float,
        cache_key: str = "default",
        noisy_frame_mask: torch.Tensor | None = None,
        max_text_seq_len: int | None = None,
        current_timestep: int | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run transformer forward pass.

        Args:
            latents: Noisy latent tensor
            timestep: Current diffusion timestep
            text_ids: Tokenized text input
            text_mask: Attention mask for text
            video_shape: (T, H, W) in latent space
            fps: Video frame rate
            cache_key: Key for the UND K/V cache. Use "cond" for conditional
                and "uncond" for unconditional to enable cache reuse across steps.
            noisy_frame_mask: Optional [B, 1, T, 1, 1] I2V / V2V conditioning mask.
        """
        if current_timestep is None:
            current_timestep = int(timestep.flatten()[0].item())
        with set_forward_context(current_timestep=current_timestep, attn_metadata=None):
            return self.transformer(
                hidden_states=latents,
                encoder_hidden_states=None,  # Not used by Cosmos3
                timestep=timestep,
                text_ids=text_ids,
                text_mask=text_mask,
                fps=fps,
                cache_key=cache_key,
                noisy_frame_mask=noisy_frame_mask,
                max_text_seq_len=max_text_seq_len,
                sound_latents=sound_latents,
                action_latents=action_latents,
                action_domain_ids=action_domain_ids,
                action_noisy_mask=action_noisy_mask,
                action_fps=action_fps,
                action_start_frame_offset=action_start_frame_offset,
            )

    def _manage_device_placement(self, server_args: ServerArgs):
        """Move transformer to GPU if CPU offload is enabled."""
        if not server_args.dit_cpu_offload:
            return

        # FSDP manages offloading internally
        if server_args.use_fsdp_inference:
            return

        device = get_local_torch_device()
        # Load the model to GPU if it's on CPU
        if next(self.transformer.parameters()).device.type == "cpu":
            self.log_info("Moving transformer to GPU for inference")
            self.transformer.to(device)

    @staticmethod
    def _cfg_active_at(t: torch.Tensor, interval: tuple[float, float] | None) -> bool:
        """Return True iff CFG should be applied at timestep ``t``.

        T2I uses a CFG window (e.g. ``[400, 1000]``) to skip guidance at low
        noise levels, where it is empirically harmful. T2V/I2V leave this
        unset and CFG is always on.
        """
        if interval is None:
            return True
        t_scalar = float(t.item()) if torch.is_tensor(t) else float(t)
        lo, hi = interval
        return lo <= t_scalar <= hi

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Run the denoising loop with CFG and optional I2V conditioning."""
        self._manage_device_placement(server_args)

        latents = batch.latents
        sound_latents = batch.audio_latents
        action_latents = getattr(batch, "action_latents", None)
        action_domain_ids = batch.extra.get("action_domain_ids")
        action_velocity_mask = batch.extra.get("action_velocity_mask")
        action_condition_latents = batch.extra.get("action_condition_latents")
        action_raw_dim = batch.extra.get("raw_action_dim")
        action_start_frame_offset = batch.extra.get("action_start_frame_offset", 1)
        action_fps = getattr(batch.sampling_params, "action_fps", None)
        timesteps = batch.timesteps
        guidance_scale = batch.guidance_scale

        cond_text_ids = batch.extra["cond_text_ids"]
        cond_text_mask = batch.extra["cond_text_mask"]
        uncond_text_ids = batch.extra["uncond_text_ids"]
        uncond_text_mask = batch.extra["uncond_text_mask"]
        video_shape = batch.extra["video_shape"]
        fps = batch.extra.get("fps", 24.0)
        velocity_mask = batch.extra.get("velocity_mask")
        condition_latents = batch.extra.get("condition_latents")
        guidance_interval = getattr(batch.sampling_params, "guidance_interval", None)

        do_cfg = guidance_scale > 1.0

        enable_cfg_parallel = server_args.enable_cfg_parallel and do_cfg
        if action_latents is not None and enable_cfg_parallel:
            raise NotImplementedError(
                "Cosmos3 action generation does not support CFG parallel yet"
            )

        # Use separate scheduler instances for action/sound: UniPC keeps a
        # per-call output history sized to the last sample, so video (5D),
        # action (3D), and sound (3D) steps must not share state.
        sound_scheduler = None
        if sound_latents is not None:
            sound_scheduler = copy.deepcopy(self.scheduler)
            sound_scheduler.set_timesteps(len(timesteps), device=timesteps.device)
        action_scheduler = None
        if action_latents is not None:
            action_scheduler = copy.deepcopy(self.scheduler)
            action_scheduler.set_timesteps(len(timesteps), device=timesteps.device)
        cfg_rank = get_classifier_free_guidance_rank() if enable_cfg_parallel else 0
        cfg_world_size = (
            get_classifier_free_guidance_world_size() if enable_cfg_parallel else 1
        )

        sp_size = get_sp_world_size()
        sp_rank = get_sp_parallel_rank() if sp_size > 1 else 0
        ulysses_enabled = sp_size > 1

        if not self._logged_parallel_config:
            self._logged_parallel_config = True
            if enable_cfg_parallel and ulysses_enabled:
                self.log_info(
                    f"CFG + Ulysses enabled: cfg_size={cfg_world_size}, cfg_rank={cfg_rank}, "
                    f"sp_size={sp_size}, sp_rank={sp_rank}"
                )
            elif enable_cfg_parallel:
                self.log_info(
                    f"CFG parallel enabled: cfg_size={cfg_world_size}, cfg_rank={cfg_rank}"
                )
            elif ulysses_enabled:
                self.log_info(f"Ulysses enabled: sp_size={sp_size}, sp_rank={sp_rank}")

        # Drop any cached UND K/V from a previous request — its text differs.
        self.transformer.reset_cache()

        self.log_info(
            f"Starting denoising with {len(timesteps)} steps, CFG={do_cfg}, "
            f"CFG_parallel={enable_cfg_parallel}, cfg_rank={cfg_rank}"
        )

        progress_bar = self.progress_bar(
            enumerate(timesteps),
            total=len(timesteps),
            desc="Denoising",
            batch=batch,
        )

        for i, t in progress_bar:
            timestep = t.unsqueeze(0) if t.dim() == 0 else t
            # Outside the CFG window the effective scale collapses to 1.0,
            # which reduces CFG to the cond branch (cfg-parallel safe).
            effective_scale = (
                guidance_scale if self._cfg_active_at(t, guidance_interval) else 1.0
            )

            if do_cfg:
                if enable_cfg_parallel:
                    noise_pred = self._predict_noise_cfg_parallel(
                        latents=latents,
                        timestep=timestep,
                        cond_text_ids=cond_text_ids,
                        cond_text_mask=cond_text_mask,
                        uncond_text_ids=uncond_text_ids,
                        uncond_text_mask=uncond_text_mask,
                        video_shape=video_shape,
                        fps=fps,
                        guidance_scale=effective_scale,
                        cfg_rank=cfg_rank,
                        noisy_frame_mask=velocity_mask,
                        cond_text_seq_len=batch.extra["cond_text_seq_len"],
                        uncond_text_seq_len=batch.extra["uncond_text_seq_len"],
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
                elif effective_scale == 1.0:
                    noise_pred = self._run_transformer(
                        latents=latents,
                        timestep=timestep,
                        text_ids=cond_text_ids,
                        text_mask=cond_text_mask,
                        video_shape=video_shape,
                        fps=fps,
                        cache_key="cond",
                        noisy_frame_mask=velocity_mask,
                        max_text_seq_len=batch.extra["cond_text_seq_len"],
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
                else:
                    noise_pred = self._predict_noise_cfg_batched(
                        latents=latents,
                        timestep=timestep,
                        cond_text_ids=cond_text_ids,
                        cond_text_mask=cond_text_mask,
                        uncond_text_ids=uncond_text_ids,
                        uncond_text_mask=uncond_text_mask,
                        video_shape=video_shape,
                        fps=fps,
                        guidance_scale=effective_scale,
                        noisy_frame_mask=velocity_mask,
                        max_text_seq_len=max(
                            batch.extra["cond_text_seq_len"],
                            batch.extra["uncond_text_seq_len"],
                        ),
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
            else:
                noise_pred = self._run_transformer(
                    latents=latents,
                    timestep=timestep,
                    text_ids=cond_text_ids,
                    text_mask=cond_text_mask,
                    video_shape=video_shape,
                    fps=fps,
                    cache_key="cond",
                    noisy_frame_mask=velocity_mask,
                    max_text_seq_len=batch.extra["cond_text_seq_len"],
                    current_timestep=i,
                    sound_latents=sound_latents,
                    action_latents=action_latents,
                    action_domain_ids=action_domain_ids,
                    action_noisy_mask=action_velocity_mask,
                    action_fps=action_fps,
                    action_start_frame_offset=action_start_frame_offset,
                )

            # Unpack multi-modality outputs; ordering is (video[, action][, sound]).
            action_noise_pred = None
            sound_noise_pred = None
            if isinstance(noise_pred, tuple):
                out_idx = 1
                video_noise_pred = noise_pred[0]
                if action_latents is not None:
                    action_noise_pred = noise_pred[out_idx]
                    out_idx += 1
                if sound_latents is not None:
                    sound_noise_pred = noise_pred[out_idx]
                noise_pred = video_noise_pred

            # I2V / V2V: zero-velocity at conditioned frames so the scheduler
            # keeps them clean; UniPC's predictor-corrector still rescales the
            # sample, so we re-blend the clean condition latents below.
            if velocity_mask is not None:
                noise_pred = noise_pred * velocity_mask

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=False,
            )[0]

            if action_noise_pred is not None:
                # Zero the velocity at conditioned (clean) action tokens and at
                # padding dims so the scheduler only denoises the active slots,
                # then re-blend the clean condition after the step.
                if action_velocity_mask is not None:
                    action_noise_pred = action_noise_pred * action_velocity_mask
                if (
                    action_raw_dim is not None
                    and action_raw_dim < action_noise_pred.shape[-1]
                ):
                    action_noise_pred[..., action_raw_dim:] = 0.0
                action_latents = action_scheduler.step(
                    action_noise_pred,
                    t,
                    action_latents,
                    return_dict=False,
                )[0]
                if (
                    action_condition_latents is not None
                    and action_velocity_mask is not None
                ):
                    action_latents = (
                        action_velocity_mask * action_latents
                        + (1.0 - action_velocity_mask) * action_condition_latents
                    )

            if sound_noise_pred is not None:
                sound_latents = sound_scheduler.step(
                    sound_noise_pred,
                    t,
                    sound_latents,
                    return_dict=False,
                )[0]

            if condition_latents is not None and velocity_mask is not None:
                latents = (
                    velocity_mask * latents + (1.0 - velocity_mask) * condition_latents
                )

            if batch.profile and not batch.is_warmup:
                self.step_profile()

        batch.latents = latents
        if action_latents is not None:
            batch.action_latents = action_latents
        if sound_latents is not None:
            batch.audio_latents = sound_latents
        self.log_info("Denoising complete")
        return batch

    def _predict_noise_cfg_batched(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_text_ids: torch.Tensor,
        cond_text_mask: torch.Tensor,
        uncond_text_ids: torch.Tensor,
        uncond_text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float,
        guidance_scale: float,
        noisy_frame_mask: torch.Tensor | None = None,
        max_text_seq_len: int | None = None,
        current_timestep: int | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run CFG by stacking both branches into a batch_size=2 forward.

        Halves the kernel-launch count vs running cond and uncond serially.
        Order is ``[uncond, cond]`` so the chunk-and-combine math below
        matches the standard CFG formula.
        """
        latents_batched = torch.cat([latents, latents], dim=0)
        text_ids_batched = torch.cat([uncond_text_ids, cond_text_ids], dim=0)
        text_mask_batched = torch.cat([uncond_text_mask, cond_text_mask], dim=0)
        timestep_batched = timestep.expand(2)
        mask_batched = (
            torch.cat([noisy_frame_mask, noisy_frame_mask], dim=0)
            if noisy_frame_mask is not None
            else None
        )
        sound_batched = (
            torch.cat([sound_latents, sound_latents], dim=0)
            if sound_latents is not None
            else None
        )
        action_batched = (
            torch.cat([action_latents, action_latents], dim=0)
            if action_latents is not None
            else None
        )
        action_domain_ids_batched = (
            torch.cat([action_domain_ids, action_domain_ids], dim=0)
            if action_domain_ids is not None
            else None
        )
        action_noisy_mask_batched = (
            torch.cat([action_noisy_mask, action_noisy_mask], dim=0)
            if action_noisy_mask is not None
            else None
        )

        out = self._run_transformer(
            latents=latents_batched,
            timestep=timestep_batched,
            text_ids=text_ids_batched,
            text_mask=text_mask_batched,
            video_shape=video_shape,
            fps=fps,
            cache_key="cfg_batched",
            noisy_frame_mask=mask_batched,
            max_text_seq_len=max_text_seq_len,
            current_timestep=current_timestep,
            sound_latents=sound_batched,
            action_latents=action_batched,
            action_domain_ids=action_domain_ids_batched,
            action_noisy_mask=action_noisy_mask_batched,
            action_fps=action_fps,
            action_start_frame_offset=action_start_frame_offset,
        )

        def _cfg_combine(pred: torch.Tensor) -> torch.Tensor:
            uncond, cond = pred.chunk(2, dim=0)
            return uncond + guidance_scale * (cond - uncond)

        if isinstance(out, tuple):
            return tuple(_cfg_combine(p) for p in out)
        return _cfg_combine(out)

    def _predict_noise_cfg_parallel(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        cond_text_ids: torch.Tensor,
        cond_text_mask: torch.Tensor,
        uncond_text_ids: torch.Tensor,
        uncond_text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float,
        guidance_scale: float,
        cfg_rank: int,
        noisy_frame_mask: torch.Tensor | None = None,
        cond_text_seq_len: int | None = None,
        uncond_text_seq_len: int | None = None,
        current_timestep: int | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run CFG with one branch per CFG rank, combined by all-reduce.

        Rank 0 runs the conditional branch and contributes ``g·cond`` to the
        sum; rank 1 runs the unconditional branch and contributes
        ``(1−g)·uncond``. The all-reduce sum is exactly the standard CFG
        result. Each rank keeps its own UND K/V cache (``"cond"`` /
        ``"uncond"``). When sound/action modalities are present the forward
        returns a per-modality tuple; each branch scales every modality by its
        coefficient and the reduction combines them element-wise.
        """
        if cfg_rank == 0:
            text_ids, text_mask, cache_key = cond_text_ids, cond_text_mask, "cond"
            text_seq_len = cond_text_seq_len
            coeff = guidance_scale
        else:
            text_ids, text_mask, cache_key = uncond_text_ids, uncond_text_mask, "uncond"
            text_seq_len = uncond_text_seq_len
            coeff = 1.0 - guidance_scale

        out = self._run_transformer(
            latents=latents,
            timestep=timestep,
            text_ids=text_ids,
            text_mask=text_mask,
            video_shape=video_shape,
            fps=fps,
            cache_key=cache_key,
            noisy_frame_mask=noisy_frame_mask,
            max_text_seq_len=text_seq_len,
            current_timestep=current_timestep,
            sound_latents=sound_latents,
            action_latents=action_latents,
            action_domain_ids=action_domain_ids,
            action_noisy_mask=action_noisy_mask,
            action_fps=action_fps,
            action_start_frame_offset=action_start_frame_offset,
        )

        if isinstance(out, tuple):
            return tuple(cfg_model_parallel_all_reduce(coeff * p) for p in out)
        return cfg_model_parallel_all_reduce(coeff * out)


class Cosmos3DecodingStage(PipelineStage):
    """
    VAE decoding stage for Cosmos3.

    Decodes latents to pixel space using the VAE.
    Returns OutputBatch instead of Req to signal pipeline completion.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, vae, guardrails: bool = False, sound_tokenizer=None):
        super().__init__()
        self.vae = vae
        self._latents_mean = None
        self._latents_std = None
        self._guardrails = guardrails
        self.sound_tokenizer = sound_tokenizer
        if guardrails:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
                _init_guardrails,
            )

            _init_guardrails()

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("latents", batch.latents, V.is_tensor)
        return result

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to video frames. Returns tensor in [B, C, T, H, W] format."""
        device = latents.device
        # Get VAE dtype from its parameters
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(vae_dtype)

        # Apply latent normalization if configured
        if hasattr(self.vae.config, "latents_mean") and hasattr(
            self.vae.config, "latents_std"
        ):
            if self._latents_mean is None:
                self._latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, -1, 1, 1, 1)
                    .to(device, vae_dtype)
                )
                self._latents_std = (
                    torch.tensor(self.vae.config.latents_std)
                    .view(1, -1, 1, 1, 1)
                    .to(device, vae_dtype)
                )
            latents = (latents * self._latents_std) + self._latents_mean
        else:
            scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
            latents = latents / scaling_factor

        # Decode - returns [B, C, T, H, W]
        video = self.vae.decode(latents)
        # Handle both dict return and direct tensor return
        if hasattr(video, "sample"):
            video = video.sample
        elif isinstance(video, tuple):
            video = video[0]

        return video

    @staticmethod
    def _postprocess_tensor(decoded: torch.Tensor) -> torch.Tensor:
        return decoded.mul_(0.5).add_(0.5).clamp_(0, 1).float()

    @staticmethod
    def _postprocess_video_np(video: torch.Tensor, is_image_gen: bool) -> np.ndarray:
        if is_image_gen:
            return video.squeeze(2).permute(0, 2, 3, 1).cpu().numpy()
        return video.permute(0, 2, 3, 4, 1).cpu().numpy()

    def forward(self, batch: Req, server_args: ServerArgs):
        """Decode latents to video, or to a single image for T2I."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
        )

        is_image_gen = batch.data_type == DataType.IMAGE
        self.log_info(
            "Decoding latents to image..."
            if is_image_gen
            else "Decoding latents to video..."
        )

        device = batch.latents.device
        if server_args.vae_cpu_offload:
            self.vae.to(device)

        with torch.no_grad():
            decoded = self._decode_latents(batch.latents)

        if server_args.vae_cpu_offload and not getattr(batch, "is_warmup", False):
            self.vae.to("cpu", non_blocking=True)

        self.log_info(f"Decoded tensor shape: {decoded.shape}")
        output = self._postprocess_tensor(decoded)

        if self._guardrails and batch.use_guardrails is not False:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
                check_video_safety,
            )

            output = self._postprocess_video_np(output, is_image_gen)
            if is_image_gen:
                # check_video_safety expects [B, T, H, W, C]; wrap then unwrap.
                output = check_video_safety(output[:, np.newaxis, ...])[:, 0, ...]
            else:
                output = check_video_safety(output)
        elif not is_image_gen:
            self.log_info(f"Postprocessed video tensor shape: {output.shape}")

        audio = None
        audio_sample_rate = None
        if self.sound_tokenizer is not None and batch.audio_latents is not None:
            if server_args.vae_cpu_offload:
                self.sound_tokenizer.to(device)
            with torch.no_grad():
                decoded_audio = self.sound_tokenizer.decode(
                    batch.audio_latents.to(device)
                )
            audio = decoded_audio.float().cpu()
            audio_sample_rate = self.sound_tokenizer.sample_rate
            if server_args.vae_cpu_offload and not getattr(batch, "is_warmup", False):
                self.sound_tokenizer.to("cpu", non_blocking=True)
            self.log_info(
                f"Decoded audio tensor shape: {tuple(audio.shape)} @ {audio_sample_rate} Hz"
            )

        action_pred = None
        if getattr(batch, "action_latents", None) is not None:
            raw_action_dim = batch.extra.get("raw_action_dim")
            action_pred = batch.action_latents.float().cpu()
            if raw_action_dim is not None:
                action_pred = action_pred[:, :, :raw_action_dim]
            stats_path = getattr(batch.sampling_params, "action_stats_path", None)
            if stats_path is not None:
                method = getattr(
                    batch.sampling_params, "action_normalization", "quantile"
                )
                action_pred = denormalize_action(
                    action_pred, method, load_action_stats(stats_path)
                )
            self.log_info(f"Action predictions shape: {tuple(action_pred.shape)}")

        return OutputBatch(
            output=output,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            action_pred=action_pred,
            action_mode=getattr(batch.sampling_params, "action_mode", None),
            action_domain_id=getattr(batch.sampling_params, "domain_id", None),
            action_raw_action_dim=batch.extra.get("raw_action_dim") if getattr(batch, "extra", None) else None,
            metrics=batch.metrics if hasattr(batch, "metrics") else None,
        )
