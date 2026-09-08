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
import math
from typing import Any

import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    ACTION_MODE_FORWARD_DYNAMICS,
    ACTION_MODE_INVERSE_DYNAMICS,
    ACTION_MODE_POLICY,
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
from sglang.multimodal_gen.runtime.post_training.rollout_denoising_mixin import (
    RolloutDenoisingMixin,
)
from sglang.multimodal_gen.runtime.post_training.rollout_scheduler import (
    prepare_rollout_request_scheduler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.profiler import SGLDiffusionProfiler
from sglang.multimodal_gen.runtime.utils.vision import load_image, load_video
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
# config leave flow_shift unset.
COSMOS3_T2I_FLOW_SHIFT = 3.0
COSMOS3_I2V_FLOW_SHIFT = 10.0
COSMOS3_T2V_FLOW_SHIFT = 10.0
COSMOS3_V2V_FLOW_SHIFT = 10.0
COSMOS3_ACTION_FLOW_SHIFT = 10.0
# Edge uses a single low flow-shift for every video mode (t2v/i2v/v2v).
COSMOS3_EDGE_VIDEO_FLOW_SHIFT = 3.0


def _inject_caption_metadata(
    prompt: str, num_frames: int, fps: float, height: int, width: int
) -> str | None:
    """Add the generation metadata that Cosmos3's structured captions carry.

    Training captions always ship ``resolution``, plus ``duration``/``fps`` for
    video, so a JSON caption without them is out of distribution. Returns
    ``None`` when the prompt is not a JSON object; those prompts carry the same
    metadata as trailing prose via the duration template instead.
    """
    try:
        caption = json.loads(prompt)
    except (TypeError, ValueError):
        return None
    if not isinstance(caption, dict):
        return None

    caption["resolution"] = {"H": int(height), "W": int(width)}
    if num_frames > 1:
        caption["duration"] = f"{int(num_frames / fps) if fps > 0 else 0}s"
        caption["fps"] = float(fps)
    else:
        caption.pop("duration", None)
        caption.pop("fps", None)
    return json.dumps(caption)


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


def _pil_to_uint8_tensor(image: PIL.Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _resize_center_crop_uint8_cthw(
    frames: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Resize and center-crop ``uint8 [3, T, H, W]`` transfer frames."""
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Transfer frames must have shape [3, T, H, W], got {tuple(frames.shape)}"
        )
    orig_h, orig_w = int(frames.shape[2]), int(frames.shape[3])
    scale = max(width / orig_w, height / orig_h)
    resize_h = int(np.ceil(scale * orig_h))
    resize_w = int(np.ceil(scale * orig_w))
    frames_tchw = frames.permute(1, 0, 2, 3).to(dtype=torch.float32)
    resized = F.interpolate(
        frames_tchw,
        size=(resize_h, resize_w),
        mode="bilinear",
        align_corners=False,
    )
    top = (resize_h - height) // 2
    left = (resize_w - width) // 2
    cropped = resized[:, :, top : top + height, left : left + width]
    return (
        cropped.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
    )


def _pad_transfer_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Pad ``[1, 3, T, H, W]`` with reflected temporal content."""
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(
            f"Transfer video must have shape [1, 3, T, H, W], got {tuple(video.shape)}"
        )
    if target_frames <= 0:
        raise ValueError("Transfer target frame count must be positive")
    video = video[:, :, :target_frames]
    if video.shape[2] == 0:
        raise ValueError("Transfer video cannot be empty")
    while video.shape[2] < target_frames:
        remaining = target_frames - video.shape[2]
        reflected = video.flip(dims=[2])
        pad_len = min(max(video.shape[2] - 1, 1), remaining)
        video = torch.cat([video, reflected[:, :, :pad_len]], dim=2)
    return video.contiguous()


class Cosmos3ImagePreprocessStage(PipelineStage):
    """Load, aspect-resize, and center-crop the conditioning input.

    For I2V: writes ``[1, 3, H, W]`` to ``batch.preprocessed_image``. Batched
    policy requests write ``[B, 3, H, W]``; regular visual generation remains
    single-image conditioned.
    For V2V: writes ``[1, 3, T_in, H, W]`` to ``batch.preprocessed_video``.
    For transfer: writes ``[1, 3, T, H, W]`` control pixels to
    ``batch.extra["preprocessed_control"]`` (independent of I2V / V2V).
    No-op for T2V / T2I.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    def _load_control_video(
        self,
        control_path: str,
        target_w: int,
        target_h: int,
        max_frames: int,
    ) -> torch.Tensor:
        """Load transfer media as CPU ``uint8 [1, 3, T, H, W]``."""
        frames = load_video(control_path)
        if not frames:
            raise ValueError(f"No frames decoded from transfer video: {control_path!r}")
        frames = frames[:max_frames]
        frames_cthw = torch.stack(
            [_pil_to_uint8_tensor(frame.convert("RGB")) for frame in frames],
            dim=1,
        )
        return (
            _resize_center_crop_uint8_cthw(frames_cthw, height=target_h, width=target_w)
            .unsqueeze(0)
            .contiguous()
        )

    @staticmethod
    def _get_transfer_num_chunks(
        total_frames: int, frames_per_chunk: int, conditional_frames: int
    ) -> tuple[int, int]:
        if total_frames <= frames_per_chunk:
            return 1, frames_per_chunk
        stride = frames_per_chunk - conditional_frames
        if stride <= 0:
            raise ValueError(
                "num_conditional_frames must be smaller than num_video_frames_per_chunk"
            )
        remaining = total_frames - frames_per_chunk
        return 1 + math.ceil(remaining / stride), stride

    def _prepare_transfer_plan(
        self,
        batch: Req,
        control_paths: list[str],
        source_video_path: str | None,
    ) -> None:
        max_frames = int(batch.sampling_params.max_frames)
        controls = [
            self._load_control_video(
                path, batch.width, batch.height, max_frames=max_frames
            )
            for path in control_paths
        ]
        total_frames = min(batch.num_frames, controls[0].shape[2], max_frames)
        if total_frames <= 0:
            raise ValueError("Cosmos3 transfer requires at least one control frame")

        requested_chunk_frames = int(batch.sampling_params.num_video_frames_per_chunk)
        chunk_frames = (
            1
            if total_frames == 1
            else (math.ceil((requested_chunk_frames - 1) / 4) * 4 + 1)
        )
        num_chunks, stride = self._get_transfer_num_chunks(
            total_frames,
            chunk_frames,
            int(batch.sampling_params.num_conditional_frames),
        )
        padded_frames = max(total_frames, chunk_frames)
        controls = [
            _pad_transfer_frames(control, padded_frames) for control in controls
        ]

        source_video = None
        if source_video_path:
            source_video = self._load_control_video(
                source_video_path,
                batch.width,
                batch.height,
                max_frames=max_frames,
            )
            source_video = _pad_transfer_frames(source_video, padded_frames)
        if (
            batch.sampling_params.num_first_chunk_conditional_frames > 0
            and source_video is None
        ):
            raise ValueError(
                "num_first_chunk_conditional_frames > 0 requires video_path"
            )

        batch.num_frames = total_frames
        batch.extra["preprocessed_control"] = controls
        batch.extra["preprocessed_transfer_video"] = source_video
        batch.extra["transfer_plan"] = {
            "total_frames": total_frames,
            "chunk_frames": chunk_frames,
            "num_chunks": num_chunks,
            "stride": stride,
        }
        self.log_info(
            f"Prepared transfer plan with {len(controls)} control(s), "
            f"{total_frames} output frames, {num_chunks} chunk(s) of "
            f"{chunk_frames} frames"
        )

    @staticmethod
    def _normalize_control_paths(control_path: Any) -> list[str]:
        """Normalize ``control_path`` (str / list / None) to a list of paths.

        Multiple paths drive multi-hint transfer (e.g. ``[edge.mp4, depth.mp4]``):
        each is VAE-encoded into its own control-latent block and all blocks
        prefix the target clip in the GEN sequence.
        """
        if control_path is None:
            return []
        if isinstance(control_path, str):
            if not control_path.strip():
                raise ValueError("control_path is an empty string")
            return [control_path]
        if isinstance(control_path, (list, tuple)):
            paths: list[str] = []
            for i, p in enumerate(control_path):
                if not isinstance(p, str) or not p.strip():
                    raise ValueError(
                        f"control_path[{i}] must be a non-empty string, got {p!r}"
                    )
                paths.append(p)
            return paths
        raise ValueError(
            "control_path must be a string or list of strings, got "
            f"{type(control_path).__name__}"
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image_path = batch.image_path
        video_path = batch.video_path
        is_action_policy = (
            batch.data_type == DataType.ACTION
            and getattr(batch.sampling_params, "action_mode", None)
            == ACTION_MODE_POLICY
        )
        if isinstance(image_path, list) and not is_action_policy:
            image_path = image_path[0] if image_path else None
        if isinstance(video_path, list):
            video_path = video_path[0] if video_path else None

        control_paths = self._normalize_control_paths(
            getattr(batch.sampling_params, "control_path", None)
        )
        if control_paths:
            if image_path is not None:
                raise ValueError(
                    "Cosmos3 transfer accepts an optional source video, not an image"
                )
            self._prepare_transfer_plan(batch, control_paths, video_path)
            return batch

        if image_path and video_path:
            raise ValueError(
                "Cosmos3 accepts either --image-path (I2V) or --video-path "
                "(V2V), not both"
            )

        target_h, target_w = batch.height, batch.width

        if image_path is not None:
            image_sources = (
                list(image_path)
                if isinstance(image_path, (list, tuple))
                else [image_path]
            )
            if not image_sources:
                raise ValueError("Cosmos3 I2V image list is empty")
            tensors: list[torch.Tensor] = []
            for src in image_sources:
                image = load_image(src)
                image = _resize_crop_pil(image, target_w, target_h)
                tensors.append(_pil_to_normalized_tensor(image))
            batch.preprocessed_image = torch.stack(tensors, dim=0).contiguous()
            self.log_info(
                f"Preprocessed {len(tensors)} conditioning image(s) to "
                f"{target_w}x{target_h}"
            )
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
        text: str | list[str],
        max_sequence_length: int,
        device: torch.device,
        use_system_prompt: bool = False,
        system_prompt: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Tokenize a prompt using Qwen2 chat template.

        Returns (input_ids, attention_mask, seq_len) as [B, S] tensors.
        """
        texts = text if isinstance(text, (list, tuple)) else [text]
        if not texts:
            raise ValueError("Cosmos3 prompt batch must not be empty")
        input_id_lists: list[list[int]] = []
        attention_mask_lists: list[list[int]] = []
        seq_lens: list[int] = []
        pad_token_id = self.tokenizer.pad_token_id or 0
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        for text_item in texts:
            conversations = []
            if use_system_prompt:
                conversations.append(
                    {
                        "role": "system",
                        "content": system_prompt or COSMOS3_VIDEO_SYSTEM_PROMPT,
                    }
                )
            conversations.append({"role": "user", "content": text_item})

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
            if vision_start_id is not None:
                token_ids.append(vision_start_id)

            seq_len = len(token_ids)
            pad_len = max_sequence_length - seq_len
            attention_mask = [1] * seq_len + [0] * pad_len
            token_ids = token_ids + [pad_token_id] * pad_len
            input_id_lists.append(token_ids)
            attention_mask_lists.append(attention_mask)
            seq_lens.append(seq_len)

        if len(set(seq_lens)) != 1:
            raise ValueError(
                "Cosmos3 batched prompts must tokenize to the same length because "
                "GEN cross-attention does not mask padded text K/V; split prompts "
                f"into equal-length batches instead (lengths={seq_lens})"
            )
        input_ids = torch.tensor(input_id_lists, dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            attention_mask_lists, dtype=torch.long, device=device
        )
        return input_ids, attention_mask, seq_lens[0]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Tokenize prompt and negative prompt."""
        device = get_local_torch_device()
        prompt = batch.prompt
        negative_prompt = batch.negative_prompt or COSMOS3_DEFAULT_NEGATIVE_PROMPT

        # Get parameters
        max_sequence_length = getattr(batch, "max_sequence_length", None) or 4096
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
        num_frames = batch.extra.get("transfer_plan", {}).get(
            "chunk_frames", batch.num_frames
        )
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
        else:
            structured_caption = _inject_caption_metadata(
                prompt, num_frames, fps, batch.height, batch.width
            )
            if structured_caption is not None:
                # The metadata is already in the caption; appending the prose
                # template too would state the duration twice.
                prompt = structured_caption
                use_duration_template = False

        # Apply duration template if enabled (no temporal concept for T2I).
        if use_duration_template and not is_image_gen and num_frames > 1:
            duration = num_frames / fps
            suffix = COSMOS3_DURATION_TEMPLATE.format(duration=duration, fps=fps)
            prompt = f"{prompt} {suffix}"
            self.log_info(f"Prompt with duration: '{prompt}'")

        # Tokenize prompts
        if isinstance(prompt, list) and not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt] * len(prompt)

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

        transfer_plan = batch.extra.get("transfer_plan")
        pixel_num_frames = (
            transfer_plan["chunk_frames"] if transfer_plan else batch.num_frames
        )
        num_channels_latents = self.transformer.latent_channel
        num_latent_frames = (pixel_num_frames - 1) // vae_scale_factor_temporal + 1
        height_latent = batch.height // vae_scale_factor_spatial
        width_latent = batch.width // vae_scale_factor_spatial

        if batch.preprocessed_image is not None:
            batch_dim = int(batch.preprocessed_image.shape[0])
        else:
            batch_dim = 1

        shape = (
            batch_dim,
            num_channels_latents,
            num_latent_frames,
            height_latent,
            width_latent,
        )

        if transfer_plan is not None:
            batch.latents = torch.zeros(shape, device=device, dtype=dtype)
            batch.raw_latent_shape = shape
            batch.extra["video_shape"] = (
                num_latent_frames,
                height_latent,
                width_latent,
            )
            batch.extra["vae_scale_factor_temporal"] = vae_scale_factor_temporal
            batch.extra["vae_scale_factor_spatial"] = vae_scale_factor_spatial
            self.log_info(f"Prepared transfer latent shape {shape}")
            return batch

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)
            # The rollout SDE step draws its variance noise from this generator.
            batch.generator = generator

        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        uses_visual_latents = batch.data_type in (DataType.VIDEO, DataType.ACTION)
        has_image_cond = batch.preprocessed_image is not None and uses_visual_latents
        has_video_cond = batch.preprocessed_video is not None and uses_visual_latents

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

            with self.use_declared_component(component_name="vae", module=self.vae):
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
                batch_dim, 1, num_latent_frames, 1, 1, device=device, dtype=dtype
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

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]

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
        batch_dim = (
            int(batch.raw_latent_shape[0])
            if getattr(batch, "raw_latent_shape", None)
            else 1
        )
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
                batch_dim, action_chunk_size, action_dim, device=device, dtype=dtype
            )

        raw_action_dim = int(raw_action_dim)
        if not 0 < raw_action_dim <= action_dim:
            raise ValueError(
                f"raw_action_dim must be in [1, {action_dim}], got {raw_action_dim}"
            )

        # condition_mask marks clean (given) action tokens. forward_dynamics
        # conditions on the whole action sequence; the others denoise it fully.
        condition_mask = torch.zeros(
            batch_dim, action_chunk_size, 1, device=device, dtype=dtype
        )
        if mode == ACTION_MODE_FORWARD_DYNAMICS:
            condition_mask[:] = 1.0

        noise = torch.randn(
            batch_dim,
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
            [domain_id] * batch_dim, dtype=torch.long, device=device
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

    def _default_flow_shift_for_mode(self, batch: Req, is_edge: bool) -> float | None:
        """Resolve the per-mode default flow_shift for the request."""
        if getattr(batch.sampling_params, "action_mode", None) is not None:
            return COSMOS3_ACTION_FLOW_SHIFT
        if batch.data_type == DataType.IMAGE:
            return COSMOS3_T2I_FLOW_SHIFT
        if is_edge:
            return COSMOS3_EDGE_VIDEO_FLOW_SHIFT
        if batch.preprocessed_image is not None:
            return COSMOS3_I2V_FLOW_SHIFT
        if batch.preprocessed_video is not None:
            return COSMOS3_V2V_FLOW_SHIFT
        return COSMOS3_T2V_FLOW_SHIFT

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Prepare scheduler timesteps."""
        device = get_local_torch_device()

        pipeline_config = server_args.pipeline_config
        distilled_sigmas = pipeline_config.distilled_sigmas
        if distilled_sigmas is not None:
            # Distilled checkpoints carry an explicit fixed-step sigma schedule
            # with the shift already baked in; drive the scheduler from it
            # directly (step count == len(sigmas), num_inference_steps ignored).
            # Reset shift so set_timesteps does not re-shift the baked-in sigmas.
            if hasattr(self.scheduler, "set_shift"):
                self.scheduler.set_shift(1.0)
            self.scheduler.set_timesteps(sigmas=distilled_sigmas, device=device)
            batch.timesteps = self.scheduler.timesteps
            self.log_info(
                f"Prepared {len(batch.timesteps)} distilled timesteps "
                f"(sigmas={distilled_sigmas})"
            )
            return batch

        num_inference_steps = batch.num_inference_steps
        explicit_flow_shift = getattr(batch, "flow_shift", None)
        if explicit_flow_shift is None:
            explicit_flow_shift = pipeline_config.flow_shift
        flow_shift = explicit_flow_shift
        if flow_shift is None:
            flow_shift = self._default_flow_shift_for_mode(
                batch, bool(pipeline_config.is_edge)
            )
        if flow_shift is not None and hasattr(self.scheduler, "set_shift"):
            self.scheduler.set_shift(float(flow_shift))

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        batch.timesteps = self.scheduler.timesteps

        if batch.rollout:
            prepare_rollout_request_scheduler(
                batch,
                self.scheduler,
                explicit_shift=explicit_flow_shift,
                num_inference_steps=num_inference_steps,
                device=device,
            )

        self.log_info(
            f"Prepared {len(batch.timesteps)} timesteps (flow_shift={flow_shift})"
        )
        return batch


class Cosmos3DenoisingStage(PipelineStage, RolloutDenoisingMixin):
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

    def __init__(
        self,
        transformer,
        scheduler,
        server_args: ServerArgs | None = None,
        vae=None,
    ):
        super().__init__()
        self.transformer = transformer
        self.scheduler = scheduler
        self.server_args = server_args
        self.vae = vae
        self._logged_parallel_config = False
        self._logged_cfg_split = False

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
            transformer._gen_layers_torch_compiled = True
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
        control_latents: list[torch.Tensor] | None = None,
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
                control_latents=control_latents,
                transfer_share_vision_temporal_positions=getattr(
                    self, "_share_vision_temporal_positions", True
                ),
            )

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

    def _normalize_transfer_video(
        self, video: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return video.to(device=device, dtype=dtype).div(127.5).sub(1.0)

    def _encode_transfer_video(
        self, video: torch.Tensor, output_dtype: torch.dtype
    ) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        with torch.no_grad():
            latent = self.vae.encode(video.to(dtype=vae_dtype)).mode()
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
        return ((latent - mean) / std).to(output_dtype)

    def _decode_transfer_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        latents = latents.to(vae_dtype)
        mean = (
            torch.as_tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(latents.device, vae_dtype)
        )
        std = (
            torch.as_tensor(self.vae.config.latents_std)
            .view(1, -1, 1, 1, 1)
            .to(latents.device, vae_dtype)
        )
        with torch.no_grad():
            decoded = self.vae.decode(latents * std + mean)
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        elif isinstance(decoded, tuple):
            decoded = decoded[0]
        return decoded

    def _prepare_transfer_chunk(
        self,
        batch: Req,
        chunk_id: int,
        previous_output: torch.Tensor | None,
        generator: torch.Generator | None,
    ) -> int:
        plan = batch.extra["transfer_plan"]
        chunk_frames = int(plan["chunk_frames"])
        start_frame = chunk_id * int(plan["stride"])
        end_frame = min(start_frame + chunk_frames, int(plan["total_frames"]))
        device = batch.latents.device
        latent_dtype = batch.latents.dtype
        vae_dtype = next(self.vae.parameters()).dtype

        control_norms = [
            self._normalize_transfer_video(
                _pad_transfer_frames(
                    control[:, :, start_frame:end_frame], chunk_frames
                ),
                device,
                vae_dtype,
            )
            for control in batch.extra["preprocessed_control"]
        ]
        target_norm = torch.zeros_like(control_norms[0])
        current_conditional_frames = 0

        source_video = batch.extra.get("preprocessed_transfer_video")
        if (
            chunk_id == 0
            and batch.sampling_params.num_first_chunk_conditional_frames > 0
        ):
            current_conditional_frames = min(
                int(batch.sampling_params.num_first_chunk_conditional_frames),
                source_video.shape[2],
                chunk_frames,
            )
            source_norm = self._normalize_transfer_video(
                source_video[:, :, :current_conditional_frames], device, vae_dtype
            )
            target_norm[:, :, :current_conditional_frames] = source_norm
        elif chunk_id > 0 and previous_output is not None:
            current_conditional_frames = min(
                int(batch.sampling_params.num_conditional_frames),
                previous_output.shape[2],
                chunk_frames,
            )
            if current_conditional_frames > 0:
                target_norm[:, :, :current_conditional_frames] = previous_output[
                    :, :, -current_conditional_frames:
                ].to(target_norm)

        if 0 < current_conditional_frames < chunk_frames:
            target_norm[:, :, current_conditional_frames:] = target_norm[
                :, :, current_conditional_frames - 1 : current_conditional_frames
            ].expand(-1, -1, chunk_frames - current_conditional_frames, -1, -1)

        control_latents = [
            self._encode_transfer_video(control, latent_dtype)
            for control in control_norms
        ]
        condition_latents = self._encode_transfer_video(target_norm, latent_dtype)
        noise = torch.randn(
            condition_latents.shape,
            generator=generator,
            device=device,
            dtype=latent_dtype,
        )
        condition_mask = torch.zeros(
            1,
            1,
            condition_latents.shape[2],
            1,
            1,
            device=device,
            dtype=latent_dtype,
        )
        if current_conditional_frames > 0:
            temporal_scale = int(batch.extra["vae_scale_factor_temporal"])
            latent_conditional_frames = (
                current_conditional_frames - 1
            ) // temporal_scale + 1
            condition_mask[:, :, :latent_conditional_frames] = 1.0

        velocity_mask = 1.0 - condition_mask
        batch.latents = condition_mask * condition_latents + velocity_mask * noise
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.extra["video_shape"] = tuple(batch.latents.shape[2:])
        batch.extra["condition_latents"] = condition_mask * condition_latents
        batch.extra["velocity_mask"] = velocity_mask
        batch.extra["control_latents"] = control_latents
        return current_conditional_frames

    def _forward_transfer(self, batch: Req, server_args: ServerArgs) -> Req:
        if self.vae is None:
            raise RuntimeError("Cosmos3 Transfer denoising requires the pipeline VAE")

        device = batch.latents.device
        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)

        plan = batch.extra["transfer_plan"]
        output_chunks = []
        previous_output = None
        for chunk_id in range(int(plan["num_chunks"])):
            with self.use_declared_component(
                component_name="vae",
                module=self.vae,
                phase="transfer_encode",
            ):
                current_conditional_frames = self._prepare_transfer_chunk(
                    batch, chunk_id, previous_output, generator
                )
            self.scheduler.set_timesteps(
                batch.num_inference_steps, device=batch.latents.device
            )
            batch.timesteps = self.scheduler.timesteps
            with self.use_declared_component(
                component_name="transformer",
                module=self.transformer,
                phase="denoise",
            ):
                self._denoise_once(batch, server_args, generator=generator)
            with self.use_declared_component(
                component_name="vae",
                module=self.vae,
                phase="transfer_decode",
            ):
                previous_output = self._decode_transfer_latents(batch.latents).clamp(
                    -1, 1
                )
            if chunk_id == 0:
                output_chunks.append(previous_output)
            else:
                output_chunks.append(previous_output[:, :, current_conditional_frames:])

        batch.extra["transfer_decoded_output"] = torch.cat(output_chunks, dim=2)[
            :, :, : int(plan["total_frames"])
        ]
        return batch

    def _denoise_once(
        self,
        batch: Req,
        server_args: ServerArgs,
        generator: torch.Generator | None = None,
    ) -> Req:
        """Run one denoising loop with CFG and optional conditioning."""
        self._share_vision_temporal_positions = bool(
            getattr(
                batch.sampling_params,
                "share_vision_temporal_positions",
                True,
            )
        )
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

        # Seed the scheduler's stochastic (SDE) noise from the request seed so it
        # is identical on every sequence-parallel rank; otherwise each rank draws
        # its own noise and the sharded latents diverge at the shard boundary.
        if generator is None:
            generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=latents.device).manual_seed(batch.seed)

        cond_text_ids = batch.extra["cond_text_ids"]
        cond_text_mask = batch.extra["cond_text_mask"]
        uncond_text_ids = batch.extra["uncond_text_ids"]
        uncond_text_mask = batch.extra["uncond_text_mask"]
        video_shape = batch.extra["video_shape"]
        fps = batch.extra.get("fps", 24.0)
        velocity_mask = batch.extra.get("velocity_mask")
        condition_latents = batch.extra.get("condition_latents")
        control_latents = batch.extra.get("control_latents")
        guidance_interval = getattr(batch.sampling_params, "guidance_interval", None)
        control_guidance = getattr(batch.sampling_params, "control_guidance", 1.0)
        if control_guidance is None:
            control_guidance = 1.0
        control_guidance_interval = getattr(
            batch.sampling_params, "control_guidance_interval", None
        )

        # Rollout requests carry a per-request scheduler bound by the timestep stage.
        scheduler = batch.scheduler if batch.scheduler is not None else self.scheduler
        if batch.rollout:
            if velocity_mask is not None or condition_latents is not None:
                raise ValueError(
                    "Cosmos3 rollout supports T2V/T2I only; I2V/V2V "
                    "conditioned-frame re-blending breaks the Gaussian "
                    "transition assumption of the SDE log-prob math."
                )
            if action_latents is not None or sound_latents is not None:
                raise ValueError(
                    "Cosmos3 rollout does not support action/sound modalities."
                )
            self._maybe_prepare_rollout(batch)
            self._maybe_init_denoising_env_collection(
                batch=batch,
                pipeline_config=server_args.pipeline_config,
                image_kwargs={},
                pos_cond_kwargs={
                    "text_ids": cond_text_ids,
                    "text_mask": cond_text_mask,
                    "fps": fps,
                },
                neg_cond_kwargs={
                    "text_ids": uncond_text_ids,
                    "text_mask": uncond_text_mask,
                    "fps": fps,
                },
                guidance=None,
            )

        do_cfg = guidance_scale > 1.0
        # Control-CFG runs even when text guidance is off (its own extra
        # control-dropped forward), so it can drive CFG parallel on its own.
        any_control_cfg = control_latents is not None and control_guidance != 1.0

        enable_cfg_parallel = server_args.enable_cfg_parallel and (
            do_cfg or any_control_cfg
        )
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
            # Precision is chosen once per step, before any transformer call,
            # so all CFG branches of the step share the same selection.
            self.transformer.set_denoising_step(step_index=i, num_steps=len(timesteps))
            batch_dim = batch.latents.shape[0] if batch.latents is not None else 1
            timestep = t.unsqueeze(0).expand(batch_dim) if t.dim() == 0 else t
            # Outside the CFG window the effective scale collapses to 1.0,
            # which reduces CFG to the cond branch (cfg-parallel safe).
            effective_scale = (
                guidance_scale if self._cfg_active_at(t, guidance_interval) else 1.0
            )
            # Transfer control-CFG: active only when a control video is present,
            # ``control_guidance != 1.0``, and the step is inside the (optional)
            # control window. It needs a second control-dropped forward, so it
            # owns the prediction for the step and composes text CFG internally.
            control_cfg_active = (
                control_latents is not None
                and control_guidance != 1.0
                and self._cfg_active_at(t, control_guidance_interval)
            )

            if control_cfg_active:
                # Control-CFG owns the step: 2 branches (text guidance off) or 3
                # (text guidance on), distributed across CFG ranks and reduced by
                # ``_predict_noise_cfg`` (sequential per rank, no batching).
                branches = self._control_cfg_branches(
                    cond_text_ids,
                    cond_text_mask,
                    uncond_text_ids,
                    uncond_text_mask,
                    cond_text_seq_len=batch.extra["cond_text_seq_len"],
                    uncond_text_seq_len=batch.extra["uncond_text_seq_len"],
                    control_latents=control_latents,
                    text_guidance_scale=effective_scale,
                    control_guidance_scale=control_guidance,
                )
                noise_pred = self._predict_noise_cfg(
                    branches,
                    latents=latents,
                    timestep=timestep,
                    video_shape=video_shape,
                    fps=fps,
                    cfg_rank=cfg_rank,
                    cfg_world_size=cfg_world_size,
                    noisy_frame_mask=velocity_mask,
                    current_timestep=i,
                    sound_latents=sound_latents,
                    action_latents=action_latents,
                    action_domain_ids=action_domain_ids,
                    action_noisy_mask=action_velocity_mask,
                    action_fps=action_fps,
                    action_start_frame_offset=action_start_frame_offset,
                )
            elif do_cfg and effective_scale != 1.0:
                cond_text_seq_len = batch.extra["cond_text_seq_len"]
                uncond_text_seq_len = batch.extra["uncond_text_seq_len"]
                text_seq_lens_differ = cond_text_seq_len != uncond_text_seq_len
                if (
                    text_seq_lens_differ
                    and not self._logged_cfg_split
                    and not self._current_batch_is_warmup
                ):
                    self._logged_cfg_split = True
                    self.log_info(
                        "Prompt and negative prompt tokenize to different lengths "
                        f"({cond_text_seq_len} vs {uncond_text_seq_len}); running "
                        "the CFG branches in separate forwards to keep padding "
                        "out of the cross-attention"
                    )
                single_cfg_rank_control_free = (
                    cfg_world_size == 1 and control_latents is None
                )
                can_batch_text_cfg = (
                    single_cfg_rank_control_free and not text_seq_lens_differ
                )
                if can_batch_text_cfg:
                    # Single-CFG-rank, control-free text CFG: one batched forward
                    # (lower launch overhead, no control tokens to duplicate).
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
                        max_text_seq_len=cond_text_seq_len,
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
                elif single_cfg_rank_control_free:
                    # Keep each branch at its native text length, but preserve the
                    # canonical CFG operation order used by the batched path. The
                    # algebraically equivalent coefficient sum rounds differently
                    # in BF16 and changes deterministic generation results.
                    noise_pred = self._predict_noise_text_cfg_serial(
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
                        cond_text_seq_len=cond_text_seq_len,
                        uncond_text_seq_len=uncond_text_seq_len,
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
                else:
                    # CFG parallel or control passthrough: distribute unbatched
                    # branches across ranks. Separate forwards preserve each
                    # branch's native text length.
                    branches = self._text_cfg_branches(
                        cond_text_ids,
                        cond_text_mask,
                        uncond_text_ids,
                        uncond_text_mask,
                        guidance_scale=effective_scale,
                        cond_text_seq_len=cond_text_seq_len,
                        uncond_text_seq_len=uncond_text_seq_len,
                        control_latents=control_latents,
                    )
                    noise_pred = self._predict_noise_cfg(
                        branches,
                        latents=latents,
                        timestep=timestep,
                        video_shape=video_shape,
                        fps=fps,
                        cfg_rank=cfg_rank,
                        cfg_world_size=cfg_world_size,
                        noisy_frame_mask=velocity_mask,
                        current_timestep=i,
                        sound_latents=sound_latents,
                        action_latents=action_latents,
                        action_domain_ids=action_domain_ids,
                        action_noisy_mask=action_velocity_mask,
                        action_fps=action_fps,
                        action_start_frame_offset=action_start_frame_offset,
                    )
            else:
                # No CFG this step (guidance off or outside the CFG window): a
                # single conditional forward, run identically on every rank.
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
                    control_latents=control_latents,
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

            if batch.rollout:
                # Capture the pre-step x_{t_i} before the scheduler advances it.
                batch._rollout_loop_step_index = i
                self._maybe_append_dit_trajectory_step(
                    batch=batch,
                    latents=latents,
                    timestep_value=t,
                    step_index=i,
                )
                latents = scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=batch.generator,
                    batch=batch,
                    return_dict=False,
                )[0]
            else:
                latents = scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    generator=generator,
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

        # Hygiene only: the set_denoising_step at each loop head is what
        # actually selects precision, so stale state cannot leak into the
        # next request's steps.
        self.transformer.reset_denoising_step()

        if batch.rollout:
            self._postprocess_rollout_outputs(
                batch=batch,
                latents=latents,
                num_inference_steps=len(timesteps),
                final_timestep=timesteps.new_zeros(()).cpu(),
                server_args=server_args,
            )

        batch.latents = latents
        if action_latents is not None:
            batch.action_latents = action_latents
        if sound_latents is not None:
            batch.audio_latents = sound_latents
        self.log_info("Denoising complete")
        return batch

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.extra.get("transfer_plan") is not None:
            return self._forward_transfer(batch, server_args)
        with self.use_declared_component(
            component_name="transformer",
            module=self.transformer,
            phase="denoise",
        ):
            return self._denoise_once(batch, server_args)

    def _predict_noise_cfg(
        self,
        branches: list[dict],
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float,
        cfg_rank: int,
        cfg_world_size: int,
        noisy_frame_mask: torch.Tensor | None = None,
        current_timestep: int | None = None,
        sound_latents: torch.Tensor | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_fps: float | None = None,
        action_start_frame_offset: int = 1,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Combine CFG branches as the weighted sum ``sum_b coeff_b * f(branch_b)``.

        Both text CFG and transfer control-CFG are linear in their per-branch
        forwards, so each is expressed as a list of branches (text ids/mask,
        control latents, UND cache key, coeff) and reduced here. ``branches`` is
        identical on every CFG rank.

        Branches are distributed round-robin across the ``cfg_world_size`` CFG
        ranks. A rank runs its branches sequentially and a final sum all-reduce
        combines ranks. Forwards are never batched, which preserves each text
        branch's native length and bounds activation memory with control inputs.
        """
        acc = None
        for i, branch in enumerate(branches):
            if i % cfg_world_size != cfg_rank:
                continue
            out = self._run_transformer(
                latents=latents,
                timestep=timestep,
                text_ids=branch["text_ids"],
                text_mask=branch["text_mask"],
                video_shape=video_shape,
                fps=fps,
                cache_key=branch["cache_key"],
                noisy_frame_mask=noisy_frame_mask,
                max_text_seq_len=branch["text_seq_len"],
                current_timestep=current_timestep,
                sound_latents=sound_latents,
                action_latents=action_latents,
                action_domain_ids=action_domain_ids,
                action_noisy_mask=action_noisy_mask,
                action_fps=action_fps,
                action_start_frame_offset=action_start_frame_offset,
                control_latents=branch["control_latents"],
            )
            coeff = branch["coeff"]
            if isinstance(out, tuple):
                scaled = tuple(coeff * prediction for prediction in out)
                acc = (
                    scaled
                    if acc is None
                    else tuple(
                        total + contribution
                        for total, contribution in zip(acc, scaled, strict=True)
                    )
                )
            else:
                scaled = coeff * out
                acc = scaled if acc is None else acc + scaled

        if acc is None:
            # More ranks than branches: contribute zeros to the all-reduce.
            acc = self._zero_like_output(latents, action_latents, sound_latents)

        if cfg_world_size > 1:
            if isinstance(acc, tuple):
                return tuple(
                    cfg_model_parallel_all_reduce(prediction) for prediction in acc
                )
            return cfg_model_parallel_all_reduce(acc)
        return acc

    def _predict_noise_text_cfg_serial(
        self,
        *,
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
        """Run control-free text CFG as separate native-length forwards.

        Keep the canonical uncond + g * (cond - uncond) operation order.
        Although a coefficient-weighted sum is algebraically equivalent, it is
        not numerically equivalent in BF16 and would make results depend on
        whether the two prompts happen to tokenize to the same length.
        """
        common_kwargs = {
            "latents": latents,
            "timestep": timestep,
            "video_shape": video_shape,
            "fps": fps,
            "noisy_frame_mask": noisy_frame_mask,
            "current_timestep": current_timestep,
            "sound_latents": sound_latents,
            "action_latents": action_latents,
            "action_domain_ids": action_domain_ids,
            "action_noisy_mask": action_noisy_mask,
            "action_fps": action_fps,
            "action_start_frame_offset": action_start_frame_offset,
            "control_latents": None,
        }
        cond = self._run_transformer(
            text_ids=cond_text_ids,
            text_mask=cond_text_mask,
            cache_key="cond",
            max_text_seq_len=cond_text_seq_len,
            **common_kwargs,
        )
        uncond = self._run_transformer(
            text_ids=uncond_text_ids,
            text_mask=uncond_text_mask,
            cache_key="uncond",
            max_text_seq_len=uncond_text_seq_len,
            **common_kwargs,
        )

        def _combine(
            cond_pred: torch.Tensor, uncond_pred: torch.Tensor
        ) -> torch.Tensor:
            return uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        if isinstance(cond, tuple):
            return tuple(_combine(c, u) for c, u in zip(cond, uncond, strict=True))
        return _combine(cond, uncond)

    def _predict_noise_cfg_batched(
        self,
        *,
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
        """Run CFG as one ``batch_size=2`` forward stacking both branches (``[uncond, cond]``).

        Kept only for the non-parallel, control-free text path: one batched
        forward has lower kernel-launch overhead than two serial ones, and
        doubling the GEN tokens is cheap here. The caller does not route control
        through this (batching the larger control-in forwards risks OOM on big
        models — that path uses ``_predict_noise_cfg`` instead) and CFG parallel
        splits branches across ranks rather than batching.
        """
        latents_b = torch.cat([latents, latents], dim=0)
        text_ids_b = torch.cat([uncond_text_ids, cond_text_ids], dim=0)
        text_mask_b = torch.cat([uncond_text_mask, cond_text_mask], dim=0)
        timestep_b = torch.cat([timestep, timestep], dim=0)
        mask_b = (
            torch.cat([noisy_frame_mask, noisy_frame_mask], dim=0)
            if noisy_frame_mask is not None
            else None
        )
        sound_b = (
            torch.cat([sound_latents, sound_latents], dim=0)
            if sound_latents is not None
            else None
        )
        action_b = (
            torch.cat([action_latents, action_latents], dim=0)
            if action_latents is not None
            else None
        )
        action_domain_b = (
            torch.cat([action_domain_ids, action_domain_ids], dim=0)
            if action_domain_ids is not None
            else None
        )
        action_mask_b = (
            torch.cat([action_noisy_mask, action_noisy_mask], dim=0)
            if action_noisy_mask is not None
            else None
        )

        out = self._run_transformer(
            latents=latents_b,
            timestep=timestep_b,
            text_ids=text_ids_b,
            text_mask=text_mask_b,
            video_shape=video_shape,
            fps=fps,
            cache_key="cfg_batched",
            noisy_frame_mask=mask_b,
            max_text_seq_len=max_text_seq_len,
            current_timestep=current_timestep,
            sound_latents=sound_b,
            action_latents=action_b,
            action_domain_ids=action_domain_b,
            action_noisy_mask=action_mask_b,
            action_fps=action_fps,
            action_start_frame_offset=action_start_frame_offset,
            control_latents=None,
        )

        def _combine(o: torch.Tensor) -> torch.Tensor:
            uncond, cond = o.chunk(2, dim=0)
            return uncond + guidance_scale * (cond - uncond)

        if isinstance(out, tuple):
            return tuple(_combine(p) for p in out)
        return _combine(out)

    @staticmethod
    def _zero_like_output(
        latents: torch.Tensor,
        action_latents: torch.Tensor | None,
        sound_latents: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Zero prediction matching the forward's (video[, action][, sound]) layout."""
        zeros = [torch.zeros_like(latents)]
        if action_latents is not None:
            zeros.append(torch.zeros_like(action_latents))
        if sound_latents is not None:
            zeros.append(torch.zeros_like(sound_latents))
        return zeros[0] if len(zeros) == 1 else tuple(zeros)

    @staticmethod
    def _text_cfg_branches(
        cond_text_ids: torch.Tensor,
        cond_text_mask: torch.Tensor,
        uncond_text_ids: torch.Tensor,
        uncond_text_mask: torch.Tensor,
        *,
        guidance_scale: float,
        cond_text_seq_len: int | None,
        uncond_text_seq_len: int | None,
        control_latents: list[torch.Tensor] | None,
    ) -> list[dict]:
        """Standard text CFG as two branches: ``g*cond + (1-g)*uncond``.

        Control latents (if any) pass through both branches unchanged — text CFG
        does not drop the control map; that is what control-CFG adds.
        """
        return [
            {
                "cache_key": "cond",
                "text_ids": cond_text_ids,
                "text_mask": cond_text_mask,
                "text_seq_len": cond_text_seq_len,
                "control_latents": control_latents,
                "coeff": guidance_scale,
            },
            {
                "cache_key": "uncond",
                "text_ids": uncond_text_ids,
                "text_mask": uncond_text_mask,
                "text_seq_len": uncond_text_seq_len,
                "control_latents": control_latents,
                "coeff": 1.0 - guidance_scale,
            },
        ]

    @staticmethod
    def _control_cfg_branches(
        cond_text_ids: torch.Tensor,
        cond_text_mask: torch.Tensor,
        uncond_text_ids: torch.Tensor,
        uncond_text_mask: torch.Tensor,
        *,
        cond_text_seq_len: int | None,
        uncond_text_seq_len: int | None,
        control_latents: list[torch.Tensor] | None,
        text_guidance_scale: float,
        control_guidance_scale: float,
    ) -> list[dict]:
        """Transfer control-CFG (optionally composed with text CFG) as branches.

        Two conditional forwards share the cond-text branch but differ in whether
        the control map is packed in:

        - ``cond_full`` — control clips in (the standard transfer forward)
        - ``cond_nc``   — control clips dropped (``control_latents=None``)

        mixed on the generated span as ``cond = cond_nc + cg*(cond_full -
        cond_nc)``. When text CFG is also active a third (uncond, control-in)
        forward composes the text blend ``pred = uncond + g*(cond - uncond)`` on
        top. Expanding both gives the coefficient-weighted sum reduced by
        ``_predict_noise_cfg``::

            pred = g*cg*cond_full + g*(1-cg)*cond_nc + (1-g)*uncond

        (with ``g = 1`` collapsing to ``cg*cond_full + (1-cg)*cond_nc``).

        Branch order places the two control-in forwards first and ``cond_nc``
        second so the round-robin split in ``_predict_noise_cfg`` lands the
        control-in pair on rank 0 and ``cond_nc`` on rank 1 under 2-rank CFG.
        ``cond_full`` and ``cond_nc`` reuse distinct UND cache keys (``"cond"`` /
        ``"cond_nc"``) because their GEN rope layout differs.
        """
        g = text_guidance_scale
        cg = control_guidance_scale
        cond_full = {
            "cache_key": "cond",
            "text_ids": cond_text_ids,
            "text_mask": cond_text_mask,
            "text_seq_len": cond_text_seq_len,
            "control_latents": control_latents,
        }
        cond_nc = {
            "cache_key": "cond_nc",
            "text_ids": cond_text_ids,
            "text_mask": cond_text_mask,
            "text_seq_len": cond_text_seq_len,
            "control_latents": None,
        }
        if g == 1.0:
            cond_full["coeff"] = cg
            cond_nc["coeff"] = 1.0 - cg
            return [cond_full, cond_nc]
        cond_full["coeff"] = g * cg
        cond_nc["coeff"] = g * (1.0 - cg)
        uncond = {
            "cache_key": "uncond",
            "text_ids": uncond_text_ids,
            "text_mask": uncond_text_mask,
            "text_seq_len": uncond_text_seq_len,
            "control_latents": control_latents,
            "coeff": 1.0 - g,
        }
        return [cond_full, cond_nc, uncond]

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses = [
            ComponentUse(
                stage_name,
                "transformer",
                phase="denoise",
                preferred_ready_after_request=True,
                memory_intensive=True,
                start_at_stage_entry=False,
            )
        ]
        if self.vae is not None:
            uses = [
                ComponentUse(
                    stage_name,
                    "vae",
                    phase="transfer_encode",
                    allow_prefetch=False,
                    keep_ready_after_warmup=True,
                    start_at_stage_entry=False,
                ),
                *uses,
                ComponentUse(
                    stage_name,
                    "vae",
                    phase="transfer_decode",
                    allow_prefetch=False,
                    keep_ready_after_warmup=True,
                    start_at_stage_entry=False,
                ),
            ]
        return uses


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

    @staticmethod
    def _compose_transfer_display(output: torch.Tensor, batch: Req) -> torch.Tensor:
        plan = batch.extra.get("transfer_plan")
        if plan is None:
            return output
        total_frames = int(plan["total_frames"])
        if batch.sampling_params.show_control_condition:
            controls = batch.extra["preprocessed_control"]
            controls = controls if isinstance(controls, list) else [controls]
            normalized_controls = [
                control[:, :, :total_frames]
                .to(device=output.device, dtype=output.dtype)
                .div(255.0)
                for control in controls
            ]
            output = torch.cat([*normalized_controls, output], dim=-1)
        source = batch.extra.get("preprocessed_transfer_video")
        if batch.sampling_params.show_input and source is not None:
            normalized_source = (
                source[:, :, :total_frames]
                .to(device=output.device, dtype=output.dtype)
                .div(255.0)
            )
            output = torch.cat([normalized_source, output], dim=-1)
        return output

    def forward(self, batch: Req, server_args: ServerArgs):
        """Decode latents to video, or to a single image for T2I."""
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
            OutputBatch,
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

        action_domain_ids = batch.extra.get("action_domain_ids")
        action_domain_id = (
            int(action_domain_ids[0].item()) if action_domain_ids is not None else None
        )
        action_metadata = {
            "action_mode": getattr(batch.sampling_params, "action_mode", None),
            "action_domain_id": action_domain_id,
            "action_raw_action_dim": (
                batch.extra.get("raw_action_dim")
                if getattr(batch, "extra", None)
                else None
            ),
        }
        if batch.data_type == DataType.ACTION:
            if action_pred is None:
                raise RuntimeError("Cosmos3 action request produced no action tensor")
            payload_actions = (
                action_pred[0] if action_pred.shape[0] == 1 else action_pred
            )
            payload = {
                "request_id": batch.request_id,
                "actions": payload_actions.numpy(),
                "action_mode": action_metadata["action_mode"],
                "domain_id": action_metadata["action_domain_id"],
                "raw_action_dim": action_metadata["action_raw_action_dim"],
                "parameters": {
                    "num_inference_steps": batch.num_inference_steps,
                    "num_frames": batch.num_frames,
                },
            }
            return OutputBatch(
                output=[payload],
                action_pred=action_pred,
                metrics=batch.metrics if hasattr(batch, "metrics") else None,
                **action_metadata,
            )

        is_image_gen = batch.data_type == DataType.IMAGE
        self.log_info(
            "Decoding latents to image..."
            if is_image_gen
            else "Decoding latents to video..."
        )

        device = batch.latents.device
        decoded = batch.extra.get("transfer_decoded_output")
        if decoded is None:
            with self.use_declared_component(component_name="vae", module=self.vae):
                with torch.no_grad():
                    decoded = self._decode_latents(batch.latents)

        self.log_debug("Decoded tensor shape: %s", decoded.shape)
        output = self._postprocess_tensor(decoded)
        output = self._compose_transfer_display(output, batch)

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
            self.log_debug("Postprocessed video tensor shape: %s", output.shape)

        audio = None
        audio_sample_rate = None
        if self.sound_tokenizer is not None and batch.audio_latents is not None:
            with self.use_declared_component(
                component_name="sound_tokenizer", module=self.sound_tokenizer
            ) as sound_tokenizer:
                assert sound_tokenizer is not None
                with torch.no_grad():
                    decoded_audio = sound_tokenizer.decode(
                        batch.audio_latents.to(device)
                    )
            audio = decoded_audio.float().cpu()
            audio_sample_rate = sound_tokenizer.sample_rate
            self.log_debug(
                "Decoded audio tensor shape: %s @ %s Hz",
                tuple(audio.shape),
                audio_sample_rate,
            )

        return OutputBatch(
            output=output,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            action_pred=action_pred,
            metrics=batch.metrics if hasattr(batch, "metrics") else None,
            rollout_trajectory_data=batch.rollout_trajectory_data,
            **action_metadata,
        )

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        stage_name = self._component_stage_name(stage_name)
        uses = [ComponentUse(stage_name, "vae", keep_ready_after_warmup=True)]
        if self.sound_tokenizer is not None:
            uses.append(ComponentUse(stage_name, "sound_tokenizer"))
        return uses
