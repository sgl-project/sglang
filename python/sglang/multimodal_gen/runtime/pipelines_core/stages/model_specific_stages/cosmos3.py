# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 pipeline stages: image preprocess, tokenization, latent / timestep
prep, denoising, decode.

Cosmos3 has no separate text encoder — text is tokenized with Qwen2's chat
template and embedded inside the transformer's UND pathway. The same
``Cosmos3Pipeline`` serves T2V, I2V, and T2I; mode is dispatched per-request
from ``batch.data_type`` and the presence of ``batch.preprocessed_image``.
"""

import copy
import json
import math
from pathlib import Path
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
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
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

# Action helpers. The per-embodiment normalization stats are dataset-derived and
# not shipped in the checkpoint, so the server takes an optional stats file.

# Target (width, height) per resolution tier and aspect ratio.
VIDEO_RES_SIZE_INFO: dict[str, dict[str, tuple[int, int]]] = {
    "256": {"1,1": (256, 256), "4,3": (320, 256), "3,4": (256, 320), "16,9": (320, 192), "9,16": (192, 320)},
    "480": {"1,1": (640, 640), "4,3": (736, 544), "3,4": (544, 736), "16,9": (832, 480), "9,16": (480, 832)},
    "704": {"1,1": (960, 960), "4,3": (1088, 832), "3,4": (832, 1088), "16,9": (1280, 704), "9,16": (704, 1280)},
    "720": {"1,1": (960, 960), "4,3": (1104, 832), "3,4": (832, 1104), "16,9": (1280, 720), "9,16": (720, 1280)},
}

VIEWPOINT_TEMPLATES: dict[str, str] = {
    "ego_view": "This video is captured from a first-person perspective looking at the scene.",
    "third_person_view": "This video is captured from a third-person perspective looking towards the agent from the front.",
    "wrist_view": "This video is captured from a wrist-mounted camera.",
    "concat_view": "This video contains concatenated views from multiple camera perspectives.",
}

# Embodiment -> domain id (selects the per-domain projection weights).
EMBODIMENT_TO_DOMAIN_ID: dict[str, int] = {
    "av": 1,
    "camera_pose": 2,
    "pusht": 4,
    "umi": 6,
    "bridge_orig_lerobot": 7,
    "droid_lerobot": 8,
    "robomind-franka": 8,
    "robomind-franka-dual": 12,
    "robomind-ur": 13,
    "agibotworld": 15,
    "fractal": 20,
}

# Embodiment -> real (unpadded) action channel count. Extra channels are zero-padding.
EMBODIMENT_TO_RAW_ACTION_DIM: dict[str, int] = {
    "av": 9,
    "camera_pose": 9,
    "pusht": 2,
    "umi": 10,
    "bridge_orig_lerobot": 10,
    "droid_lerobot": 10,
    "robomind-franka": 10,
    "robomind-franka-dual": 20,
    "robomind-ur": 10,
    "agibotworld": 29,
    "fractal": 10,
}

_STAT_KEYS = {"mean", "std", "min", "max", "q01", "q99"}


def find_closest_target_size(h: int, w: int, resolution: str | int) -> tuple[int, int]:
    """Closest predefined ``(target_w, target_h)`` for the input aspect ratio."""
    resolution = str(resolution)
    if resolution not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Resolution {resolution!r} not in VIDEO_RES_SIZE_INFO. Available: "
            f"{list(VIDEO_RES_SIZE_INFO)}."
        )
    input_ratio = h / w
    cands = VIDEO_RES_SIZE_INFO[resolution]
    return min(cands.values(), key=lambda wh: abs(input_ratio - wh[1] / wh[0]))


def canonical_aspect_ratio(width: int, height: int) -> str:
    """Canonical ``"W,H"`` aspect string for the action caption."""
    for sizes in VIDEO_RES_SIZE_INFO.values():
        for aspect, (cand_w, cand_h) in sizes.items():
            if width == cand_w and height == cand_h:
                return aspect
    divisor = math.gcd(width, height)
    return f"{width // divisor},{height // divisor}"


def build_action_prompt(
    description: str,
    view_point: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str:
    """Build the Cosmos3 action JSON caption, matching ActionPromptJsonFormatter."""
    duration_seconds = num_frames / fps
    minutes, secs = divmod(round(duration_seconds), 60)
    if description and description[-1] not in ".!?":
        description = description + "."
    prompt = {
        "cinematography": {
            "framing": VIEWPOINT_TEMPLATES.get(view_point, VIEWPOINT_TEMPLATES["ego_view"])
        },
        "actions": [
            {"time": f"0:00-{minutes}:{secs:02d}", "description": description}
        ],
        "duration": f"{int(duration_seconds)}s",
        "fps": float(fps),
        "resolution": {"H": int(height), "W": int(width)},
        "aspect_ratio": canonical_aspect_ratio(int(width), int(height)),
    }
    return json.dumps(prompt)


def get_domain_id(embodiment: str) -> int:
    key = embodiment.lower().strip()
    if key not in EMBODIMENT_TO_DOMAIN_ID:
        raise ValueError(
            f"Unknown Cosmos3 embodiment {embodiment!r}. Expected one of "
            f"{sorted(EMBODIMENT_TO_DOMAIN_ID)}."
        )
    return EMBODIMENT_TO_DOMAIN_ID[key]


def get_raw_action_dim(embodiment: str) -> int:
    key = embodiment.lower().strip()
    if key not in EMBODIMENT_TO_RAW_ACTION_DIM:
        raise ValueError(
            f"No raw action dim for Cosmos3 embodiment {embodiment!r}. Expected one "
            f"of {sorted(EMBODIMENT_TO_RAW_ACTION_DIM)}."
        )
    return EMBODIMENT_TO_RAW_ACTION_DIM[key]


def load_action_stats(stats_path: str, stats_key: str = "global") -> dict[str, torch.Tensor]:
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Action normalization stats not found at {stats_path}.")
    raw = json.loads(path.read_text())
    if stats_key in raw:
        raw = raw[stats_key]
    return {
        k: torch.as_tensor(np.array(v, dtype=np.float32))
        for k, v in raw.items()
        if k in _STAT_KEYS
    }


def normalize_action(action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    if method == "quantile":
        q01, q99 = stats["q01"].to(action), stats["q99"].to(action)
        return (2.0 * (action - q01) / (q99 - q01).clamp(min=1e-8) - 1.0).clamp(-1.0, 1.0)
    if method == "meanstd":
        return (action - stats["mean"].to(action)) / stats["std"].to(action).clamp(min=1e-8)
    if method == "minmax":
        lo, hi = stats["min"].to(action), stats["max"].to(action)
        return (2.0 * (action - lo) / (hi - lo).clamp(min=1e-8) - 1.0).clamp(-1.0, 1.0)
    raise ValueError(f"Unknown action normalization method {method!r}.")


def denormalize_action(action: torch.Tensor, method: str, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    if method == "quantile":
        q01, q99 = stats["q01"].to(action), stats["q99"].to(action)
        return (action + 1.0) / 2.0 * (q99 - q01) + q01
    if method == "meanstd":
        return action * stats["std"].to(action) + stats["mean"].to(action)
    if method == "minmax":
        lo, hi = stats["min"].to(action), stats["max"].to(action)
        return (action + 1.0) / 2.0 * (hi - lo) + lo
    raise ValueError(f"Unknown action normalization method {method!r}.")


class Cosmos3ImagePreprocessStage(PipelineStage):
    """Load, aspect-resize, and center-crop the I2V conditioning image.

    No-op when the request has no image (T2V / T2I). The output is a
    ``[1, 3, H, W]`` tensor in ``[-1, 1]`` written to ``batch.preprocessed_image``.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        return VerificationResult()

    @staticmethod
    def _resize_crop(image: PIL.Image.Image, target_w: int, target_h: int) -> torch.Tensor:
        scale = max(target_w / image.width, target_h / image.height)
        resize_w = int(np.ceil(scale * image.width))
        resize_h = int(np.ceil(scale * image.height))
        image = image.resize((resize_w, resize_h), PIL.Image.Resampling.LANCZOS)
        left = (resize_w - target_w) // 2
        top = (resize_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


    _VIDEO_EXTS = (".mp4", ".avi", ".mov", ".webm", ".mkv")

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if getattr(batch, "action_mode", None) is not None:
            return self._preprocess_action_vision(batch)

        image_path = batch.image_path
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else None
        if not isinstance(image_path, str) or not image_path:
            return batch

        target_h, target_w = batch.height, batch.width
        if Path(image_path).suffix.lower() in self._VIDEO_EXTS:
            return self._preprocess_v2v_vision(batch, image_path, target_h, target_w)

        image = PIL.Image.open(image_path).convert("RGB")
        tensor = self._resize_crop(image, target_w, target_h).unsqueeze(0)
        batch.preprocessed_image = tensor
        self.log_info(f"Preprocessed conditioning image to {target_w}x{target_h}")
        return batch

    def _preprocess_v2v_vision(
        self, batch: Req, vision_path: str, target_h: int, target_w: int
    ) -> Req:
        """V2V: keep the first ``condition_frame_indexes_vision`` latent frames of
        the input video as clean conditioning (default [0, 1])."""
        cond_idx = getattr(batch, "condition_frame_indexes_vision", None) or [0, 1]
        tcf = 4  # Wan2.2 VAE temporal compression
        num_pixel = max(cond_idx) * tcf + 1
        frames = self._read_vision_frames(vision_path, num_pixel)[:num_pixel]
        clip = torch.stack(
            [self._resize_crop(f, target_w, target_h) for f in frames], dim=1
        ).unsqueeze(0)
        batch.extra["v2v_cond_clip"] = clip
        batch.extra["condition_frame_indexes_vision"] = list(cond_idx)
        self.log_info(
            f"Preprocessed V2V conditioning: {len(frames)} pixel frame(s) -> "
            f"{target_w}x{target_h}, condition latent frames={list(cond_idx)}"
        )
        return batch

    def _preprocess_action_vision(self, batch: Req) -> Req:
        """Read the action observation, set the tier resolution and clip length.

        Inverse dynamics keeps the full pixel clip (all frames condition). Policy
        and forward dynamics only need frame 0.
        """
        vision_path = batch.image_path
        if isinstance(vision_path, list):
            vision_path = vision_path[0] if vision_path else None
        if not isinstance(vision_path, str) or not vision_path:
            raise ValueError("Cosmos3 action mode requires image_path (the observation).")

        chunk = getattr(batch, "action_chunk_size", 16)
        num_frames = chunk + 1
        frames = self._read_vision_frames(vision_path, num_frames)
        in_h, in_w = frames[0].height, frames[0].width
        target_w, target_h = find_closest_target_size(
            in_h, in_w, getattr(batch, "action_resolution", "480")
        )

        batch.height = target_h
        batch.width = target_w
        batch.num_frames = num_frames

        clip = torch.stack(
            [self._resize_crop(f, target_w, target_h) for f in frames], dim=1
        ).unsqueeze(0)
        batch.preprocessed_image = clip[:, :, 0]
        if batch.action_mode == "inverse_dynamics":
            batch.extra["action_cond_pixels"] = clip
        self.log_info(
            f"Preprocessed action vision: {len(frames)} frame(s) -> "
            f"{target_w}x{target_h}, mode={batch.action_mode}"
        )
        return batch

    @staticmethod
    def _read_vision_frames(path: str, max_frames: int) -> list[PIL.Image.Image]:
        suffix = Path(path).suffix.lower()
        if suffix in Cosmos3ImagePreprocessStage._VIDEO_EXTS:
            import imageio.v3 as iio

            video = iio.imread(path)
            frames = [
                PIL.Image.fromarray(np.asarray(video[i])).convert("RGB")
                for i in range(min(len(video), max_frames))
            ]
        else:
            frames = [PIL.Image.open(path).convert("RGB")]
        last = frames[-1]
        while len(frames) < max_frames:
            frames.append(last.copy())
        return frames


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

        # Action mode uses a structured JSON caption (no system prompt / suffix).
        if getattr(batch, "action_mode", None) is not None:
            prompt = build_action_prompt(
                prompt,
                getattr(batch, "action_view_point", "ego_view"),
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

    T2V / T2I produce pure Gaussian noise. I2V VAE-encodes the conditioning
    image, replaces frame 0 of the latent with the encoded image, and stashes
    a per-frame velocity mask plus the clean frame-0 latent for the denoiser
    to re-inject after each scheduler step.
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

        is_i2v = (
            batch.preprocessed_image is not None and batch.data_type == DataType.VIDEO
        )
        action_cond_pixels = batch.extra.get("action_cond_pixels")
        v2v_clip = batch.extra.get("v2v_cond_clip")

        if action_cond_pixels is not None:
            # Inverse dynamics: all video frames clean, velocity fully masked.
            vae_dtype = next(self.vae.parameters()).dtype
            with torch.no_grad():
                cond_latent = self._vae_encode(
                    action_cond_pixels.to(device=device, dtype=vae_dtype)
                ).to(dtype)
            latents = cond_latent
            batch.extra["velocity_mask"] = torch.zeros(
                1, 1, num_latent_frames, 1, 1, device=device, dtype=dtype
            )
            self.log_info("Prepared inverse-dynamics latents (all frames clean)")
        elif is_i2v:
            vae_dtype = next(self.vae.parameters()).dtype
            pixel_video = batch.preprocessed_image.unsqueeze(2).to(
                device=device, dtype=vae_dtype
            )
            with torch.no_grad():
                cond_latent = self._vae_encode(pixel_video).to(dtype)

            condition_mask = torch.zeros(
                1, 1, num_latent_frames, 1, 1, device=device, dtype=dtype
            )
            condition_mask[:, :, 0, :, :] = 1.0
            latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
            batch.image_latent = cond_latent[:, :, 0:1, :, :].clone()
            batch.extra["velocity_mask"] = 1.0 - condition_mask
            self.log_info("Prepared I2V latents with frame-0 conditioning")
        elif v2v_clip is not None:
            # V2V: condition_frame_indexes latent frames are clean, rest generated.
            cond_idx = batch.extra.get("condition_frame_indexes_vision", [0, 1])
            cond_idx = [fi for fi in cond_idx if fi < num_latent_frames]
            vae_dtype = next(self.vae.parameters()).dtype
            t_cond = v2v_clip.shape[2]
            pixel_video = torch.zeros(
                1, 3, batch.num_frames, batch.height, batch.width,
                device=device, dtype=vae_dtype,
            )
            t_fill = min(t_cond, batch.num_frames)
            pixel_video[:, :, :t_fill] = v2v_clip[:, :, :t_fill].to(device=device, dtype=vae_dtype)
            if t_fill < batch.num_frames:
                pixel_video[:, :, t_fill:] = pixel_video[:, :, t_fill - 1 : t_fill].expand(
                    -1, -1, batch.num_frames - t_fill, -1, -1
                )
            with torch.no_grad():
                cond_latent = self._vae_encode(pixel_video).to(dtype)
            condition_mask = torch.zeros(
                1, 1, num_latent_frames, 1, 1, device=device, dtype=dtype
            )
            for fi in cond_idx:
                condition_mask[:, :, fi, :, :] = 1.0
            latents = condition_mask * cond_latent + (1.0 - condition_mask) * noise
            batch.image_latent = cond_latent.clone()
            batch.extra["cond_latent_frames"] = cond_idx
            batch.extra["velocity_mask"] = 1.0 - condition_mask
            self.log_info(f"Prepared V2V latents conditioning latent frames {cond_idx}")
        else:
            latents = noise

        batch.latents = latents
        batch.raw_latent_shape = shape

        batch.extra["video_shape"] = (num_latent_frames, height_latent, width_latent)
        batch.extra["vae_scale_factor_temporal"] = vae_scale_factor_temporal
        batch.extra["vae_scale_factor_spatial"] = vae_scale_factor_spatial

        self.log_info(f"Prepared latents with shape {shape}")
        return batch


class Cosmos3SoundLatentPreparationStage(PipelineStage):
    """Prepare the initial sound-latent noise for joint video+audio generation.

    Runs only when the request sets ``generate_sound`` and the checkpoint has a
    sound head. The sound latent is ``[1, sound_dim, T_sound]`` with
    ``T_sound = round(num_frames / fps * sound_latent_fps)``, matching the audio
    AVAE's latent frame rate.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if not getattr(batch, "generate_sound", False):
            return batch
        if not self.transformer.sound_gen:
            raise ValueError(
                "generate_sound was requested but this Cosmos3 checkpoint has no "
                "sound head (sound_gen is false)."
            )
        device = get_local_torch_device()
        fps = batch.fps or 24.0
        t_sound = max(
            1, round(batch.num_frames / fps * self.transformer.sound_latent_fps)
        )
        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(batch.seed)
        shape = (1, self.transformer.sound_dim, t_sound)
        batch.extra["sound_latents"] = torch.randn(
            shape, generator=generator, device=device, dtype=torch.bfloat16
        )
        self.log_info(f"Prepared sound latents with shape {shape}")
        return batch


class Cosmos3ActionLatentPreparationStage(PipelineStage):
    """Prepare the action latent track for forward/inverse dynamics and policy.

    Policy and inverse dynamics generate the action, so the track starts as
    Gaussian noise. Forward dynamics conditions on a clean (normalized) action
    loaded from ``action_path``. Channels beyond the embodiment's raw action dim
    are padding held at zero throughout, and ``action_noisy_mask`` gates the
    time embedding so only generated tokens see the diffusion timestep.
    """

    parallelism_type = StageParallelismType.REPLICATED

    VALID_MODES = ("forward_dynamics", "inverse_dynamics", "policy")

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        mode = getattr(batch, "action_mode", None)
        if mode is None:
            return batch
        if not self.transformer.action_gen:
            raise ValueError(
                "action_mode was requested but this Cosmos3 checkpoint has no "
                "action head (action_gen is false)."
            )
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown Cosmos3 action_mode {mode!r}. Expected one of "
                f"{list(self.VALID_MODES)}."
            )

        device = get_local_torch_device()
        dtype = torch.bfloat16
        embodiment = getattr(batch, "action_embodiment", "droid_lerobot")
        domain_id = get_domain_id(embodiment)
        raw_dim = get_raw_action_dim(embodiment)
        max_dim = self.transformer.action_dim
        t_action = getattr(batch, "action_chunk_size", 16)
        is_noisy = mode != "forward_dynamics"

        if is_noisy:
            generator = batch.generator
            if generator is None and batch.seed is not None:
                generator = torch.Generator(device=device).manual_seed(batch.seed)
            action_latents = torch.randn(
                (1, t_action, max_dim),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            action_latents[..., raw_dim:] = 0
            noisy_mask = torch.ones(1, t_action, 1, device=device, dtype=dtype)
        else:
            action = self._load_clean_action(batch, t_action, raw_dim)
            action_latents = torch.zeros(
                1, t_action, max_dim, device=device, dtype=dtype
            )
            action_latents[:, :, :raw_dim] = action.to(device=device, dtype=dtype)
            noisy_mask = torch.zeros(1, t_action, 1, device=device, dtype=dtype)

        batch.extra["action_latents"] = action_latents
        batch.extra["action_domain_id"] = torch.tensor(
            [domain_id], device=device, dtype=torch.long
        )
        batch.extra["action_noisy_mask"] = noisy_mask
        batch.extra["action_raw_dim"] = raw_dim
        batch.extra["action_is_noisy"] = is_noisy
        self.log_info(
            f"Prepared action latents: mode={mode} embodiment={embodiment} "
            f"T_action={t_action} raw_dim={raw_dim} noisy={is_noisy}"
        )
        return batch

    def _load_clean_action(
        self, batch: Req, t_action: int, raw_dim: int
    ) -> torch.Tensor:
        path = getattr(batch, "action_path", None)
        if path is None:
            raise ValueError(
                "forward_dynamics requires action_path (a JSON action array)."
            )
        raw = torch.tensor(json.loads(Path(path).read_text()), dtype=torch.float32)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.shape[-1] != raw_dim:
            raise ValueError(
                f"Action file last dim {raw.shape[-1]} does not match embodiment "
                f"raw action dim {raw_dim}."
            )
        if raw.shape[0] < t_action:
            pad = raw[-1:].repeat(t_action - raw.shape[0], 1)
            raw = torch.cat([raw, pad], dim=0)
        elif raw.shape[0] > t_action:
            raw = raw[:t_action]
        stats_path = getattr(batch, "action_stats_path", None)
        if stats_path is not None:
            method = getattr(batch, "action_normalization", "quantile")
            raw = normalize_action(raw, method, load_action_stats(stats_path))
        return raw


class Cosmos3TimestepPreparationStage(PipelineStage):
    """
    Timestep preparation stage for Cosmos3.

    Sets up the diffusion scheduler timesteps.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, scheduler, transformer=None):
        super().__init__()
        self.scheduler = scheduler
        # Super (hidden_size 5120) uses a flat shift. Nano (4096) is per-mode.
        self.is_super = getattr(transformer, "hidden_size", 0) >= 5120

    def _default_shift_for_mode(self, batch: Req) -> float:
        """Native per-mode FlowUniPC sampling shift. Nano: T2I 3, I2V 5, T2V 10.
        Super: a flat 5 for all modes. Action: a flat 10 across modes.

        IMAGE data_type is T2I, a set conditioning image is I2V, and the
        remaining video case is T2V.
        """
        if getattr(batch, "action_mode", None) is not None:
            return 10.0
        if self.is_super:
            return 5.0
        if batch.data_type == DataType.IMAGE:
            return 3.0
        if batch.preprocessed_image is not None:
            return 5.0
        return 10.0

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Prepare scheduler timesteps."""
        device = get_local_torch_device()
        num_inference_steps = batch.num_inference_steps
        flow_shift = getattr(batch, "flow_shift", None)
        if flow_shift is None:
            flow_shift = server_args.pipeline_config.flow_shift
        if flow_shift is None:
            flow_shift = self._default_shift_for_mode(batch)
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
        sound_latent: torch.Tensor | None = None,
        action_latent: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
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
            noisy_frame_mask: Optional [B, 1, T, 1, 1] I2V conditioning mask.
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
                sound_latent=sound_latent,
                action_latent=action_latent,
                action_domain_id=action_domain_id,
                action_noisy_mask=action_noisy_mask,
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
        timesteps = batch.timesteps
        guidance_scale = batch.guidance_scale

        cond_text_ids = batch.extra["cond_text_ids"]
        cond_text_mask = batch.extra["cond_text_mask"]
        uncond_text_ids = batch.extra["uncond_text_ids"]
        uncond_text_mask = batch.extra["uncond_text_mask"]
        video_shape = batch.extra["video_shape"]
        fps = batch.extra.get("fps", 24.0)
        velocity_mask = batch.extra.get("velocity_mask")
        image_latent = batch.image_latent
        cond_latent_frames = batch.extra.get("cond_latent_frames")
        guidance_interval = getattr(batch.sampling_params, "guidance_interval", None)

        do_cfg = guidance_scale > 1.0

        enable_cfg_parallel = server_args.enable_cfg_parallel and do_cfg
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

        # Sound runs a parallel latent track on its own UniPC, stepped in lockstep.
        sound_latents = batch.extra.get("sound_latents")
        generate_sound = sound_latents is not None
        if generate_sound and (enable_cfg_parallel or ulysses_enabled):
            raise NotImplementedError(
                "Cosmos3 sound generation requires a single GPU (no cfg-parallel "
                "or sequence parallelism)."
            )
        sound_scheduler = copy.deepcopy(self.scheduler) if generate_sound else None
        if generate_sound:
            self.log_info(
                f"Joint sound generation enabled (T_sound={sound_latents.shape[-1]})"
            )

        action_latents = batch.extra.get("action_latents")
        has_action = action_latents is not None
        if has_action and (enable_cfg_parallel or ulysses_enabled):
            raise NotImplementedError(
                "Cosmos3 action conditioning requires a single GPU."
            )
        action_domain_id = batch.extra.get("action_domain_id")
        action_noisy_mask = batch.extra.get("action_noisy_mask")
        action_raw_dim = batch.extra.get("action_raw_dim")
        action_is_noisy = batch.extra.get("action_is_noisy", True)
        action_scheduler = (
            copy.deepcopy(self.scheduler) if has_action and action_is_noisy else None
        )
        if has_action:
            self.log_info(
                f"Action mode enabled (T_action={action_latents.shape[1]}, "
                f"noisy={action_is_noisy})"
            )

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
                    prediction = self._predict_noise_cfg_parallel(
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
                    )
                elif effective_scale == 1.0:
                    prediction = self._run_transformer(
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
                        sound_latent=sound_latents,
                        action_latent=action_latents,
                        action_domain_id=action_domain_id,
                        action_noisy_mask=action_noisy_mask,
                    )
                elif generate_sound or has_action:
                    # Omni runs serial CFG: the batched path returns one tensor.
                    prediction = self._predict_noise_cfg_serial(
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
                        cond_text_seq_len=batch.extra["cond_text_seq_len"],
                        uncond_text_seq_len=batch.extra["uncond_text_seq_len"],
                        sound_latent=sound_latents,
                        action_latent=action_latents,
                        action_domain_id=action_domain_id,
                        action_noisy_mask=action_noisy_mask,
                        current_timestep=i,
                    )
                else:
                    prediction = self._predict_noise_cfg_batched(
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
                    )
            else:
                prediction = self._run_transformer(
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
                    sound_latent=sound_latents,
                    action_latent=action_latents,
                    action_domain_id=action_domain_id,
                    action_noisy_mask=action_noisy_mask,
                )

            # The transformer returns (video, [sound], [action]) in that order.
            parts = list(prediction) if isinstance(prediction, tuple) else [prediction]
            noise_pred = parts[0]
            idx = 1
            sound_pred = None
            action_pred = None
            if generate_sound:
                sound_pred = parts[idx]
                idx += 1
            if has_action:
                action_pred = parts[idx]
                idx += 1

            # I2V: zero-velocity at conditioned frames so the scheduler keeps
            # them clean; UniPC's predictor-corrector still rescales the
            # sample, so we re-inject the clean image latent below.
            if velocity_mask is not None:
                noise_pred = noise_pred * velocity_mask

            latents = self.scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=False,
            )[0]

            if image_latent is not None:
                if cond_latent_frames is not None:
                    for fi in cond_latent_frames:
                        latents[:, :, fi : fi + 1, :, :] = image_latent[:, :, fi : fi + 1, :, :]
                else:
                    latents[:, :, 0:1, :, :] = image_latent

            if generate_sound:
                sound_latents = sound_scheduler.step(
                    sound_pred, t, sound_latents, return_dict=False
                )[0]

            # Forward dynamics conditions on a clean action, so only the noisy
            # action track (policy, inverse dynamics) is denoised.
            if has_action and action_is_noisy:
                if action_raw_dim is not None:
                    action_pred[..., action_raw_dim:] = 0
                action_latents = action_scheduler.step(
                    action_pred, t, action_latents, return_dict=False
                )[0]
                if action_raw_dim is not None:
                    action_latents[..., action_raw_dim:] = 0

            if batch.profile and not batch.is_warmup:
                self.step_profile()

        batch.latents = latents
        if generate_sound:
            batch.extra["sound_latents"] = sound_latents
        if has_action:
            batch.extra["action_latents"] = action_latents
        self.log_info("Denoising complete")
        return batch

    def _predict_noise_cfg_serial(
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
        cond_text_seq_len: int | None = None,
        uncond_text_seq_len: int | None = None,
        sound_latent: torch.Tensor | None = None,
        action_latent: torch.Tensor | None = None,
        action_domain_id: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        current_timestep: int | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """CFG as two batch=1 forwards, combining every returned velocity
        component (video, and any of sound / action)."""
        extra = dict(
            sound_latent=sound_latent,
            action_latent=action_latent,
            action_domain_id=action_domain_id,
            action_noisy_mask=action_noisy_mask,
            current_timestep=current_timestep,
        )
        out_cond = self._run_transformer(
            latents=latents, timestep=timestep, text_ids=cond_text_ids,
            text_mask=cond_text_mask, video_shape=video_shape, fps=fps,
            cache_key="cond", noisy_frame_mask=noisy_frame_mask,
            max_text_seq_len=cond_text_seq_len, **extra,
        )
        out_uncond = self._run_transformer(
            latents=latents, timestep=timestep, text_ids=uncond_text_ids,
            text_mask=uncond_text_mask, video_shape=video_shape, fps=fps,
            cache_key="uncond", noisy_frame_mask=noisy_frame_mask,
            max_text_seq_len=uncond_text_seq_len, **extra,
        )
        cond = out_cond if isinstance(out_cond, tuple) else (out_cond,)
        uncond = out_uncond if isinstance(out_uncond, tuple) else (out_uncond,)
        return tuple(u + guidance_scale * (c - u) for c, u in zip(cond, uncond))

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
    ) -> torch.Tensor:
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

        noise_pred = self._run_transformer(
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
        )

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
        # CFG: uncond + g·(cond − uncond).
        return noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

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
    ) -> torch.Tensor:
        """Run CFG with one branch per CFG rank, combined by all-reduce.

        Rank 0 runs the conditional branch and contributes ``g·cond`` to the
        sum; rank 1 runs the unconditional branch and contributes
        ``(1−g)·uncond``. The all-reduce sum is exactly the standard CFG
        result. Each rank keeps its own UND K/V cache (``"cond"`` /
        ``"uncond"``).
        """
        if cfg_rank == 0:
            noise_pred = self._run_transformer(
                latents=latents,
                timestep=timestep,
                text_ids=cond_text_ids,
                text_mask=cond_text_mask,
                video_shape=video_shape,
                fps=fps,
                cache_key="cond",
                noisy_frame_mask=noisy_frame_mask,
                max_text_seq_len=cond_text_seq_len,
                current_timestep=current_timestep,
            )
            partial = guidance_scale * noise_pred
        else:
            noise_pred = self._run_transformer(
                latents=latents,
                timestep=timestep,
                text_ids=uncond_text_ids,
                text_mask=uncond_text_mask,
                video_shape=video_shape,
                fps=fps,
                cache_key="uncond",
                noisy_frame_mask=noisy_frame_mask,
                max_text_seq_len=uncond_text_seq_len,
                current_timestep=current_timestep,
            )
            partial = (1.0 - guidance_scale) * noise_pred

        return cfg_model_parallel_all_reduce(partial)


class Cosmos3DecodingStage(PipelineStage):
    """
    VAE decoding stage for Cosmos3.

    Decodes latents to pixel space using the VAE.
    Returns OutputBatch instead of Req to signal pipeline completion.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, vae, guardrails: bool = False, sound_vae=None):
        super().__init__()
        self.vae = vae
        self.sound_vae = sound_vae
        self._latents_mean = None
        self._latents_std = None
        self._guardrails = guardrails
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

        result = OutputBatch(
            output=output,
            metrics=batch.metrics if hasattr(batch, "metrics") else None,
        )

        # Decode the joint sound latent (if any). The save path muxes it as AAC.
        sound_latents = batch.extra.get("sound_latents")
        if sound_latents is not None and self.sound_vae is not None:
            sv_dtype = next(self.sound_vae.parameters()).dtype
            sv_device = next(self.sound_vae.parameters()).device
            with torch.no_grad():
                waveform = self.sound_vae.decode(
                    sound_latents.to(device=sv_device, dtype=sv_dtype)
                )
            # Keep the batch dim [B, 2, N] so the save path indexes per-output
            # audio and preserves both channels (indexing [0] here collapsed the
            # stereo pair to mono downstream).
            result.audio = waveform.float().cpu()
            result.audio_sample_rate = self.sound_vae.sample_rate

        # Slice the predicted action latent back to the embodiment's raw
        # channels and de-normalize into physical units when stats are given.
        action_latents = batch.extra.get("action_latents")
        if action_latents is not None:
            raw_dim = batch.extra.get("action_raw_dim", action_latents.shape[-1])
            action = action_latents[..., :raw_dim].float().cpu()
            stats_path = getattr(batch, "action_stats_path", None)
            if stats_path is not None:
                method = getattr(batch, "action_normalization", "quantile")
                action = denormalize_action(
                    action, method, load_action_stats(stats_path)
                )
            result.action = action

        return result
