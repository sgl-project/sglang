# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams-Transfer stages: control-video conditioned causal rollout.

Mirrors imaginaire4's ``video_transfer`` autoregressive path. The control clip
(edge, blur, depth, or segmentation video) is resized to a trained canvas,
VAE-encoded, and committed chunk by chunk as clean vision K/V that shares the
target frames' temporal positions. Each target chunk is then denoised against
``[text | history | current control chunk]`` and its clean frames are
committed afterwards. Nothing is evicted; the artifact declares
``no_eviction``.
"""

import json
import math
from typing import Any

import imageio.v3 as iio
import msgspec
import numpy as np
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    CONTROL_VIDEO_CONDITIONING_MODE,
    TEXT_TOKENS_TRAINING_MAX,
    CosmosDreamsManifest,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos_dreams import (
    CosmosDreamsTransformer,
    KVPair,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    canonical_aspect_ratio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos_dreams import (
    EXTRA_GEOMETRY,
    EXTRA_TARGET_LATENT_FRAMES,
    EXTRA_TEXT_IDS,
    EXTRA_TEXT_MASK,
    CosmosDreamsGeometry,
    closest_canvas,
    commit_clean_kv,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    latent_frame_count,
    resolve_geometry,
    run_fixed_step_sde,
    single_prompt,
    validate_fixed_step_scheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_video

logger = init_logger(__name__)

EXTRA_CONTROL_LATENT = "cosmos_dreams_control_latent"
EXTRA_CONTROL_HINT = "cosmos_dreams_control_hint"
EXTRA_CONTROL_FPS = "cosmos_dreams_control_fps"
# Read by Cosmos3DecodingStage to place the control clip beside the output.
EXTRA_PREPROCESSED_CONTROL = "preprocessed_control"
EXTRA_TRANSFER_PLAN = "transfer_plan"

TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates images or videos following the "
    "user's instructions and control signals (edge maps, blur, depth, or segmentation)."
)
# Caption augmentors of the Transfer training data (imaginaire4
# duration_fps_text_timestamps.py and resolution_text_info.py defaults).
DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
CONTROL_EMPHASIS_TEMPLATE = (
    " Follow the {hint} control video precisely: shape, contour, silhouette, position,"
    " and motion of every visible structure must align with the {hint} signal at every frame."
)
# Longest clip of the legacy chunkwise 480p Transfer recipe (601 pixel frames).
MAX_TRAINED_LATENT_FRAMES = 151


def align_pixel_frames_to_chunks(
    num_pixel_frames: int, *, temporal_compression_factor: int, chunk_size: int
) -> int:
    """Largest pixel-frame count whose latents fit the trained ``[1, C, C, ...]`` partition."""
    latent = latent_frame_count(num_pixel_frames, temporal_compression_factor)
    if latent < 1 + chunk_size:
        raise ValueError(
            "Cosmos-Dreams-Transfer needs at least one full latent chunk after the first "
            f"frame: {num_pixel_frames} pixel frames give {latent} latent frames but "
            f"{1 + chunk_size} are required "
            f"({1 + chunk_size * temporal_compression_factor} pixel frames)."
        )
    aligned_latent = 1 + ((latent - 1) // chunk_size) * chunk_size
    return 1 + (aligned_latent - 1) * temporal_compression_factor


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def format_transfer_prompt(
    prompt: str,
    *,
    hint: str,
    fps: int,
    num_frames: int,
    height: int,
    width: int,
    emphasize_control: bool,
) -> str:
    """Caption as the Transfer training data saw it.

    JSON-object prompts receive ``duration``/``fps``/``resolution``/``aspect_ratio``
    fields; plain prompts receive the duration and resolution sentences. The
    control-adherence sentence names the hint so the model gets the exact
    control type.
    """
    if fps <= 0 or num_frames <= 0:
        raise ValueError(
            f"Cosmos-Dreams-Transfer prompt metadata needs positive fps and num_frames, got fps={fps}, num_frames={num_frames}."
        )
    text = prompt.strip()
    parsed = _json_object(text)
    if parsed is not None:
        parsed.update(
            {
                "duration": f"{int(num_frames / fps)}s",
                "fps": float(fps),
                "resolution": {"H": int(height), "W": int(width)},
                "aspect_ratio": canonical_aspect_ratio(int(width), int(height)),
            }
        )
        text = json.dumps(parsed)
    else:
        sentences = (
            DURATION_TEMPLATE.format(duration=num_frames / fps, fps=fps),
            RESOLUTION_TEMPLATE.format(height=height, width=width),
        )
        for sentence in sentences:
            text = f"{text.rstrip('.')}. {sentence}" if text else sentence
    if emphasize_control:
        text = text.rstrip() + CONTROL_EMPHASIS_TEMPLATE.format(hint=hint)
    return text


def tokenize_transfer_prompt(
    tokenizer: Any, prompt: str, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2 chat template with the Transfer system prompt, plus EOS and vision start."""
    result = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": TRANSFER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = list(result) if isinstance(result, list) else list(result["input_ids"])
    token_ids = token_ids[:TEXT_TOKENS_TRAINING_MAX]
    token_ids.append(tokenizer.eos_token_id)
    token_ids.append(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return text_ids, torch.ones_like(text_ids)


def read_video_fps(path: str) -> float | None:
    """Frame rate from the container metadata, or None when the backend has none."""
    try:
        fps = iio.immeta(path).get("fps")
    except (OSError, ValueError, RuntimeError, KeyError):
        return None
    return float(fps) if fps else None


def load_control_frames(path: str, *, max_frames: int) -> torch.Tensor:
    """First ``max_frames`` frames of the control clip as ``uint8 [3, T, H, W]``."""
    frames = load_video(path)
    if not frames:
        raise ValueError(f"No frames decoded from control video {path!r}.")
    arrays = [
        np.asarray(frame.convert("RGB"), dtype=np.uint8)
        for frame in frames[:max_frames]
    ]
    return torch.from_numpy(np.stack(arrays, axis=0)).permute(3, 0, 1, 2).contiguous()


def resize_center_crop_uint8(
    frames: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Aspect-preserving resize then center crop of ``uint8 [3, T, H, W]`` frames."""
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Control frames must have shape [3, T, H, W], got {tuple(frames.shape)}."
        )
    orig_h, orig_w = int(frames.shape[2]), int(frames.shape[3])
    scale = max(width / orig_w, height / orig_h)
    resize_h = math.ceil(scale * orig_h)
    resize_w = math.ceil(scale * orig_w)
    resized = F.interpolate(
        frames.permute(1, 0, 2, 3).to(torch.float32),
        size=(resize_h, resize_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resize_h - height) // 2
    left = (resize_w - width) // 2
    cropped = resized[:, :, top : top + height, left : left + width]
    return (
        cropped.round().clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()
    )


def synthetic_control_frames(
    *, num_frames: int, height: int, width: int
) -> torch.Tensor:
    """Black ``uint8 [3, T, H, W]`` clip standing in for a control video on warmup requests."""
    if num_frames <= 0 or height <= 0 or width <= 0:
        raise ValueError(
            f"Synthetic control frames need positive dimensions, got {num_frames}x{height}x{width}."
        )
    return torch.zeros((3, num_frames, height, width), dtype=torch.uint8)


def encode_control_latent(
    vae: Any, frames: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """VAE-encode ``uint8 [3, T, H, W]`` frames to normalized bf16 ``[1, C, T_lat, h, w]``."""
    vae_dtype = next(vae.parameters()).dtype
    # The reference normalizes to [-1, 1] directly in the checkpoint dtype.
    pixels = frames.unsqueeze(0).to(device=device, dtype=vae_dtype) / 127.5 - 1.0
    with torch.no_grad():
        latent = vae.encode(pixels).mode()
    mean = torch.as_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latent)
    std = torch.as_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent)
    return ((latent - mean) / std).to(torch.bfloat16)


class CosmosDreamsControlVideoStage(PipelineStage):
    """Load the control clip, snap the canvas, and trim it to the chunk partition."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self, *, manifest: CosmosDreamsManifest, canvas_tier: str, max_pixels: int
    ) -> None:
        super().__init__()
        self.manifest = manifest
        self.canvas_tier = canvas_tier
        self.max_pixels = max_pixels

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def _resolve_fps(self, batch: Req, control_path: str) -> float:
        if batch.sampling_params.is_explicit("fps"):
            return float(batch.fps)
        fps = read_video_fps(control_path)
        if fps is None:
            raise ValueError(
                f"Could not read the frame rate of {control_path!r}; pass fps explicitly."
            )
        return fps

    def _load_control(self, batch: Req) -> tuple[torch.Tensor, str, float, str]:
        """Return ``(frames, hint, fps, source)``; warmup requests get a synthetic clip."""
        manifest = self.manifest
        sampling_params = batch.sampling_params
        if batch.is_warmup:
            frames = synthetic_control_frames(
                num_frames=int(batch.num_frames), height=batch.height, width=batch.width
            )
            return (
                frames,
                manifest.control_contract.hints[0],
                float(batch.fps),
                "<warmup>",
            )
        control_path = sampling_params.resolved_control_path
        hint = sampling_params.resolved_control_hint
        allowed_hints = manifest.control_contract.hints
        if hint not in allowed_hints:
            raise ValueError(
                f"Checkpoint {manifest.checkpoint_id} accepts control hints {list(allowed_hints)}, got {hint!r}."
            )
        frames = load_control_frames(
            control_path, max_frames=int(sampling_params.max_frames)
        )
        return frames, hint, self._resolve_fps(batch, control_path), control_path

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        manifest = self.manifest
        sampling_params = batch.sampling_params
        frames, hint, fps, control_path = self._load_control(batch)
        if sampling_params.canvas_from_control:
            canvas_width, canvas_height = closest_canvas(
                height=int(frames.shape[2]),
                width=int(frames.shape[3]),
                tier=self.canvas_tier,
            )
            batch.height, batch.width = canvas_height, canvas_width
        frames = resize_center_crop_uint8(
            frames, height=batch.height, width=batch.width
        )
        total_frames = align_pixel_frames_to_chunks(
            min(int(frames.shape[1]), int(batch.num_frames)),
            temporal_compression_factor=manifest.temporal_compression_factor,
            chunk_size=manifest.chunk_size,
        )
        frames = frames[:, :total_frames]
        geometry = resolve_geometry(
            height=batch.height,
            width=batch.width,
            manifest=manifest,
            max_pixels=self.max_pixels,
        )
        target_frame = latent_frame_count(
            total_frames, manifest.temporal_compression_factor
        )
        if target_frame > MAX_TRAINED_LATENT_FRAMES:
            self.log_warning(
                f"{target_frame} latent frames exceed the trained Transfer horizon of "
                f"{MAX_TRAINED_LATENT_FRAMES}; quality may degrade past it."
            )
        batch.num_frames = total_frames
        batch.fps = int(round(fps))
        batch.extra[EXTRA_CONTROL_FPS] = fps
        batch.extra[EXTRA_CONTROL_HINT] = hint
        batch.extra[EXTRA_PREPROCESSED_CONTROL] = [frames.unsqueeze(0)]
        chunk_pixel_frames = manifest.chunk_size * manifest.temporal_compression_factor
        batch.extra[EXTRA_TRANSFER_PLAN] = {
            "total_frames": total_frames,
            "chunk_frames": chunk_pixel_frames,
            "num_chunks": 1 + (target_frame - 1) // manifest.chunk_size,
            "stride": chunk_pixel_frames,
        }
        batch.extra[EXTRA_GEOMETRY] = geometry
        batch.extra[EXTRA_TARGET_LATENT_FRAMES] = target_frame
        self.log_info(
            f"Control clip {control_path} ({hint}): {total_frames} frames at {fps:.3g} fps on "
            f"canvas {batch.height}x{batch.width}, {target_frame} latent frames"
        )
        return batch


class CosmosDreamsTransferPrepareStage(PipelineStage):
    """Format and tokenize the prompt; VAE-encode the control clip."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self, *, vae: Any, tokenizer: Any, manifest: CosmosDreamsManifest
    ) -> None:
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.manifest = manifest

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("geometry", batch.extra.get(EXTRA_GEOMETRY), V.not_none)
        result.add_check(
            "control_frames", batch.extra.get(EXTRA_PREPROCESSED_CONTROL), V.not_none
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]

    def _format_prompt(self, batch: Req, geometry: CosmosDreamsGeometry) -> str:
        emphasize = batch.sampling_params.emphasize_control_in_prompt
        if emphasize is None:
            emphasize = self.manifest.control_contract.emphasize_control_in_prompt
        return format_transfer_prompt(
            single_prompt(batch.prompt),
            hint=batch.extra[EXTRA_CONTROL_HINT],
            fps=int(batch.fps),
            num_frames=int(batch.num_frames),
            height=geometry.height,
            width=geometry.width,
            emphasize_control=emphasize,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        manifest = self.manifest
        geometry: CosmosDreamsGeometry = batch.extra[EXTRA_GEOMETRY]
        target_frame: int = batch.extra[EXTRA_TARGET_LATENT_FRAMES]

        prompt = self._format_prompt(batch, geometry)
        self.log_info(f"Prompt: {prompt}")
        text_ids, text_mask = tokenize_transfer_prompt(
            self.tokenizer, prompt, device=device
        )

        frames = batch.extra[EXTRA_PREPROCESSED_CONTROL][0][0]
        with self.use_declared_component(component_name="vae", module=self.vae):
            latent = encode_control_latent(self.vae, frames, device=device)
        expected = (
            1,
            latent.shape[1],
            target_frame,
            geometry.latent_height,
            geometry.latent_width,
        )
        if tuple(latent.shape) != expected:
            raise ValueError(
                "Encoded control clip does not match the resolved geometry: "
                f"expected {expected}, got {tuple(latent.shape)}."
            )
        batch.extra[EXTRA_CONTROL_LATENT] = latent
        batch.extra[EXTRA_TEXT_IDS] = text_ids
        batch.extra[EXTRA_TEXT_MASK] = text_mask
        batch.raw_latent_shape = expected
        self.log_info(
            f"Prepared Cosmos-Dreams-Transfer request: {geometry.height}x{geometry.width}, "
            f"{target_frame} latent frames, {text_ids.shape[1]} text tokens"
        )
        return batch


class _TransferRolloutContext(msgspec.Struct):
    """Per-request constants shared by every chunk of one rollout."""

    text_kv: list[KVPair]
    fps: float
    tokens_per_frame: int
    latent_channels: int
    geometry: CosmosDreamsGeometry
    device: torch.device
    dtype: torch.dtype


class CosmosDreamsTransferRolloutStage(PipelineStage):
    """Chunked denoising against committed control and clean-RGB K/V history."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        *,
        transformer: CosmosDreamsTransformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        manifest: CosmosDreamsManifest,
    ) -> None:
        super().__init__()
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"Cosmos-Dreams requires CosmosDreamsTransformer, got {type(transformer).__name__}."
            )
        if manifest.conditioning_mode != CONTROL_VIDEO_CONDITIONING_MODE:
            raise ValueError(
                "CosmosDreamsTransferRolloutStage needs a control_video checkpoint, got "
                f"{manifest.conditioning_mode!r}."
            )
        self.transformer = transformer
        self.scheduler = validate_fixed_step_scheduler(scheduler, manifest)
        self.manifest = manifest

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("text_ids", batch.extra.get(EXTRA_TEXT_IDS), V.is_tensor)
        result.add_check(
            "control_latent", batch.extra.get(EXTRA_CONTROL_LATENT), V.is_tensor
        )
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [
            ComponentUse(
                self._component_stage_name(stage_name),
                "transformer",
                phase="denoise",
                preferred_ready_after_request=True,
                memory_intensive=True,
                start_at_stage_entry=False,
            )
        ]

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise"
        ):
            with torch.no_grad():
                batch.latents = self._rollout(batch)
        return batch

    def _commit(
        self,
        context: _TransferRolloutContext,
        history: list[KVPair] | None,
        latent: torch.Tensor,
        *,
        frame_start: int,
    ) -> list[KVPair]:
        # no_eviction: every control/RGB pair stays for the whole rollout.
        return commit_clean_kv(
            self.transformer,
            latent,
            history=history,
            dtype=context.dtype,
            tokens_per_frame=context.tokens_per_frame,
            sink_frames=0,
            window_frames=None,
            text_kv=context.text_kv,
            frame_start=frame_start,
            fps=context.fps,
        )

    def _rollout(self, batch: Req) -> torch.Tensor:
        manifest = self.manifest
        device = get_local_torch_device()
        text_kv, _ = self.transformer.encode_und_kv(
            batch.extra[EXTRA_TEXT_IDS], batch.extra[EXTRA_TEXT_MASK]
        )
        geometry: CosmosDreamsGeometry = batch.extra[EXTRA_GEOMETRY]
        context = _TransferRolloutContext(
            text_kv=text_kv,
            fps=float(batch.extra[EXTRA_CONTROL_FPS]),
            tokens_per_frame=geometry.vision_tokens_per_frame,
            latent_channels=self.transformer.latent_channel,
            geometry=geometry,
            device=device,
            dtype=torch.bfloat16,
        )
        control: torch.Tensor = batch.extra[EXTRA_CONTROL_LATENT].to(
            device=device, dtype=context.dtype
        )
        target_frame: int = batch.extra[EXTRA_TARGET_LATENT_FRAMES]
        seed = batch.seed
        if not isinstance(seed, int):
            raise ValueError(
                f"Cosmos-Dreams requires a single integer seed, got {seed!r}."
            )

        history: list[KVPair] | None = None
        latents: list[torch.Tensor] = []
        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            0, target_frame, manifest.chunk_size
        ):
            history = self._commit(
                context,
                history,
                control[:, :, chunk_start:chunk_end],
                frame_start=chunk_start,
            )
            generator = torch.Generator(device=device).manual_seed(seed + chunk_start)
            # The reference draws checkpoint-dtype noise, then promotes it to fp32.
            noise = torch.randn(
                (
                    1,
                    context.latent_channels,
                    chunk_end - chunk_start,
                    geometry.latent_height,
                    geometry.latent_width,
                ),
                generator=generator,
                device=device,
                dtype=context.dtype,
            )
            clean_chunk = run_fixed_step_sde(
                self.transformer,
                self.scheduler,
                noise,
                t_list=manifest.t_list,
                seed=seed,
                frame_start=chunk_start,
                dtype=context.dtype,
                text_kv=context.text_kv,
                fps=context.fps,
                history_kv=history,
            )
            for local_idx, frame_idx in iter_clean_commit_frames(
                chunk_start, chunk_end, target_frame=target_frame
            ):
                history = self._commit(
                    context,
                    history,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_start=frame_idx,
                )
            latents.append(clean_chunk)
            self.log_info(
                f"Committed latent frames [{chunk_start}, {chunk_end}) of {target_frame}"
            )
        return torch.cat(latents, dim=2)
