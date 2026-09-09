# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline stages.

One request carries every camera of the rig. The stages pack all cameras
camera-major along time (all frames of camera 0, then camera 1, ...), encode
and decode each camera separately through the temporally causal Wan VAE, and
hand the transformer a ``MultiviewLayout`` plus the temporal wrap period that
make the GEN cross-attention sparse across cameras. Denoising reuses
``Cosmos3DenoisingStage``: the WSM control item rides the existing clean
control-prefix mechanism and the image anchors ride the I2V velocity mask.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    Cosmos3MultiviewDeploymentConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_HEIGHT,
    COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH,
    COSMOS3_MULTIVIEW_WIDTH,
    MultiviewViewInput,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview_attention import (
    DEFAULT_MAX_UND_TOKENS,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3TokenizationStage,
    _pil_to_uint8_tensor,
    _resize_center_crop_uint8_cthw,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_image, load_video

logger = init_logger(__name__)

# ``batch.extra`` keys shared by the multiview stages.
EXTRA_CONTROL_PIXELS = "multiview_control_pixels"
EXTRA_VISION_PIXELS = "multiview_vision_pixels"
EXTRA_NUM_VIEWS = "multiview_num_views"
EXTRA_FRAMES_PER_VIEW = "multiview_frames_per_view"
EXTRA_LATENT_FRAMES_PER_VIEW = "multiview_latent_frames_per_view"
EXTRA_LOCAL_CONDITION_INDEXES = "multiview_local_condition_indexes"
EXTRA_CAMERAS = "multiview_cameras"
# Consumed by Cosmos3DenoisingStage and forwarded to every transformer call.
EXTRA_TRANSFORMER_KWARGS = "transformer_extra_kwargs"

COSMOS3_TRANSFER_SYSTEM_PROMPT = (
    "You are a helpful assistant that generates images or videos following the "
    "user's instructions and control signals (edge maps, blur, depth, or segmentation)."
)
COSMOS3_MULTIVIEW_EMPHASIS = (
    "Follow the wsm control videos precisely for every camera view: shape, contour, "
    "position, and motion must align with the wsm signal at every frame."
)
# Caption augmentors of the transfer training data.
DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."
INVERSE_DURATION_TEMPLATE = (
    "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
)
INVERSE_RESOLUTION_TEMPLATE = "This video is not of {height}x{width} resolution."
# The canvas is fixed at 832x480, so the metadata aspect ratio is too.
COSMOS3_MULTIVIEW_ASPECT_RATIO = "16,9"

# Rates and frame counts outside these bounds are allowed with a warning.
COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE = (10.0, 30.0)
COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE = (24, 300)
# Server warmup carries no per-camera media; a short black clip exercises every
# stage and kernel at a fraction of a real request's cost.
COSMOS3_MULTIVIEW_WARMUP_NUM_FRAMES = 5

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
)


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------
def media_kind(path: str) -> str:
    """``"image"`` or ``"video"`` from the file extension."""
    return "image" if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS else "video"


def load_media_uint8_cthw(
    path: str, *, height: int, width: int, max_frames: int
) -> torch.Tensor:
    """Load an image or the first ``max_frames`` of a clip as ``uint8 [3, T, H, W]``.

    Frames are aspect-resized and center-cropped to ``height x width``.
    """
    if media_kind(path) == "image":
        frames = [load_image(path).convert("RGB")]
    else:
        decoded = load_video(path)
        if not decoded:
            raise ValueError(
                f"No frames decoded from Cosmos3 multiview media {path!r}."
            )
        frames = [frame.convert("RGB") for frame in decoded[:max_frames]]
    cthw = torch.stack([_pil_to_uint8_tensor(frame) for frame in frames], dim=1)
    return _resize_center_crop_uint8_cthw(cthw, height=height, width=width)


def pad_view_frames_uint8(frames: torch.Tensor, *, num_frames: int) -> torch.Tensor:
    """Truncate or last-frame-pad one camera's ``uint8 [3, T, H, W]`` to ``num_frames``.

    The reference replicates the last decoded frame over the tail; a clip that
    decoded to zero frames is an error rather than a silently gray camera.
    """
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Cosmos3 multiview view frames must have shape [3, T, H, W], got {tuple(frames.shape)}."
        )
    fill = min(int(frames.shape[1]), int(num_frames))
    if fill <= 0:
        raise ValueError("Cosmos3 multiview view media decoded to zero frames.")
    video = frames[:, :fill]
    if fill < num_frames:
        video = torch.cat(
            [video, video[:, -1:].expand(-1, num_frames - fill, -1, -1)], dim=1
        )
    return video.contiguous()


def synthetic_multiview_pixels(
    *, num_views: int, num_frames: int, height: int, width: int
) -> torch.Tensor:
    """Black camera-major ``uint8 [1, 3, V*F, H, W]`` clip for warmup requests."""
    if min(num_views, num_frames, height, width) <= 0:
        raise ValueError("Synthetic multiview pixels need positive dimensions.")
    return torch.zeros((1, 3, num_views * num_frames, height, width), dtype=torch.uint8)


# ---------------------------------------------------------------------------
# Prompt formatting (mirrors the reference Cosmos3 metadata templates)
# ---------------------------------------------------------------------------
def apply_metadata_templates(
    prompt: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    duration_template: str | None,
    resolution_template: str | None,
    force_duration_template: bool = False,
) -> str:
    """Append duration and resolution sentences to a prose caption."""
    prompt = prompt.strip()
    if duration_template is None and resolution_template is None:
        return prompt
    parts: list[str] = []
    head = prompt.rstrip(".").strip()
    if head:
        parts.append(head)
    if duration_template is not None and (num_frames > 1 or force_duration_template):
        duration = num_frames / fps
        parts.append(duration_template.format(duration=duration, fps=fps).rstrip("."))
    if resolution_template is not None:
        parts.append(resolution_template.format(height=height, width=width).rstrip("."))
    if not parts:
        return ""
    return ". ".join(parts) + "."


def format_json_caption(
    prompt: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    aspect_ratio: str | None,
) -> str | None:
    """Inject training-time metadata into a JSON-object caption; None if not JSON."""
    try:
        caption = json.loads(prompt)
    except (TypeError, ValueError):
        return None
    if not isinstance(caption, dict):
        return None
    metadata: dict[str, Any] = {}
    if num_frames > 1:
        duration_seconds = int(num_frames / fps) if fps > 0 else 0
        metadata.update({"duration": f"{duration_seconds}s", "fps": float(fps)})
    else:
        caption.pop("duration", None)
        caption.pop("fps", None)
    metadata["resolution"] = {"H": int(height), "W": int(width)}
    if aspect_ratio is not None:
        metadata["aspect_ratio"] = aspect_ratio
    caption.update(metadata)
    return json.dumps(caption)


def format_multiview_prompts(
    prompt: str,
    negative_prompt: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    negative_metadata_mode: str = "same",
) -> tuple[str, str]:
    """Positive caption with metadata plus the WSM emphasis, and the negative caption.

    ``num_frames`` is the per-camera frame count. JSON-object captions receive
    the metadata as fields; prose captions receive the duration and resolution
    sentences. ``negative_metadata_mode`` selects no metadata, the same
    metadata, or the inverse sentences for the negative prompt.
    """
    mode = negative_metadata_mode.strip().lower()
    if mode not in {"none", "same", "inverse"}:
        raise ValueError(
            "Cosmos3 negative_metadata_mode must be one of 'none', 'same', or "
            f"'inverse'; got {negative_metadata_mode!r}."
        )
    json_prompt = format_json_caption(
        prompt,
        num_frames=num_frames,
        fps=fps,
        height=height,
        width=width,
        aspect_ratio=COSMOS3_MULTIVIEW_ASPECT_RATIO,
    )
    if json_prompt is not None:
        positive = json_prompt
    else:
        positive = apply_metadata_templates(
            prompt,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
            duration_template=DURATION_TEMPLATE,
            resolution_template=RESOLUTION_TEMPLATE,
        )
    positive = f"{positive.rstrip()} {COSMOS3_MULTIVIEW_EMPHASIS}".strip()

    if mode == "none":
        negative_duration, negative_resolution = None, None
    elif mode == "same":
        negative_duration, negative_resolution = DURATION_TEMPLATE, RESOLUTION_TEMPLATE
    else:
        negative_duration = INVERSE_DURATION_TEMPLATE
        negative_resolution = INVERSE_RESOLUTION_TEMPLATE
    negative = apply_metadata_templates(
        negative_prompt,
        num_frames=num_frames,
        fps=fps,
        height=height,
        width=width,
        duration_template=negative_duration,
        resolution_template=negative_resolution,
        force_duration_template=mode == "inverse",
    )
    return positive, negative


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
class Cosmos3MultiviewInputStage(PipelineStage):
    """Load every camera's WSM control clip and optional RGB vision input.

    Writes camera-major ``uint8 [1, 3, V*F, H, W]`` pixel tensors to
    ``batch.extra`` together with the per-camera frame count and the local
    latent condition indexes the latent stage expands.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, deployment: Cosmos3MultiviewDeploymentConfig) -> None:
        super().__init__()
        self.deployment = deployment

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        result.add_check("height", batch.height, V.positive_int_divisible(16))
        result.add_check("width", batch.width, V.positive_int_divisible(16))
        return result

    def _resolve_views(self, batch: Req) -> list[MultiviewViewInput] | None:
        cameras = self.deployment.cameras
        views = batch.sampling_params.resolve_views()
        if not views:
            if batch.is_warmup:
                return None
            raise ValueError(
                "Cosmos3 multiview requests need multiview.views with one WSM "
                "control clip per camera (or a control_path list in checkpoint "
                "camera order for text-to-video)."
            )
        if len(views) != len(cameras):
            raise ValueError(
                f"Cosmos3 multiview requires exactly {len(cameras)} camera views in "
                f"checkpoint order, got {len(views)}."
            )
        camera_keys = [view.camera_key for view in views]
        if any(key is not None for key in camera_keys):
            if tuple(camera_keys) != cameras:
                raise ValueError(
                    "Cosmos3 multiview camera order must exactly match the exported "
                    f"checkpoint order: expected={list(cameras)}, got={camera_keys}."
                )
        else:
            views = [
                MultiviewViewInput(
                    camera_key=camera, control=view.control, vision=view.vision
                )
                for camera, view in zip(cameras, views, strict=True)
            ]
        if any("lidar" in str(view.camera_key).lower() for view in views):
            raise ValueError("Cosmos3 multiview v1 does not support LiDAR items.")
        control_kinds = {media_kind(view.control) for view in views}
        if len(control_kinds) != 1:
            raise ValueError(
                "Cosmos3 multiview control inputs must be all images or all videos, "
                f"got {sorted(control_kinds)}."
            )
        vision_kinds = {
            media_kind(view.vision) for view in views if view.vision is not None
        }
        if len(vision_kinds) > 1:
            raise ValueError(
                "Cosmos3 multiview vision inputs must be all images or all videos, "
                f"got {sorted(vision_kinds)}."
            )
        return views

    def _pack_camera_major(
        self,
        views: list[MultiviewViewInput],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
    ) -> torch.Tensor:
        clips = []
        for view in views:
            path = view.control if field == "control" else view.vision
            if path is None:
                raise ValueError(
                    f"Cosmos3 multiview camera {view.camera_key!r} is missing {field} input."
                )
            frames = load_media_uint8_cthw(
                path,
                height=height,
                width=width,
                max_frames=1 if keep_first else num_frames,
            )
            clips.append(pad_view_frames_uint8(frames, num_frames=num_frames))
        return torch.cat(clips, dim=1).unsqueeze(0).contiguous()

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        sampling_params = batch.sampling_params
        height, width = int(batch.height), int(batch.width)
        if (width, height) != (COSMOS3_MULTIVIEW_WIDTH, COSMOS3_MULTIVIEW_HEIGHT):
            raise ValueError(
                "Cosmos3 multiview v1 is fixed at "
                f"{COSMOS3_MULTIVIEW_WIDTH}x{COSMOS3_MULTIVIEW_HEIGHT}, got {width}x{height}."
            )
        cameras = self.deployment.cameras
        views = self._resolve_views(batch)

        if views is None:
            num_frames = COSMOS3_MULTIVIEW_WARMUP_NUM_FRAMES
            batch.num_frames = num_frames
            batch.extra[EXTRA_CONTROL_PIXELS] = synthetic_multiview_pixels(
                num_views=len(cameras),
                num_frames=num_frames,
                height=height,
                width=width,
            )
            batch.extra[EXTRA_VISION_PIXELS] = None
            local_indexes: list[int] = []
        else:
            num_frames = int(batch.num_frames)
            fps = float(batch.fps)
            low, high = COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE
            if not low <= fps <= high:
                self.log_warning(
                    f"Cosmos3 multiview fps {fps} is outside the recommended range "
                    f"[{low}, {high}]; the model was trained at 30 FPS."
                )
            low_frames, high_frames = COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE
            if not low_frames <= num_frames <= high_frames:
                self.log_warning(
                    f"Cosmos3 multiview num_frames {num_frames} is outside the "
                    f"recommended range [{low_frames}, {high_frames}]."
                )
            has_vision = views[0].vision is not None
            condition_video_as_image = (
                sampling_params.resolved_condition_video_as_image()
            )
            batch.extra[EXTRA_CONTROL_PIXELS] = self._pack_camera_major(
                views,
                field="control",
                height=height,
                width=width,
                num_frames=num_frames,
                keep_first=False,
            )
            batch.extra[EXTRA_VISION_PIXELS] = (
                self._pack_camera_major(
                    views,
                    field="vision",
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    keep_first=condition_video_as_image,
                )
                if has_vision
                else None
            )
            explicit_indexes = sampling_params.resolved_local_condition_indexes()
            if explicit_indexes is not None:
                if not has_vision and explicit_indexes:
                    raise ValueError(
                        "Cosmos3 multiview condition_frame_indexes_vision requires "
                        "per-camera vision inputs."
                    )
                local_indexes = explicit_indexes
            elif not has_vision:
                local_indexes = []
            elif condition_video_as_image or media_kind(views[0].vision) == "image":
                local_indexes = [0]
            else:
                local_indexes = [0, 1]

        batch.extra[EXTRA_NUM_VIEWS] = len(cameras)
        batch.extra[EXTRA_FRAMES_PER_VIEW] = num_frames
        batch.extra[EXTRA_LOCAL_CONDITION_INDEXES] = local_indexes
        batch.extra[EXTRA_CAMERAS] = tuple(cameras)
        mode = "text2video" if not local_indexes else "image2video"
        self.log_info(
            f"Prepared {len(cameras)} camera views x {num_frames} frames "
            f"({mode}, condition latent frames per camera {local_indexes})"
        )
        return batch


class Cosmos3MultiviewTokenizationStage(Cosmos3TokenizationStage):
    """Transfer-style prompt with WSM emphasis, tokenized with the Qwen2 template."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        prompt = batch.prompt
        if isinstance(prompt, (list, tuple)):
            if len(prompt) != 1:
                raise ValueError(
                    "Cosmos3 multiview supports exactly one prompt per request."
                )
            prompt = prompt[0]
        prompt = str(prompt or "")
        negative_prompt = batch.negative_prompt
        if isinstance(negative_prompt, (list, tuple)):
            negative_prompt = negative_prompt[0] if negative_prompt else ""
        negative_prompt = str(negative_prompt or "")

        fps = float(batch.fps)
        num_frames = int(batch.num_frames)
        mode = batch.sampling_params.negative_metadata_mode
        prompt, negative_prompt = format_multiview_prompts(
            prompt,
            negative_prompt,
            num_frames=num_frames,
            fps=fps,
            height=int(batch.height),
            width=int(batch.width),
            negative_metadata_mode=mode,
        )

        requested = batch.max_sequence_length or COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        max_sequence_length = min(int(requested), COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH)
        # ``_tokenize_prompt`` reserves two slots inside its cap for the eos and
        # vision_start framing tokens. The reference caps the caption itself, so
        # widen by two; the sparse attention's UND capacity (4096 + 2) matches.
        tokenizer_cap = max_sequence_length + 2

        cond_ids, cond_mask, cond_seq_len = self._tokenize_prompt(
            text=prompt,
            max_sequence_length=tokenizer_cap,
            device=device,
            use_system_prompt=True,
            system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
        )
        uncond_ids, uncond_mask, uncond_seq_len = self._tokenize_prompt(
            text=negative_prompt,
            max_sequence_length=tokenizer_cap,
            device=device,
            use_system_prompt=True,
            system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
        )
        shared_seq_len = max(cond_seq_len, uncond_seq_len)
        batch.extra["cond_text_ids"] = cond_ids[:, :shared_seq_len]
        batch.extra["cond_text_mask"] = cond_mask[:, :shared_seq_len]
        batch.extra["uncond_text_ids"] = uncond_ids[:, :shared_seq_len]
        batch.extra["uncond_text_mask"] = uncond_mask[:, :shared_seq_len]
        batch.extra["cond_text_seq_len"] = cond_seq_len
        batch.extra["uncond_text_seq_len"] = uncond_seq_len
        batch.extra["fps"] = fps
        batch.is_prompt_processed = True
        if not batch.is_warmup:
            self.log_info(
                f"Multiview prompt ({cond_seq_len} tokens, negative {uncond_seq_len}): "
                f"{prompt[:240]!r}"
            )
        return batch


class Cosmos3MultiviewLatentStage(PipelineStage):
    """Per-camera VAE encode, camera-major latents, anchors, and the attention layout."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        vae,
        transformer,
        deployment: Cosmos3MultiviewDeploymentConfig,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.deployment = deployment
        self.attention_backend = attention_backend

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            EXTRA_CONTROL_PIXELS, batch.extra.get(EXTRA_CONTROL_PIXELS), V.is_tensor
        )
        result.add_check("height", batch.height, V.positive_int_divisible(16))
        result.add_check("width", batch.width, V.positive_int_divisible(16))
        return result

    def _encode_camera_major(
        self,
        pixels: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode ``uint8 [1, 3, V*F, H, W]`` one camera at a time into ``[1, C, V*Fl, h, w]``.

        The Wan VAE is causal in time, so the camera-major stream must never be
        encoded as one long clip.
        """
        expected = num_views * frames_per_view
        if pixels.ndim != 5 or pixels.shape[2] != expected:
            raise ValueError(
                "Cosmos3 multiview pixel video must be camera-major [1, 3, V*F, H, W]: "
                f"shape={tuple(pixels.shape)}, V={num_views}, F={frames_per_view}."
            )
        vae_dtype = next(self.vae.parameters()).dtype
        mean = torch.as_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        latents = []
        for view in range(num_views):
            clip = pixels[:, :, view * frames_per_view : (view + 1) * frames_per_view]
            clip = clip.to(device=device, dtype=vae_dtype).div(127.5).sub(1.0)
            with torch.no_grad():
                latent = self.vae.encode(clip).mode()
            latent = (latent - mean.to(latent)) / std.to(latent)
            latents.append(latent.to(dtype))
        latent_frames = {int(latent.shape[2]) for latent in latents}
        if len(latent_frames) != 1:
            raise ValueError(
                f"Cosmos3 multiview per-camera VAE encodes have unequal lengths: {latent_frames}."
            )
        return torch.cat(latents, dim=2)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        dtype = torch.bfloat16
        deployment = self.deployment
        num_views = int(batch.extra[EXTRA_NUM_VIEWS])
        frames_per_view = int(batch.extra[EXTRA_FRAMES_PER_VIEW])
        temporal_factor = int(self.vae.config.scale_factor_temporal)
        spatial_factor = int(self.vae.config.scale_factor_spatial)
        latent_frames_per_view = (frames_per_view - 1) // temporal_factor + 1
        latent_t = num_views * latent_frames_per_view
        latent_h = int(batch.height) // spatial_factor
        latent_w = int(batch.width) // spatial_factor
        shape = (
            1,
            int(self.transformer.latent_channel),
            latent_t,
            latent_h,
            latent_w,
        )

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(batch.seed))
            batch.generator = generator
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        local_indexes = [
            int(i) for i in (batch.extra.get(EXTRA_LOCAL_CONDITION_INDEXES) or [])
        ]
        vision_pixels = batch.extra.get(EXTRA_VISION_PIXELS)
        with self.use_declared_component(component_name="vae", module=self.vae):
            control_latents = self._encode_camera_major(
                batch.extra[EXTRA_CONTROL_PIXELS],
                num_views=num_views,
                frames_per_view=frames_per_view,
                device=device,
                dtype=dtype,
            )
            encoded_vision = None
            if vision_pixels is not None and local_indexes:
                encoded_vision = self._encode_camera_major(
                    vision_pixels,
                    num_views=num_views,
                    frames_per_view=frames_per_view,
                    device=device,
                    dtype=dtype,
                )
        if tuple(control_latents.shape) != shape:
            raise ValueError(
                "Cosmos3 multiview WSM and target latent shapes must match: "
                f"control={tuple(control_latents.shape)}, target={shape}."
            )

        condition_indexes = expand_multiview_condition_frame_indexes(
            local_indexes, num_views, latent_t
        )
        if condition_indexes:
            if encoded_vision is None:
                raise ValueError(
                    "Cosmos3 multiview condition indexes require per-camera vision inputs."
                )
            if tuple(encoded_vision.shape) != shape:
                raise ValueError(
                    "Cosmos3 multiview target VAE latent shape mismatch: "
                    f"expected={shape}, got={tuple(encoded_vision.shape)}."
                )
            condition_mask = torch.zeros(
                1, 1, latent_t, 1, 1, device=device, dtype=dtype
            )
            condition_latents = torch.zeros_like(noise)
            for index in condition_indexes:
                condition_mask[:, :, index] = 1.0
                condition_latents[:, :, index] = encoded_vision[:, :, index]
            latents = (
                condition_mask * condition_latents + (1.0 - condition_mask) * noise
            )
            batch.extra["condition_latents"] = condition_latents
            batch.extra["velocity_mask"] = 1.0 - condition_mask
        else:
            latents = noise
            batch.extra.pop("condition_latents", None)
            batch.extra.pop("velocity_mask", None)

        patch_h, patch_w, _, _ = self.transformer._pad_to_patch_size(latent_h, latent_w)
        fps = float(batch.extra.get("fps") or batch.fps)
        layout = MultiviewLayout(
            num_views=num_views,
            latent_frames=latent_t,
            patch_height=patch_h,
            patch_width=patch_w,
            attention_scope=deployment.attention_scope,  # type: ignore[arg-type]
            decomposed_temporal_window_seconds=deployment.decomposed_temporal_window_seconds,
            control_attends_sensor=deployment.control_attends_sensor,
            seconds_per_frame=temporal_factor / fps,
            backend=self.attention_backend,
            max_und_tokens=DEFAULT_MAX_UND_TOKENS,
        )
        temporal_position_period = (
            latent_frames_per_view
            if deployment.align_temporal_positions_across_views
            else None
        )

        batch.latents = latents
        batch.raw_latent_shape = shape
        batch.extra["video_shape"] = (latent_t, latent_h, latent_w)
        batch.extra["vae_scale_factor_temporal"] = temporal_factor
        batch.extra["vae_scale_factor_spatial"] = spatial_factor
        batch.extra["control_latents"] = [control_latents]
        batch.extra[EXTRA_LATENT_FRAMES_PER_VIEW] = latent_frames_per_view
        batch.extra[EXTRA_TRANSFORMER_KWARGS] = {
            "temporal_position_period": temporal_position_period,
            "multiview_layout": layout,
        }
        self.log_info(
            f"Prepared multiview latents {shape} ({num_views} cameras x "
            f"{latent_frames_per_view} latent frames, {layout.gen_tokens} GEN tokens, "
            f"{len(condition_indexes)} anchored frames, backend={self.attention_backend})"
        )
        return batch

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]


class Cosmos3MultiviewDecodingStage(Cosmos3DecodingStage):
    """Decode each camera's latent slice separately and re-pack camera-major.

    The output is one ``[1, 3, V*F, H, W]`` video: all frames of camera 0, then
    camera 1, and so on in checkpoint camera order. Split it by
    ``frames_per_view`` on the client to get per-camera clips.
    """

    def forward(self, batch: Req, server_args: ServerArgs):
        num_views = int(batch.extra[EXTRA_NUM_VIEWS])
        latent_frames_per_view = int(batch.extra[EXTRA_LATENT_FRAMES_PER_VIEW])
        latents = batch.latents
        if latents.shape[2] != num_views * latent_frames_per_view:
            raise ValueError(
                "Cosmos3 multiview latents must be camera-major before decode: "
                f"shape={tuple(latents.shape)}, V={num_views}, F={latent_frames_per_view}."
            )
        with self.use_declared_component(component_name="vae", module=self.vae):
            with torch.no_grad():
                decoded = torch.cat(
                    [
                        self._decode_latents(
                            latents[
                                :,
                                :,
                                view * latent_frames_per_view : (view + 1)
                                * latent_frames_per_view,
                            ]
                        )
                        for view in range(num_views)
                    ],
                    dim=2,
                )
        # The base stage consumes a pre-decoded tensor from this key (it is the
        # transfer pipeline's hand-off) and only post-processes it.
        batch.extra["transfer_decoded_output"] = decoded
        frames_per_view = int(decoded.shape[2]) // num_views
        cameras = batch.extra.get(EXTRA_CAMERAS)
        if not batch.is_warmup:
            self.log_info(
                f"Decoded camera-major video: {num_views} cameras x {frames_per_view} "
                f"frames at {batch.fps} fps; camera order {list(cameras or [])}"
            )
        return super().forward(batch, server_args)
