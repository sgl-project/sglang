# SPDX-License-Identifier: Apache-2.0
"""Cosmos-Dreams stages: request preparation and the causal rollout.

Latent frame 0 comes from the conditioning image (or is generated as its own
one-frame chunk); later frames are produced in chunks of
``manifest.chunk_size`` with the checkpoint's four-step distilled SDE sampler.
After a chunk is denoised, every frame is re-run clean (``timestep=0``, no time
embedding) so its K/V is committed as history for the following chunks.
"""

import json
import math
import os
from collections.abc import Iterator, Sequence
from typing import Any

import msgspec
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    TEXT_TOKENS_TRAINING_MAX,
    AffineTransform,
    CosmosDreamsManifest,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
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
    VIDEO_RES_SIZE_INFO,
    VIEWPOINT_TEMPLATES,
    canonical_aspect_ratio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import load_image

logger = init_logger(__name__)

RESOLUTION_ALIGNMENT = 16
# Trained canvas family: 16:9 through 9:16 around 704x1280.
MIN_ASPECT_RATIO = 704 / 1280
MAX_ASPECT_RATIO = 1280 / 704

# ``Req.extra`` keys shared by the preparation and rollout stages.
EXTRA_GEOMETRY = "cosmos_dreams_geometry"
EXTRA_TEXT_IDS = "cosmos_dreams_text_ids"
EXTRA_TEXT_MASK = "cosmos_dreams_text_mask"
EXTRA_ACTION_ROWS = "cosmos_dreams_action_rows"
EXTRA_DOMAIN_ID = "cosmos_dreams_domain_id"
EXTRA_TARGET_LATENT_FRAMES = "cosmos_dreams_target_latent_frames"
# (content_height, content_width) of the conditioning image inside the canvas.
EXTRA_CONTENT_SIZE = "cosmos_dreams_content_size"


class CosmosDreamsGeometry(msgspec.Struct, frozen=True):
    """Pixel, latent, and patch-grid sizes of one request."""

    height: int
    width: int
    latent_height: int
    latent_width: int
    grid_height: int
    grid_width: int

    @property
    def vision_tokens_per_frame(self) -> int:
        return self.grid_height * self.grid_width

    def tokens_per_frame(self, action_tokens_per_frame: int) -> int:
        return self.vision_tokens_per_frame + action_tokens_per_frame


class PreparedConditioning(msgspec.Struct, frozen=True):
    """Request-level inputs shared by the offline rollout and realtime ticks."""

    geometry: CosmosDreamsGeometry
    canvas: tuple[int, int]
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    embodiment: str
    domain_id: int
    image_latent: torch.Tensor | None


def resolve_geometry(
    *, height: int, width: int, manifest: CosmosDreamsManifest, max_pixels: int
) -> CosmosDreamsGeometry:
    if height % RESOLUTION_ALIGNMENT or width % RESOLUTION_ALIGNMENT:
        raise ValueError(
            f"Cosmos-Dreams dimensions must be multiples of {RESOLUTION_ALIGNMENT}, "
            f"got {height}x{width}."
        )
    if height * width > max_pixels:
        raise ValueError(
            f"Cosmos-Dreams dimensions exceed max_pixels={max_pixels}: "
            f"{height}x{width}={height * width}."
        )
    aspect = width / height
    if not MIN_ASPECT_RATIO <= aspect <= MAX_ASPECT_RATIO:
        raise ValueError(
            "Cosmos-Dreams width/height aspect ratio must be in "
            f"[{MIN_ASPECT_RATIO:.6g}, {MAX_ASPECT_RATIO:.6g}], got {aspect:.6g}."
        )
    spatial = manifest.vae_spatial_compression_factor
    patch = manifest.latent_patch_size
    latent_height = math.ceil(height / spatial)
    latent_width = math.ceil(width / spatial)
    return CosmosDreamsGeometry(
        height=height,
        width=width,
        latent_height=latent_height,
        latent_width=latent_width,
        grid_height=math.ceil(latent_height / patch),
        grid_width=math.ceil(latent_width / patch),
    )


def crop_geometry_to_content(
    geometry: CosmosDreamsGeometry,
    *,
    content_size: tuple[int, int],
    manifest: CosmosDreamsManifest,
) -> CosmosDreamsGeometry:
    """Geometry of the content region the model generates.

    Training and the reference inference encode the reflection-padded canvas,
    then drop the padded latent rows/columns (``latent[..., :h // s, :w // s]``
    with ``s`` the VAE spatial stride), so the padding never reaches the
    transformer and the decoded video has the content size.
    """
    spatial = manifest.vae_spatial_compression_factor
    patch = manifest.latent_patch_size
    content_height, content_width = (int(v) for v in content_size)
    if content_height <= 0 or content_width <= 0:
        raise ValueError(f"Content size must be positive, got {content_size}.")
    latent_height = max(content_height // spatial, 1)
    latent_width = max(content_width // spatial, 1)
    if latent_height > geometry.latent_height or latent_width > geometry.latent_width:
        raise ValueError(
            f"Content {content_height}x{content_width} exceeds the canvas "
            f"{geometry.height}x{geometry.width}."
        )
    return CosmosDreamsGeometry(
        height=latent_height * spatial,
        width=latent_width * spatial,
        latent_height=latent_height,
        latent_width=latent_width,
        grid_height=math.ceil(latent_height / patch),
        grid_width=math.ceil(latent_width / patch),
    )


def latent_frame_count(num_pixel_frames: int, temporal_compression_factor: int) -> int:
    if num_pixel_frames <= 0:
        raise ValueError(
            f"Cosmos-Dreams num_frames must be positive, got {num_pixel_frames}."
        )
    return (num_pixel_frames - 1) // temporal_compression_factor + 1


def iter_ar_chunk_ranges(
    start_frame: int, num_frames: int, chunk_size: int
) -> Iterator[tuple[int, int]]:
    """Yield the training-aligned latent partition ``[0,1), [1,1+C), [1+C,1+2C), ...``."""
    if start_frame < 0 or num_frames < 0 or start_frame > num_frames:
        raise ValueError(
            f"Invalid Cosmos-Dreams frame range [{start_frame}, {num_frames})."
        )
    if chunk_size <= 0:
        raise ValueError(
            f"Cosmos-Dreams chunk_size must be positive, got {chunk_size}."
        )
    frame = start_frame
    while frame < num_frames:
        if frame == 0:
            chunk_end = 1
        else:
            chunk_end = 1 + ((frame - 1) // chunk_size + 1) * chunk_size
        chunk_end = min(chunk_end, num_frames)
        yield frame, chunk_end
        frame = chunk_end


def iter_clean_commit_frames(
    chunk_start: int, chunk_end: int, *, target_frame: int
) -> Iterator[tuple[int, int]]:
    """Yield ``(local, absolute)`` frames to clean-refresh, in commit order.

    Every frame is refreshed individually so later frames of the same chunk see
    clean, committed history. The globally final frame has no reader in a full
    rollout and is skipped.
    """
    if chunk_start < 0 or chunk_end <= chunk_start or target_frame < chunk_end:
        raise ValueError(
            "Invalid Cosmos-Dreams clean-commit range: "
            f"chunk=[{chunk_start}, {chunk_end}), target={target_frame}."
        )
    for local_idx, frame_idx in enumerate(range(chunk_start, chunk_end)):
        if frame_idx == target_frame - 1:
            continue
        yield local_idx, frame_idx


def _load_action_file(path: str) -> Any:
    extension = os.path.splitext(path)[1].lower()
    if extension == ".json":
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    if extension == ".npy":
        return np.load(path)
    if extension == ".pt":
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(
        f"Cosmos-Dreams action file must be .json, .npy, or .pt, got {path!r}."
    )


def load_action_rows(action: Any) -> torch.Tensor:
    """Read raw action rows ``[T, D]`` (float32) from a request value.

    Accepts nested lists, arrays, tensors, a JSON string, or a path to a
    ``.json`` / ``.npy`` / ``.pt`` file.
    """
    if isinstance(action, str):
        text = action.strip()
        action = _load_action_file(text) if os.path.isfile(text) else json.loads(text)
    if isinstance(action, torch.Tensor):
        rows = action.detach().to(dtype=torch.float32, device="cpu")
    else:
        rows = torch.as_tensor(np.asarray(action, dtype=np.float32))
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows.squeeze(0)
    if rows.ndim != 2:
        raise ValueError(
            f"Cosmos-Dreams action must have shape [T, D], got {tuple(rows.shape)}."
        )
    return rows


def normalize_action_rows(
    rows: torch.Tensor, transform: AffineTransform
) -> torch.Tensor:
    """Apply the contract's unclamped affine normalizer in float32."""
    if rows.shape[-1] != len(transform.offset):
        raise ValueError(
            "Cosmos-Dreams raw action dimension does not match the action contract: "
            f"{rows.shape[-1]} != {len(transform.offset)}."
        )
    rows = rows.to(dtype=torch.float32)
    if not torch.isfinite(rows).all():
        raise ValueError("Cosmos-Dreams raw actions must contain only finite values.")
    normalized = (rows - rows.new_tensor(transform.offset)) / rows.new_tensor(
        transform.scale
    )
    if not torch.isfinite(normalized).all():
        raise ValueError(
            "Cosmos-Dreams normalized actions must contain only finite values."
        )
    return normalized


def pad_action_rows(rows: torch.Tensor, model_action_dim: int) -> torch.Tensor:
    if rows.shape[-1] > model_action_dim:
        raise ValueError(
            f"Cosmos-Dreams action dimension {rows.shape[-1]} exceeds model_action_dim="
            f"{model_action_dim}."
        )
    padding = rows.new_zeros(*rows.shape[:-1], model_action_dim - rows.shape[-1])
    return torch.cat([rows, padding], dim=-1)


def actions_for_frames(
    rows: torch.Tensor | None,
    *,
    frame_start: int,
    frame_end: int,
    action_tokens_per_frame: int,
    model_action_dim: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Select the ``[1, F * A, D]`` action block for latent frames ``[start, end)``.

    Row block ``[(f-1)*A, f*A)`` conditions latent frame ``f``; frame 0 has no
    preceding motion and every frame is null when no actions were supplied.
    Returns the block and the chunk-local indexes of null-action frames.
    """
    frame_count = frame_end - frame_start
    action_count = action_tokens_per_frame
    if rows is None:
        zeros = torch.zeros(1, frame_count * action_count, model_action_dim)
        return zeros, tuple(range(frame_count))
    required_rows = (frame_end - 1) * action_count
    if rows.shape[0] < required_rows or rows.shape[1] != model_action_dim:
        raise ValueError(
            f"Cosmos-Dreams action rows {tuple(rows.shape)} cannot condition latent frames "
            f"[{frame_start}, {frame_end}); need at least {required_rows} rows of width "
            f"{model_action_dim}."
        )
    blocks: list[torch.Tensor] = []
    null_indexes: list[int] = []
    for local_idx, frame_idx in enumerate(range(frame_start, frame_end)):
        if frame_idx == 0:
            blocks.append(rows.new_zeros(action_count, model_action_dim))
            null_indexes.append(local_idx)
            continue
        start = (frame_idx - 1) * action_count
        blocks.append(rows[start : start + action_count])
    return torch.cat(blocks, dim=0).unsqueeze(0), tuple(null_indexes)


def _bounded_append(
    old: torch.Tensor, new: torch.Tensor, *, sink_tokens: int, tail_tokens: int
) -> torch.Tensor:
    if old.shape[1] + new.shape[1] <= sink_tokens + tail_tokens:
        return torch.cat([old, new], dim=1)
    parts: list[torch.Tensor] = []
    sink_from_old = min(sink_tokens, old.shape[1])
    if sink_from_old:
        parts.append(old[:, :sink_from_old])
    if sink_tokens - sink_from_old:
        parts.append(new[:, : sink_tokens - sink_from_old])
    tail_from_new = min(tail_tokens, new.shape[1])
    tail_from_old = tail_tokens - tail_from_new
    if tail_from_old:
        parts.append(old[:, -tail_from_old:])
    if tail_from_new:
        parts.append(new[:, -tail_from_new:])
    return parts[0].clone() if len(parts) == 1 else torch.cat(parts, dim=1)


def append_kv_history(
    history: list[KVPair] | None,
    current_kv: list[KVPair],
    *,
    tokens_per_frame: int,
    sink_frames: int,
    window_frames: int | None,
) -> list[KVPair]:
    """Append committed K/V per layer, keeping ``sink`` + latest ``window`` frames.

    ``window_frames=None`` keeps everything (no eviction).
    """
    if not current_kv:
        raise ValueError("Cosmos-Dreams K/V update must contain at least one layer.")
    for key, value in current_kv:
        if (
            key.shape != value.shape
            or key.shape[1] <= 0
            or key.shape[1] % tokens_per_frame
        ):
            raise ValueError(
                "Cosmos-Dreams K/V must be matching tensors whose token length is a positive "
                f"multiple of tokens_per_frame={tokens_per_frame}, got {tuple(key.shape)}."
            )
    if history is None:
        if window_frames is not None and any(
            key.shape[1] > (sink_frames + window_frames) * tokens_per_frame
            for key, _ in current_kv
        ):
            raise ValueError(
                "The initial Cosmos-Dreams K/V block exceeds the history window."
            )
        return [(key.detach(), value.detach()) for key, value in current_kv]
    if len(history) != len(current_kv):
        raise ValueError(
            "Cosmos-Dreams K/V layer count changed within a rollout: "
            f"history={len(history)}, current={len(current_kv)}."
        )
    if window_frames is None:
        return [
            (
                torch.cat([old_k, new_k], dim=1).detach(),
                torch.cat([old_v, new_v], dim=1).detach(),
            )
            for (old_k, old_v), (new_k, new_v) in zip(history, current_kv, strict=True)
        ]
    sink_tokens = sink_frames * tokens_per_frame
    tail_tokens = window_frames * tokens_per_frame
    updated: list[KVPair] = []
    for (old_k, old_v), (new_k, new_v) in zip(history, current_kv, strict=True):
        updated.append(
            (
                _bounded_append(
                    old_k, new_k, sink_tokens=sink_tokens, tail_tokens=tail_tokens
                ).detach(),
                _bounded_append(
                    old_v, new_v, sink_tokens=sink_tokens, tail_tokens=tail_tokens
                ).detach(),
            )
        )
    return updated


def closest_canvas(*, height: int, width: int, tier: str) -> tuple[int, int]:
    """Trained ``(width, height)`` canvas of ``tier`` whose aspect is closest to the input."""
    if tier not in VIDEO_RES_SIZE_INFO:
        raise ValueError(
            f"Unknown Cosmos-Dreams canvas tier {tier!r}; expected one of {sorted(VIDEO_RES_SIZE_INFO)}."
        )
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size must be positive, got {height}x{width}.")
    input_ratio = height / width
    best: tuple[int, int] | None = None
    best_diff = float("inf")
    for canvas_w, canvas_h in VIDEO_RES_SIZE_INFO[tier].values():
        diff = abs(input_ratio - canvas_h / canvas_w)
        if diff < best_diff:
            best_diff = diff
            best = (canvas_w, canvas_h)
    assert best is not None
    return best


def fit_image_to_canvas(
    image: torch.Tensor, *, target_height: int, target_width: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Aspect-preserving resize into the canvas, then pad bottom/right like training.

    ``image`` is ``[3, H, W]`` in ``[0, 1]``. The training transform resizes
    with antialiased bicubic filtering so the content fits inside the canvas,
    then reflection-pads (edge-pads when the pad would exceed the content) at
    the bottom and right; content stays top-left. Unlike the training
    transform, smaller inputs are upscaled so the content fills the canvas the
    way the >= 480p training clips did. Returns the canvas tensor and the
    ``(content_height, content_width)`` region.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB image [3, H, W], got {tuple(image.shape)}.")
    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(target_width / width, target_height / height)
    content_height = int(scale * height + 0.5)
    content_width = int(scale * width + 0.5)
    fitted = image.unsqueeze(0).to(torch.float32)
    if (content_height, content_width) != (height, width):
        fitted = F.interpolate(
            fitted,
            size=(content_height, content_width),
            mode="bicubic",
            antialias=True,
            align_corners=False,
        ).clamp_(0.0, 1.0)
    pad_right = target_width - content_width
    pad_bottom = target_height - content_height
    if pad_right or pad_bottom:
        mode = (
            "replicate"
            if pad_right >= content_width or pad_bottom >= content_height
            else "reflect"
        )
        fitted = F.pad(fitted, (0, pad_right, 0, pad_bottom), mode=mode)
    return fitted.squeeze(0), (content_height, content_width)


def format_dreams_prompt(
    prompt: str, *, view_point: str, height: int, width: int
) -> str:
    """Wrap a plain prompt in the JSON caption the checkpoint was trained on.

    The training stream ran the camera captions through the action prompt
    formatter with duration/fps timestamps disabled, giving
    ``{"cinematography": {"framing": <viewpoint>}, "actions": [{"description":
    <caption>}], "resolution": {"H", "W"}, "aspect_ratio": "W,H"}`` serialized
    with default ``json.dumps`` separators. An empty prompt stays empty (the
    training caption dropout), and a prompt that already is a JSON object is
    kept, with resolution and aspect ratio filled in when missing.
    """
    text = prompt.strip()
    if not text:
        return text
    if view_point not in VIEWPOINT_TEMPLATES:
        raise ValueError(
            f"Unknown Cosmos-Dreams action_view_point {view_point!r}; expected one of "
            f"{sorted(VIEWPOINT_TEMPLATES)}."
        )
    resolution = {"H": int(height), "W": int(width)}
    aspect_ratio = canonical_aspect_ratio(int(width), int(height))
    try:
        structured = json.loads(text)
    except ValueError:
        structured = None
    if isinstance(structured, dict):
        structured.setdefault("resolution", resolution)
        structured.setdefault("aspect_ratio", aspect_ratio)
        return json.dumps(structured)
    description = text if text.endswith((".", "!", "?")) else f"{text}."
    caption = {
        "cinematography": {"framing": VIEWPOINT_TEMPLATES[view_point]},
        "actions": [{"description": description}],
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
    }
    return json.dumps(caption)


def tokenize_dreams_prompt(
    tokenizer: Any, prompt: str, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2 chat-template tokens for the prompt plus EOS and vision start.

    Cosmos-Dreams uses neither the Cosmos3 system prompt nor the duration
    suffix, and the K/V is trimmed to the real length, so nothing is padded.
    """
    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    token_ids = list(result) if isinstance(result, list) else list(result["input_ids"])
    token_ids = token_ids[:TEXT_TOKENS_TRAINING_MAX]
    token_ids.append(tokenizer.eos_token_id)
    token_ids.append(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    text_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    return text_ids, torch.ones_like(text_ids)


def single_prompt(prompt: Any) -> str:
    if isinstance(prompt, (list, tuple)):
        if len(prompt) != 1:
            raise ValueError("Cosmos-Dreams supports exactly one prompt per request.")
        prompt = prompt[0]
    if not isinstance(prompt, str):
        raise ValueError(
            f"Cosmos-Dreams prompt must be a string, got {type(prompt).__name__}."
        )
    return prompt


def _pil_to_unit_tensor(image: PIL.Image.Image) -> torch.Tensor:
    """PIL RGB to ``[3, H, W]`` float32 in ``[0, 1]``."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _single_image_path(image_path: Any) -> str | None:
    if isinstance(image_path, (list, tuple)):
        if len(image_path) > 1:
            raise ValueError("Cosmos-Dreams accepts exactly one conditioning image.")
        return image_path[0] if image_path else None
    return image_path


class CosmosDreamsImageStage(PipelineStage):
    """Load the conditioning image and fit it to a trained canvas.

    Mirrors the training transform: pick the canvas whose aspect is closest to
    the image (when the request left height/width unset), resize to fit, and
    reflection-pad bottom/right. Writes ``[1, 3, H, W]`` in ``[-1, 1]`` to
    ``batch.preprocessed_image``; a text-only start leaves it ``None``.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(self, *, canvas_tier: str) -> None:
        super().__init__()
        if canvas_tier not in VIDEO_RES_SIZE_INFO:
            raise ValueError(
                f"Unknown Cosmos-Dreams canvas tier {canvas_tier!r}; expected one of "
                f"{sorted(VIDEO_RES_SIZE_INFO)}."
            )
        self.canvas_tier = canvas_tier

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        image_path = _single_image_path(batch.image_path)
        if image_path is None:
            return batch
        image = _pil_to_unit_tensor(load_image(image_path))
        if batch.sampling_params.canvas_from_image:
            canvas_width, canvas_height = closest_canvas(
                height=int(image.shape[1]),
                width=int(image.shape[2]),
                tier=self.canvas_tier,
            )
            batch.height, batch.width = canvas_height, canvas_width
        fitted, content_size = fit_image_to_canvas(
            image, target_height=batch.height, target_width=batch.width
        )
        batch.preprocessed_image = (fitted * 2.0 - 1.0).unsqueeze(0).contiguous()
        batch.extra[EXTRA_CONTENT_SIZE] = content_size
        self.log_info(
            f"Fitted conditioning image {tuple(image.shape[1:])} into canvas "
            f"{batch.height}x{batch.width} (content {content_size[0]}x{content_size[1]})"
        )
        return batch


class CosmosDreamsPrepareStage(PipelineStage):
    """Validate the request and build every input the rollout needs.

    Writes the tokenized prompt, resolved geometry, normalized zero-padded
    action rows, and the VAE latent of the conditioning image into ``batch``.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self, *, vae, tokenizer, manifest: CosmosDreamsManifest, max_pixels: int
    ) -> None:
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.max_pixels = max_pixels

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("prompt", batch.prompt, V.string_or_list_strings)
        result.add_check(
            "height", batch.height, V.positive_int_divisible(RESOLUTION_ALIGNMENT)
        )
        result.add_check(
            "width", batch.width, V.positive_int_divisible(RESOLUTION_ALIGNMENT)
        )
        result.add_check("num_frames", batch.num_frames, V.positive_int)
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        return [ComponentUse(self._component_stage_name(stage_name), "vae")]

    def _prepare_action_rows(
        self, action: Any, *, embodiment: str, target_frame: int
    ) -> torch.Tensor | None:
        if action is None:
            return None
        contract = self.manifest.action_contract.embodiments[embodiment]
        rows = load_action_rows(action)
        if rows.shape[-1] != contract.raw_action_dim:
            raise ValueError(
                f"Cosmos-Dreams embodiment {embodiment!r} requires raw action dimension "
                f"{contract.raw_action_dim}, got {rows.shape[-1]}."
            )
        rows = normalize_action_rows(rows, contract.normalizer.transform)
        rows = pad_action_rows(rows, self.manifest.max_action_dim)
        required_rows = (target_frame - 1) * self.manifest.action_tokens_per_frame
        if rows.shape[0] < required_rows:
            raise ValueError(
                f"Cosmos-Dreams action has {rows.shape[0]} rows but {target_frame} latent "
                f"frames need {required_rows} (one row per pixel step after frame 0)."
            )
        return rows

    def _encode_image_latent(
        self, image: torch.Tensor, geometry: CosmosDreamsGeometry, device: torch.device
    ) -> torch.Tensor:
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(
                f"Cosmos-Dreams expects one conditioning image [1, 3, H, W], got {tuple(image.shape)}."
            )
        vae_dtype = next(self.vae.parameters()).dtype
        with self.use_declared_component(component_name="vae", module=self.vae):
            with torch.no_grad():
                latent = self.vae.encode(
                    image.unsqueeze(2).to(device=device, dtype=vae_dtype)
                ).mode()
        mean = (
            torch.as_tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(latent)
        )
        std = (
            torch.as_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latent)
        )
        latent = ((latent - mean) / std).to(torch.bfloat16)
        expected = (
            1,
            latent.shape[1],
            1,
            geometry.latent_height,
            geometry.latent_width,
        )
        if tuple(latent.shape) != expected:
            raise ValueError(
                "Cosmos-Dreams encoded conditioning image does not match the resolved geometry: "
                f"expected {expected}, got {tuple(latent.shape)}."
            )
        return latent

    def _prepare_conditioning(
        self, batch: Req, device: torch.device
    ) -> PreparedConditioning:
        """Resolve canvas, prompt tokens, embodiment, and the conditioning latent."""
        manifest = self.manifest
        geometry = resolve_geometry(
            height=batch.height,
            width=batch.width,
            manifest=manifest,
            max_pixels=self.max_pixels,
        )
        prompt = single_prompt(batch.prompt)
        if batch.sampling_params.format_prompt_as_json:
            prompt = format_dreams_prompt(
                prompt,
                view_point=batch.sampling_params.action_view_point,
                height=geometry.height,
                width=geometry.width,
            )
        self.log_info(f"Prompt: {prompt}")
        text_ids, text_mask = tokenize_dreams_prompt(
            self.tokenizer, prompt, device=device
        )
        sampling_params = batch.sampling_params
        contract = manifest.action_contract
        embodiment = contract.resolve_embodiment(
            sampling_params.domain_name, sampling_params.domain_id
        )
        canvas = (geometry.height, geometry.width)
        image_latent = None
        if batch.preprocessed_image is not None:
            latent = self._encode_image_latent(
                batch.preprocessed_image, geometry, device
            )
            # The prompt above names the canvas; the model only sees the content.
            geometry = crop_geometry_to_content(
                geometry,
                content_size=batch.extra[EXTRA_CONTENT_SIZE],
                manifest=manifest,
            )
            image_latent = latent[
                ..., : geometry.latent_height, : geometry.latent_width
            ].contiguous()
        return PreparedConditioning(
            geometry=geometry,
            canvas=canvas,
            text_ids=text_ids,
            text_mask=text_mask,
            embodiment=embodiment,
            domain_id=contract.embodiments[embodiment].domain_id,
            image_latent=image_latent,
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        manifest = self.manifest
        prepared = self._prepare_conditioning(batch, device)
        geometry = prepared.geometry
        target_frame = latent_frame_count(
            batch.num_frames, manifest.temporal_compression_factor
        )
        rows = self._prepare_action_rows(
            batch.sampling_params.action,
            embodiment=prepared.embodiment,
            target_frame=target_frame,
        )
        if rows is None:
            self.log_warning(
                "No action supplied; every latent frame is conditioned on the null action."
            )
        if prepared.image_latent is not None:
            batch.image_latent = prepared.image_latent
            batch.height, batch.width = geometry.height, geometry.width

        batch.extra[EXTRA_GEOMETRY] = geometry
        batch.extra[EXTRA_TEXT_IDS] = prepared.text_ids
        batch.extra[EXTRA_TEXT_MASK] = prepared.text_mask
        batch.extra[EXTRA_ACTION_ROWS] = (
            None if rows is None else rows.to(device=device)
        )
        batch.extra[EXTRA_DOMAIN_ID] = prepared.domain_id
        batch.extra[EXTRA_TARGET_LATENT_FRAMES] = target_frame
        batch.raw_latent_shape = (
            1,
            self.vae.config.z_dim,
            target_frame,
            geometry.latent_height,
            geometry.latent_width,
        )
        self.log_info(
            f"Prepared Cosmos-Dreams request: canvas {prepared.canvas[0]}x{prepared.canvas[1]}, "
            f"generated {geometry.height}x{geometry.width} "
            f"(latent {geometry.latent_height}x{geometry.latent_width}), "
            f"{target_frame} latent frames, embodiment={prepared.embodiment} (domain {prepared.domain_id}), "
            f"{prepared.text_ids.shape[1]} text tokens, image={'yes' if batch.image_latent is not None else 'no'}"
        )
        return batch


def validate_fixed_step_scheduler(
    scheduler: Any, manifest: CosmosDreamsManifest
) -> FlowMatchEulerDiscreteScheduler:
    if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
        raise TypeError(
            "Cosmos-Dreams requires FlowMatchEulerDiscreteScheduler, got "
            f"{type(scheduler).__name__}."
        )
    if not scheduler.config.stochastic_sampling:
        raise ValueError(
            "Cosmos-Dreams requires a scheduler with stochastic_sampling=True."
        )
    if int(scheduler.config.num_train_timesteps) != manifest.num_train_timesteps:
        raise ValueError(
            "Cosmos-Dreams scheduler and manifest disagree on num_train_timesteps: "
            f"{scheduler.config.num_train_timesteps} != {manifest.num_train_timesteps}."
        )
    return scheduler


# The reference mixes seed, chunk start, and step into one SDE noise seed
# (omni_mot_causal_model._run_distilled_ar_sampler); matching it keeps parity
# runs comparable at the latent level.
SDE_FRAME_SEED_STRIDE = 1_000_003
SDE_STEP_SEED_STRIDE = 9_176


def sde_step_generator(
    *, seed: int, frame_start: int, step_index: int, device: torch.device
) -> torch.Generator:
    """Generator for the SDE noise injected after ``step_index`` of the chunk at ``frame_start``."""
    step_seed = (
        seed + frame_start * SDE_FRAME_SEED_STRIDE + step_index * SDE_STEP_SEED_STRIDE
    )
    return torch.Generator(device=device).manual_seed(step_seed)


def run_fixed_step_sde(
    transformer: CosmosDreamsTransformer,
    scheduler: FlowMatchEulerDiscreteScheduler,
    noise: torch.Tensor,
    *,
    t_list: Sequence[float],
    seed: int,
    frame_start: int,
    dtype: torch.dtype,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Fixed-step SDE denoising of one chunk; history K/V is read but never written."""
    scheduler.set_shift(1.0)
    scheduler.set_timesteps(sigmas=list(t_list), device=noise.device)
    latents = noise.float()
    for step_index, timestep in enumerate(scheduler.timesteps):
        with set_forward_context(current_timestep=step_index, attn_metadata=None):
            output = transformer(
                latents.to(dtype),
                timestep.reshape(1),
                frame_start=frame_start,
                condition_vision=False,
                **forward_kwargs,
            )
        latents = scheduler.step(
            output.video.float(),
            timestep,
            latents,
            generator=sde_step_generator(
                seed=seed,
                frame_start=frame_start,
                step_index=step_index,
                device=noise.device,
            ),
            return_dict=False,
        )[0]
    return latents.to(dtype)


def commit_clean_kv(
    transformer: CosmosDreamsTransformer,
    latent: torch.Tensor,
    *,
    history: list[KVPair] | None,
    dtype: torch.dtype,
    tokens_per_frame: int,
    sink_frames: int,
    window_frames: int | None,
    **forward_kwargs: Any,
) -> list[KVPair]:
    """Clean-refresh ``latent`` (timestep 0, no time embedding) and commit its K/V."""
    with set_forward_context(current_timestep=0, attn_metadata=None):
        output = transformer(
            latent.to(dtype),
            torch.zeros(1, device=latent.device, dtype=torch.float32),
            history_kv=history,
            condition_vision=True,
            **forward_kwargs,
        )
    return append_kv_history(
        history,
        output.current_kv,
        tokens_per_frame=tokens_per_frame,
        sink_frames=sink_frames,
        window_frames=window_frames,
    )


class _RolloutContext(msgspec.Struct):
    """Per-request constants shared by every chunk of one rollout."""

    text_kv: list[KVPair]
    fps: float
    domain_ids: torch.Tensor
    tokens_per_frame: int
    latent_channels: int
    geometry: CosmosDreamsGeometry
    device: torch.device
    dtype: torch.dtype


class CosmosDreamsRolloutStage(PipelineStage):
    """Autoregressive chunked denoising with committed clean K/V history."""

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
        self.transformer = transformer
        self.scheduler = validate_fixed_step_scheduler(scheduler, manifest)
        self.manifest = manifest

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("text_ids", batch.extra.get(EXTRA_TEXT_IDS), V.is_tensor)
        result.add_check("geometry", batch.extra.get(EXTRA_GEOMETRY), V.not_none)
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

    def _rollout(self, batch: Req) -> torch.Tensor:
        manifest = self.manifest
        device = get_local_torch_device()
        text_kv, _ = self.transformer.encode_und_kv(
            batch.extra[EXTRA_TEXT_IDS], batch.extra[EXTRA_TEXT_MASK]
        )
        geometry: CosmosDreamsGeometry = batch.extra[EXTRA_GEOMETRY]
        context = _RolloutContext(
            text_kv=text_kv,
            fps=float(batch.fps),
            domain_ids=torch.tensor(
                [batch.extra[EXTRA_DOMAIN_ID]], device=device, dtype=torch.long
            ),
            tokens_per_frame=geometry.tokens_per_frame(
                manifest.action_tokens_per_frame
            ),
            latent_channels=self.transformer.latent_channel,
            geometry=geometry,
            device=device,
            dtype=torch.bfloat16,
        )
        rows: torch.Tensor | None = batch.extra[EXTRA_ACTION_ROWS]
        target_frame: int = batch.extra[EXTRA_TARGET_LATENT_FRAMES]
        seed = batch.seed
        if not isinstance(seed, int):
            raise ValueError(
                f"Cosmos-Dreams requires a single integer seed, got {seed!r}."
            )

        history: list[KVPair] | None = None
        latents: list[torch.Tensor] = []
        next_frame = 0
        if batch.image_latent is not None:
            initial_latent = batch.image_latent.to(device=device, dtype=context.dtype)
            action, null_indexes = self._actions(
                rows, frame_start=0, frame_end=1, context=context
            )
            if target_frame > 1:
                history = self._commit_clean_frame(
                    context,
                    history,
                    initial_latent,
                    frame_idx=0,
                    action=action,
                    null_action=bool(null_indexes),
                )
            latents.append(initial_latent)
            next_frame = 1

        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            next_frame, target_frame, manifest.chunk_size
        ):
            action, null_indexes = self._actions(
                rows, frame_start=chunk_start, frame_end=chunk_end, context=context
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
            clean_chunk = self._denoise_chunk(
                context,
                noise,
                seed=seed,
                history=history,
                frame_start=chunk_start,
                action=action,
                null_indexes=null_indexes,
            )
            for local_idx, frame_idx in iter_clean_commit_frames(
                chunk_start, chunk_end, target_frame=target_frame
            ):
                action_count = manifest.action_tokens_per_frame
                history = self._commit_clean_frame(
                    context,
                    history,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_idx=frame_idx,
                    action=action[
                        :, local_idx * action_count : (local_idx + 1) * action_count
                    ],
                    null_action=local_idx in null_indexes,
                )
            latents.append(clean_chunk)
            self.log_info(
                f"Committed latent frames [{chunk_start}, {chunk_end}) of {target_frame}"
            )
        return torch.cat(latents, dim=2)

    def _actions(
        self,
        rows: torch.Tensor | None,
        *,
        frame_start: int,
        frame_end: int,
        context: _RolloutContext,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        action, null_indexes = actions_for_frames(
            rows,
            frame_start=frame_start,
            frame_end=frame_end,
            action_tokens_per_frame=self.manifest.action_tokens_per_frame,
            model_action_dim=self.manifest.max_action_dim,
        )
        return action.to(device=context.device, dtype=context.dtype), null_indexes

    def _denoise_chunk(
        self,
        context: _RolloutContext,
        initial_noise: torch.Tensor,
        *,
        seed: int,
        history: list[KVPair] | None,
        frame_start: int,
        action: torch.Tensor,
        null_indexes: tuple[int, ...],
    ) -> torch.Tensor:
        """Four fixed-step SDE updates; history K/V is read but never written."""
        return run_fixed_step_sde(
            self.transformer,
            self.scheduler,
            initial_noise,
            t_list=self.manifest.t_list,
            seed=seed,
            frame_start=frame_start,
            dtype=context.dtype,
            text_kv=context.text_kv,
            fps=context.fps,
            action_latents=action,
            action_domain_ids=context.domain_ids,
            history_kv=history,
            null_action_frame_indexes=null_indexes,
        )

    def _commit_clean_frame(
        self,
        context: _RolloutContext,
        history: list[KVPair] | None,
        latent: torch.Tensor,
        *,
        frame_idx: int,
        action: torch.Tensor,
        null_action: bool,
    ) -> list[KVPair]:
        """Refresh one clean frame (timestep 0, no time embedding) and commit its K/V."""
        return commit_clean_kv(
            self.transformer,
            latent,
            history=history,
            dtype=context.dtype,
            tokens_per_frame=context.tokens_per_frame,
            sink_frames=self.manifest.sink_frames,
            window_frames=self.manifest.window_frames,
            text_kv=context.text_kv,
            frame_start=frame_idx,
            fps=context.fps,
            action_latents=action,
            action_domain_ids=context.domain_ids,
            null_action_frame_indexes=(0,) if null_action else (),
        )
