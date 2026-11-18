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
from contextlib import contextmanager, nullcontext
from typing import Any, NamedTuple

import msgspec
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.dits.cosmos_dreams import (
    HISTORY_MODE_FULL,
    TEXT_TOKENS_TRAINING_MAX,
    UNIFIED_ACTION_DIM,
    UNIFIED_ACTION_SPACE,
    UNIFIED_CAMERA_SLOTS,
    AffineTransform,
    CosmosDreamsInferenceProfile,
    CosmosDreamsManifest,
    CosmosDreamsUnifiedActionContract,
    load_unified_normalizer,
)
from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows
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
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
    finish_streamed_encode,
    streamed_video_output_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    denormalize_vae_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_action import (
    VIDEO_RES_SIZE_INFO,
    VIEWPOINT_TEMPLATES,
    canonical_aspect_ratio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.frustum import (
    FrustumHistorySelector,
    describe_selection,
    resolve_frustum_selector,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.kv_history import (
    CosmosDreamsKVHistory,
    SelectedKVHistory,
    new_kv_history,
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
# Per-action-row domain ids ``[T]`` (unified checkpoints), or None for one domain.
EXTRA_ACTION_DOMAIN_IDS = "cosmos_dreams_action_domain_ids"
# Decoded video handed to the decoding stage when the rollout overlapped decoding
# (the Cosmos3 stage already honours this key for pre-decoded output).
EXTRA_DECODED_OUTPUT = "transfer_decoded_output"
# Paths of the MP4 the rollout encoded while it decoded; the decoding stage
# returns them instead of frames.
EXTRA_STREAMED_OUTPUT_PATHS = "cosmos_dreams_streamed_output_paths"
EXTRA_TARGET_LATENT_FRAMES = "cosmos_dreams_target_latent_frames"
# FrustumHistorySelector of the request, None without retrieval.
EXTRA_FRUSTUM_SELECTOR = "cosmos_dreams_frustum_selector"
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


def load_action_payload(action: Any) -> tuple[torch.Tensor, list[str] | None]:
    """Rows plus optional per-row embodiment names from a request value.

    Besides the bare row forms of :func:`load_action_rows`, accepts the
    imaginaire4 unified sidecar ``{"action_space": "normalized_unified_v1",
    "action": rows, "domain_names": [...]}``.
    """
    if isinstance(action, str):
        text = action.strip()
        if os.path.isfile(text) and text.endswith(".json"):
            with open(text, encoding="utf-8") as handle:
                action = json.load(handle)
        elif text.startswith("{"):
            action = json.loads(text)
    if isinstance(action, dict):
        space = action.get("action_space")
        if space is not None and space != UNIFIED_ACTION_SPACE:
            raise ValueError(
                f"Cosmos-Dreams action payload action_space must be {UNIFIED_ACTION_SPACE!r}, "
                f"got {space!r}."
            )
        if "action" not in action:
            raise ValueError(
                "Cosmos-Dreams action payload object needs an `action` list."
            )
        names = action.get("domain_names")
        if names is not None and (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) for name in names)
        ):
            raise ValueError(
                "Cosmos-Dreams action payload domain_names must be a non-empty list of names."
            )
        return load_action_rows(action["action"]), names
    return load_action_rows(action), None


def row_domain_ids(
    names: list[str] | None,
    *,
    contract: CosmosDreamsUnifiedActionContract,
    rows: int,
    default_embodiment: str,
) -> torch.Tensor | None:
    """``[rows]`` domain ids for per-row routing, or None when one domain applies."""
    if names is None or len(names) == 1 and len(names) != rows:
        if names is not None:
            contract.resolve_embodiment(names[0], None)
        return None
    if len(names) != rows:
        raise ValueError(
            f"Cosmos-Dreams domain_names must name one embodiment or one per action row; "
            f"got {len(names)} names for {rows} rows."
        )
    ids = [
        contract.embodiment_to_domain[contract.resolve_embodiment(name, None)]
        for name in names
    ]
    if (
        len(set(ids)) == 1
        and ids[0] == contract.embodiment_to_domain[default_embodiment]
    ):
        return None
    return torch.tensor(ids, dtype=torch.long)


def check_unified_rows(
    rows: torch.Tensor,
    *,
    contract: CosmosDreamsUnifiedActionContract,
    domain_ids: torch.Tensor | None,
    scalar_domain: int,
) -> None:
    """Width, finiteness and the camera rule (only ego-pose slots may be non-zero)."""
    if rows.shape[-1] != UNIFIED_ACTION_DIM:
        raise ValueError(
            f"Cosmos-Dreams unified action rows must have {UNIFIED_ACTION_DIM} slots, "
            f"got {rows.shape[-1]}."
        )
    if not torch.isfinite(rows).all():
        raise ValueError(
            "Cosmos-Dreams normalized actions must contain only finite values."
        )
    camera = contract.camera_domain_id
    if camera is None:
        return
    if domain_ids is None:
        camera_rows = rows if scalar_domain == camera else rows[:0]
    else:
        camera_rows = rows[domain_ids == camera]
    if camera_rows.numel() and torch.count_nonzero(
        camera_rows[:, UNIFIED_CAMERA_SLOTS:]
    ):
        raise ValueError(
            f"Cosmos-Dreams camera rows must have zeros outside the first {UNIFIED_CAMERA_SLOTS} slots."
        )


def domains_for_frames(
    row_domains: torch.Tensor | None,
    *,
    scalar_domain: int,
    frame_start: int,
    frame_end: int,
    action_tokens_per_frame: int,
) -> torch.Tensor:
    """Domain id per action token of latent frames ``[start, end)``.

    Returns ``[1]`` when one domain covers the chunk, else ``[1, F * A]``.
    Frame 0 carries null actions and takes the first real row's domain, as the
    reference packer does.
    """
    if row_domains is None:
        return torch.tensor([scalar_domain], dtype=torch.long)
    blocks: list[torch.Tensor] = []
    for frame_idx in range(frame_start, frame_end):
        if frame_idx == 0:
            blocks.append(row_domains[:1].expand(action_tokens_per_frame))
        else:
            start = (frame_idx - 1) * action_tokens_per_frame
            blocks.append(row_domains[start : start + action_tokens_per_frame])
    selected = torch.cat(blocks)
    if bool((selected == selected[0]).all()):
        return selected[:1].clone()
    return selected.unsqueeze(0)


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


def prepare_action_rows(
    action: Any,
    *,
    manifest: CosmosDreamsManifest,
    embodiment: str,
    target_frame: int,
    pre_normalized: bool,
    domain_names: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Model-width action rows for the rollout plus per-row domain ids (or None).

    Native contracts (schema 3) take rows in the embodiment's raw units and
    apply the exported affine unless ``pre_normalized``. Unified contracts
    (schema 5) take normalized 59-D rows only; ``domain_names`` from the
    request or the sidecar route rows to embodiments individually.
    """
    rows, sidecar_names = load_action_payload(action)
    if domain_names is None:
        domain_names = sidecar_names
    contract = manifest.action_contract
    action_count = manifest.action_tokens_per_frame
    domain_ids = None
    if isinstance(contract, CosmosDreamsUnifiedActionContract):
        domain_ids = row_domain_ids(
            domain_names,
            contract=contract,
            rows=rows.shape[0],
            default_embodiment=embodiment,
        )
        rows = rows.to(dtype=torch.float32)
        check_unified_rows(
            rows,
            contract=contract,
            domain_ids=domain_ids,
            scalar_domain=contract.embodiment_to_domain[embodiment],
        )
    else:
        if domain_names is not None and len(set(domain_names)) > 1:
            raise ValueError(
                "Per-row domain_names need a unified (schema-5) checkpoint; this "
                "checkpoint routes one embodiment per request."
            )
        embodiment_contract = contract.embodiments[embodiment]
        if rows.shape[-1] != embodiment_contract.raw_action_dim:
            raise ValueError(
                "Cosmos-Dreams action rows do not match the embodiment's raw action dim "
                f"{embodiment_contract.raw_action_dim}, got {rows.shape[-1]}."
            )
        if pre_normalized:
            rows = rows.to(dtype=torch.float32)
            if not torch.isfinite(rows).all():
                raise ValueError(
                    "Cosmos-Dreams normalized actions must contain only finite values."
                )
        else:
            rows = normalize_action_rows(rows, embodiment_contract.normalizer.transform)
    rows = pad_action_rows(rows, manifest.max_action_dim)
    required_rows = (target_frame - 1) * action_count
    if rows.shape[0] < required_rows:
        raise ValueError(
            f"Cosmos-Dreams action has {rows.shape[0]} rows but {target_frame} latent "
            f"frames need {required_rows} (one row per pixel step after frame 0)."
        )
    return rows, domain_ids


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
    the bottom and right; content stays top-left. Smaller inputs are never
    upscaled (``reflection_pad_to_target`` caps the scale at 1.0), so a
    640x360 robot frame keeps its native size inside an 832x480 canvas.
    Returns the canvas tensor and the ``(content_height, content_width)`` region.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected an RGB image [3, H, W], got {tuple(image.shape)}.")
    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(target_width / width, target_height / height, 1.0)
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
    tokenizer: Any,
    prompt: str,
    *,
    device: torch.device,
    system_prompt: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen2 chat-template tokens for the prompt plus EOS and vision start.

    Bimanual has no system prompt; Transfer supplies its trained system prompt.
    K/V is trimmed to the real length, so nothing is padded.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    result = tokenizer.apply_chat_template(
        messages,
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


def frame_domain_ids(
    chunk_domains: torch.Tensor, local_idx: int, action_tokens_per_frame: int
) -> torch.Tensor:
    """The chunk's domain ids restricted to one frame (``[1]`` stays ``[1]``)."""
    if chunk_domains.ndim == 1:
        return chunk_domains
    start = local_idx * action_tokens_per_frame
    return chunk_domains[:, start : start + action_tokens_per_frame]


class CausalDecodeQueue:
    """Decode committed latent chunks with the causal Wan VAE on a side stream.

    Each chunk waits for the producing stream and decodes with the decoder's
    feature cache carried across chunks. Its frames go to ``encoder`` when there
    is one, otherwise they land in pinned host memory asynchronously; leaving the
    context drains the stream and resets the VAE, then ``finish`` concatenates
    the completed frames.
    """

    def __init__(
        self,
        vae: Any,
        *,
        device: torch.device,
        stream: torch.cuda.Stream | None = None,
        fast_path: bool = False,
        encoder: EncodeWhileDecoding | None = None,
    ) -> None:
        self.vae = vae
        self.device = device
        self.fast_path = fast_path
        self.encoder = encoder
        self.stream = stream or torch.cuda.Stream(device=device)
        self.outputs: list[torch.Tensor] = []
        self.submitted = 0
        self.vae.reset_causal_decode_state()
        config = vae.config
        dtype = next(vae.parameters()).dtype
        self._mean = torch.tensor(config.latents_mean, device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )
        self._std = torch.tensor(config.latents_std, device=device, dtype=dtype).view(
            1, -1, 1, 1, 1
        )

    def __enter__(self) -> "CausalDecodeQueue":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.vae.reset_causal_decode_state()
            if exc_type is not None:
                self.outputs.clear()
                if self.encoder is not None:
                    self.encoder.abort()
        finally:
            # Drain last: freeing pinned outputs records host-allocator events
            # on this stream, so they must land before the final sync.
            self.stream.synchronize()

    def _denormalize(self, latents: torch.Tensor) -> torch.Tensor:
        return denormalize_vae_latents(latents, self._mean, self._std)

    def submit(self, latents: torch.Tensor) -> None:
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        latents = latents.detach()
        latents.record_stream(self.stream)
        with (
            torch.cuda.stream(self.stream),
            torch.no_grad(),
            use_vae_fast_path(self.vae, self.fast_path),
        ):
            video = self.vae.causal_decode(self._denormalize(latents))
            if self.encoder is not None:
                # Postprocesses ``video`` in place and queues its frames.
                self.encoder(video)
            else:
                host = torch.empty(
                    video.shape, dtype=video.dtype, device="cpu", pin_memory=True
                )
                # Keep the destination referenced before the copy is enqueued.
                self.outputs.append(host)
                host.copy_(video, non_blocking=True)
        self.submitted += 1

    def finish(self) -> torch.Tensor:
        return torch.cat(self.outputs, dim=2)


@contextmanager
def rollout_stream(
    device: torch.device, stream: torch.cuda.Stream | None
) -> Iterator[None]:
    """Issue the block's CUDA work on ``stream``, a higher-priority one (None: stay put).

    A rollout that overlaps its VAE decode shares the GPU with it. On a stream
    that outranks the decode's, the rollout's kernels take freed SMs first, so
    the contention lands on the decode, which has slack. Scheduling only: every
    kernel runs with the same inputs in the same per-stream order.
    """
    if stream is None:
        yield
        return
    entry = torch.cuda.current_stream(device)
    stream.wait_stream(entry)
    try:
        with torch.cuda.stream(stream):
            yield
    finally:
        # Later stages run on the entry stream and must see the block's work done.
        entry.wait_stream(stream)


class CosmosDreamsPrepareStage(PipelineStage):
    """Validate the request and build every input the rollout needs.

    Writes the tokenized prompt, resolved geometry, normalized zero-padded
    action rows, and the VAE latent of the conditioning image into ``batch``.
    """

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        *,
        vae,
        tokenizer,
        manifest: CosmosDreamsManifest,
        max_pixels: int,
        profile: CosmosDreamsInferenceProfile | None = None,
        model_path: str | None = None,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.tokenizer = tokenizer
        self.manifest = manifest
        self.max_pixels = max_pixels
        self.profile = profile
        self.normalizer_stats = unified_normalizer_stats(manifest, model_path)

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
        self,
        action: Any,
        *,
        embodiment: str,
        target_frame: int,
        pre_normalized: bool = False,
        domain_names: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if action is None:
            return None, None
        return prepare_action_rows(
            action,
            manifest=self.manifest,
            embodiment=embodiment,
            target_frame=target_frame,
            pre_normalized=pre_normalized,
            domain_names=domain_names,
        )

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
        rows, row_domains = self._prepare_action_rows(
            batch.sampling_params.action,
            embodiment=prepared.embodiment,
            target_frame=target_frame,
            pre_normalized=batch.sampling_params.actions_pre_normalized,
            domain_names=batch.sampling_params.domain_names,
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
        batch.extra[EXTRA_ACTION_DOMAIN_IDS] = row_domains
        batch.extra[EXTRA_DOMAIN_ID] = prepared.domain_id
        batch.extra[EXTRA_TARGET_LATENT_FRAMES] = target_frame
        batch.extra[EXTRA_FRUSTUM_SELECTOR] = self._frustum_selector(
            batch,
            rows=rows,
            target_frame=target_frame,
            has_image=prepared.image_latent is not None,
        )
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

    def _frustum_selector(
        self,
        batch: Req,
        *,
        rows: torch.Tensor | None,
        target_frame: int,
        has_image: bool,
    ) -> FrustumHistorySelector | None:
        geometry = batch.sampling_params.frustum_geometry
        if geometry is None:
            return None
        if self.profile is None or self.profile.history_mode != HISTORY_MODE_FULL:
            raise ValueError(
                "Cosmos-Dreams frustum_geometry reads the complete committed archive: "
                "set history_mode='full' in the pipeline config (the default 'auto' "
                "slides when the export records a K/V window) and leave "
                "kv_cache_inference_size/attention_sink_size unset."
            )
        if not has_image:
            raise ValueError(
                "Cosmos-Dreams frustum_geometry needs a conditioning image."
            )
        if target_frame - 1 > self.profile.sink_frames + self.profile.window_frames:
            raise ValueError(
                "Cosmos-Dreams frustum_geometry needs the whole rollout in the archive; "
                f"raise history_max_frames to cover {batch.num_frames} pixel frames."
            )
        selector = resolve_frustum_selector(
            geometry,
            rows=None if rows is None else rows.detach().to("cpu"),
            normalizer_stats=self.normalizer_stats,
            num_latent_frames=target_frame,
            temporal_compression_factor=self.manifest.temporal_compression_factor,
            fps=float(batch.fps),
        )
        sinks, retrieved, recent = selector.sizes
        self.log_info(
            f"Frustum K/V retrieval: {sinks} sinks + {retrieved} retrieved + {recent} recent "
            f"past latent frames over {target_frame}; poses "
            + (
                "recovered from the unified rows"
                if "pose_source" in geometry
                else "from c2w"
            )
        )
        return selector


def unified_normalizer_stats(
    manifest: CosmosDreamsManifest, model_path: str | None
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """``(offset, scale)`` of the schema-5 normalizer; None for native contracts or
    when the export path is unknown."""
    contract = manifest.conditioning
    if model_path is None or not isinstance(
        contract, CosmosDreamsUnifiedActionContract
    ):
        return None
    normalizer = load_unified_normalizer(model_path, contract)
    return tuple(normalizer.offset), tuple(normalizer.scale)


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


class _PreparedSchedule(NamedTuple):
    timesteps: torch.Tensor
    sigmas: torch.Tensor
    begin_index: int
    # One (scalar, [1]) view pair of ``timesteps`` per step: what the scheduler
    # and the transformer take.
    steps: tuple[tuple[torch.Tensor, torch.Tensor], ...]


class FixedStepSchedules:
    """The scheduler's state for each sigma list of a rollout, built once.

    ``set_timesteps`` uploads the sigmas with a blocking copy, and the first
    ``step`` after it searches the device for its index and reads it back.
    """

    def __init__(self, scheduler: FlowMatchEulerDiscreteScheduler) -> None:
        self.scheduler = scheduler
        self._prepared: dict[
            tuple[tuple[float, ...], torch.device], _PreparedSchedule
        ] = {}

    def begin_chunk(
        self, t_list: Sequence[float], device: torch.device
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        scheduler = self.scheduler
        key = (tuple(float(sigma) for sigma in t_list), device)
        prepared = self._prepared.get(key)
        if prepared is None:
            scheduler.set_shift(1.0)
            scheduler.set_timesteps(sigmas=list(key[0]), device=device)
            prepared = _PreparedSchedule(
                timesteps=scheduler.timesteps,
                sigmas=scheduler.sigmas,
                # What the first step would otherwise look up on the device.
                begin_index=scheduler.index_for_timestep(scheduler.timesteps[0]),
                steps=tuple(
                    (timestep, timestep.reshape(1)) for timestep in scheduler.timesteps
                ),
            )
            self._prepared[key] = prepared
        else:
            scheduler.timesteps = prepared.timesteps
            scheduler.sigmas = prepared.sigmas
            scheduler.num_inference_steps = len(prepared.steps)
            scheduler._step_index = None
        scheduler.set_begin_index(prepared.begin_index)
        return prepared.steps


# The reference mixes seed, chunk start, and step into one SDE noise seed
# (omni_mot_causal_model._run_distilled_ar_sampler); matching it keeps parity
# runs comparable at the latent level.
SDE_FRAME_SEED_STRIDE = 1_000_003
SDE_STEP_SEED_STRIDE = 9_176


def sde_step_generator(
    *, seed: int, frame_start: int, step_index: int, device: torch.device
) -> torch.Generator:
    """Generator for the SDE noise injected after ``step_index`` of the chunk at ``frame_start``.

    Reinjection steps count from one (imaginaire4 a64c10426a2): with the initial
    chunk noise drawn from ``seed + frame_start``, a zero-based step at frame 0
    would reuse that noise.
    """
    step_seed = (
        seed
        + frame_start * SDE_FRAME_SEED_STRIDE
        + (step_index + 1) * SDE_STEP_SEED_STRIDE
    )
    return torch.Generator(device=device).manual_seed(step_seed)


def run_fixed_step_sde(
    transformer: CosmosDreamsTransformer,
    schedules: FixedStepSchedules,
    noise: torch.Tensor,
    *,
    t_list: Sequence[float],
    seed: int,
    frame_start: int,
    dtype: torch.dtype,
    forward_batch: Req | None = None,
    **forward_kwargs: Any,
) -> torch.Tensor:
    """Fixed-step SDE denoising of one chunk; history K/V is read but never committed."""
    scheduler = schedules.scheduler
    latents = noise.float()
    for step_index, (timestep, forward_timestep) in enumerate(
        schedules.begin_chunk(t_list, noise.device)
    ):
        with set_forward_context(
            current_timestep=step_index, attn_metadata=None, forward_batch=forward_batch
        ):
            output = transformer(
                latents.to(dtype),
                forward_timestep,
                frame_start=frame_start,
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
    history: CosmosDreamsKVHistory | SelectedKVHistory,
    dtype: torch.dtype,
    tokens_per_frame: int,
    window_frames: int | None,
    forward_batch: Req | None = None,
    **forward_kwargs: Any,
) -> None:
    """Clean-refresh ``latent`` (no time embedding) and commit its K/V.

    The pass attends to ``history`` and writes behind it; ``window_frames`` bounds
    what the history keeps after its sink (None keeps everything).
    """
    chunk_tokens = latent.shape[2] * tokens_per_frame
    with set_forward_context(
        current_timestep=0, attn_metadata=None, forward_batch=forward_batch
    ):
        transformer.write_clean_kv(
            latent.to(dtype), kv_slots=history.slots(chunk_tokens), **forward_kwargs
        )
    history.commit(
        chunk_tokens,
        tail_tokens=None if window_frames is None else window_frames * tokens_per_frame,
    )


class _RolloutContext(msgspec.Struct):
    """Per-request constants shared by every chunk of one rollout."""

    text_kv: list[KVPair]
    fps: float
    domain_id: int
    # ``[1]`` request-level domain; ``row_domain_ids`` (``[T]``) overrides it per row.
    domain_ids: torch.Tensor
    tokens_per_frame: int
    latent_channels: int
    geometry: CosmosDreamsGeometry
    device: torch.device
    dtype: torch.dtype
    forward_batch: Req | None = None
    row_domain_ids: torch.Tensor | None = None

    @property
    def text_length(self) -> int:
        return int(self.text_kv[0][0].shape[1])

    def chunk_domain_ids(
        self, frame_start: int, frame_end: int, action_count: int
    ) -> torch.Tensor:
        if self.row_domain_ids is None:
            return self.domain_ids
        return domains_for_frames(
            self.row_domain_ids,
            scalar_domain=self.domain_id,
            frame_start=frame_start,
            frame_end=frame_end,
            action_tokens_per_frame=action_count,
        ).to(device=self.device)


class CosmosDreamsRolloutStage(PipelineStage):
    """Autoregressive chunked denoising with committed clean K/V history."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        *,
        transformer: CosmosDreamsTransformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        manifest: CosmosDreamsManifest,
        profile: CosmosDreamsInferenceProfile | None = None,
        vae: Any = None,
        overlap_vae_decode: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"Cosmos-Dreams requires CosmosDreamsTransformer, got {type(transformer).__name__}."
            )
        self.transformer = transformer
        self.scheduler = validate_fixed_step_scheduler(scheduler, manifest)
        self.schedules = FixedStepSchedules(self.scheduler)
        self.manifest = manifest
        # With a VAE the rollout decodes committed chunks on a side stream; the
        # decoding stage then only post-processes.
        self.vae = vae
        self.overlap_vae_decode = bool(overlap_vae_decode and vae is not None)
        # Streams of the overlapped decode and, when enabled, of the rollout that
        # outranks it, made on first use and kept: the CUDA allocator caches freed
        # memory per stream, so a new stream per request strands the previous one's.
        self._decode_stream: torch.cuda.Stream | None = None
        self._priority_stream: torch.cuda.Stream | None = None
        # Without a resolved profile the stage runs the artifact as exported:
        # its t_list on every chunk and its K/V window.
        self.profile = profile or CosmosDreamsInferenceProfile(
            frame_sigma_schedules=(tuple(manifest.t_list),),
            window_frames=manifest.window_frames,
            sink_frames=manifest.sink_frames,
            history_mode="sliding",
        )

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check("text_ids", batch.extra.get(EXTRA_TEXT_IDS), V.is_tensor)
        result.add_check("geometry", batch.extra.get(EXTRA_GEOMETRY), V.not_none)
        return result

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        name = self._component_stage_name(stage_name)
        uses = [
            ComponentUse(
                name,
                "transformer",
                phase="denoise",
                preferred_ready_after_request=True,
                memory_intensive=True,
                start_at_stage_entry=False,
            )
        ]
        if self.overlap_vae_decode and not server_args.should_start_component_on_cpu(
            "vae"
        ):
            uses.append(ComponentUse(name, "vae", keep_ready_after_warmup=True))
        return uses

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        with self.use_declared_component(
            component_name="transformer", module=self.transformer, phase="denoise"
        ):
            with torch.no_grad():
                batch.latents = self._rollout(batch)
        return batch

    def _decode_queue(
        self, device: torch.device, batch: Req
    ) -> "CausalDecodeQueue | None":
        if (
            not self.overlap_vae_decode
            or device.type != "cuda"
            or self.server_args.should_start_component_on_cpu("vae")
        ):
            return None
        if self._decode_stream is None:
            self._decode_stream = torch.cuda.Stream(device=device)
            if envs.SGLANG_DIFFUSION_COSMOS_DREAMS_ROLLOUT_STREAM_PRIORITY:
                # 0 is the lowest CUDA stream priority; lower numbers rank higher.
                self._priority_stream = torch.cuda.Stream(device=device, priority=-1)
        return CausalDecodeQueue(
            self.vae,
            device=device,
            stream=self._decode_stream,
            fast_path=quality_allows(batch.sampling_params.quality, "lossless"),
            encoder=self._streamed_encoder(batch),
        )

    def _streamed_encoder(self, batch: Req) -> EncodeWhileDecoding | None:
        save_file_path = streamed_video_output_path(batch, self.server_args, 1)
        if save_file_path is None:
            return None
        return EncodeWhileDecoding(
            save_file_path,
            batch,
            finish_frames=Cosmos3DecodingStage._postprocess_tensor,
        )

    def _rollout_context(self, batch: Req, device: torch.device) -> _RolloutContext:
        text_kv, _ = self.transformer.encode_und_kv(
            batch.extra[EXTRA_TEXT_IDS], batch.extra[EXTRA_TEXT_MASK]
        )
        geometry: CosmosDreamsGeometry = batch.extra[EXTRA_GEOMETRY]
        row_domain_ids = batch.extra.get(EXTRA_ACTION_DOMAIN_IDS)
        return _RolloutContext(
            text_kv=text_kv,
            fps=float(batch.fps),
            domain_id=batch.extra[EXTRA_DOMAIN_ID],
            domain_ids=torch.tensor(
                [batch.extra[EXTRA_DOMAIN_ID]], device=device, dtype=torch.long
            ),
            tokens_per_frame=geometry.tokens_per_frame(
                self.manifest.action_tokens_per_frame
            ),
            latent_channels=self.transformer.latent_channel,
            geometry=geometry,
            device=device,
            dtype=torch.bfloat16,
            forward_batch=batch,
            row_domain_ids=None if row_domain_ids is None else row_domain_ids.to("cpu"),
        )

    def _new_history(
        self, context: _RolloutContext, *, rollout_frames: int | None
    ) -> CosmosDreamsKVHistory:
        return new_kv_history(
            context.text_kv,
            tokens_per_frame=context.tokens_per_frame,
            sink_frames=self.profile.sink_frames,
            window_frames=self.profile.window_frames,
            chunk_frames=self.manifest.chunk_size,
            rollout_frames=rollout_frames,
        )

    def _rollout(self, batch: Req) -> torch.Tensor:
        device = get_local_torch_device()
        target_frame: int = batch.extra[EXTRA_TARGET_LATENT_FRAMES]
        if not isinstance(batch.seed, int):
            raise ValueError(
                f"Cosmos-Dreams requires a single integer seed, got {batch.seed!r}."
            )
        if target_frame - 1 > self.profile.sink_frames + self.profile.window_frames:
            self.log_info(
                f"Rollout of {target_frame} latent frames exceeds the {self.profile.describe()}; "
                "the oldest committed frames are evicted from the K/V history once it fills."
            )
        decode_queue = self._decode_queue(device, batch)
        with decode_queue or nullcontext():
            # Only a rollout that shares the GPU with its decode has anything to
            # outrank; the stream exists only when the priority is switched on.
            with rollout_stream(
                device, None if decode_queue is None else self._priority_stream
            ):
                latents = self._rollout_chunks(
                    batch,
                    self._rollout_context(batch, device),
                    target_frame=target_frame,
                    decode_queue=decode_queue,
                )
        if decode_queue is not None:
            self._hand_over_decode(batch, decode_queue)
        return latents

    def _hand_over_decode(self, batch: Req, decode_queue: "CausalDecodeQueue") -> None:
        """Tell the decoding stage what the rollout already produced."""
        self.log_info("Decoded %d chunks during the rollout", decode_queue.submitted)
        output_file_paths, encode_failed = finish_streamed_encode(decode_queue.encoder)
        if encode_failed:
            # A failed encode leaves nothing behind, so the decoding stage decodes
            # the latents itself and the worker saves the frames. Ranks without
            # the encoder keep their frames back too, to take part in that decode.
            return
        if output_file_paths is not None:
            batch.extra[EXTRA_STREAMED_OUTPUT_PATHS] = output_file_paths
        else:
            batch.extra[EXTRA_DECODED_OUTPUT] = decode_queue.finish()

    def _rollout_chunks(
        self,
        batch: Req,
        context: _RolloutContext,
        *,
        target_frame: int,
        decode_queue: "CausalDecodeQueue | None",
    ) -> torch.Tensor:
        manifest = self.manifest
        rows: torch.Tensor | None = batch.extra[EXTRA_ACTION_ROWS]
        seed: int = batch.seed
        action_count = manifest.action_tokens_per_frame
        selector: FrustumHistorySelector | None = batch.extra[EXTRA_FRUSTUM_SELECTOR]
        history = self._new_history(context, rollout_frames=target_frame)
        # Frustum retrieval reads a gathered subset; its commits still land in ``history``.
        retrieval = (
            None
            if selector is None
            else SelectedKVHistory(
                history,
                context.text_kv,
                selected_tokens=selector.budget * context.tokens_per_frame,
                chunk_tokens=manifest.chunk_size * context.tokens_per_frame,
            )
        )
        latents: list[torch.Tensor] = []
        if batch.image_latent is not None:
            latents.append(
                self._commit_image_frame(
                    batch, context, history, rows=rows, commit=target_frame > 1
                )
            )
            if decode_queue is not None:
                decode_queue.submit(latents[0])

        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            len(latents), target_frame, manifest.chunk_size
        ):
            action, null_indexes = self._actions(
                rows, frame_start=chunk_start, frame_end=chunk_end, context=context
            )
            chunk_domains = context.chunk_domain_ids(
                chunk_start, chunk_end, action_count
            )
            # One retrieval per chunk, shared by every denoising step and refresh;
            # the framewise refresh also sees the chunk's already-committed frames.
            selected = (
                None if selector is None else selector.select(chunk_start, chunk_end)
            )
            reader = (
                history
                if retrieval is None
                else retrieval.select(
                    selected, tokens_per_frame=context.tokens_per_frame
                )
            )
            clean_chunk = self._denoise_chunk(
                context,
                self._chunk_noise(context, seed=seed, frames=(chunk_start, chunk_end)),
                seed=seed,
                history=reader,
                frame_start=chunk_start,
                action=action,
                null_indexes=null_indexes,
                domain_ids=chunk_domains,
            )
            for local_idx, frame_idx in iter_clean_commit_frames(
                chunk_start, chunk_end, target_frame=target_frame
            ):
                self._commit_clean_frame(
                    context,
                    reader,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_idx=frame_idx,
                    action=action[
                        :, local_idx * action_count : (local_idx + 1) * action_count
                    ],
                    null_action=local_idx in null_indexes,
                    domain_ids=frame_domain_ids(chunk_domains, local_idx, action_count),
                )
            latents.append(clean_chunk)
            if decode_queue is not None:
                decode_queue.submit(clean_chunk)
            retrieved = (
                ""
                if selected is None
                else "; frustum history "
                + describe_selection(selector, chunk_start, selected)
            )
            self.log_info(
                f"Committed latent frames [{chunk_start}, {chunk_end}) of {target_frame} "
                f"({len(self.profile.sigmas_for_frame(chunk_start))} denoise steps){retrieved}"
            )
        return torch.cat(latents, dim=2)

    def _commit_image_frame(
        self,
        batch: Req,
        context: _RolloutContext,
        history: CosmosDreamsKVHistory,
        *,
        rows: torch.Tensor | None,
        commit: bool,
    ) -> torch.Tensor:
        latent = batch.image_latent.to(device=context.device, dtype=context.dtype)
        if commit:
            action, null_indexes = self._actions(
                rows, frame_start=0, frame_end=1, context=context
            )
            self._commit_clean_frame(
                context,
                history,
                latent,
                frame_idx=0,
                action=action,
                null_action=bool(null_indexes),
                domain_ids=context.chunk_domain_ids(
                    0, 1, self.manifest.action_tokens_per_frame
                ),
            )
        return latent

    @staticmethod
    def _chunk_noise(
        context: _RolloutContext, *, seed: int, frames: tuple[int, int]
    ) -> torch.Tensor:
        chunk_start, chunk_end = frames
        generator = torch.Generator(device=context.device).manual_seed(
            seed + chunk_start
        )
        # The reference draws checkpoint-dtype noise, then promotes it to fp32.
        return torch.randn(
            (
                1,
                context.latent_channels,
                chunk_end - chunk_start,
                context.geometry.latent_height,
                context.geometry.latent_width,
            ),
            generator=generator,
            device=context.device,
            dtype=context.dtype,
        )

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
        history: CosmosDreamsKVHistory | SelectedKVHistory,
        frame_start: int,
        action: torch.Tensor,
        null_indexes: tuple[int, ...],
        domain_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fixed-step SDE updates over the chunk's schedule; history K/V is read but never committed."""
        return run_fixed_step_sde(
            self.transformer,
            self.schedules,
            initial_noise,
            t_list=self.profile.sigmas_for_frame(frame_start),
            seed=seed,
            frame_start=frame_start,
            dtype=context.dtype,
            forward_batch=context.forward_batch,
            text_length=context.text_length,
            fps=context.fps,
            action_latents=action,
            action_domain_ids=context.domain_ids if domain_ids is None else domain_ids,
            kv_slots=history.slots(initial_noise.shape[2] * context.tokens_per_frame),
            null_action_frame_indexes=null_indexes,
        )

    def _commit_clean_frame(
        self,
        context: _RolloutContext,
        history: CosmosDreamsKVHistory | SelectedKVHistory,
        latent: torch.Tensor,
        *,
        frame_idx: int,
        action: torch.Tensor,
        null_action: bool,
        domain_ids: torch.Tensor | None = None,
    ) -> None:
        """Refresh one clean frame (no time embedding) and commit its K/V."""
        commit_clean_kv(
            self.transformer,
            latent,
            history=history,
            dtype=context.dtype,
            tokens_per_frame=context.tokens_per_frame,
            window_frames=self.profile.window_frames,
            forward_batch=context.forward_batch,
            text_length=context.text_length,
            frame_start=frame_idx,
            fps=context.fps,
            action_latents=action,
            action_domain_ids=context.domain_ids if domain_ids is None else domain_ids,
            null_action_frame_indexes=(0,) if null_action else (),
        )


class CosmosDreamsDecodingStage(Cosmos3DecodingStage):
    def forward(self, batch: Req, server_args: ServerArgs):
        output_file_paths = batch.extra.get(EXTRA_STREAMED_OUTPUT_PATHS)
        if output_file_paths is None:
            return super().forward(batch, server_args)
        return OutputBatch(
            output=None,
            output_file_paths=output_file_paths,
            metrics=batch.metrics,
            rollout_trajectory_data=batch.rollout_trajectory_data,
        )
