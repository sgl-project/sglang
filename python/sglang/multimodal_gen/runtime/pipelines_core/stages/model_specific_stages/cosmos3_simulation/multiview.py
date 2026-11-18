# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV pipeline stages.

One request carries every camera of the rig. The stages pack all cameras
camera-major along time (all frames of camera 0, then camera 1, ...), encode
and decode each camera separately through the temporally causal Wan VAE, and
hand the transformer a ``MultiviewLayout`` plus the temporal wrap period that
drive the masked cross-camera attention. Joint checkpoints add an
HD-map range-map control and a LiDAR target of their own geometry; the
denoised state is one flat packing of every target so the shared scheduler
steps cameras and LiDAR together. Denoising reuses ``Cosmos3DenoisingStage``:
the WSM control item rides the existing clean control-prefix mechanism and the
image anchors ride the I2V velocity mask.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from itertools import chain, islice
from typing import Any

import numpy as np
import PIL.Image
import torch
import torchvision.transforms.functional as TF

from sglang.multimodal_gen.configs.pipeline_configs.cosmos3_multiview import (
    COSMOS3_MADS_CAMERA_ATTRIBUTES,
    Cosmos3MultiviewDeploymentConfig,
)
from sglang.multimodal_gen.configs.sample.cosmos3_multiview import (
    COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH,
    MultiviewViewInput,
    closest_multiview_aspect_ratio,
    multiview_canvas,
)
from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_rank
from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentUse,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_multiview import (
    pack_state,
    unpack_state,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_layout import (
    DEFAULT_MAX_UND_TOKENS,
    MaskItem,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.decoder import (
    Cosmos3LidarDecoder,
)
from sglang.multimodal_gen.runtime.models.vaes.cosmos3_lidar.encoder import (
    Cosmos3LidarEncoder,
    pad_lidar_sweeps,
    required_lidar_sweeps,
)
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    PipelineStage,
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.encode_while_decoding import (
    EncodeWhileDecoding,
    finish_streamed_encode,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3TokenizationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.lidar_outputs import (
    lidar_output_payload,
    write_lidar_outputs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.validators import (
    VerificationResult,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.vision import (
    iter_video_frames,
    load_image,
    load_video,
    rgb_hwc_uint8,
)

logger = init_logger(__name__)

# Most clips a request decodes at once (one ffmpeg process each); its
# ``CameraLoadGate`` starts fewer while the encoder keeps up.
# Arbitrary; 8 keeps a 22-clip rig within a few seconds on a 24-core host.
COSMOS3_MULTIVIEW_MEDIA_THREADS = 8
# Frames resized at a time by ``fit_uint8_cthw``. The resize is per frame, so
# the chunk only bounds the float32 temporaries: a 297-frame 1924x1084 MADS
# control needs 15 GB of host memory resized whole and under 2 GB in chunks of
# 16, with bit-identical output. Whole-clip resizes of every camera at once,
# on every rank, took a four-GPU replica past its 660 GB limit.
COSMOS3_MULTIVIEW_RESIZE_CHUNK_FRAMES = 16
# Byte bound on one chunk's float32 source frames. Measured on seven 121-frame
# 1280x720 clips: the rig loads in 2.3 s at two frames a chunk, 3.0 s at 16.
COSMOS3_MULTIVIEW_RESIZE_CHUNK_BYTES = 32 << 20

# ``batch.extra`` keys shared by the multiview stages.
EXTRA_CONTROL_PIXELS = "multiview_control_pixels"
EXTRA_VISION_PIXELS = "multiview_vision_pixels"
EXTRA_NUM_VIEWS = "multiview_num_views"
EXTRA_FRAMES_PER_VIEW = "multiview_frames_per_view"
EXTRA_LATENT_FRAMES_PER_VIEW = "multiview_latent_frames_per_view"
EXTRA_LOCAL_CONDITION_INDEXES = "multiview_local_condition_indexes"
EXTRA_KNOWN_VIEWS = "multiview_known_views"
EXTRA_CAMERAS = "multiview_cameras"
EXTRA_VIEW_PROMPTS = "multiview_view_prompts"
EXTRA_LIDAR_FRAMES = "multiview_lidar_frames"
EXTRA_LIDAR_CONDITION_FRAMES = "multiview_lidar_condition_frames"
EXTRA_LIDAR_LATENTS = "multiview_lidar_latents"
EXTRA_PACKED_SHAPES = "multiview_packed_shapes"
EXTRA_CAPTION_LENGTHS = "multiview_caption_lengths_by_cache_key"
# Consumed by Cosmos3DenoisingStage and forwarded to every transformer call.
EXTRA_TRANSFORMER_KWARGS = "transformer_extra_kwargs"

# System prompts, verbatim from the training text tokenizer augmentors (the
# Sep-15-2026 wording, imaginaire4 86041fb1b52: WSM named explicitly plus a
# control-adherence paragraph), which every servable export trained under.
COSMOS3_AV_WSM_CONTROL_INSTRUCTION = (
    "Follow WSM controls for vehicles (including trucks), cyclists, pedestrians, "
    "traffic lights, traffic signs, road markings, lane boundaries, and road boundaries. "
    "Do not add objects or road features in these categories that are absent from WSM. "
    "Use captions for appearance and unconstrained background details; WSM takes "
    "precedence in any conflict."
)
COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT_WSM = (
    "You are a helpful assistant that generates temporally synchronized, geometrically "
    "consistent autonomous-driving videos from per-camera scene descriptions and World "
    "Scenario Map (WSM) control videos depicting the controlled objects and road layout. "
    "Treat all camera views as simultaneous observations of the same driving scene, "
    "preserving each camera's viewpoint, shared ego motion, road layout, object identity "
    "and motion, weather, lighting, and cross-view consistency.\n\n"
    f"{COSMOS3_AV_WSM_CONTROL_INSTRUCTION}"
)
COSMOS3_AV_JOINT_TRANSFER_SYSTEM_PROMPT_WSM = (
    "You are a helpful assistant that jointly generates temporally synchronized, "
    "geometrically consistent autonomous-driving camera videos and LiDAR range-view "
    "sequences from per-camera scene descriptions and provided control signals: "
    "per-camera World Scenario Map (WSM) control videos depicting the controlled objects "
    "and road layout, and an HD-map control for LiDAR. Treat all camera views and LiDAR "
    "sweeps as synchronized observations of the same driving scene, preserving each "
    "camera's viewpoint, shared ego motion, road layout, object identity and motion, "
    "weather, lighting, cross-view consistency, and camera-LiDAR alignment.\n\n"
    f"{COSMOS3_AV_WSM_CONTROL_INSTRUCTION}"
)
# Control-adherence sentences appended to every caption after the metadata.
COSMOS3_MULTIVIEW_EMPHASIS = (
    "Follow the wsm control videos precisely for every camera view: shape, contour, "
    "position, and motion must align with the wsm signal at every frame."
)
COSMOS3_JOINT_EMPHASIS = (
    "Follow the wsm and lidar control videos precisely: every camera view must align "
    "with its world-scenario map, and the LiDAR rangemap must align with the HD-map "
    "rangemap, at every frame."
)
# Caption augmentors of the transfer training data.
DURATION_TEMPLATE = "The video is {duration:.1f} seconds long and is of {fps:.0f} FPS."
RESOLUTION_TEMPLATE = "This video is of {height}x{width} resolution."

# Rates and frame counts outside these bounds are allowed with a warning.
COSMOS3_MULTIVIEW_RECOMMENDED_FPS_RANGE = (10.0, 30.0)
COSMOS3_MULTIVIEW_RECOMMENDED_NUM_FRAMES_RANGE = (24, 300)
# Server warmup carries no per-camera media; a short black clip exercises every
# stage and kernel at a fraction of a real request's cost.
COSMOS3_MULTIVIEW_WARMUP_NUM_FRAMES = 5
# Cameras without a vision clip in view completion get this gray canvas.
COSMOS3_MULTIVIEW_GRAY_LEVEL = 128

IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
)


# ---------------------------------------------------------------------------
# Media helpers
# ---------------------------------------------------------------------------
def media_kind(path: str) -> str:
    """``"image"`` or ``"video"`` from the file extension."""
    return "image" if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS else "video"


def media_hw(path: str) -> tuple[int, int]:
    """``(height, width)`` of an image or of a clip's first frame."""
    if media_kind(path) == "image":
        with PIL.Image.open(path) as image:
            return int(image.height), int(image.width)
    try:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            if stream.height and stream.width:
                return int(stream.height), int(stream.width)
    except Exception:  # PyAV missing or metadata unreadable: decode instead.
        pass
    decoded = load_video(path, max_frames=1)
    if not decoded:
        raise ValueError(f"No frames decoded from Cosmos3 multiview media {path!r}.")
    first = decoded[0]
    return int(first.height), int(first.width)


def load_media_uint8_cthw(
    path: str,
    *,
    height: int,
    width: int,
    max_frames: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Load an image or the first ``max_frames`` of a clip as ``uint8 [3, T, H, W]``.

    Frames are aspect-resized and center-cropped to ``height x width`` as they
    decode, so only one resize chunk and the canvas-sized result are ever in
    memory. ``out`` is an optional ``uint8 [3, >= T, height, width]``
    destination, filled in place and returned narrowed to the ``T`` decoded
    frames.
    """
    if media_kind(path) == "image":
        frames = [load_image(path)]
    else:
        frames = iter_video_frames(path, max_frames=max_frames)
    fitted = _fit_frames_uint8_cthw(
        frames, max_frames=max_frames, height=height, width=width, out=out
    )
    if fitted.shape[1] == 0:
        raise ValueError(f"No frames decoded from Cosmos3 multiview media {path!r}.")
    return fitted


def _fit_geometry(orig_h: int, orig_w: int, height: int, width: int) -> tuple[int, int]:
    """Resize target before the center crop: the aspect-preserving cover of the canvas."""
    scale = max(width / orig_w, height / orig_h)
    return int(math.ceil(scale * orig_h)), int(math.ceil(scale * orig_w))


def _resize_chunk_frames(orig_h: int, orig_w: int, chunk_frames: int) -> int:
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    float_frame_bytes = orig_h * orig_w * 3 * 4
    by_bytes = COSMOS3_MULTIVIEW_RESIZE_CHUNK_BYTES // max(float_frame_bytes, 1)
    return max(1, min(chunk_frames, by_bytes))


def _fit_frames_uint8_cthw(
    frames: Iterable[np.ndarray | PIL.Image.Image],
    *,
    max_frames: int,
    height: int,
    width: int,
    chunk_frames: int = COSMOS3_MULTIVIEW_RESIZE_CHUNK_FRAMES,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fit up to ``max_frames`` decoded frames onto the canvas: ``uint8 [3, T, height, width]``.

    Same arithmetic as imaginaire4's ``_resize_and_center_crop`` (torchvision
    resize with antialias, crop offset rounded half to even). The single-view
    helper in cosmos3.py keeps vLLM-Omni's floor offset without antialias,
    which lands one row off on 1720x1080 fisheye sources.

    Resize, crop, round and clamp act on each frame alone, so consuming the
    frames a chunk at a time changes no value.
    """
    if out is None:
        dest = torch.empty((3, max_frames, height, width), dtype=torch.uint8)
    else:
        if (
            out.ndim != 4
            or out.shape[0] != 3
            or tuple(out.shape[2:]) != (height, width)
            or out.dtype != torch.uint8
        ):
            raise ValueError(
                f"Multiview frame destination must be uint8 [3, T, {height}, {width}], "
                f"got {out.dtype} {tuple(out.shape)}"
            )
        dest = out
    arrays = map(rgb_hwc_uint8, islice(frames, min(max_frames, int(dest.shape[1]))))
    first = next(arrays, None)
    count = 0
    if first is not None:
        orig_h, orig_w = int(first.shape[0]), int(first.shape[1])
        arrays = _same_size_frames(chain([first], arrays), orig_h, orig_w)
        # At scale 1 the antialiased resize and the crop are the identity.
        if (orig_h, orig_w) == (height, width):
            count = _copy_frames(arrays, dest)
        else:
            count = _resize_frames(
                arrays, dest, orig_h=orig_h, orig_w=orig_w, chunk_frames=chunk_frames
            )
    if out is None and count < max_frames:
        return dest[:, :count].contiguous()
    return dest[:, :count]


def _same_size_frames(arrays: Iterable[np.ndarray], orig_h: int, orig_w: int):
    for array in arrays:
        if array.shape != (orig_h, orig_w, 3):
            raise ValueError(
                f"Multiview frames must all decode to [{orig_h}, {orig_w}, 3], got {tuple(array.shape)}"
            )
        yield array


def _copy_frames(arrays: Iterable[np.ndarray], dest: torch.Tensor) -> int:
    dest_np = dest.numpy()
    count = 0
    for array in arrays:
        dest_np[:, count] = array.transpose(2, 0, 1)
        count += 1
    return count


def _resize_frames(
    arrays: Iterable[np.ndarray],
    dest: torch.Tensor,
    *,
    orig_h: int,
    orig_w: int,
    chunk_frames: int,
) -> int:
    height, width = int(dest.shape[2]), int(dest.shape[3])
    resize_h, resize_w = _fit_geometry(orig_h, orig_w, height, width)
    chunk_frames = _resize_chunk_frames(orig_h, orig_w, chunk_frames)
    staging = np.empty((chunk_frames, orig_h, orig_w, 3), dtype=np.uint8)
    source = torch.empty((chunk_frames, 3, orig_h, orig_w), dtype=torch.float32)
    arrays = iter(arrays)
    count = 0
    while True:
        filled = 0
        for array in islice(arrays, chunk_frames):
            staging[filled] = array
            filled += 1
        if filled == 0:
            return count
        chunk = source[:filled]
        chunk.copy_(torch.from_numpy(staging[:filled]).permute(0, 3, 1, 2))
        resized = TF.resize(chunk, [resize_h, resize_w])
        cropped = TF.center_crop(resized, [height, width])
        dest[:, count : count + filled].copy_(
            cropped.round_().clamp_(0, 255).permute(1, 0, 2, 3)
        )
        count += filled


def fit_uint8_cthw(
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
    chunk_frames: int = COSMOS3_MULTIVIEW_RESIZE_CHUNK_FRAMES,
) -> torch.Tensor:
    """Aspect-preserving antialiased resize plus center crop of ``uint8 [3, T, H, W]``."""
    if frames.ndim != 4 or frames.shape[0] != 3:
        raise ValueError(
            f"Multiview frames must have shape [3, T, H, W], got {tuple(frames.shape)}"
        )
    if chunk_frames < 1:
        raise ValueError(f"chunk_frames must be positive, got {chunk_frames}")
    return _fit_frames_uint8_cthw(
        frames.permute(1, 2, 3, 0).numpy(),
        max_frames=int(frames.shape[1]),
        height=height,
        width=width,
        chunk_frames=chunk_frames,
    )


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


class CameraLoadGate:
    """Starts a request's camera loads in order, as many at once as their consumer needs.

    A load slows the thread that drives the GPU, so one runs at a time while
    the consumer keeps up, and each time it has to wait for a load after the
    first, one more may run alongside.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._tickets = 0
        self._next = 0
        self._running = 0
        self._limit = 1

    def ticket(self) -> int:
        # Tickets are issued from one thread, so this needs no lock.
        ticket = self._tickets
        self._tickets += 1
        return ticket

    @contextmanager
    def turn(self, ticket: int) -> Iterator[None]:
        with self._changed:
            self._changed.wait_for(
                lambda: ticket == self._next and self._running < self._limit
            )
            self._next += 1
            self._running += 1
            self._changed.notify_all()
        try:
            yield
        finally:
            with self._changed:
                self._running -= 1
                self._changed.notify_all()

    def consumer_waits(self, ticket: int) -> None:
        # Waiting for the first load says nothing about the loads falling behind.
        if ticket == 0:
            return
        with self._changed:
            self._limit += 1
            self._changed.notify_all()


class PendingCameraMajorPixels:
    """A camera-major ``uint8 [1, 3, V*F, H, W]`` rig whose cameras are still loading.

    A failed load is raised by the call that waits for its camera.
    """

    def __init__(
        self,
        rig: torch.Tensor,
        loads: Sequence[Future],
        *,
        gate: CameraLoadGate,
        tickets: Sequence[int],
    ) -> None:
        self._rig = rig
        self._loads = tuple(loads)
        self._gate = gate
        self._tickets = tuple(tickets)
        self._frames_per_view = int(rig.shape[2]) // len(self._loads)

    @property
    def shape(self) -> torch.Size:
        return self._rig.shape

    def camera(self, view: int) -> torch.Tensor:
        load = self._loads[view]
        if not load.done():
            self._gate.consumer_waits(self._tickets[view])
        load.result()
        frames = self._frames_per_view
        return self._rig[:, :, view * frames : (view + 1) * frames]

    def result(self) -> torch.Tensor:
        for load in self._loads:
            load.result()
        return self._rig


def _is_camera_major_pixels(value: Any) -> bool:
    return isinstance(value, PendingCameraMajorPixels) or V.is_tensor(value)


def load_lidar_control_frames(path: str) -> torch.Tensor:
    """A prepared range-map clip ``float32 [3, T, 128, W]``: metric range, unit intensity, validity."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        tensors = load_file(path)
        if "frames" not in tensors:
            raise ValueError(
                f"Cosmos3 LiDAR control {path!r} must contain a single tensor named 'frames'."
            )
        frames = tensors["frames"]
    else:
        frames = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f"Cosmos3 LiDAR control {path!r} must contain a tensor.")
    if frames.ndim == 5:
        if frames.shape[0] != 1:
            raise ValueError(
                f"Cosmos3 LiDAR control must hold one clip, got {tuple(frames.shape)}."
            )
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(
            f"Cosmos3 LiDAR control must be [3, T, H, W], got {tuple(frames.shape)}."
        )
    if frames.shape[0] != 3 and frames.shape[1] == 3:
        frames = frames.permute(1, 0, 2, 3)
    if frames.shape[0] != 3:
        raise ValueError(
            f"Cosmos3 LiDAR control needs range, intensity, and validity channels, got {tuple(frames.shape)}."
        )
    frames = frames.float().contiguous()
    if not torch.isfinite(frames).all() or bool((frames[0] < 0).any()):
        raise ValueError(
            "Cosmos3 LiDAR control must be finite with non-negative ranges."
        )
    return frames


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
    truncate_duration: bool = False,
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
        duration: float = num_frames / fps
        if truncate_duration:
            # The training augmentor (DurationFPSTextTimeStamps) writes whole
            # seconds unless fractional_duration is set, which the multiview
            # datasets do not; 17 frames at 30 fps reads "0.0 seconds".
            duration = float(int(duration))
        parts.append(duration_template.format(duration=duration, fps=fps).rstrip("."))
    if resolution_template is not None:
        parts.append(resolution_template.format(height=height, width=width).rstrip("."))
    if not parts:
        return ""
    return ". ".join(parts) + "."


def _camera_identity(camera: str) -> str:
    attributes = COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]
    role = str(attributes["camera_role"]).replace("_", "-")
    camera_type = str(attributes["camera_type"]).replace("_", "-")
    camera_type = {"standard": "", "wide": "wide-angle"}.get(camera_type, camera_type)
    return " ".join(part for part in (role, camera_type) if part) + " camera"


def _camera_facing(camera: str) -> str:
    return str(COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]["facing"]).replace("_", "-")


def format_separate_view_captions(
    captions: list[str], cameras: list[str] | tuple[str, ...]
) -> list[str]:
    """Prefix every per-camera caption with the selected rig and its own camera header.

    This is the separate-view-caption training layout: the rig sentence lists
    every selected camera, the header names the caption's camera, then the raw
    dataset caption follows. Unknown cameras are rejected by the deployment
    config, so the attribute table covers every camera here.
    """
    if len(captions) != len(cameras):
        raise ValueError(
            "Cosmos3 multiview per-camera captions must match the selected cameras: "
            f"captions={len(captions)}, cameras={len(cameras)}."
        )
    descriptions = [
        f"{_camera_identity(camera)} ({_camera_facing(camera)}-facing, "
        f"{COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]['fov_degrees']}° FOV)"
        for camera in cameras
    ]
    camera_word = "camera" if len(cameras) == 1 else "cameras"
    rig_prefix = (
        "This multiview driving sequence contains time-aligned recordings from "
        f"{len(cameras)} vehicle-mounted {camera_word}: {'; '.join(descriptions)}."
    )
    return [
        f"{rig_prefix}\n\nThe description below is for the {_camera_identity(camera)} "
        f"mounted on the vehicle. This camera is facing {_camera_facing(camera)} and has a "
        f"{COSMOS3_MADS_CAMERA_ATTRIBUTES[camera]['fov_degrees']}° field of view:\n\n"
        f"{caption}"
        for camera, caption in zip(cameras, captions, strict=True)
    ]


def format_per_view_negative_prompt(
    negative: str,
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
) -> str:
    """The shared negative caption of a per-camera-caption request, as the model reads it.

    Mirrors the reference ``_format_sample_prompt(per_view_negative_prompt, ...,
    truncate_duration=True)``: the duration and resolution sentences every
    positive caption carries, but neither the rig header nor the emphasis.
    """
    return apply_metadata_templates(
        negative,
        num_frames=num_frames,
        fps=fps,
        height=height,
        width=width,
        duration_template=DURATION_TEMPLATE,
        resolution_template=RESOLUTION_TEMPLATE,
        truncate_duration=True,
    )


def format_per_view_prompts(
    captions: list[str],
    cameras: list[str] | tuple[str, ...],
    *,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    emphasis: str | None,
) -> list[str]:
    """Per-camera prompts as the model reads them: rig header, metadata, emphasis."""
    prompts = []
    for caption in format_separate_view_captions(captions, cameras):
        prompt = apply_metadata_templates(
            caption,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
            duration_template=DURATION_TEMPLATE,
            resolution_template=RESOLUTION_TEMPLATE,
            truncate_duration=True,
        )
        if emphasis:
            prompt = f"{prompt.rstrip()} {emphasis}"
        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
class Cosmos3MultiviewInputStage(PipelineStage):
    """Load every camera's WSM control clip, optional RGB vision input, and LiDAR control.

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
        return result

    def _resolve_views(self, batch: Req) -> list[MultiviewViewInput] | None:
        deployment = self.deployment
        cameras = deployment.cameras
        views = batch.sampling_params.resolve_views()
        if not views:
            if batch.is_warmup:
                return None
            raise ValueError(
                "Cosmos3 multiview requests need multiview.views with one WSM "
                "control clip per camera (or a control_path list in checkpoint "
                "camera order for text-to-video)."
            )
        camera_keys = [view.camera_key for view in views]
        if any(key is not None for key in camera_keys):
            unknown = [key for key in camera_keys if key not in cameras]
            if unknown:
                raise ValueError(
                    "Cosmos3 multiview cameras must be exported checkpoint cameras: "
                    f"unknown={unknown}, exported={list(cameras)}."
                )
        else:
            if len(views) != len(cameras):
                raise ValueError(
                    f"Cosmos3 multiview requires exactly {len(cameras)} camera views in "
                    f"checkpoint order when camera_key is omitted, got {len(views)}."
                )
            views = [
                MultiviewViewInput(
                    camera_key=camera,
                    control=view.control,
                    vision=view.vision,
                    prompt=view.prompt,
                )
                for camera, view in zip(cameras, views, strict=True)
            ]
        if any(view.prompt is None or not view.prompt.strip() for view in views):
            raise ValueError(
                "This checkpoint tokenizes one caption per camera; supply a non-empty "
                "multiview.views[*].prompt for every camera."
            )
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

    def _resolve_canvas(
        self, batch: Req, views: list[MultiviewViewInput] | None
    ) -> None:
        """Detect the aspect bucket from the first camera's WSM when the request said auto."""
        sampling_params = batch.sampling_params
        if views is None or sampling_params.resolved_aspect_ratio() != "auto":
            return
        resolution = str(sampling_params.resolution)
        source_h, source_w = media_hw(views[0].control)
        ratio = closest_multiview_aspect_ratio(source_h, source_w, resolution)
        width, height = multiview_canvas(resolution, ratio)
        if (int(batch.width), int(batch.height)) != (width, height):
            self.log_info(
                f"Selected {resolution}p bucket {ratio} ({width}x{height}) from the first WSM "
                f"input ({source_w}x{source_h})"
            )
        batch.width = width
        batch.height = height
        sampling_params.width = width
        sampling_params.height = height
        sampling_params.aspect_ratio = ratio

    def _pack_camera_major(
        self,
        views: list[MultiviewViewInput],
        *,
        pool: ThreadPoolExecutor,
        gate: CameraLoadGate,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
        require_complete: bool = False,
    ) -> PendingCameraMajorPixels:
        """One field of the rig as camera-major ``uint8 [1, 3, V*F, H, W]``, loading on ``pool``.

        ``F`` is ``num_frames``, every clip truncated or padded with its last
        frame, or 1 with ``keep_first``.
        """
        frames_per_view = 1 if keep_first else num_frames
        rig = torch.empty(
            (1, 3, len(views) * frames_per_view, height, width), dtype=torch.uint8
        )

        # The stage's inference mode is thread-local; the workers need their own
        # to write the rig in place.
        @torch.inference_mode()
        def fill(index: int) -> None:
            view = views[index]
            dest = rig[0, :, index * frames_per_view : (index + 1) * frames_per_view]
            path = view.control if field == "control" else view.vision
            if path is None:
                if field != "vision":
                    raise ValueError(
                        f"Cosmos3 multiview camera {view.camera_key!r} is missing {field} input."
                    )
                dest.fill_(COSMOS3_MULTIVIEW_GRAY_LEVEL)
                return
            decoded = int(
                load_media_uint8_cthw(
                    path,
                    height=height,
                    width=width,
                    max_frames=frames_per_view,
                    out=dest,
                ).shape[1]
            )
            if require_complete and decoded < num_frames:
                raise ValueError(
                    f"Known camera {view.camera_key!r} requires a complete RGB video of "
                    f"{num_frames} frames for view completion."
                )
            if decoded < frames_per_view:
                dest[:, decoded:] = dest[:, decoded - 1 : decoded]

        def load(index: int, ticket: int) -> None:
            with gate.turn(ticket):
                fill(index)

        # Each clip decodes in its own ffmpeg subprocess and the resize releases
        # the GIL, so the rig's clips overlap on threads, each in its own slice.
        tickets = [gate.ticket() for _ in views]
        loads = [
            pool.submit(load, index, ticket) for index, ticket in enumerate(tickets)
        ]
        return PendingCameraMajorPixels(rig, loads, gate=gate, tickets=tickets)

    def _start_pixel_loads(
        self,
        views: list[MultiviewViewInput],
        *,
        height: int,
        width: int,
        num_frames: int,
        vision_keep_first: bool,
        completion: bool,
    ) -> tuple[PendingCameraMajorPixels, PendingCameraMajorPixels | None]:
        """Start loading the rig's control and RGB clips; returns before they finish.

        The latent stage encodes the cameras in the order they are queued here,
        controls first, so its GPU work overlaps the clips still decoding; one
        gate paces both rigs' loads to that consumer.
        """
        pool = ThreadPoolExecutor(
            max_workers=COSMOS3_MULTIVIEW_MEDIA_THREADS,
            thread_name_prefix="cosmos3-multiview-load",
        )
        gate = CameraLoadGate()
        try:
            control = self._pack_camera_major(
                views,
                pool=pool,
                gate=gate,
                field="control",
                height=height,
                width=width,
                num_frames=num_frames,
                keep_first=False,
            )
            vision = None
            if any(view.vision is not None for view in views):
                vision = self._pack_camera_major(
                    views,
                    pool=pool,
                    gate=gate,
                    field="vision",
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    keep_first=vision_keep_first,
                    require_complete=completion,
                )
        finally:
            # Queued loads still run; the threads exit after the last one.
            pool.shutdown(wait=False)
        return control, vision

    def _load_lidar(
        self, batch: Req, num_frames: int, fps: float
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """The HD-map control clip and, when requested, the measured prefix sweeps."""
        lidar = batch.sampling_params.resolved_lidar()
        if lidar is None:
            return None, None
        deployment = self.deployment
        if not deployment.supports_lidar:
            raise ValueError(
                "Joint camera/LiDAR requests require a checkpoint with the LiDAR encoder."
            )
        lidar_fps = float(deployment.lidar["fps"])
        sweeps = required_lidar_sweeps(num_frames, fps, lidar_fps)
        frames = load_lidar_control_frames(lidar["control_path"])
        if frames.shape[1] < sweeps:
            self.log_warning(
                f"LiDAR control has {frames.shape[1]} sweeps; padding to the {sweeps} the "
                f"{num_frames}-frame clip at {fps:g} FPS covers at {lidar_fps:g} Hz."
            )
        frames = pad_lidar_sweeps(frames, sweeps)
        condition = None
        if lidar.get("condition_path"):
            count = int(lidar["num_conditional_sweeps"])
            if count >= sweeps:
                raise ValueError(
                    f"Cosmos3 lidar.num_conditional_sweeps={count} must leave at least one "
                    f"of the request's {sweeps} LiDAR sweeps to generate."
                )
            measured = load_lidar_control_frames(lidar["condition_path"])
            if measured.shape[1] < count:
                raise ValueError(
                    f"Cosmos3 lidar.condition_path holds {measured.shape[1]} sweeps; "
                    f"{count} conditional sweeps were requested."
                )
            if tuple(measured.shape[2:]) != tuple(frames.shape[2:]):
                raise ValueError(
                    "Cosmos3 measured LiDAR sweeps must share the control's range-map grid: "
                    f"measured={tuple(measured.shape[2:])}, control={tuple(frames.shape[2:])}."
                )
            condition = measured[:, :count].contiguous()
        return frames, condition

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        sampling_params = batch.sampling_params
        cameras = self.deployment.cameras
        views = self._resolve_views(batch)
        self._resolve_canvas(batch, views)
        height, width = int(batch.height), int(batch.width)
        if height % 16 or width % 16:
            raise ValueError(
                f"Cosmos3 multiview canvas must be a multiple of 16, got {width}x{height}."
            )

        known_views: list[int] = []
        lidar_frames = None
        lidar_condition = None
        if views is None:
            num_frames = COSMOS3_MULTIVIEW_WARMUP_NUM_FRAMES
            batch.num_frames = num_frames
            selected = list(cameras)
            batch.extra[EXTRA_CONTROL_PIXELS] = synthetic_multiview_pixels(
                num_views=len(cameras),
                num_frames=num_frames,
                height=height,
                width=width,
            )
            batch.extra[EXTRA_VISION_PIXELS] = None
            batch.extra[EXTRA_VIEW_PROMPTS] = [""] * len(cameras)
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
            selected = [str(view.camera_key) for view in views]
            known_views = [
                index for index, view in enumerate(views) if view.vision is not None
            ]
            has_vision = bool(known_views)
            completion = has_vision and len(known_views) < len(views)
            condition_video_as_image = (
                sampling_params.resolved_condition_video_as_image()
            )
            vision_is_image = (
                has_vision and media_kind(views[known_views[0]].vision) == "image"
            )
            lidar_frames, lidar_condition = self._load_lidar(batch, num_frames, fps)
            if lidar_frames is not None and completion:
                raise ValueError(
                    "Joint camera/LiDAR RGB conditions must cover every camera or none."
                )
            if completion and (
                media_kind(views[known_views[0]].vision) != "video"
                or condition_video_as_image
            ):
                raise ValueError(
                    "View completion requires complete RGB videos for the known cameras."
                )
            explicit_indexes = sampling_params.resolved_local_condition_indexes()
            if completion:
                if explicit_indexes:
                    raise ValueError(
                        "View completion conditions all frames of known views; omit "
                        "condition frame indexes."
                    )
                local_indexes = []
            elif explicit_indexes is not None:
                if not has_vision and explicit_indexes:
                    raise ValueError(
                        "Cosmos3 multiview condition_frame_indexes_vision requires "
                        "per-camera vision inputs."
                    )
                local_indexes = explicit_indexes
            elif not has_vision:
                local_indexes = []
            elif condition_video_as_image or vision_is_image:
                local_indexes = [0]
            else:
                local_indexes = [0, 1]
            (
                batch.extra[EXTRA_CONTROL_PIXELS],
                batch.extra[EXTRA_VISION_PIXELS],
            ) = self._start_pixel_loads(
                views,
                height=height,
                width=width,
                num_frames=num_frames,
                vision_keep_first=condition_video_as_image or vision_is_image,
                completion=completion,
            )
            batch.extra[EXTRA_VIEW_PROMPTS] = [view.prompt or "" for view in views]
            if not completion:
                known_views = []

        batch.extra[EXTRA_NUM_VIEWS] = len(selected)
        batch.extra[EXTRA_FRAMES_PER_VIEW] = num_frames
        batch.extra[EXTRA_LOCAL_CONDITION_INDEXES] = local_indexes
        batch.extra[EXTRA_KNOWN_VIEWS] = known_views
        batch.extra[EXTRA_CAMERAS] = tuple(selected)
        batch.extra[EXTRA_LIDAR_FRAMES] = lidar_frames
        batch.extra[EXTRA_LIDAR_CONDITION_FRAMES] = lidar_condition
        mode = "text2video" if not (local_indexes or known_views) else "image2video"
        if known_views:
            mode = "view_completion"
        self.log_info(
            f"Prepared {len(selected)} camera views x {num_frames} frames at {width}x{height} "
            f"({mode}, condition latent frames per camera {local_indexes}"
            f"{', LiDAR sweeps %d' % lidar_frames.shape[1] if lidar_frames is not None else ''}"
            f"{', measured LiDAR prefix %d' % lidar_condition.shape[1] if lidar_condition is not None else ''})"
        )
        return batch


class Cosmos3MultiviewTokenizationStage(Cosmos3TokenizationStage):
    """Transfer-style prompts tokenized with the Qwen2 template.

    Unversioned exports read one prompt for the whole rig (plus the negative
    prompt). Separate-view-caption exports read one caption per camera, each
    encoded on its own so no caption attends another; their unconditional
    branch is an empty caption per camera, the way training's caption dropout
    produced it, so a negative prompt is ignored.
    """

    def __init__(self, tokenizer, deployment: Cosmos3MultiviewDeploymentConfig) -> None:
        super().__init__(tokenizer)
        self.deployment = deployment

    def _tokenizer_cap(self, batch: Req) -> int:
        requested = batch.max_sequence_length or COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        max_sequence_length = min(int(requested), COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH)
        # ``_tokenize_prompt`` reserves two slots inside its cap for the eos and
        # vision_start framing tokens. The reference caps the caption itself, so
        # widen by two.
        return max_sequence_length + 2

    def _tokenize_compact(
        self, texts: list[str], cap: int, device: torch.device, system_prompt: str
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Tokenize each text separately and concatenate the real tokens."""
        pieces = []
        lengths = []
        for text in texts:
            ids, _mask, seq_len = self._tokenize_prompt(
                text=text,
                max_sequence_length=cap,
                device=device,
                use_system_prompt=True,
                system_prompt=system_prompt,
            )
            pieces.append(ids[:, :seq_len])
            lengths.append(int(seq_len))
        return torch.cat(pieces, dim=1), tuple(lengths)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        fps = float(batch.fps)
        num_frames = int(batch.num_frames)
        height, width = int(batch.height), int(batch.width)
        cap = self._tokenizer_cap(batch)
        joint = batch.extra.get(EXTRA_LIDAR_FRAMES) is not None
        emphasize = batch.sampling_params.resolved_emphasize_control(
            bool(self.deployment.inference_default("emphasize_control_in_prompt", True))
        )

        cameras = list(batch.extra[EXTRA_CAMERAS])
        captions = list(batch.extra[EXTRA_VIEW_PROMPTS])
        emphasis = None
        if emphasize:
            emphasis = COSMOS3_JOINT_EMPHASIS if joint else COSMOS3_MULTIVIEW_EMPHASIS
        prompts = format_per_view_prompts(
            captions,
            cameras,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
            emphasis=emphasis,
        )
        system_prompt = (
            COSMOS3_AV_JOINT_TRANSFER_SYSTEM_PROMPT_WSM
            if joint
            else COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT_WSM
        )
        cond_ids, cond_lengths = self._tokenize_compact(
            prompts, cap, device, system_prompt
        )
        per_view_negative = batch.sampling_params.per_view_negative_prompt
        if per_view_negative is not None and per_view_negative.strip():
            # The reference formats one shared negative caption with each
            # clip's duration/resolution metadata (no rig header, no
            # emphasis) and tokenizes it for every camera.
            negative_caption = format_per_view_negative_prompt(
                per_view_negative,
                num_frames=num_frames,
                fps=fps,
                height=height,
                width=width,
            )
            uncond_captions = [negative_caption] * len(prompts)
            if not batch.is_warmup:
                self.log_info(
                    f"Using an explicit negative caption for each of {len(prompts)} camera views."
                )
        else:
            uncond_captions = [""] * len(prompts)
            if batch.negative_prompt and not batch.is_warmup:
                self.log_warning(
                    "Ignoring negative_prompt: this checkpoint tokenizes one caption per "
                    "camera and its unconditional branch is an empty caption per camera, "
                    "as training's caption dropout produced; set per_view_negative_prompt "
                    "to opt into a shared negative caption."
                )
        uncond_ids, uncond_lengths = self._tokenize_compact(
            uncond_captions, cap, device, system_prompt
        )
        preview = prompts[0]

        batch.extra["cond_text_ids"] = cond_ids
        batch.extra["cond_text_mask"] = torch.ones_like(cond_ids)
        batch.extra["uncond_text_ids"] = uncond_ids
        batch.extra["uncond_text_mask"] = torch.ones_like(uncond_ids)
        batch.extra["cond_text_seq_len"] = int(cond_ids.shape[1])
        batch.extra["uncond_text_seq_len"] = int(uncond_ids.shape[1])
        batch.extra[EXTRA_CAPTION_LENGTHS] = {
            "cond": cond_lengths,
            "uncond": uncond_lengths,
        }
        batch.extra["fps"] = fps
        batch.is_prompt_processed = True
        if not batch.is_warmup:
            self.log_info(
                f"Multiview prompt ({cond_ids.shape[1]} tokens in {len(cond_lengths)} caption(s), "
                f"negative {uncond_ids.shape[1]}): {preview[:240]!r}"
            )
        return batch


class Cosmos3MultiviewLatentStage(PipelineStage):
    """Per-camera VAE encode, LiDAR encode, packed targets, anchors, and the attention layout."""

    parallelism_type = StageParallelismType.REPLICATED

    def __init__(
        self,
        vae,
        transformer,
        deployment: Cosmos3MultiviewDeploymentConfig,
        lidar_encoder: Cosmos3LidarEncoder | None,
        attention_backend: str,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.transformer = transformer
        self.deployment = deployment
        self.lidar_encoder = lidar_encoder
        self.attention_backend = attention_backend

    def _encode_lidar_prefix(
        self,
        frames: torch.Tensor,
        *,
        total_sweeps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Clean latents of the measured sweeps that start the generated LiDAR.

        The streaming tokenizer is frame-causal within a chunk, but how much
        history a chunk keeps depends on that chunk's length, so encoding the
        bare prefix would change the latents of a trailing partial chunk. Zero-
        padding the prefix to the target's chunk boundary reproduces the full
        clip's chunk lengths for every chunk holding prefix sweeps, which is how
        the reference encodes the prefix (inside an otherwise empty target clip);
        the causal padding cannot affect the prefix latents.
        """
        count = int(frames.shape[1])
        chunk = int(self.lidar_encoder.config["streaming_chunk_frames"])
        padded = min(math.ceil(count / chunk) * chunk, total_sweeps)
        if padded > count:
            frames = torch.cat(
                [
                    frames,
                    frames.new_zeros(
                        frames.shape[0], padded - count, *frames.shape[2:]
                    ),
                ],
                dim=1,
            )
        latents = self.lidar_encoder(frames)[:, :, :count]
        return latents.to(device=device, dtype=dtype)

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        result = VerificationResult()
        result.add_check(
            EXTRA_CONTROL_PIXELS,
            batch.extra.get(EXTRA_CONTROL_PIXELS),
            _is_camera_major_pixels,
        )
        result.add_check("height", batch.height, V.positive_int_divisible(16))
        result.add_check("width", batch.width, V.positive_int_divisible(16))
        return result

    def _encode_camera_major(
        self,
        pixels: torch.Tensor | PendingCameraMajorPixels,
        *,
        num_views: int,
        frames_per_view: int,
        device: torch.device,
        dtype: torch.dtype,
        latent_frames: int | None = None,
    ) -> torch.Tensor:
        """Encode camera-major ``uint8 [1, 3, V*P, H, W]`` one camera at a time into ``[1, C, V*Fl, h, w]``.

        The Wan VAE is causal in time, so the camera-major stream must never be
        encoded as one long clip.
        """
        pending = isinstance(pixels, PendingCameraMajorPixels)
        if len(pixels.shape) != 5 or num_views < 1 or pixels.shape[2] % num_views:
            raise ValueError(
                "Cosmos3 multiview pixel video must be camera-major [1, 3, V*P, H, W]: "
                f"shape={tuple(pixels.shape)}, V={num_views}."
            )
        packed_frames = pixels.shape[2] // num_views
        vae_dtype = next(self.vae.parameters()).dtype
        mean = torch.as_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.as_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        encode_kwargs = (
            {} if latent_frames is None else {"latent_frames": latent_frames}
        )
        latents = []
        for view in range(num_views):
            clip = (
                pixels.camera(view)
                if pending
                else pixels[:, :, view * packed_frames : (view + 1) * packed_frames]
            )
            if packed_frames != frames_per_view:
                clip = pad_view_frames_uint8(
                    clip[0], num_frames=frames_per_view
                ).unsqueeze(0)
            # Cast on the device: 0..255 is exact in the VAE dtype, and the
            # upload stays one byte per value.
            clip = clip.to(device).to(vae_dtype).div(127.5).sub(1.0)
            with torch.no_grad():
                latent = self.vae.encode(clip, **encode_kwargs).mode()
            latent = (latent - mean.to(latent)) / std.to(latent)
            latents.append(latent.to(dtype))
        latent_frames = {int(latent.shape[2]) for latent in latents}
        if len(latent_frames) != 1:
            raise ValueError(
                f"Cosmos3 multiview per-camera VAE encodes have unequal lengths: {latent_frames}."
            )
        return torch.cat(latents, dim=2)

    def _encode_vision(
        self,
        pixels: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int,
        latent_frames_per_view: int,
        condition_indexes: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode the RGB conditions. Only the latents at ``condition_indexes`` are
        meaningful, and the rest may be filler."""
        needed = 1 + max(index % latent_frames_per_view for index in condition_indexes)
        latent_frames = None
        if needed < latent_frames_per_view and self.vae.use_feature_cache:
            frames_per_view = self.vae.causal_prefix_frames(needed)
            latent_frames = latent_frames_per_view
        return self._encode_camera_major(
            pixels,
            num_views=num_views,
            frames_per_view=frames_per_view,
            device=device,
            dtype=dtype,
            latent_frames=latent_frames,
        )

    def _condition_indexes(
        self, batch: Req, *, num_views: int, latent_frames_per_view: int, latent_t: int
    ) -> list[int]:
        known_views = [int(v) for v in (batch.extra.get(EXTRA_KNOWN_VIEWS) or [])]
        if known_views:
            return [
                view * latent_frames_per_view + frame
                for view in known_views
                for frame in range(latent_frames_per_view)
            ]
        local_indexes = [
            int(i) for i in (batch.extra.get(EXTRA_LOCAL_CONDITION_INDEXES) or [])
        ]
        if any(index < 0 or index >= latent_frames_per_view for index in local_indexes):
            raise ValueError(
                "Cosmos3 multiview condition frame index is outside the generated latent clip."
            )
        return expand_multiview_condition_frame_indexes(
            local_indexes, num_views, latent_t
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        device = get_local_torch_device()
        # The sampler state (noise, clean conditions, velocity masks and the
        # solver's samples) stays float32 as in the reference; the transformer
        # casts its inputs to its own dtype, so model-input-only latents (the
        # controls) are kept in that dtype.
        dtype = torch.float32
        model_dtype = torch.bfloat16
        deployment = self.deployment
        num_views = int(batch.extra[EXTRA_NUM_VIEWS])
        frames_per_view = int(batch.extra[EXTRA_FRAMES_PER_VIEW])
        temporal_factor = int(self.vae.config.scale_factor_temporal)
        spatial_factor = int(self.vae.config.scale_factor_spatial)
        latent_frames_per_view = (frames_per_view - 1) // temporal_factor + 1
        latent_t = num_views * latent_frames_per_view
        latent_h = int(batch.height) // spatial_factor
        latent_w = int(batch.width) // spatial_factor
        shape = (1, int(self.transformer.latent_channel), latent_t, latent_h, latent_w)

        generator = batch.generator
        if generator is None and batch.seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(batch.seed))
            batch.generator = generator
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)

        vision_pixels = batch.extra.get(EXTRA_VISION_PIXELS)
        condition_indexes = self._condition_indexes(
            batch,
            num_views=num_views,
            latent_frames_per_view=latent_frames_per_view,
            latent_t=latent_t,
        )
        with self.use_declared_component(component_name="vae", module=self.vae):
            control_latents = self._encode_camera_major(
                batch.extra[EXTRA_CONTROL_PIXELS],
                num_views=num_views,
                frames_per_view=frames_per_view,
                device=device,
                dtype=model_dtype,
            )
            encoded_vision = None
            if vision_pixels is not None and condition_indexes:
                encoded_vision = self._encode_vision(
                    vision_pixels,
                    num_views=num_views,
                    frames_per_view=frames_per_view,
                    latent_frames_per_view=latent_frames_per_view,
                    condition_indexes=condition_indexes,
                    device=device,
                    dtype=dtype,
                )
        # Later stages get plain tensors; waiting here also raises the failed
        # load of a clip that no encode read.
        for key in (EXTRA_CONTROL_PIXELS, EXTRA_VISION_PIXELS):
            pixels = batch.extra.get(key)
            if isinstance(pixels, PendingCameraMajorPixels):
                batch.extra[key] = pixels.result()
        if tuple(control_latents.shape) != shape:
            raise ValueError(
                "Cosmos3 multiview WSM and target latent shapes must match: "
                f"control={tuple(control_latents.shape)}, target={shape}."
            )

        condition_mask = torch.zeros(1, 1, latent_t, 1, 1, device=device, dtype=dtype)
        condition_latents = torch.zeros_like(noise)
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
            for index in condition_indexes:
                condition_mask[:, :, index] = 1.0
                condition_latents[:, :, index] = encoded_vision[:, :, index]
        camera_latents = (
            condition_mask * condition_latents + (1.0 - condition_mask) * noise
        )
        camera_velocity_mask = (1.0 - condition_mask).expand_as(noise)

        patch_h, patch_w, _, _ = self.transformer._pad_to_patch_size(latent_h, latent_w)
        fps = float(batch.extra.get("fps") or batch.fps)
        camera_shape = (latent_t, patch_h, patch_w)
        camera_rate = temporal_factor / fps
        items = [
            MaskItem(
                camera_shape, num_views, is_control=True, seconds_per_frame=camera_rate
            ),
            MaskItem(camera_shape, num_views, seconds_per_frame=camera_rate),
        ]
        targets = [camera_latents]
        masks = [camera_velocity_mask]
        conditions = [condition_latents]

        lidar_frames = batch.extra.get(EXTRA_LIDAR_FRAMES)
        lidar_control_latents = None
        lidar_fps = None
        lidar_tcf = 1
        if lidar_frames is not None:
            if self.lidar_encoder is None or deployment.lidar is None:
                raise ValueError(
                    "Joint camera/LiDAR requests require the LiDAR encoder to be loaded."
                )
            lidar_fps = float(deployment.lidar["fps"])
            lidar_tcf = int(deployment.lidar["temporal_compression_factor"])
            self.begin_declared_component_use(
                component_name="lidar_encoder", module=self.lidar_encoder
            )
            lidar_control_latents = self.lidar_encoder(lidar_frames).to(
                device=device, dtype=model_dtype
            )
            # Continue the request RNG after the camera noise so a seed
            # reproduces both streams.
            lidar_noise = torch.randn(
                lidar_control_latents.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            lidar_latents = lidar_noise
            lidar_mask = torch.ones_like(lidar_noise)
            lidar_condition = torch.zeros_like(lidar_noise)
            condition_frames = batch.extra.get(EXTRA_LIDAR_CONDITION_FRAMES)
            if condition_frames is not None:
                # Measured sweeps start the LiDAR target clean, the way the
                # cameras' first frame anchors the RGB targets: no noise, zero
                # velocity, restored after every solver step.
                prefix = self._encode_lidar_prefix(
                    condition_frames,
                    total_sweeps=int(lidar_noise.shape[2]),
                    device=device,
                    dtype=dtype,
                )
                count = int(prefix.shape[2])
                expected = (lidar_noise.shape[1], count, *lidar_noise.shape[3:])
                if tuple(prefix.shape[1:]) != expected:
                    raise ValueError(
                        "Cosmos3 LiDAR condition latents must have shape "
                        f"{expected}, got {tuple(prefix.shape[1:])}."
                    )
                lidar_latents = lidar_noise.clone()
                lidar_latents[:, :, :count] = prefix
                lidar_mask[:, :, :count] = 0.0
                lidar_condition[:, :, :count] = prefix
            targets.append(lidar_latents)
            masks.append(lidar_mask)
            conditions.append(lidar_condition)
            lt, lh, lw = lidar_noise.shape[2:]
            lhp, lwp = self.transformer.lidar_patch_grid(lh, lw)
            lidar_rate = lidar_tcf / lidar_fps
            for is_control in (True, False):
                items.append(
                    MaskItem(
                        (lt, lhp, lwp),
                        1,
                        view_offset=num_views,
                        is_control=is_control,
                        seconds_per_frame=lidar_rate,
                        is_lidar=True,
                    )
                )

        layout = MultiviewLayout(
            num_views=num_views,
            latent_frames=latent_t,
            patch_height=patch_h,
            patch_width=patch_w,
            # The architecture this model has: decomposed scope with a past-only
            # cross-view window, controls reading their own view's targets, LiDAR
            # reading every caption.
            attention_scope="decomposed",
            decomposed_temporal_window_seconds=deployment.cross_view_past_window_seconds,
            control_attends_sensor=True,
            seconds_per_frame=camera_rate,
            backend=self.attention_backend,
            # The kernels pad the text stream to one fixed capacity per caption
            # so prompts of every length share one compiled kernel.
            max_und_tokens=DEFAULT_MAX_UND_TOKENS * num_views,
            items=tuple(items),
            lidar_attends_captions=True,
        )
        # Physical rig ids of the request's cameras; subsets and reordered views
        # keep each camera's trained row.
        rig_view_ids = torch.tensor(
            deployment.rig_view_ids(batch.extra[EXTRA_CAMERAS]),
            dtype=torch.long,
            device=device,
        )
        # Temporal positions wrap per view so every camera shares one time axis.
        temporal_position_period = latent_frames_per_view
        packed_shapes = tuple(
            tuple(int(d) for d in tensor.shape[1:]) for tensor in targets
        )

        batch.latents = pack_state(targets)
        batch.raw_latent_shape = tuple(batch.latents.shape)
        batch.extra["video_shape"] = (latent_t, latent_h, latent_w)
        batch.extra["vae_scale_factor_temporal"] = temporal_factor
        batch.extra["vae_scale_factor_spatial"] = spatial_factor
        batch.extra["control_latents"] = [control_latents]
        batch.extra["condition_latents"] = pack_state(conditions)
        batch.extra["velocity_mask"] = pack_state(masks)
        batch.extra[EXTRA_LATENT_FRAMES_PER_VIEW] = latent_frames_per_view
        batch.extra[EXTRA_PACKED_SHAPES] = packed_shapes
        batch.extra[EXTRA_LIDAR_LATENTS] = None
        batch.extra[EXTRA_TRANSFORMER_KWARGS] = {
            "temporal_position_period": temporal_position_period,
            "multiview_layout": layout,
            "packed_shapes": packed_shapes,
            "caption_lengths_by_cache_key": batch.extra.get(EXTRA_CAPTION_LENGTHS),
            "lidar_control_latents": lidar_control_latents,
            "lidar_fps": lidar_fps,
            "lidar_temporal_compression_factor": lidar_tcf,
            "rig_view_ids": rig_view_ids,
        }
        lidar_note = ""
        if lidar_control_latents is not None:
            lidar_note = f", LiDAR latents {tuple(lidar_control_latents.shape)}"
            if batch.extra.get(EXTRA_LIDAR_CONDITION_FRAMES) is not None:
                lidar_note += f" with {int(batch.extra[EXTRA_LIDAR_CONDITION_FRAMES].shape[1])} measured prefix sweeps"
        self.log_info(
            f"Prepared multiview latents {shape} ({num_views} cameras x "
            f"{latent_frames_per_view} latent frames, {layout.gen_tokens} GEN tokens, "
            f"{len(condition_indexes)} anchored frames, attention backend "
            f"{self.attention_backend}{lidar_note})"
        )
        return batch

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        name = self._component_stage_name(stage_name)
        uses = [ComponentUse(name, "vae")]
        if self.lidar_encoder is not None:
            uses.append(ComponentUse(name, "lidar_encoder", start_at_stage_entry=False))
        return uses


class Cosmos3MultiviewDecodingStage(Cosmos3DecodingStage):
    """Decode each camera's latent slice separately and re-pack camera-major.

    The output is one ``[1, 3, V*F, H, W]`` video: all frames of camera 0, then
    camera 1, and so on in request camera order. Split it by
    ``frames_per_view`` on the client to get per-camera clips. When the request
    sets ``lidar.decode``, the denoised LiDAR latents are decoded to metric range
    maps and reported in the output's ``lidar`` block (files next to the video,
    arrays when frames are returned); otherwise a joint request returns only video.
    """

    def __init__(
        self,
        vae,
        guardrails: bool = False,
        sound_tokenizer=None,
        lidar_decoder: Cosmos3LidarDecoder | None = None,
    ) -> None:
        super().__init__(vae, guardrails=guardrails, sound_tokenizer=sound_tokenizer)
        self.lidar_decoder = lidar_decoder

    def component_uses(
        self, server_args: ServerArgs, stage_name: str | None = None
    ) -> list[ComponentUse]:
        uses = super().component_uses(server_args, stage_name)
        if self.lidar_decoder is not None:
            uses.append(
                ComponentUse(
                    self._component_stage_name(stage_name),
                    "lidar_decoder",
                    start_at_stage_entry=False,
                )
            )
        return uses

    def _decode_lidar(self, batch: Req, latents: torch.Tensor) -> dict[str, Any] | None:
        lidar_request = batch.sampling_params.resolved_lidar() or {}
        if not lidar_request.get("decode", False):
            return None
        if self.lidar_decoder is None:
            raise ValueError(
                "lidar.decode=true needs the LiDAR decoder, which this server did not load."
            )
        projection = self.lidar_decoder.projection
        min_range_m = float(projection["min_range_m"])
        max_range_m = float(projection["max_range_m"])
        with self.use_declared_component(
            component_name="lidar_decoder", module=self.lidar_decoder
        ):
            with torch.no_grad():
                clip = self.lidar_decoder(latents)[0].float().cpu()
        files: dict[str, str] = {}
        video_path = batch.output_file_path() if batch.save_output else None
        if video_path and get_world_rank() == 0:
            files = write_lidar_outputs(
                clip,
                directory=os.path.dirname(video_path) or ".",
                stem=os.path.splitext(os.path.basename(video_path))[0],
                fps=self.lidar_decoder.fps,
                min_range_m=min_range_m,
                max_range_m=max_range_m,
            )
        payload = lidar_output_payload(
            clip,
            fps=self.lidar_decoder.fps,
            min_range_m=min_range_m,
            max_range_m=max_range_m,
            files=files,
            include_arrays=not files or bool(batch.return_frames),
        )
        self.log_info(
            f"Decoded LiDAR: {payload['sweeps']} sweeps x {payload['height']}x{payload['width']}, "
            f"{payload['valid_fraction']:.1%} rays valid"
            f"{', files ' + ', '.join(sorted(files.values())) if files else ''}"
        )
        return payload

    def _encode_while_decoding(
        self, batch: Req, server_args: ServerArgs
    ) -> EncodeWhileDecoding | None:
        lidar_request = batch.sampling_params.resolved_lidar() or {}
        if lidar_request.get("decode", False):
            return None
        return super()._encode_while_decoding(batch, server_args)

    def _decode_views(
        self,
        latents: torch.Tensor,
        *,
        num_views: int,
        latent_frames_per_view: int,
        on_frames: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor | None:
        """The camera-major clip, or ``None`` when ``on_frames`` took the frames."""
        clips = []
        for view in range(num_views):
            clip = self._decode_latents(
                latents[
                    :,
                    :,
                    view * latent_frames_per_view : (view + 1) * latent_frames_per_view,
                ],
                on_frames=on_frames,
            )
            if on_frames is None:
                clips.append(clip)
        return torch.cat(clips, dim=2) if clips else None

    def forward(self, batch: Req, server_args: ServerArgs):
        num_views = int(batch.extra[EXTRA_NUM_VIEWS])
        latent_frames_per_view = int(batch.extra[EXTRA_LATENT_FRAMES_PER_VIEW])
        packed_shapes = batch.extra[EXTRA_PACKED_SHAPES]
        unpacked = unpack_state(batch.latents, packed_shapes)
        latents = unpacked[0]
        lidar_latents = unpacked[1] if len(unpacked) > 1 else None
        if lidar_latents is not None:
            batch.extra[EXTRA_LIDAR_LATENTS] = lidar_latents
        if latents.shape[2] != num_views * latent_frames_per_view:
            raise ValueError(
                "Cosmos3 multiview latents must be camera-major before decode: "
                f"shape={tuple(latents.shape)}, V={num_views}, F={latent_frames_per_view}."
            )
        batch.latents = latents
        batch.raw_latent_shape = tuple(latents.shape)
        # Frames reach the encoder camera by camera, the order of the clip.
        stream = self._encode_while_decoding(batch, server_args)
        with self.use_declared_component(component_name="vae", module=self.vae):
            with (
                torch.no_grad(),
                use_vae_fast_path(
                    self.vae, quality_allows(batch.sampling_params.quality, "lossless")
                ),
            ):
                try:
                    decoded = self._decode_views(
                        latents,
                        num_views=num_views,
                        latent_frames_per_view=latent_frames_per_view,
                        on_frames=stream,
                    )
                    output_file_paths, encode_failed = finish_streamed_encode(stream)
                except BaseException:
                    if stream is not None:
                        stream.abort()
                    raise
                if encode_failed:
                    # The worker saves the clip instead. Ranks without the encoder
                    # drop the clip they kept and repeat the decode with the output rank.
                    del decoded
                    decoded = self._decode_views(
                        latents,
                        num_views=num_views,
                        latent_frames_per_view=latent_frames_per_view,
                    )
        if not batch.is_warmup:
            frames_per_view = 1 + (latent_frames_per_view - 1) * int(
                self.vae.config.scale_factor_temporal
            )
            self.log_info(
                f"Decoded camera-major video: {num_views} cameras x {frames_per_view} "
                f"frames at {batch.fps} fps; camera order {list(batch.extra.get(EXTRA_CAMERAS) or [])}"
            )
        if output_file_paths is not None:
            output_batch = OutputBatch(
                output=None,
                output_file_paths=output_file_paths,
                metrics=batch.metrics,
                rollout_trajectory_data=batch.rollout_trajectory_data,
            )
        else:
            # The base stage consumes a pre-decoded tensor from this key (it is
            # the transfer pipeline's hand-off) and only post-processes it.
            batch.extra["transfer_decoded_output"] = decoded
            output_batch = super().forward(batch, server_args)
        if lidar_latents is not None and not batch.is_warmup:
            output_batch.lidar = self._decode_lidar(batch, lidar_latents)
        return output_batch
