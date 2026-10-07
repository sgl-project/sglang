# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV sampling parameters.

One request generates all cameras of the rig in a single bidirectional pass.
Per-camera inputs travel in the ``multiview`` object, mirroring the reference
input JSON::

    {
      "multiview": {
        "views": [
          {"camera_key": "camera_front_wide_120fov",
           "prompt": "A car travels along a tree-lined road.",
           "control_path": "wsm/front_wide.mp4",
           "vision_path": "rgb/front_wide.png"},
          ...
        ],
        "condition_video_as_image": true,
        "resolution": "480",
        "aspect_ratio": "auto"
      },
      "wsm": {},
      "lidar": {"control_path": "hdmap_rangemap.safetensors"}
    }

Every view needs a pre-computed WSM ``control_path``. ``vision_path`` (an RGB
still or clip) is optional but must be given for every camera or none; with it
the request is image-to-video, without it text-to-video. Every camera takes its
caption in ``views[].prompt``. ``lidar`` selects joint camera/LiDAR generation on exports that ship
the LiDAR encoder. As a CLI convenience, a ``control_path`` list in checkpoint
camera order selects text-to-video without the ``multiview`` object.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import msgspec

from sglang.multimodal_gen.configs.sample.cosmos3 import (
    Cosmos3SamplingParams,
    _parse_request_value,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

COSMOS3_MULTIVIEW_WIDTH = 832
COSMOS3_MULTIVIEW_HEIGHT = 480
# Per-camera frame default: the training window (200 camera frames, 67 LiDAR
# sweeps) unless the request says otherwise.
COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES = 201
# Training rate: the MADS WSM transfer recipes read clips at native 30 FPS.
COSMOS3_MULTIVIEW_DEFAULT_FPS = 30
COSMOS3_MULTIVIEW_DEFAULT_GUIDANCE_SCALE = 6.0
COSMOS3_MULTIVIEW_DEFAULT_NUM_INFERENCE_STEPS = 35
COSMOS3_MULTIVIEW_DEFAULT_FLOW_SHIFT = 10.0
# Prompt cap per caption, the reference tokenizer's truncation limit; the two
# framing tokens the tokenizer appends come on top of it.
COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH = 4096
COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODES = ("none", "same", "inverse")
COSMOS3_MULTIVIEW_RESOLUTIONS = ("480", "720")
# Canonical Cosmos3 video buckets as (width, height); all cameras share one.
COSMOS3_MULTIVIEW_RESOLUTION_BUCKETS: dict[str, dict[str, tuple[int, int]]] = {
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    },
    "720": {
        "1,1": (960, 960),
        "4,3": (1104, 832),
        "3,4": (832, 1104),
        "16,9": (1280, 720),
        "9,16": (720, 1280),
    },
}
COSMOS3_MULTIVIEW_ASPECT_RATIOS = ("1,1", "4,3", "3,4", "16,9", "9,16")
COSMOS3_MULTIVIEW_DEFAULT_ASPECT_RATIO = "16,9"
COSMOS3_LIDAR_CONTROL_SUFFIXES = frozenset({".safetensors", ".pt", ".pth"})

_ALLOWED_MULTIVIEW_FIELDS = frozenset(
    {
        "views",
        "condition_video_as_image",
        "condition_frame_indexes_vision",
        "num_frames",
        "resolution",
        "aspect_ratio",
    }
)
_ALLOWED_VIEW_FIELDS = frozenset(
    {"camera_key", "vision_path", "control_path", "vision", "control", "prompt"}
)


class MultiviewViewInput(msgspec.Struct, frozen=True):
    """One camera's request inputs. ``camera_key`` is None when implied by order."""

    camera_key: str | None
    control: str
    vision: str | None
    prompt: str | None = None


def normalize_multiview_aspect_ratio(value: Any) -> str:
    """The canonical ``"w,h"`` bucket label, or ``"auto"`` for detection."""
    if value is None or value == "auto":
        return "auto"
    parts = str(value).strip().replace(":", ",").split(",")
    if len(parts) == 2:
        try:
            width, height = (int(part.strip()) for part in parts)
        except ValueError:
            width = height = 0
        if width > 0 and height > 0:
            divisor = math.gcd(width, height)
            ratio = f"{width // divisor},{height // divisor}"
            if ratio in COSMOS3_MULTIVIEW_ASPECT_RATIOS:
                return ratio
    raise ValueError(
        f"Unsupported Cosmos3 multiview aspect_ratio={value!r}; expected auto, "
        "1:1, 4:3, 3:4, 16:9, or 9:16."
    )


def normalize_multiview_resolution(value: Any) -> str:
    resolution = str(value).strip()
    if resolution not in COSMOS3_MULTIVIEW_RESOLUTIONS:
        raise ValueError(
            "Cosmos3 multiview resolution must be one of "
            f"{list(COSMOS3_MULTIVIEW_RESOLUTIONS)}, got {value!r}."
        )
    return resolution


def multiview_canvas(resolution: str, aspect_ratio: str) -> tuple[int, int]:
    """``(width, height)`` of one canonical bucket."""
    return COSMOS3_MULTIVIEW_RESOLUTION_BUCKETS[
        normalize_multiview_resolution(resolution)
    ][aspect_ratio]


def closest_multiview_aspect_ratio(height: int, width: int, resolution: str) -> str:
    """The bucket whose height/width ratio is nearest the source media's."""
    if height <= 0 or width <= 0:
        raise ValueError(
            f"Cosmos3 multiview source media needs positive dimensions, got {height}x{width}."
        )
    buckets = COSMOS3_MULTIVIEW_RESOLUTION_BUCKETS[
        normalize_multiview_resolution(resolution)
    ]
    source_ratio = height / width
    return min(
        buckets,
        key=lambda ratio: abs(source_ratio - buckets[ratio][1] / buckets[ratio][0]),
    )


def parse_local_condition_indexes(value: Any) -> list[int] | None:
    """Per-view latent frame indexes from an int, list, or CSV string."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, bool):
        raise TypeError("condition_frame_indexes_vision must not be a boolean.")
    elif isinstance(value, int):
        parts = [value]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise TypeError(
            "condition_frame_indexes_vision must be an int, list, or CSV string."
        )
    return sorted({int(index) for index in parts})


def _as_optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False if normalized else None
    raise ValueError(f"{name} must be a boolean, got {value!r}.")


def _view_value(view: dict[str, Any], name: str) -> Any:
    return view.get(f"{name}_path", view.get(name))


def validate_multiview_request(multiview: Any) -> list[dict[str, Any]]:
    """Validate the ``multiview`` request object and return its views.

    Checkpoint-dependent rules (camera subsets, required per-camera prompts)
    are applied by the input stage, which knows the deployment contract.
    """
    if not isinstance(multiview, dict):
        raise ValueError(
            f"multiview must be a JSON object, got {type(multiview).__name__}."
        )
    unknown = set(multiview) - _ALLOWED_MULTIVIEW_FIELDS
    if unknown:
        raise ValueError(f"Unsupported Cosmos3 multiview fields: {sorted(unknown)}.")
    views = multiview.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("multiview.views must contain at least one camera view.")
    for index, view in enumerate(views):
        if not isinstance(view, dict):
            raise ValueError(f"multiview.views[{index}] must be an object.")
        unknown_view = set(view) - _ALLOWED_VIEW_FIELDS
        if unknown_view:
            raise ValueError(
                f"Unsupported Cosmos3 multiview view {index} fields: {sorted(unknown_view)}."
            )
        control = _view_value(view, "control")
        if not isinstance(control, str) or not control.strip():
            raise ValueError(
                f"multiview.views[{index}] requires a pre-computed WSM control_path."
            )
        vision = _view_value(view, "vision")
        if vision is not None and (not isinstance(vision, str) or not vision.strip()):
            raise ValueError(
                f"multiview.views[{index}].vision_path must be a non-empty string."
            )
        prompt = view.get("prompt")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"multiview.views[{index}].prompt must be a string.")
    camera_keys = [view.get("camera_key") for view in views]
    if any(key is not None for key in camera_keys):
        if not all(isinstance(key, str) and key for key in camera_keys):
            raise ValueError(
                "Cosmos3 multiview camera_key must be set for every view or none."
            )
        if len(set(camera_keys)) != len(camera_keys):
            raise ValueError(
                f"Cosmos3 multiview camera_key values must be unique: {camera_keys}."
            )
    prompts_present = [view.get("prompt") is not None for view in views]
    if any(prompts_present) and not all(prompts_present):
        raise ValueError(
            "Cosmos3 multiview per-camera prompts must be supplied for every view or none."
        )
    _as_optional_bool(
        multiview.get("condition_video_as_image"), "multiview.condition_video_as_image"
    )
    parse_local_condition_indexes(multiview.get("condition_frame_indexes_vision"))
    if multiview.get("resolution") is not None:
        normalize_multiview_resolution(multiview["resolution"])
    normalize_multiview_aspect_ratio(multiview.get("aspect_ratio"))
    num_frames = multiview.get("num_frames")
    if num_frames is not None and (
        isinstance(num_frames, bool)
        or not isinstance(num_frames, int)
        or num_frames <= 1
    ):
        raise ValueError(
            f"multiview.num_frames must be an integer greater than 1, got {num_frames!r}."
        )
    return views


# Measured sweeps conditioned when ``lidar.condition_path`` omits a count, as in
# the reference inference.
COSMOS3_LIDAR_DEFAULT_CONDITION_SWEEPS = 1


def validate_lidar_request(lidar: Any) -> dict[str, Any]:
    """Validate the joint camera/LiDAR ``lidar`` object.

    ``control_path`` is the prepared HD-map range-map clip every joint request
    needs. ``decode`` (default false) asks for the LiDAR output as well: the
    denoised LiDAR latents are decoded to range maps and written next to the
    camera video. ``condition_path`` (optional) holds measured sweeps in the same
    tensor format; the first ``num_conditional_sweeps`` (default 1) of them are
    kept clean as the start of the generated LiDAR, the way the camera's first
    frame anchors the RGB targets.
    """
    if not isinstance(lidar, dict):
        raise ValueError(f"lidar must be a JSON object, got {type(lidar).__name__}.")
    allowed = {"control_path", "decode", "condition_path", "num_conditional_sweeps"}
    unknown = set(lidar) - allowed
    if unknown or not isinstance(lidar.get("control_path"), str):
        raise ValueError(
            "Cosmos3 lidar requires a control_path string and accepts only the "
            f"optional decode, condition_path and num_conditional_sweeps fields; "
            f"got keys {sorted(lidar)}."
        )
    path = lidar["control_path"].strip()
    if os.path.splitext(path)[1].lower() not in COSMOS3_LIDAR_CONTROL_SUFFIXES:
        raise ValueError(
            "Cosmos3 lidar.control_path must be a prepared range-map tensor "
            f"({sorted(COSMOS3_LIDAR_CONTROL_SUFFIXES)}), got {path!r}."
        )
    decode = lidar.get("decode", False)
    if isinstance(decode, bool) is False:
        raise ValueError(f"Cosmos3 lidar.decode must be a boolean, got {decode!r}.")
    condition = lidar.get("condition_path")
    if condition is not None:
        if (
            not isinstance(condition, str)
            or os.path.splitext(condition.strip())[1].lower()
            not in COSMOS3_LIDAR_CONTROL_SUFFIXES
        ):
            raise ValueError(
                "Cosmos3 lidar.condition_path must be a prepared range-map tensor "
                f"({sorted(COSMOS3_LIDAR_CONTROL_SUFFIXES)}), got {condition!r}."
            )
        condition = condition.strip()
    count = lidar.get("num_conditional_sweeps")
    if count is not None:
        if condition is None:
            raise ValueError(
                "Cosmos3 lidar.num_conditional_sweeps requires lidar.condition_path."
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(
                f"Cosmos3 lidar.num_conditional_sweeps must be a positive integer, got {count!r}."
            )
    elif condition is not None:
        count = COSMOS3_LIDAR_DEFAULT_CONDITION_SWEEPS
    return {
        "control_path": path,
        "decode": decode,
        "condition_path": condition,
        "num_conditional_sweeps": count,
    }


@dataclass
class Cosmos3MultiviewSamplingParams(Cosmos3SamplingParams):
    # Resolved from ``resolution`` and ``aspect_ratio`` on adjustment; an
    # explicit canvas must match the selected bucket.
    height: int | None = None
    width: int | None = None
    # Per-camera pixel frames; rounded up to the VAE's 4k+1 grid on adjustment.
    num_frames: int = COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES
    fps: int = COSMOS3_MULTIVIEW_DEFAULT_FPS

    guidance_scale: float = COSMOS3_MULTIVIEW_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = COSMOS3_MULTIVIEW_DEFAULT_NUM_INFERENCE_STEPS

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (COSMOS3_MULTIVIEW_WIDTH, COSMOS3_MULTIVIEW_HEIGHT),
            *[
                size
                for buckets in COSMOS3_MULTIVIEW_RESOLUTION_BUCKETS.values()
                for size in buckets.values()
                if size != (COSMOS3_MULTIVIEW_WIDTH, COSMOS3_MULTIVIEW_HEIGHT)
            ],
        ]
    )

    # Per-camera inputs; see the module docstring for the schema.
    multiview: dict[str, Any] | None = None
    # The only supported control hint. ``{}``/``true`` select it explicitly;
    # None is accepted because every view already carries a WSM control clip.
    wsm: Any = None
    # Joint camera/LiDAR: one prepared HD-map range-map control clip.
    lidar: dict[str, Any] | None = None
    # Output bucket: ``"480"`` or ``"720"`` (default from the checkpoint) and
    # ``"auto"`` / ``"16:9"`` / ... (default auto, detected from the first WSM).
    resolution: str | int | None = None
    aspect_ratio: str | None = None
    # Use only the first frame of each camera's vision clip (top-level alias of
    # ``multiview.condition_video_as_image``).
    condition_video_as_image: bool | None = None
    # Metadata sentences carried by the negative prompt: none, same, or inverse.
    negative_metadata_mode: str = "same"
    # Per-camera-caption exports ignore negative_prompt: their unconditional
    # branch is one empty caption per camera, as training's caption dropout
    # produced. This opts into a shared negative caption instead, formatted with
    # the same duration/resolution metadata as every positive caption and
    # tokenized once per camera.
    per_view_negative_prompt: str | None = None
    # Append the control-adherence sentence to every caption (checkpoint default).
    emphasize_control_in_prompt: bool | None = None
    # Rescale the guided velocity to the conditional branch's norm.
    normalize_cfg: bool | None = None

    @classmethod
    def video_prompt_optional(cls) -> bool:
        # Captions are read per camera from multiview.views[].prompt.
        return True

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return super().video_request_extra_fields() | frozenset(
            {
                "multiview",
                "wsm",
                "lidar",
                "resolution",
                "aspect_ratio",
                "condition_video_as_image",
                "negative_metadata_mode",
                "per_view_negative_prompt",
                "emphasize_control_in_prompt",
                "normalize_cfg",
            }
        )

    @classmethod
    def lower_video_request_kwargs(
        cls, request: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        kwargs = super().lower_video_request_kwargs(request, dict(kwargs))
        for name in ("multiview", "wsm", "lidar"):
            if name in kwargs:
                value = _parse_request_value(kwargs[name])
                if value is None or (isinstance(value, str) and not value.strip()):
                    kwargs.pop(name)
                else:
                    kwargs[name] = value
        for name in (
            "condition_video_as_image",
            "emphasize_control_in_prompt",
            "normalize_cfg",
        ):
            if name in kwargs:
                value = _as_optional_bool(_parse_request_value(kwargs[name]), name)
                if value is None:
                    kwargs.pop(name)
                else:
                    kwargs[name] = value
        for name in (
            "negative_metadata_mode",
            "resolution",
            "aspect_ratio",
            "per_view_negative_prompt",
        ):
            if name in kwargs:
                value = kwargs[name]
                if value is None or not str(value).strip():
                    kwargs.pop(name)
                else:
                    kwargs[name] = str(value).strip()
        if "negative_metadata_mode" in kwargs:
            kwargs["negative_metadata_mode"] = kwargs["negative_metadata_mode"].lower()
        return kwargs

    def is_explicit(self, name: str) -> bool:
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and name in explicit

    def _is_unset(self, name: str, default: Any) -> bool:
        """A field the request neither passed nor changed from its class default."""
        return not self.is_explicit(name) and getattr(self, name) == default

    # -- Request views -----------------------------------------------------

    def resolve_views(self) -> list[MultiviewViewInput]:
        """Per-camera inputs, or an empty list when the request carries none."""
        if self.multiview is not None:
            views = validate_multiview_request(self.multiview)
            return [
                MultiviewViewInput(
                    camera_key=(
                        str(view["camera_key"])
                        if view.get("camera_key") is not None
                        else None
                    ),
                    control=str(_view_value(view, "control")),
                    vision=(
                        str(_view_value(view, "vision"))
                        if _view_value(view, "vision") is not None
                        else None
                    ),
                    prompt=(
                        str(view["prompt"]) if view.get("prompt") is not None else None
                    ),
                )
                for view in views
            ]
        control_paths = self._resolve_control_paths()
        return [
            MultiviewViewInput(camera_key=None, control=path, vision=None)
            for path in control_paths
        ]

    def resolved_lidar(self) -> dict[str, Any] | None:
        return validate_lidar_request(self.lidar) if self.lidar is not None else None

    def resolved_condition_video_as_image(self) -> bool:
        nested = None
        if isinstance(self.multiview, dict):
            nested = _as_optional_bool(
                self.multiview.get("condition_video_as_image"),
                "multiview.condition_video_as_image",
            )
        if nested is not None:
            return nested
        return bool(self.condition_video_as_image)

    def resolved_local_condition_indexes(self) -> list[int] | None:
        """Explicit per-view latent condition indexes, else None for the mode default."""
        if isinstance(self.multiview, dict):
            nested = parse_local_condition_indexes(
                self.multiview.get("condition_frame_indexes_vision")
            )
            if nested is not None:
                return nested
        if self.condition_frame_indexes:
            return sorted({int(index) for index in self.condition_frame_indexes})
        return None

    def resolved_resolution(self, default: str = "480") -> str:
        """``multiview.resolution``, then the top-level field, then ``default``."""
        nested = (
            self.multiview.get("resolution")
            if isinstance(self.multiview, dict)
            else None
        )
        value = nested if nested is not None else self.resolution
        return normalize_multiview_resolution(default if value is None else value)

    def resolved_aspect_ratio(self) -> str:
        """``"auto"`` or the canonical ``"w,h"`` bucket label."""
        nested = (
            self.multiview.get("aspect_ratio")
            if isinstance(self.multiview, dict)
            else None
        )
        return normalize_multiview_aspect_ratio(
            nested if nested is not None else self.aspect_ratio
        )

    def resolved_emphasize_control(self, default: bool = True) -> bool:
        return (
            default
            if self.emphasize_control_in_prompt is None
            else bool(self.emphasize_control_in_prompt)
        )

    # -- Validation and adjustment ------------------------------------------

    def _validate(self) -> None:
        super()._validate()
        if self.multiview is not None:
            validate_multiview_request(self.multiview)
        if self.lidar is not None:
            validate_lidar_request(self.lidar)
        if self.wsm is not None and self.wsm is not True:
            if not isinstance(self.wsm, dict) or set(self.wsm) - {"weight"}:
                raise ValueError(
                    "Cosmos3 multiview WSM controls are supplied per view; the "
                    "top-level wsm must be true or an empty object."
                )
        if self.resolution is not None:
            normalize_multiview_resolution(self.resolution)
        normalize_multiview_aspect_ratio(self.aspect_ratio)
        mode = str(self.negative_metadata_mode).strip().lower()
        if mode not in COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODES:
            raise ValueError(
                "negative_metadata_mode must be one of "
                f"{list(COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODES)}, "
                f"got {self.negative_metadata_mode!r}."
            )
        self.negative_metadata_mode = mode
        if (
            self.max_sequence_length is not None
            and int(self.max_sequence_length) > COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        ):
            raise ValueError(
                "Cosmos3 multiview max_sequence_length cannot exceed the caption cap: "
                f"requested={self.max_sequence_length}, "
                f"ceiling={COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH}."
            )
        if self.guidance_scale is not None and float(self.guidance_scale) < 0.0:
            raise ValueError(
                f"guidance_scale must be non-negative, got {self.guidance_scale}."
            )
        for name in (
            "condition_video_as_image",
            "emphasize_control_in_prompt",
            "normalize_cfg",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean or null.")
        if self.per_view_negative_prompt is not None and not isinstance(
            self.per_view_negative_prompt, str
        ):
            raise ValueError("per_view_negative_prompt must be a string or null.")

    def _apply_deployment_defaults(self, deployment: Any) -> None:
        """Fill request fields the caller left unset from the export's defaults."""
        nested_num_frames = (
            self.multiview.get("num_frames")
            if isinstance(self.multiview, dict)
            else None
        )
        if nested_num_frames is not None and not self.is_explicit("num_frames"):
            self.num_frames = int(nested_num_frames)
        if deployment is None:
            return
        if self._is_unset("fps", COSMOS3_MULTIVIEW_DEFAULT_FPS):
            self.fps = int(round(float(deployment.inference_default("fps", self.fps))))
        if self._is_unset(
            "num_inference_steps", COSMOS3_MULTIVIEW_DEFAULT_NUM_INFERENCE_STEPS
        ):
            self.num_inference_steps = int(
                deployment.inference_default("num_steps", self.num_inference_steps)
            )
        if self._is_unset("guidance_scale", COSMOS3_MULTIVIEW_DEFAULT_GUIDANCE_SCALE):
            self.guidance_scale = float(
                deployment.inference_default("guidance", self.guidance_scale)
            )
        if self.flow_shift is None:
            self.flow_shift = float(
                deployment.inference_default(
                    "shift", COSMOS3_MULTIVIEW_DEFAULT_FLOW_SHIFT
                )
            )
        if self._is_unset("control_guidance", 1.0):
            self.control_guidance = float(
                deployment.inference_default("control_guidance", 1.0)
            )
        if self.normalize_cfg is None:
            self.normalize_cfg = bool(
                deployment.inference_default("normalize_cfg", False)
            )
        if self.emphasize_control_in_prompt is None:
            self.emphasize_control_in_prompt = bool(
                deployment.inference_default("emphasize_control_in_prompt", True)
            )

    def _resolve_canvas(self, deployment: Any) -> None:
        """Select the output bucket; ``auto`` keeps a 16:9 placeholder for the input stage."""
        default_resolution = (
            deployment.inference_default("resolution", "480")
            if deployment is not None
            else "480"
        )
        resolution = self.resolved_resolution(str(default_resolution))
        aspect_ratio = self.resolved_aspect_ratio()
        bucket = (
            COSMOS3_MULTIVIEW_DEFAULT_ASPECT_RATIO
            if aspect_ratio == "auto"
            else aspect_ratio
        )
        width, height = multiview_canvas(resolution, bucket)
        if aspect_ratio != "auto":
            for name, expected in (("width", width), ("height", height)):
                requested = getattr(self, name)
                if requested is not None and int(requested) != expected:
                    raise ValueError(
                        f"Cosmos3 multiview resolution={resolution!r} aspect_ratio="
                        f"{aspect_ratio!r} requires {name}={expected}, got {requested}."
                    )
        elif self.width is not None or self.height is not None:
            # An explicit canvas selects its bucket instead of detection.
            if self.width is None or self.height is None:
                raise ValueError(
                    "Cosmos3 multiview requires both width and height when either is given."
                )
            ratio = closest_multiview_aspect_ratio(
                int(self.height), int(self.width), resolution
            )
            width, height = multiview_canvas(resolution, ratio)
            if (int(self.width), int(self.height)) != (width, height):
                raise ValueError(
                    f"Cosmos3 multiview canvas {self.width}x{self.height} is not a "
                    f"{resolution}p bucket; use one of "
                    f"{sorted(COSMOS3_MULTIVIEW_RESOLUTION_BUCKETS[resolution].values())}."
                )
            aspect_ratio = ratio
        self.width = width
        self.height = height
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio

    def _adjust(self, server_args) -> None:
        deployment = getattr(server_args.pipeline_config, "multiview_deployment", None)
        self._apply_deployment_defaults(deployment)
        if self.lidar is not None and (
            deployment is None or not deployment.supports_lidar
        ):
            raise ValueError(
                "Joint camera/LiDAR requests require a checkpoint that ships the LiDAR "
                "encoder and projections (transformer config multiview.lidar)."
            )
        # Select the canvas before the base adjustment, which would otherwise
        # fill an unset width/height from the first supported resolution.
        self._resolve_canvas(deployment)

        # Skip Cosmos3SamplingParams._adjust: its transfer branch rejects image
        # inputs and applies single-view per-hint defaults (the WSM hint would
        # pull the request to 101 frames at 10 FPS and guidance 1.0).
        SamplingParams._adjust(self, server_args)

        if self.num_frames == 1:
            raise ValueError(
                "Cosmos3 multiview generates video; num_frames must exceed 1."
            )
        if self.action_mode is not None or self.action is not None:
            raise ValueError(
                "Cosmos3 multiview cannot be combined with action streams."
            )
        if float(self.sound_duration or 0.0) > 0.0:
            raise ValueError(
                "Cosmos3 multiview cannot be combined with sound generation."
            )
        if self.image_path is not None or self.video_path is not None:
            raise ValueError(
                "Cosmos3 multiview takes per-camera vision inputs through "
                "multiview.views[*].vision_path, not image_path/video_path."
            )
        self._apply_guidance_policy(deployment)

    def _apply_guidance_policy(self, deployment: Any) -> None:
        del deployment  # The export's defaults were applied; guidance is not clamped.
        if self.control_guidance is None:
            self.control_guidance = 1.0
        # No chunked long-video transfer in multiview.
        self.share_vision_temporal_positions = True
