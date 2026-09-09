# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 Multiview-AV sampling parameters.

One request generates all cameras of the rig in a single bidirectional pass.
Per-camera inputs travel in the ``multiview`` object, mirroring the reference
input JSON::

    {
      "multiview": {
        "views": [
          {"camera_key": "camera_front_wide_120fov",
           "control_path": "wsm/front_wide.mp4",
           "vision_path": "rgb/front_wide.png"},
          ...
        ],
        "condition_video_as_image": true
      },
      "wsm": {}
    }

Every view needs a pre-computed WSM ``control_path``. ``vision_path`` (an RGB
still or clip) is optional but must be given for every camera or none; with it
the request is image-to-video, without it text-to-video. As a CLI convenience,
a ``control_path`` list in checkpoint camera order selects text-to-video
without the ``multiview`` object.
"""

from __future__ import annotations

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
COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES = 93
# Training rate: the MADS WSM transfer recipes read clips at native 30 FPS.
COSMOS3_MULTIVIEW_DEFAULT_FPS = 30
COSMOS3_MULTIVIEW_DEFAULT_GUIDANCE_SCALE = 6.0
COSMOS3_MULTIVIEW_MAX_GUIDANCE_SCALE = 7.0
COSMOS3_MULTIVIEW_DEFAULT_NUM_INFERENCE_STEPS = 35
# Prompt cap; the sparse attention pads text keys to this plus the two framing
# tokens, and the compiled kernel sees one shape for the life of the process.
COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH = 4096
COSMOS3_MULTIVIEW_NEGATIVE_METADATA_MODES = ("none", "same", "inverse")

_ALLOWED_MULTIVIEW_FIELDS = frozenset(
    {
        "views",
        "condition_video_as_image",
        "condition_frame_indexes_vision",
        "num_frames",
        "resolution",
    }
)
_ALLOWED_VIEW_FIELDS = frozenset(
    {"camera_key", "vision_path", "control_path", "vision", "control"}
)


class MultiviewViewInput(msgspec.Struct, frozen=True):
    """One camera's request inputs. ``camera_key`` is None when implied by order."""

    camera_key: str | None
    control: str
    vision: str | None


def clamp_multiview_guidance_scale(value: float) -> float:
    """The reference clamps guidance into ``[0, 7]`` rather than rejecting it."""
    return min(COSMOS3_MULTIVIEW_MAX_GUIDANCE_SCALE, max(0.0, float(value)))


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
    """Validate the ``multiview`` request object and return its views."""
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
    vision_present = [_view_value(view, "vision") is not None for view in views]
    if any(vision_present) and not all(vision_present):
        raise ValueError(
            "Cosmos3 multiview vision inputs must be supplied for every camera or none."
        )
    _as_optional_bool(
        multiview.get("condition_video_as_image"), "multiview.condition_video_as_image"
    )
    parse_local_condition_indexes(multiview.get("condition_frame_indexes_vision"))
    resolution = multiview.get("resolution")
    if resolution is not None and str(resolution) != "480":
        raise ValueError(
            f"Cosmos3 multiview v1 supports only resolution '480', got {resolution!r}."
        )
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


@dataclass
class Cosmos3MultiviewSamplingParams(Cosmos3SamplingParams):
    # v1 is fixed at 480p 16:9; other canvases are rejected.
    height: int | None = COSMOS3_MULTIVIEW_HEIGHT
    width: int | None = COSMOS3_MULTIVIEW_WIDTH
    # Per-camera pixel frames; rounded up to the VAE's 4k+1 grid on adjustment.
    num_frames: int = COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES
    fps: int = COSMOS3_MULTIVIEW_DEFAULT_FPS

    guidance_scale: float = COSMOS3_MULTIVIEW_DEFAULT_GUIDANCE_SCALE
    num_inference_steps: int = COSMOS3_MULTIVIEW_DEFAULT_NUM_INFERENCE_STEPS

    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [(COSMOS3_MULTIVIEW_WIDTH, COSMOS3_MULTIVIEW_HEIGHT)]
    )

    # Per-camera inputs; see the module docstring for the schema.
    multiview: dict[str, Any] | None = None
    # The only supported control hint. ``{}``/``true`` select it explicitly;
    # None is accepted because every view already carries a WSM control clip.
    wsm: Any = None
    # Use only the first frame of each camera's vision clip (top-level alias of
    # ``multiview.condition_video_as_image``).
    condition_video_as_image: bool | None = None
    # Metadata sentences carried by the negative prompt: none, same, or inverse.
    negative_metadata_mode: str = "same"

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        return super().video_request_extra_fields() | frozenset(
            {"multiview", "wsm", "condition_video_as_image", "negative_metadata_mode"}
        )

    @classmethod
    def lower_video_request_kwargs(
        cls, request: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        kwargs = super().lower_video_request_kwargs(request, dict(kwargs))
        for name in ("multiview", "wsm"):
            if name in kwargs:
                value = _parse_request_value(kwargs[name])
                if value is None or (isinstance(value, str) and not value.strip()):
                    kwargs.pop(name)
                else:
                    kwargs[name] = value
        if "condition_video_as_image" in kwargs:
            value = _as_optional_bool(
                _parse_request_value(kwargs["condition_video_as_image"]),
                "condition_video_as_image",
            )
            if value is None:
                kwargs.pop("condition_video_as_image")
            else:
                kwargs["condition_video_as_image"] = value
        if "negative_metadata_mode" in kwargs:
            value = kwargs["negative_metadata_mode"]
            if value is None or not str(value).strip():
                kwargs.pop("negative_metadata_mode")
            else:
                kwargs["negative_metadata_mode"] = str(value).strip().lower()
        return kwargs

    def is_explicit(self, name: str) -> bool:
        explicit = getattr(self, "_explicit_fields", None)
        return explicit is not None and name in explicit

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
                )
                for view in views
            ]
        control_paths = self._resolve_control_paths()
        return [
            MultiviewViewInput(camera_key=None, control=path, vision=None)
            for path in control_paths
        ]

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

    # -- Validation and adjustment ------------------------------------------

    def _validate(self) -> None:
        super()._validate()
        if self.multiview is not None:
            validate_multiview_request(self.multiview)
        if self.wsm is not None and self.wsm is not True:
            if not isinstance(self.wsm, dict) or self.wsm:
                raise ValueError(
                    "Cosmos3 multiview WSM controls are supplied per view; the "
                    "top-level wsm must be true or an empty object."
                )
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
                "Cosmos3 multiview max_sequence_length cannot exceed the ceiling the "
                f"sparse attention is sized for: requested={self.max_sequence_length}, "
                f"ceiling={COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH}."
            )
        if self.guidance_scale is not None and float(self.guidance_scale) < 0.0:
            raise ValueError(
                f"guidance_scale must be non-negative, got {self.guidance_scale}."
            )
        if self.condition_video_as_image is not None and not isinstance(
            self.condition_video_as_image, bool
        ):
            raise ValueError("condition_video_as_image must be a boolean or null.")

    def _adjust(self, server_args) -> None:
        # The nested frame count is the variant-owned copy of num_frames.
        nested_num_frames = (
            self.multiview.get("num_frames")
            if isinstance(self.multiview, dict)
            else None
        )
        if nested_num_frames is not None and not self.is_explicit("num_frames"):
            self.num_frames = int(nested_num_frames)

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
                "Cosmos3 multiview v1 cannot be combined with action streams."
            )
        if float(self.sound_duration or 0.0) > 0.0:
            raise ValueError(
                "Cosmos3 multiview v1 cannot be combined with sound generation."
            )
        if self.image_path is not None or self.video_path is not None:
            raise ValueError(
                "Cosmos3 multiview takes per-camera vision inputs through "
                "multiview.views[*].vision_path, not image_path/video_path."
            )
        if (int(self.width), int(self.height)) != (
            COSMOS3_MULTIVIEW_WIDTH,
            COSMOS3_MULTIVIEW_HEIGHT,
        ):
            raise ValueError(
                "Cosmos3 multiview v1 is fixed at "
                f"{COSMOS3_MULTIVIEW_WIDTH}x{COSMOS3_MULTIVIEW_HEIGHT}, "
                f"got {self.width}x{self.height}."
            )

        clamped = clamp_multiview_guidance_scale(self.guidance_scale)
        if clamped != float(self.guidance_scale):
            logger.info(
                "Clamped Cosmos3 multiview guidance_scale from %s to %s",
                self.guidance_scale,
                clamped,
            )
        self.guidance_scale = clamped
        # No control-CFG and no chunked long-video transfer in multiview v1.
        self.control_guidance = 1.0
        self.control_guidance_interval = None
        self.share_vision_temporal_positions = True
