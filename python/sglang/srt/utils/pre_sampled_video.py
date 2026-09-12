"""Video frames sampled upstream, before model-specific spatial preprocessing."""

import base64
import io
import math
from numbers import Integral, Real
from typing import Any, Optional, Sequence

import msgspec
import numpy as np
import torch
from PIL import Image


class PreSampledVideo(msgspec.Struct):
    """RGB frames with their indices on the original constant-FPS timeline.

    In-process frames may be a uint8 THWC array/tensor or a sequence of RGB PIL
    images. The JSON representation uses ``format="pre_sampled_video"`` and a
    list of base64 image data URLs. Loading never invokes a video decoder or
    selects a subset of the supplied frames.
    """

    frames: Any
    source_fps: float
    total_num_frames: int
    frame_indices: Sequence[int]
    timestamps: Optional[Sequence[float]] = None
    do_sample_frames: bool = False

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        # HTTP inputs use the tagged dictionary; Engine inputs may hold this
        # in-process object without materializing its frames as JSON arrays.
        from pydantic_core import core_schema

        return core_schema.is_instance_schema(cls)

    def __post_init__(self):
        self._validate_metadata()

    def _validate_metadata(self):
        if self.do_sample_frames is not False:
            raise ValueError("Pre-sampled video requires do_sample_frames=False")
        if (
            isinstance(self.source_fps, bool)
            or not isinstance(self.source_fps, Real)
            or not math.isfinite(self.source_fps)
            or self.source_fps <= 0
        ):
            raise ValueError("Pre-sampled video source_fps must be finite and positive")
        if (
            isinstance(self.total_num_frames, bool)
            or not isinstance(self.total_num_frames, Integral)
            or self.total_num_frames <= 0
        ):
            raise ValueError(
                "Pre-sampled video total_num_frames must be a positive integer"
            )
        if not isinstance(self.frames, (list, tuple, np.ndarray, torch.Tensor)):
            raise ValueError(
                "Pre-sampled video frames must be a nonempty frame sequence"
            )
        if (
            isinstance(self.frames, (np.ndarray, torch.Tensor))
            and self.frames.ndim != 4
        ):
            raise ValueError("Pre-sampled video arrays must have uint8 THWC RGB layout")
        count = len(self.frames)
        if not count:
            raise ValueError("Pre-sampled video frames must not be empty")
        if not isinstance(self.frame_indices, (list, tuple, np.ndarray)):
            raise ValueError("Pre-sampled video frame_indices must be a sequence")
        if isinstance(self.frame_indices, np.ndarray) and self.frame_indices.ndim != 1:
            raise ValueError("Pre-sampled video frame_indices must be one-dimensional")
        if len(self.frame_indices) != count:
            raise ValueError(
                "Pre-sampled video frame_indices must match the frame count"
            )
        previous = -1
        for index in self.frame_indices:
            if (
                isinstance(index, bool)
                or not isinstance(index, Integral)
                or not 0 <= index < self.total_num_frames
                or index < previous
            ):
                raise ValueError(
                    "Pre-sampled video frame_indices must be nondecreasing integers "
                    "within the original video"
                )
            previous = index
        if self.timestamps is not None:
            if not isinstance(self.timestamps, (list, tuple, np.ndarray)):
                raise ValueError("Pre-sampled video timestamps must be a sequence")
            if isinstance(self.timestamps, np.ndarray) and self.timestamps.ndim != 1:
                raise ValueError("Pre-sampled video timestamps must be one-dimensional")
            if len(self.timestamps) != count:
                raise ValueError(
                    "Pre-sampled video timestamps must match the frame count"
                )
            for timestamp, index in zip(self.timestamps, self.frame_indices):
                if (
                    isinstance(timestamp, bool)
                    or not isinstance(timestamp, Real)
                    or not math.isfinite(timestamp)
                    or not math.isclose(
                        timestamp, index / self.source_fps, rel_tol=1e-6, abs_tol=1e-6
                    )
                ):
                    raise ValueError(
                        "Pre-sampled video timestamps must equal frame_indices / source_fps; "
                        "variable-frame-rate timelines are not supported"
                    )

    def to_wire(self):
        """Encode RGB frames losslessly for JSON-based serving and EPD transport."""
        loaded = load_pre_sampled_video(self)
        frames = []
        for frame in loaded.frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="PNG")
            frames.append(
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode("ascii")
            )
        result = {
            "format": "pre_sampled_video",
            "frames": frames,
            "source_fps": float(loaded.source_fps),
            "total_num_frames": int(loaded.total_num_frames),
            "frame_indices": [int(index) for index in loaded.frame_indices],
            "do_sample_frames": False,
        }
        if loaded.timestamps is not None:
            result["timestamps"] = [float(timestamp) for timestamp in loaded.timestamps]
        return result

    def to_processor_inputs(self):
        """Return loaded frames and metadata understood by HF video processors."""
        loaded = load_pre_sampled_video(self)
        return loaded.frames, {
            "fps": float(loaded.source_fps),
            "duration": loaded.total_num_frames / loaded.source_fps,
            "total_num_frames": int(loaded.total_num_frames),
            "frames_indices": [int(index) for index in loaded.frame_indices],
            "video_backend": "sglang",
        }


def is_pre_sampled_video(value):
    return isinstance(value, PreSampledVideo) or (
        isinstance(value, dict) and value.get("format") == "pre_sampled_video"
    )


def load_pre_sampled_video(value):
    """Validate an input and materialize a uint8 THWC RGB array, without sampling."""
    wire_input = isinstance(value, dict)
    if wire_input:
        if value.get("format") != "pre_sampled_video":
            raise ValueError("Expected format=pre_sampled_video")
        fields = {key: item for key, item in value.items() if key != "format"}
        try:
            value = PreSampledVideo(**fields)
        except TypeError as exc:
            raise ValueError("Invalid pre-sampled video fields") from exc
    if not isinstance(value, PreSampledVideo):
        raise ValueError("Expected a PreSampledVideo or pre_sampled_video dictionary")
    value._validate_metadata()

    frames = value.frames
    if wire_input and not (
        isinstance(frames, list)
        and all(
            isinstance(frame, str)
            and frame.startswith("data:image/")
            and ";base64," in frame
            for frame in frames
        )
    ):
        raise ValueError("Pre-sampled video JSON frames must be base64 image data URLs")

    if isinstance(frames, torch.Tensor):
        if frames.device.type != "cpu" or frames.dtype != torch.uint8:
            raise ValueError("Pre-sampled video tensors must be uint8 and on CPU")
        frames = frames.detach().numpy()
    if isinstance(frames, np.ndarray):
        if (
            frames.dtype != np.uint8
            or frames.shape[-1] != 3
            or min(frames.shape[1:3]) <= 0
        ):
            raise ValueError("Pre-sampled video arrays must have uint8 THWC RGB layout")
        return PreSampledVideo(
            frames=np.ascontiguousarray(frames),
            source_fps=value.source_fps,
            total_num_frames=value.total_num_frames,
            frame_indices=list(value.frame_indices),
            timestamps=None if value.timestamps is None else list(value.timestamps),
            do_sample_frames=False,
        )

    loaded = []
    for frame in frames:
        if isinstance(frame, str):
            if not frame.startswith("data:image/") or ";base64," not in frame:
                raise ValueError(
                    "Pre-sampled video frame strings must be base64 image data URLs"
                )
            # Reuse normal image validation, without GPU decode or video codecs.
            from sglang.srt.utils.common import load_image

            frame, _ = load_image(frame, gpu_image_decode=False)
        if isinstance(frame, Image.Image):
            if frame.mode != "RGB":
                raise ValueError("Pre-sampled video images must be RGB")
            frame = np.asarray(frame)
        elif isinstance(frame, torch.Tensor):
            if frame.device.type != "cpu" or frame.dtype != torch.uint8:
                raise ValueError("Pre-sampled video tensors must be uint8 and on CPU")
            frame = frame.detach().numpy()
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.ndim != 3
            or frame.shape[-1] != 3
            or min(frame.shape[:2]) <= 0
        ):
            raise ValueError("Pre-sampled video frames must have uint8 HWC RGB layout")
        if loaded and frame.shape != loaded[0].shape:
            raise ValueError("Pre-sampled video frames must have identical dimensions")
        loaded.append(frame)
    return PreSampledVideo(
        frames=np.stack(loaded),
        source_fps=value.source_fps,
        total_num_frames=value.total_num_frames,
        frame_indices=list(value.frame_indices),
        timestamps=None if value.timestamps is None else list(value.timestamps),
        do_sample_frames=False,
    )
