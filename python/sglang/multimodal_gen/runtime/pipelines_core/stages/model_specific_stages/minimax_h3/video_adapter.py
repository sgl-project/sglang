# SPDX-License-Identifier: Apache-2.0
"""Video API lowering and strict delivery hooks for MiniMax H3."""

from __future__ import annotations

import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import QUALITY_LEVELS
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SUPPORTED_FPS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    canonical_minimax_h3_task,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


def _extra_value(request: VideoGenerationsRequest, name: str) -> Any:
    return (request.model_extra or {}).get(name)


def _parse_extra_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _format_video_seconds(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return str(int(rounded))
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


class MiniMaxH3VideoModelAdapter:
    """Canonical request and fail-closed AV delivery hooks for MiniMax H3."""

    strict_file_delivery = True
    model_specific_fields = frozenset(
        {
            "task",
            "conditions",
            "target",
            "audio_flow_shift",
            "audio_guidance_scale",
            "quality",
            "output_mode",
            "imgvid_cond_noise_aug_for_inference",
            "audio_cond_noise_aug_for_inference",
        }
    )
    supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})

    def validate_task_gate(self, task: Any, *, provided: bool) -> None:
        if not provided or task is None:
            raise ValueError(
                "task is required for MiniMax H3; supported tasks: fl2va, ref2va, t2va"
            )
        if not isinstance(task, str):
            raise ValueError("task must be a non-empty string for MiniMax H3")
        if not task.strip():
            raise ValueError("task must be a non-empty string for MiniMax H3")
        if task not in self.supported_tasks:
            raise ValueError(
                f"unsupported MiniMax H3 task {task!r}; supported tasks: "
                "fl2va, ref2va, t2va"
            )

    @staticmethod
    def _positive_finite_extra(
        request: VideoGenerationsRequest,
        name: str,
    ) -> float | None:
        value = _parse_extra_value(_extra_value(request, name))
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive finite float")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite float") from exc
        if not math.isfinite(parsed) or parsed <= 0:
            raise ValueError(f"{name} must be a positive finite float")
        return parsed

    @staticmethod
    def _quality_extra(
        request: VideoGenerationsRequest,
        name: str,
    ) -> str | None:
        value = _extra_value(request, name)
        if value is None:
            return None
        if value not in QUALITY_LEVELS:
            raise ValueError(
                f"{name} must be one of {list(QUALITY_LEVELS)}, got {value!r}"
            )
        return value

    @staticmethod
    def _reject_retired_cfg_fields(kwargs: dict[str, Any]) -> None:
        """Fail fast on retired CFG knobs.

        MiniMax H3 serves only the CFG-distilled single-positive-branch
        checkpoint, so these fields cannot influence generation. Any
        provided value is an error; only absent (``None``) is accepted.
        """

        for field_name in (
            "guidance_scale",
            "guidance_scale_2",
            "true_cfg_scale",
            "negative_prompt",
        ):
            if kwargs.pop(field_name, None) is not None:
                raise ValueError(
                    f"{field_name} is not supported: MiniMax H3 serves only the "
                    "CFG-distilled single-positive-branch checkpoint"
                )

    @staticmethod
    def _reject_transport_timing_fields(request: VideoGenerationsRequest) -> None:
        """Fail fast on explicit transport timing fields.

        Canonical MiniMax H3 timing is resolved exclusively from
        ``target.duration_seconds`` (or the single ref2va audio source), so
        explicit ``num_frames``/``fps`` cannot influence generation. Only
        absent (``None``, the protocol default) is accepted.
        """

        for field_name in ("num_frames", "fps"):
            if getattr(request, field_name) is not None:
                raise ValueError(
                    f"{field_name} is not supported: MiniMax H3 derives the "
                    "temporal shape from target.duration_seconds"
                )

    def lower_video_request_kwargs(
        self,
        request: VideoGenerationsRequest,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self.validate_transport_options(request, model_path=None)
        kwargs = dict(kwargs)
        self._reject_transport_timing_fields(request)
        # Drop the transport-synthesized timing defaults; canonical MiniMax H3
        # timing never flows through num_frames/fps.
        kwargs.pop("num_frames", None)
        kwargs.pop("fps", None)
        self._reject_retired_cfg_fields(kwargs)
        quality = self._quality_extra(request, "quality")
        kwargs.update(
            {
                "audio_flow_shift": self._positive_finite_extra(
                    request, "audio_flow_shift"
                ),
                "task": canonical_minimax_h3_task(
                    _parse_extra_value(_extra_value(request, "task"))
                ),
                "conditions": _parse_extra_value(_extra_value(request, "conditions")),
                "target": _parse_extra_value(_extra_value(request, "target")),
                "imgvid_cond_noise_aug_for_inference": _parse_extra_value(
                    _extra_value(request, "imgvid_cond_noise_aug_for_inference")
                ),
                "audio_cond_noise_aug_for_inference": _parse_extra_value(
                    _extra_value(request, "audio_cond_noise_aug_for_inference")
                ),
            }
        )
        if quality is not None:
            kwargs["quality"] = quality
        return kwargs

    def validate_transport_options(
        self,
        request: VideoGenerationsRequest,
        *,
        model_path: str | None,
    ) -> None:
        extras = request.model_extra or {}
        self.validate_task_gate(extras.get("task"), provided="task" in extras)
        del model_path
        self._positive_finite_extra(request, "audio_flow_shift")
        if _extra_value(request, "audio_guidance_scale") is not None:
            raise ValueError(
                "audio_guidance_scale is not supported: MiniMax H3 serves only "
                "the CFG-distilled single-positive-branch checkpoint"
            )
        output_mode = _extra_value(request, "output_mode")
        if output_mode not in (None, "decoded_files"):
            raise ValueError(
                "MiniMax H3 SGLang backend only supports "
                f"output_mode='decoded_files', got {output_mode!r}"
            )
        if request.enable_frame_interpolation:
            raise ValueError(
                "MiniMax H3 does not support enable_frame_interpolation: the "
                "accepted delivery contract is the canonical 24 fps output"
            )
        if request.enable_upscaling:
            raise ValueError(
                "MiniMax H3 does not support enable_upscaling: the accepted "
                "delivery contract is the resolved target canvas"
            )

    def validate_sampling_params(self, sampling_params: SamplingParams) -> None:
        """Apply the HTTP task/delivery gate to offline requests as well."""

        task = getattr(sampling_params, "task", None)
        self.validate_task_gate(task, provided=task is not None)
        sampling_params.task = canonical_minimax_h3_task(task)
        if getattr(sampling_params, "enable_frame_interpolation", False):
            raise ValueError(
                "MiniMax H3 does not support enable_frame_interpolation: the "
                "accepted delivery contract is the canonical 24 fps output"
            )
        if getattr(sampling_params, "enable_upscaling", False):
            raise ValueError(
                "MiniMax H3 does not support enable_upscaling: the accepted "
                "delivery contract is the resolved target canvas"
            )
        if not bool(getattr(sampling_params, "save_output", False)) or not getattr(
            sampling_params, "output_path", None
        ):
            raise ValueError(
                "MiniMax H3 DiffGenerator requires save_output=True and a non-empty "
                "output_path for validated file delivery"
            )
        output_mode = getattr(sampling_params, "output_mode", None)
        if output_mode not in (None, "decoded_files"):
            raise ValueError(
                "MiniMax H3 SGLang backend only supports "
                f"output_mode='decoded_files', got {output_mode!r}"
            )

    def prepare_for_queue_sync(self, batch: Req) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            minimax_h3_prepare_for_queue,
        )

        minimax_h3_prepare_for_queue(batch)

    def cleanup_request_sync(self, batch: Req) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_cleanup_temp_dirs,
        )

        minimax_h3_cleanup_temp_dirs(batch)

    @staticmethod
    def _resolved_shape(batch: Req) -> dict[str, Any] | None:
        canonical = getattr(batch, "extra", {}).get("minimax_h3_canonical_request")
        if not isinstance(canonical, dict) or not all(
            key in canonical
            for key in ("schema", "task", "prompt", "conditions", "target")
        ):
            return None
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
            minimax_h3_plan_from_batch,
        )

        plan = minimax_h3_plan_from_batch(batch)
        if plan is None:
            return None
        shape = plan.shape
        if str(shape.get("geometry") or "") != "resolved_v2":
            raise ValueError(
                "queued MiniMax H3 jobs require pre-queue resolved_v2 geometry"
            )
        if shape.get("frame_count") is None:
            raise ValueError(
                "queued MiniMax H3 jobs require pre-queue resolved temporal dimensions"
            )
        return shape

    def project_queued_job_fields(self, batch: Req) -> dict[str, str]:
        shape = self._resolved_shape(batch)
        if shape is None:
            return {}
        fields: dict[str, str] = {}
        if shape.get("width") is not None and shape.get("height") is not None:
            fields["size"] = f"{int(shape['width'])}x{int(shape['height'])}"
        queued_frame_count = shape.get("frame_count")
        if queued_frame_count is not None:
            fields["seconds"] = _format_video_seconds(
                int(queued_frame_count) / float(shape["fps"])
            )
        quality = getattr(batch.sampling_params, "quality", None)
        explicit_fields = getattr(batch.sampling_params, "_explicit_fields", ())
        if quality and "quality" in explicit_fields:
            fields["quality"] = str(quality)
        return fields

    def validate_final_outputs_sync(
        self,
        output_paths: list[str],
        batch: Req,
    ) -> dict[str, str]:
        expected_outputs = int(getattr(batch, "num_outputs_per_prompt", 1))
        if len(output_paths) != expected_outputs:
            raise RuntimeError(
                "MiniMax H3 video generation produced "
                f"{len(output_paths)} output files, expected {expected_outputs}"
            )

        expected_frame_count = None
        expected_size = None
        shape = self._resolved_shape(batch)
        if shape is not None:
            if shape.get("frame_count") is not None:
                expected_frame_count = int(shape["frame_count"])
            if shape.get("width") is not None and shape.get("height") is not None:
                expected_size = (int(shape["width"]), int(shape["height"]))

        def probe_output(output_path: str) -> dict[str, str]:
            return _probe_minimax_h3_output_fields(
                output_path,
                expected_frame_count=expected_frame_count,
                expected_size=expected_size,
            )

        if len(output_paths) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(output_paths))) as pool:
                media_fields_by_output = list(pool.map(probe_output, output_paths))
        else:
            media_fields_by_output = [probe_output(output_paths[0])]

        final_media_fields = media_fields_by_output[0]
        for output_index, media_fields in enumerate(media_fields_by_output[1:], 1):
            if media_fields != final_media_fields:
                raise RuntimeError(
                    "generated MiniMax H3 outputs have inconsistent media metadata: "
                    f"output 0={final_media_fields}, output "
                    f"{output_index}={media_fields}"
                )
        return final_media_fields


def _probe_minimax_h3_output_fields(
    path: str,
    *,
    expected_frame_count: int | None = None,
    expected_size: tuple[int, int] | None = None,
) -> dict[str, str]:
    """Validate one final MiniMax H3 AV file and derive truthful metadata."""

    try:
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration,sample_rate,channels:format=format_name,duration",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "MiniMax H3 final output ffprobe timed out after 30 seconds"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffprobe is required to validate final MiniMax H3 output"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"ffprobe failed for final MiniMax H3 output{suffix}"
        ) from exc

    try:
        payload = json.loads(probe.stdout)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "ffprobe returned invalid JSON for MiniMax H3 output"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ffprobe returned invalid JSON for MiniMax H3 output")
    streams = payload.get("streams") or []
    if not isinstance(streams, list):
        raise RuntimeError("ffprobe returned invalid stream metadata")
    video_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "video"
    ]
    audio_streams = [
        stream
        for stream in streams
        if isinstance(stream, dict) and stream.get("codec_type") == "audio"
    ]
    if len(streams) != 2 or len(video_streams) != 1 or len(audio_streams) != 1:
        raise RuntimeError(
            "generated MiniMax H3 MP4 must contain exactly one video stream and "
            f"one audio stream, got {len(video_streams)} video, "
            f"{len(audio_streams)} audio, and {len(streams)} total streams"
        )
    video_stream = video_streams[0]
    audio_stream = audio_streams[0]
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(
            f"generated MiniMax H3 MP4 has invalid size {width}x{height}"
        )
    if expected_size is not None and (width, height) != expected_size:
        raise RuntimeError(
            "generated MiniMax H3 MP4 size does not match the resolved request: "
            f"expected {expected_size[0]}x{expected_size[1]}, got {width}x{height}"
        )
    rate_raw = str(video_stream.get("avg_frame_rate") or "")
    try:
        if "/" in rate_raw:
            numerator, denominator = rate_raw.split("/", 1)
            fps = float(numerator) / float(denominator)
        else:
            fps = float(rate_raw)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0
    if abs(fps - float(MINIMAX_H3_SUPPORTED_FPS)) > 1e-6:
        raise RuntimeError(
            "generated MiniMax H3 MP4 frame rate must be "
            f"{MINIMAX_H3_SUPPORTED_FPS} fps, got {rate_raw!r}"
        )
    try:
        audio_sample_rate = int(audio_stream.get("sample_rate") or 0)
    except (TypeError, ValueError):
        audio_sample_rate = 0
    expected_sample_rate = MiniMaxH3PipelineConfig.output_audio_sample_rate
    if expected_sample_rate is not None and audio_sample_rate != expected_sample_rate:
        raise RuntimeError(
            "generated MiniMax H3 MP4 audio sample rate must be "
            f"{expected_sample_rate} Hz, "
            f"got {audio_sample_rate}"
        )
    try:
        audio_channels = int(audio_stream.get("channels") or 0)
    except (TypeError, ValueError):
        audio_channels = 0
    expected_channels = MiniMaxH3PipelineConfig.output_audio_channels
    if expected_channels is not None and audio_channels != expected_channels:
        channel_contract = (
            "stereo (2 channels)"
            if expected_channels == 2
            else f"{expected_channels} channels"
        )
        raise RuntimeError(
            "generated MiniMax H3 MP4 audio must be "
            f"{channel_contract}, got {audio_channels}"
        )
    try:
        duration = float(video_stream.get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    frames_raw = video_stream.get("nb_frames")
    frame_count = None
    if frames_raw not in {None, "", "N/A"}:
        try:
            frame_count = int(frames_raw)
        except (TypeError, ValueError):
            frame_count = None
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise RuntimeError(
            "generated MiniMax H3 MP4 frame count does not match the resolved "
            f"request: expected {expected_frame_count}, got {frame_count}"
        )
    if frame_count is not None and "/" in rate_raw and fps > 0:
        duration = frame_count / fps
    if duration <= 0:
        try:
            format_metadata = payload.get("format")
            if not isinstance(format_metadata, dict):
                format_metadata = {}
            duration = float(format_metadata.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
    if duration <= 0:
        raise RuntimeError("generated MiniMax H3 MP4 has no positive video duration")
    try:
        audio_duration = float(audio_stream.get("duration") or 0.0)
    except (TypeError, ValueError):
        audio_duration = 0.0
    if audio_duration <= 0:
        raise RuntimeError("generated MiniMax H3 MP4 has no positive audio duration")
    drift_tolerance_s = MiniMaxH3PipelineConfig.output_av_drift_tolerance_s
    if (
        drift_tolerance_s is not None
        and abs(audio_duration - duration) > drift_tolerance_s
    ):
        raise RuntimeError(
            "generated MiniMax H3 MP4 audio/video duration drift exceeds "
            f"{drift_tolerance_s:g}s: "
            f"video={duration:.6f}s audio={audio_duration:.6f}s"
        )
    format_metadata = payload.get("format")
    if not isinstance(format_metadata, dict):
        format_metadata = {}
    format_names = {
        token.strip().lower()
        for token in str(format_metadata.get("format_name") or "").split(",")
        if token.strip()
    }
    if "mp4" not in format_names:
        raise RuntimeError(
            "generated MiniMax H3 output container must be MP4-family, got "
            f"{format_metadata.get('format_name')!r}"
        )
    video_codec = str(video_stream.get("codec_name") or "").lower()
    if video_codec != "h264":
        raise RuntimeError(
            f"generated MiniMax H3 MP4 video codec must be h264, got {video_codec!r}"
        )
    audio_codec = str(audio_stream.get("codec_name") or "").lower()
    if audio_codec != "aac":
        raise RuntimeError(
            f"generated MiniMax H3 MP4 audio codec must be aac, got {audio_codec!r}"
        )
    pixel_format = str(video_stream.get("pix_fmt") or "").lower()
    if pixel_format != "yuv420p":
        raise RuntimeError(
            f"generated MiniMax H3 MP4 pixel format must be yuv420p, got {pixel_format!r}"
        )
    return {
        "size": f"{width}x{height}",
        "seconds": _format_video_seconds(duration),
    }


__all__ = ["MiniMaxH3VideoModelAdapter"]
