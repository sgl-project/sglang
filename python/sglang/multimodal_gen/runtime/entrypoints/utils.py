# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import os
import shutil
import subprocess
import tempfile
from copy import copy
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

import imageio
import numpy as np
import torch
from PIL import Image

try:
    import scipy.io.wavfile as scipy_wavfile
except ImportError:  # pragma: no cover
    scipy_wavfile = None

try:
    import imageio_ffmpeg as _imageio_ffmpeg
except ImportError:  # pragma: no cover
    _imageio_ffmpeg = None

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger
from sglang.srt.observability.trace import TraceReqContext

logger = init_logger(__name__)


# Video outputs are written through imageio/ffmpeg as MP4/H.264 and optional
# audio muxing also targets MP4. Keep the visible path aligned with the
# container we actually create.
VIDEO_OUTPUT_EXTENSIONS = frozenset({".mp4"})

# Audio sample-rate bounds used for best-effort validation/inference.
MIN_AUDIO_SAMPLE_RATE = 8000
MAX_AUDIO_SAMPLE_RATE = 192000
DEFAULT_AUDIO_SAMPLE_RATE = 24000


@dataclass
class SetLoraReq:
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    merge_mode: Optional[str] = None


@dataclass
class MergeLoraWeightsReq:
    target: str = "all"
    strength: float = 1.0


@dataclass
class UnmergeLoraWeightsReq:
    target: str = "all"


@dataclass
class ListLorasReq:
    pass


@dataclass
class ShutdownReq:
    pass


@dataclass
class ReleaseRealtimeSessionReq:
    session_id: str


@dataclass
class GetDisaggStatsReq:
    """Request to get disagg pipeline metrics from the scheduler."""

    pass


def format_lora_message(
    lora_nickname: Union[str, List[str]],
    target: Union[str, List[str]],
    strength: Union[float, List[float]],
) -> tuple[str, str, str]:
    """Format success message for single or multiple LoRAs."""
    if isinstance(lora_nickname, list):
        nickname_str = ", ".join(lora_nickname)
        target_str = ", ".join(target) if isinstance(target, list) else target
        strength_str = (
            ", ".join(f"{s:.2f}" for s in strength)
            if isinstance(strength, list)
            else f"{strength:.2f}"
        )
    else:
        nickname_str = lora_nickname
        target_str = target if isinstance(target, str) else ", ".join(target)
        strength_str = (
            f"{strength:.2f}"
            if isinstance(strength, (int, float))
            else ", ".join(f"{s:.2f}" for s in strength)
        )
    return nickname_str, target_str, strength_str


@dataclass
class GenerationResult:
    """Result of a single generation request from DiffGenerator."""

    samples: Any = None
    frames: Any = None
    audio: Any = None
    prompt: str | None = None
    size: tuple | None = None  # (height, width, num_frames)
    generation_time: float = 0.0
    peak_memory_mb: float = 0.0
    metrics: dict = field(default_factory=dict)
    trajectory_latents: Any = None
    trajectory_timesteps: Any = None
    rollout_trajectory_data: Any = None
    trajectory_decoded: Any = None
    prompt_index: int = 0
    output_file_path: str | None = None


@dataclass
class MaterializedOutput:
    sample: Any
    frames: list[Any]
    audio: Any = None
    fps: int = 0


def normalize_output_seeds(
    seed: int | list[int],
    *,
    num_outputs_per_prompt: int,
    num_prompts: int = 1,
    prompt_index: int = 0,
) -> list[int]:
    """
    return a list of seed with size equal to `num_outputs_per_prompt`
    """
    if num_outputs_per_prompt <= 0:
        raise ValueError(
            f"num_outputs_per_prompt must be positive, got {num_outputs_per_prompt}"
        )

    if isinstance(seed, list):
        seeds = [int(item) for item in seed]
        total_outputs = num_outputs_per_prompt * num_prompts
        if len(seeds) == num_outputs_per_prompt:
            return seeds
        if len(seeds) == total_outputs:
            start = prompt_index * num_outputs_per_prompt
            return seeds[start : start + num_outputs_per_prompt]
        raise ValueError(
            "seed list length must match num_outputs_per_prompt "
            f"({num_outputs_per_prompt}) or total outputs ({total_outputs}), "
            f"got {len(seeds)}"
        )

    base_seed = int(seed)
    return [base_seed + i for i in range(num_outputs_per_prompt)]


def _with_output_index_suffix(output_file_name: str, output_index: int) -> str:
    base, ext = os.path.splitext(output_file_name)
    return f"{base}_{output_index}{ext}"


def _copy_trace_ctx_for_output(req: Req, request_id: str | None, output_index: int):
    trace_ctx = req.trace_ctx
    if output_index == 0 or not trace_ctx.tracing_enable:
        return trace_ctx

    output_trace_ctx = TraceReqContext(
        rid=request_id,
        module_name=trace_ctx.module_name,
        external_trace_header=trace_ctx.external_trace_header,
    )
    output_trace_ctx.trace_req_start()
    return output_trace_ctx


def _copy_req_for_output(
    req: Req,
    *,
    request_id: str | None,
    output_index: int,
) -> Req:
    """Create a lightweight per-output ``Req`` without deep-copying tensors."""
    output_req = copy(req)
    output_req.sampling_params = copy(req.sampling_params)
    output_req.extra = dict(req.extra)
    output_req.condition_inputs = dict(req.condition_inputs)
    output_req.trace_ctx = _copy_trace_ctx_for_output(req, request_id, output_index)
    return output_req


def expand_request_outputs(
    req: Req,
    *,
    num_prompts: int = 1,
    prompt_index: int = 0,
) -> list[Req]:
    """
    Expand a req to a list with size equal to `num_prompts`
    """
    num_outputs = int(req.num_outputs_per_prompt)
    # each req must has different seed
    seeds = normalize_output_seeds(
        req.seed,
        num_outputs_per_prompt=num_outputs,
        num_prompts=num_prompts,
        prompt_index=prompt_index,
    )

    if num_outputs == 1:
        req.seed = seeds[0]
        req.seeds = None
        req.generator = None
        return [req]

    expanded: list[Req] = []
    for output_index, seed in enumerate(seeds):
        output_request_id = (
            f"{req.request_id}:{output_index}" if req.request_id is not None else None
        )
        output_req = _copy_req_for_output(
            req, request_id=output_request_id, output_index=output_index
        )
        output_req.seed = seed
        output_req.num_outputs_per_prompt = 1
        output_req.seeds = None
        output_req.generator = None
        output_req.extra["parent_request_id"] = req.request_id
        output_req.extra["output_index"] = output_index

        if output_request_id is not None:
            output_req.request_id = output_request_id

        if req.output_file_name:
            output_req.output_file_name = _with_output_index_suffix(
                req.output_file_name, output_index
            )
        output_req.validate()
        expanded.append(output_req)

    return expanded


def _normalize_audio_to_numpy(audio: Any) -> np.ndarray | None:
    """Convert audio (torch / numpy) into a float32 numpy array in [-1, 1], best-effort."""
    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().float().clamp(-1.0, 1.0).cpu().numpy()
    elif isinstance(audio, np.ndarray):
        audio_np = audio.astype(np.float32, copy=False)
        audio_np = np.clip(audio_np, -1.0, 1.0)
    else:
        return None

    # 1. Squeeze leading singleton dimensions (Batch, etc.)
    while audio_np.ndim > 1 and audio_np.shape[0] == 1:
        audio_np = audio_np.squeeze(0)

    # 2. Handle (C, L) -> (L, C)
    if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.transpose(1, 0)

    # 3. Final safety check: if still 2D and channels (dim 1) is huge, something is wrong
    if audio_np.ndim == 2 and audio_np.shape[1] > 256 and audio_np.shape[0] == 1:
        audio_np = audio_np.flatten()

    return audio_np


def _pick_audio_sample_rate(
    *,
    audio_np: np.ndarray,
    audio_sample_rate: Optional[int],
    fps: int,
    num_frames: int,
) -> int:
    """Pick a plausible sample rate, falling back to inferring from video duration."""
    selected_sr = int(audio_sample_rate) if audio_sample_rate is not None else None
    if selected_sr is None or not (
        MIN_AUDIO_SAMPLE_RATE <= selected_sr <= MAX_AUDIO_SAMPLE_RATE
    ):
        selected_sr = DEFAULT_AUDIO_SAMPLE_RATE
        try:
            duration_s = float(num_frames) / float(fps) if fps else 0.0
            if duration_s > 0:
                audio_len = (
                    int(audio_np.shape[0])
                    if audio_np.ndim == 2
                    else int(audio_np.shape[-1])
                )
                inferred_sr = int(round(float(audio_len) / duration_s))
                if MIN_AUDIO_SAMPLE_RATE <= inferred_sr <= MAX_AUDIO_SAMPLE_RATE:
                    selected_sr = inferred_sr
        except Exception:
            pass
    return selected_sr


def _ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def _normalize_video_output_path(
    save_file_path: Optional[str], data_type: DataType
) -> Optional[str]:
    if data_type != DataType.VIDEO or not save_file_path:
        return save_file_path

    _, ext = os.path.splitext(save_file_path)
    if ext.lower() in VIDEO_OUTPUT_EXTENSIONS:
        return save_file_path

    base = save_file_path if not ext else save_file_path[: -len(ext)]
    corrected_path = f"{base}.mp4"
    logger.warning(
        "Video output path %s has non-video extension %s; saving as %s",
        save_file_path,
        ext or "<none>",
        corrected_path,
    )
    return corrected_path


def _write_video_frames(
    *,
    save_file_path: str,
    frames: list[Any],
    fps: int,
    output_compression: Optional[int],
) -> None:
    mimsave_kwargs: dict[str, Any] = {
        "fps": fps,
        "format": DataType.VIDEO.get_default_extension(),
        "codec": "libx264",
    }
    if output_compression is not None:
        mimsave_kwargs["quality"] = output_compression / 10

    try:
        imageio.mimsave(save_file_path, frames, **mimsave_kwargs)
    except TypeError:
        if "quality" not in mimsave_kwargs:
            raise
        logger.warning(
            "Video writer rejected the quality option; retrying without it",
            exc_info=True,
        )
        mimsave_kwargs.pop("quality", None)
        imageio.mimsave(save_file_path, frames, **mimsave_kwargs)


def _resolve_ffmpeg_exe() -> str:
    ffmpeg_exe = "ffmpeg"
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        ffmpeg_exe = ffmpeg_on_path
    try:
        if _imageio_ffmpeg is not None:
            ffmpeg_exe = _imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    ffmpeg_ok = False
    if ffmpeg_exe:
        if os.path.isabs(ffmpeg_exe):
            ffmpeg_ok = os.path.exists(ffmpeg_exe)
        else:
            ffmpeg_ok = shutil.which(ffmpeg_exe) is not None
    if not ffmpeg_ok:
        raise RuntimeError("ffmpeg not found")
    return ffmpeg_exe


def _mux_audio_np_into_mp4(
    *,
    save_file_path: str,
    audio_np: np.ndarray,
    sample_rate: int,
    ffmpeg_exe: str,
) -> None:
    merged_path = save_file_path.rsplit(".", 1)[0] + ".tmp_mux.mp4"
    tmp_wav_path = None
    try:
        if scipy_wavfile is None:
            raise RuntimeError(
                "scipy is required to mux audio into mp4 (pip install scipy)"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav_path = f.name
        scipy_wavfile.write(tmp_wav_path, sample_rate, audio_np)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                save_file_path,
                "-i",
                tmp_wav_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                merged_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.replace(merged_path, save_file_path)
    finally:
        if tmp_wav_path:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass
        if os.path.exists(merged_path):
            try:
                os.remove(merged_path)
            except OSError:
                pass


def _maybe_mux_audio_into_mp4(
    *,
    save_file_path: str,
    audio: Any,
    frames: list,
    fps: int,
    audio_sample_rate: Optional[int],
) -> None:
    """Best-effort mux audio into an already-written mp4 at save_file_path.

    Any failure should keep the silent video and only log a warning.
    """
    audio_np = _normalize_audio_to_numpy(audio)
    if audio_np is None:
        return
    selected_sr = _pick_audio_sample_rate(
        audio_np=audio_np,
        audio_sample_rate=audio_sample_rate,
        fps=fps,
        num_frames=len(frames),
    )

    try:
        ffmpeg_exe = _resolve_ffmpeg_exe()
        _mux_audio_np_into_mp4(
            save_file_path=save_file_path,
            audio_np=audio_np,
            sample_rate=selected_sr,
            ffmpeg_exe=ffmpeg_exe,
        )
    except Exception as e:
        logger.warning(
            "Failed to mux audio into mp4 (saved silent video): %s",
            str(e),
        )


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
    external_trace_header: dict[str, str] | None = None,
) -> Req:
    """
    Create a Req object with sampling_params as a parameter.
    """
    req = Req(
        sampling_params=sampling_params,
        VSA_sparsity=server_args.attention_backend_config.VSA_sparsity,
    )
    sampling_params.apply_request_extra(req)
    if getattr(sampling_params, "max_sequence_length", None) is not None:
        req.max_sequence_length = sampling_params.max_sequence_length

    diffusers_kwargs = getattr(sampling_params, "diffusers_kwargs", None)
    if diffusers_kwargs and "max_sequence_length" in diffusers_kwargs:
        req.max_sequence_length = diffusers_kwargs["max_sequence_length"]

    if not isinstance(req.prompt, str):
        raise TypeError(f"`prompt` must be a string, but got {type(req.prompt)}")

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req.height}, width={req.width}"
        )

    if server_args.enable_trace:
        trace_ctx = TraceReqContext(
            rid=sampling_params.request_id,
            module_name="diffusion",
            external_trace_header=external_trace_header,
        )
        trace_ctx.trace_req_start()
        req.trace_ctx = trace_ctx

    return req


def attach_audio_to_video_sample(
    sample: Any,
    audio: Any,
    output_idx: int,
) -> Any:
    """Attach per-sample audio for video outputs when available."""
    audio = select_output_audio(audio, output_idx)
    if audio is None:
        return sample
    if not (isinstance(sample, (tuple, list)) and len(sample) == 2):
        return (sample, audio)
    return sample


def select_output_audio(audio: Any, output_idx: int) -> Any:
    if isinstance(audio, torch.Tensor) and audio.ndim >= 2:
        return audio[output_idx] if audio.shape[0] > output_idx else None
    if isinstance(audio, np.ndarray) and audio.ndim >= 2:
        return audio[output_idx] if audio.shape[0] > output_idx else None
    return audio


def _split_sample_audio(sample: Any) -> tuple[Any, Any]:
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        return sample[0], sample[1]
    return sample, None


def _sample_to_uint8_frames(sample: Any) -> list[Any]:
    """return numpy frames in THCW format"""
    if isinstance(sample, torch.Tensor):
        # sample is raw tensor
        if sample.dim() == 3:
            sample = sample.unsqueeze(1)
        sample = (sample * 255).clamp(0, 255).to(torch.uint8)
        videos = sample.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        return list(videos)

    if not isinstance(sample, np.ndarray):
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    # sample is numpy frames
    arr = sample
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[None, ...]
        else:
            arr = arr[..., None]
    if arr.ndim != 4:
        raise ValueError(f"Unexpected numpy sample shape: {tuple(arr.shape)}")

    if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        t = torch.from_numpy(arr)
        if t.dim() == 3:
            t = t.unsqueeze(1)
        t = (t * 255).clamp(0, 255).to(torch.uint8)
        videos = t.permute(1, 2, 3, 0).contiguous().cpu().numpy()
        return list(videos)

    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return list(arr)


def materialize_output_sample(
    sample: Any,
    data_type: DataType,
    fps: int,
    *,
    enable_frame_interpolation: bool = False,
    frame_interpolation_exp: int = 1,
    frame_interpolation_scale: float = 1.0,
    frame_interpolation_model_path: Optional[str] = None,
    enable_upscaling: bool = False,
    upscaling_model_path: Optional[str] = None,
    upscaling_scale: int = 4,
) -> MaterializedOutput:
    """materialize samples, apply postprocessing if applicable"""
    sample_without_audio, audio = _split_sample_audio(sample)
    frames = _sample_to_uint8_frames(sample_without_audio)

    # frames are uint8 numpy arrays in THWC format at this point
    if enable_frame_interpolation and data_type == DataType.VIDEO and len(frames) > 1:
        from sglang.multimodal_gen.runtime.postprocess import (
            interpolate_video_frames,
        )

        frames, multiplier = interpolate_video_frames(
            frames,
            exp=frame_interpolation_exp,
            scale=frame_interpolation_scale,
            model_path=frame_interpolation_model_path,
        )
        fps = fps * multiplier

    if enable_upscaling and frames:
        from sglang.multimodal_gen.runtime.postprocess import upscale_frames

        frames = upscale_frames(
            frames,
            model_path=upscaling_model_path,
            scale=upscaling_scale,
        )

    return MaterializedOutput(sample=sample, frames=frames, audio=audio, fps=fps)


def save_materialized_output(
    materialized: MaterializedOutput,
    data_type: DataType,
    save_file_path: Optional[str],
    *,
    save_output: bool = True,
    audio_sample_rate: Optional[int] = None,
    output_compression: Optional[int] = None,
) -> Optional[str]:
    if not save_output:
        return save_file_path
    if not save_file_path:
        logger.info("No output path provided, output not saved")
        return save_file_path

    save_file_path = _normalize_video_output_path(save_file_path, data_type)
    _ensure_parent_dir(save_file_path)
    if data_type == DataType.VIDEO:
        _write_video_frames(
            save_file_path=save_file_path,
            frames=materialized.frames,
            fps=materialized.fps,
            output_compression=output_compression,
        )

        _maybe_mux_audio_into_mp4(
            save_file_path=save_file_path,
            audio=materialized.audio,
            frames=materialized.frames,
            fps=materialized.fps,
            audio_sample_rate=audio_sample_rate,
        )
    else:
        quality = output_compression if output_compression is not None else 75
        if len(materialized.frames) > 1:
            for i, image in enumerate(materialized.frames):
                parts = save_file_path.rsplit(".", 1)
                if len(parts) == 2:
                    indexed_path = f"{parts[0]}_{i}.{parts[1]}"
                else:
                    indexed_path = f"{save_file_path}_{i}"
                _save_image_frame(indexed_path, image, quality, output_compression)
        else:
            _save_image_frame(
                save_file_path, materialized.frames[0], quality, output_compression
            )
    logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
    return save_file_path


def _save_image_frame(
    path: str, frame: np.ndarray, quality: int | None, output_compression: int | None
) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        compress_level = 1
        if output_compression is not None and output_compression != 75:
            compress_level = max(0, min(9, round(output_compression / 100 * 9)))
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        Image.fromarray(frame).save(path, format="PNG", compress_level=compress_level)
    else:
        imageio.imwrite(path, frame, quality=quality)


def save_outputs(
    outputs: Sequence[Any],
    data_type: DataType,
    fps: int,
    save_output: bool,
    build_output_path: Callable[[int], str],
    *,
    audio: Any = None,
    audio_sample_rate: Optional[int] = None,
    samples_out: Optional[list[Any]] = None,
    audios_out: Optional[list[Any]] = None,
    frames_out: Optional[list[Any]] = None,
    output_compression: Optional[int] = None,
    enable_frame_interpolation: bool = False,
    frame_interpolation_exp: int = 1,
    frame_interpolation_scale: float = 1.0,
    frame_interpolation_model_path: Optional[str] = None,
    enable_upscaling: bool = False,
    upscaling_model_path: Optional[str] = None,
    upscaling_scale: int = 4,
) -> list[str]:
    output_paths: list[str] = []
    for idx, sample in enumerate(outputs):
        save_file_path = build_output_path(idx)
        save_file_path = _normalize_video_output_path(save_file_path, data_type)
        if data_type == DataType.VIDEO:
            sample = attach_audio_to_video_sample(sample, audio, idx)

        frames = post_process_sample(
            sample,
            data_type,
            fps,
            save_output,
            save_file_path,
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
            enable_frame_interpolation=enable_frame_interpolation,
            frame_interpolation_exp=frame_interpolation_exp,
            frame_interpolation_scale=frame_interpolation_scale,
            frame_interpolation_model_path=frame_interpolation_model_path,
            enable_upscaling=enable_upscaling,
            upscaling_model_path=upscaling_model_path,
            upscaling_scale=upscaling_scale,
        )

        if samples_out is not None:
            samples_out.append(sample)
        if audios_out is not None:
            if data_type == DataType.VIDEO:
                audios_out.append(select_output_audio(audio, idx))
            else:
                audios_out.append(audio)
        if frames_out is not None:
            frames_out.append(frames)
        output_paths.append(save_file_path)
    return output_paths


def post_process_sample(
    sample: Any,
    data_type: DataType,
    fps: int,
    save_output: bool = True,
    save_file_path: Optional[str] = None,
    audio_sample_rate: Optional[int] = None,
    output_compression: Optional[int] = None,
    enable_frame_interpolation: bool = False,
    frame_interpolation_exp: int = 1,
    frame_interpolation_scale: float = 1.0,
    frame_interpolation_model_path: Optional[str] = None,
    enable_upscaling: bool = False,
    upscaling_model_path: Optional[str] = None,
    upscaling_scale: int = 4,
) -> list[Any]:
    """materialize frames and save outputs (optional)"""
    materialized = materialize_output_sample(
        sample,
        data_type,
        fps,
        enable_frame_interpolation=enable_frame_interpolation,
        frame_interpolation_exp=frame_interpolation_exp,
        frame_interpolation_scale=frame_interpolation_scale,
        frame_interpolation_model_path=frame_interpolation_model_path,
        enable_upscaling=enable_upscaling,
        upscaling_model_path=upscaling_model_path,
        upscaling_scale=upscaling_scale,
    )
    actual_save_file_path = save_materialized_output(
        materialized,
        data_type,
        save_file_path,
        save_output=save_output,
        audio_sample_rate=audio_sample_rate,
        output_compression=output_compression,
    )
    if actual_save_file_path != save_file_path:
        logger.debug("Saved output path adjusted to %s", actual_save_file_path)
    return materialized.frames
