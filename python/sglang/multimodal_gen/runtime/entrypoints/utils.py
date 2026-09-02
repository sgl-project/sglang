# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
DiffGenerator module for sglang-diffusion.

This module provides a consolidated interface for generating videos using
diffusion models.
"""

import atexit
import json
import mmap
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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

_MAX_CACHED_CUDA_VIDEO_BUFFER_BYTES = 1024 * 1024 * 1024
_MAX_CUDA_VIDEO_CONVERSION_CHUNK_BYTES = 128 * 1024 * 1024
_MAX_PARALLEL_CUDA_VIDEO_SAVES = 2
_cuda_video_buffer_cache_lock = threading.Lock()
_cached_cuda_video_buffer: "_CudaMemfdVideoBuffer | None" = None


class _CudaMemfdVideoBuffer:
    """CUDA-registered memfd used as direct ffmpeg raw-video input."""

    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.nbytes = int(np.prod(shape, dtype=np.int64))
        self.fd = -1
        self.mapping: mmap.mmap | None = None
        self.array: np.ndarray | None = None
        self.tensor: torch.Tensor | None = None
        self._registered = False

        try:
            self.fd = os.memfd_create(
                "sglang-video-frames",
                flags=getattr(os, "MFD_CLOEXEC", 0),
            )
            os.ftruncate(self.fd, self.nbytes)
            self.mapping = mmap.mmap(
                self.fd,
                self.nbytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self.array = np.ndarray(shape, dtype=np.uint8, buffer=self.mapping)
            error = torch.cuda.cudart().cudaHostRegister(
                self.array.ctypes.data,
                self.nbytes,
                0,
            )
            if error != 0:
                raise RuntimeError(f"cudaHostRegister failed: {error}")
            self._registered = True
            self.tensor = torch.from_numpy(self.array)
            if not self.tensor.is_pinned():
                raise RuntimeError("CUDA-registered memfd is not pinned")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._registered and self.array is not None:
            try:
                torch.cuda.cudart().cudaHostUnregister(self.array.ctypes.data)
            except Exception:
                pass
            self._registered = False
        self.tensor = None
        self.array = None
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


@contextmanager
def _acquire_cuda_video_buffer(shape: tuple[int, ...]):
    global _cached_cuda_video_buffer

    buffer = None
    stale_buffer = None
    with _cuda_video_buffer_cache_lock:
        if _cached_cuda_video_buffer is not None:
            if _cached_cuda_video_buffer.shape == shape:
                buffer = _cached_cuda_video_buffer
            else:
                stale_buffer = _cached_cuda_video_buffer
            _cached_cuda_video_buffer = None

    if stale_buffer is not None:
        stale_buffer.close()

    if buffer is None:
        buffer = _CudaMemfdVideoBuffer(shape)

    try:
        yield buffer
    finally:
        with _cuda_video_buffer_cache_lock:
            if (
                buffer.nbytes <= _MAX_CACHED_CUDA_VIDEO_BUFFER_BYTES
                and _cached_cuda_video_buffer is None
            ):
                _cached_cuda_video_buffer = buffer
                buffer = None
        if buffer is not None:
            buffer.close()


def _close_cached_cuda_video_buffer() -> None:
    global _cached_cuda_video_buffer

    with _cuda_video_buffer_cache_lock:
        buffer = _cached_cuda_video_buffer
        _cached_cuda_video_buffer = None
    if buffer is not None:
        buffer.close()


atexit.register(_close_cached_cuda_video_buffer)


@dataclass
class SetLoraReq:
    lora_nickname: Union[str, List[str]]
    lora_path: Optional[Union[str, List[Optional[str]]]] = None
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    merge_mode: Optional[str] = None
    lora_alpha: Optional[Union[int, List[Optional[int]]]] = None


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
    action: Any = None  # [T, raw_action_dim] predicted action (policy/inverse_dynamics)
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
        req.sampling_params.refresh_request_extra_after_output_expansion(req)
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
        output_req.sampling_params.refresh_request_extra_after_output_expansion(
            output_req
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
    if selected_sr is None or not (8000 <= selected_sr <= 192000):
        selected_sr = 24000
        try:
            duration_s = float(num_frames) / float(fps) if fps else 0.0
            if duration_s > 0:
                audio_len = (
                    int(audio_np.shape[0])
                    if audio_np.ndim == 2
                    else int(audio_np.shape[-1])
                )
                inferred_sr = int(round(float(audio_len) / duration_s))
                if 8000 <= inferred_sr <= 192000:
                    selected_sr = inferred_sr
        except Exception:
            pass
    return selected_sr


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


def _x264_auto_thread_count(height: int) -> int:
    """Match x264's auto frame-thread count for progressive video."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    cpu_limit = max(1, cpu_count * 3 // 2)
    macroblock_rows = max(1, (height + 15) // 16)
    row_limit = max(1, macroblock_rows // 2)
    return min(cpu_limit, row_limit, 128)


def _cuda_video_conversion_chunk_frames(video: torch.Tensor) -> int:
    _, num_frames, height, width = video.shape
    temporary_bytes_per_frame = 3 * height * width * (video.element_size() + 2)
    return min(
        num_frames,
        max(1, _MAX_CUDA_VIDEO_CONVERSION_CHUNK_BYTES // temporary_bytes_per_frame),
    )


def _sendfile_all(output_fd: int, input_fd: int, count: int) -> None:
    offset = 0
    while count:
        sent = os.sendfile(output_fd, input_fd, offset, count)
        if sent <= 0:
            raise RuntimeError("sendfile made no progress")
        offset += sent
        count -= sent


def _try_save_cuda_video_direct(
    *,
    save_file_path: str,
    sample: Any,
    fps: int,
    audio_sample_rate: Optional[int],
    output_compression: Optional[int],
) -> bool:
    """Stream CUDA RGB chunks to ffmpeg through a registered memfd."""
    if not hasattr(os, "memfd_create") or not hasattr(os, "sendfile"):
        return False

    sample_without_audio, audio = _split_sample_audio(sample)
    if not (
        isinstance(sample_without_audio, torch.Tensor)
        and sample_without_audio.device.type == "cuda"
        and sample_without_audio.dim() in (3, 4)
    ):
        return False
    if os.path.splitext(save_file_path)[1].lower() != ".mp4":
        return False

    video = sample_without_audio
    if video.dim() == 3:
        video = video.unsqueeze(1)
    if video.shape[0] != 3:
        return False

    _, num_frames, height, width = video.shape
    chunk_frames = _cuda_video_conversion_chunk_frames(video)

    quality = output_compression / 10 if output_compression is not None else 5
    if not 1 <= quality <= 10:
        return False
    crf = int((1 - quality / 10.0) * 51)

    audio_np = _normalize_audio_to_numpy(audio)
    tmp_wav_path = None
    try:
        if audio_np is not None:
            if scipy_wavfile is None:
                return False
            selected_sr = _pick_audio_sample_rate(
                audio_np=audio_np,
                audio_sample_rate=audio_sample_rate,
                fps=fps,
                num_frames=num_frames,
            )
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_wav_path = f.name
            scipy_wavfile.write(tmp_wav_path, selected_sr, audio_np)

        ffmpeg_exe = _resolve_ffmpeg_exe()
        command = [
            ffmpeg_exe,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{fps:.02f}",
            "-i",
            "pipe:0",
        ]
        if tmp_wav_path is None:
            command += ["-an"]
        else:
            command += ["-i", tmp_wav_path]
        command += [
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(crf),
        ]

        macro_block_size = 16
        if width % macro_block_size or height % macro_block_size:
            output_width = (
                width
                if width % macro_block_size == 0
                else width + macro_block_size - width % macro_block_size
            )
            output_height = (
                height
                if height % macro_block_size == 0
                else height + macro_block_size - height % macro_block_size
            )
            command += ["-vf", f"scale={output_width}:{output_height}"]

        command += ["-threads", str(_x264_auto_thread_count(height))]
        if tmp_wav_path is not None:
            command += [
                "-acodec",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
            ]
        command += ["-v", "warning", save_file_path]

        with tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )
            try:
                if process.stdin is None:
                    raise RuntimeError("ffmpeg stdin pipe was not created")
                with _acquire_cuda_video_buffer(
                    (chunk_frames, height, width, 3)
                ) as buffer:
                    assert buffer.tensor is not None
                    for start in range(0, num_frames, chunk_frames):
                        end = min(start + chunk_frames, num_frames)
                        frames = (
                            (video[:, start:end] * 255).clamp_(0, 255).to(torch.uint8)
                        )
                        frames = frames.permute(1, 2, 3, 0).contiguous()
                        buffer.tensor[: end - start].copy_(frames, non_blocking=True)
                        torch.cuda.current_stream(video.device).synchronize()
                        del frames
                        _sendfile_all(
                            process.stdin.fileno(),
                            buffer.fd,
                            (end - start) * height * width * 3,
                        )
                process.stdin.close()
                process.stdin = None
                returncode = process.wait()
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if returncode:
                stderr_file.seek(0)
                raise subprocess.CalledProcessError(
                    returncode,
                    command,
                    stderr=stderr_file.read(),
                )
        return True
    except Exception:
        logger.warning_once(
            "Direct CUDA video save failed; falling back to imageio. "
            "Enable debug logging for exception details."
        )
        logger.debug("Direct CUDA video save failure", exc_info=True)
        return False
    finally:
        if tmp_wav_path:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


def _try_save_cuda_videos_direct(
    samples: Sequence[Any],
    save_file_paths: Sequence[str],
    *,
    fps: int,
    audio_sample_rate: Optional[int],
    output_compression: Optional[int],
) -> list[bool] | None:
    """Save independent CUDA videos concurrently when memory permits."""
    if len(samples) < 2 or len(samples) != len(save_file_paths):
        return None

    videos = []
    for sample, save_file_path in zip(samples, save_file_paths):
        video, _ = _split_sample_audio(sample)
        if not (
            isinstance(video, torch.Tensor)
            and video.device.type == "cuda"
            and video.dim() in (3, 4)
            and int(video.shape[0]) == 3
            and os.path.splitext(save_file_path)[1].lower() == ".mp4"
        ):
            return None
        if video.dim() == 3:
            video = video.unsqueeze(1)
        if videos and video.device != videos[0].device:
            return None
        videos.append(video)

    # Each direct save creates one multiply result and two uint8 layouts for a
    # temporal chunk. Keep two such chunks below 25% of currently free device
    # memory so postprocessing cannot turn a tight inference into an OOM.
    temporary_bytes = sorted(
        (
            _cuda_video_conversion_chunk_frames(video)
            * 3
            * int(video.shape[-2])
            * int(video.shape[-1])
            * (int(video.element_size()) + 2)
            for video in videos
        ),
        reverse=True,
    )[:_MAX_PARALLEL_CUDA_VIDEO_SAVES]
    try:
        free_bytes, _ = torch.cuda.mem_get_info(videos[0].device)
    except (RuntimeError, TypeError):
        return None
    if sum(temporary_bytes) > int(free_bytes) // 4:
        return None
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    encoder_threads = sorted(
        (_x264_auto_thread_count(int(video.shape[-2])) for video in videos),
        reverse=True,
    )[:_MAX_PARALLEL_CUDA_VIDEO_SAVES]
    if sum(encoder_threads) > available_cpus:
        return None

    def save_one(idx: int) -> bool:
        return _try_save_cuda_video_direct(
            save_file_path=save_file_paths[idx],
            sample=samples[idx],
            fps=fps,
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
        )

    try:
        with ThreadPoolExecutor(max_workers=_MAX_PARALLEL_CUDA_VIDEO_SAVES) as pool:
            return list(pool.map(save_one, range(len(samples))))
    except Exception:
        logger.warning_once(
            "Parallel CUDA video save failed; falling back to serial output. "
            "Enable debug logging for exception details."
        )
        logger.debug("Parallel CUDA video save failure", exc_info=True)
        return None


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


def _try_save_video_with_audio(
    *,
    save_file_path: str,
    frames: list,
    fps: int,
    audio: Any,
    audio_sample_rate: Optional[int],
    output_format: str,
    quality: float,
) -> bool:
    """Encode video and audio in one ffmpeg pass when audio is available."""
    audio_np = _normalize_audio_to_numpy(audio)
    if audio_np is None:
        return False

    selected_sr = _pick_audio_sample_rate(
        audio_np=audio_np,
        audio_sample_rate=audio_sample_rate,
        fps=fps,
        num_frames=len(frames),
    )
    tmp_wav_path = None
    try:
        if scipy_wavfile is None:
            raise RuntimeError(
                "scipy is required to mux audio into mp4 (pip install scipy)"
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_wav_path = f.name
        scipy_wavfile.write(tmp_wav_path, selected_sr, audio_np)
        imageio.mimsave(
            save_file_path,
            frames,
            fps=fps,
            format=output_format,
            codec="libx264",
            quality=quality,
            audio_path=tmp_wav_path,
            audio_codec="aac",
        )
        return True
    except Exception as e:
        logger.warning(
            "Failed to encode video and audio in one pass; "
            "falling back to the compatible two-pass path: %s",
            str(e),
        )
        return False
    finally:
        if tmp_wav_path:
            try:
                os.remove(tmp_wav_path)
            except OSError:
                pass


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
    external_trace_header: dict[str, str] | None = None,
) -> Req:
    """
    Create a Req object with sampling_params as a parameter.
    """
    attention_backend_config = server_args.attention_backend_config or {}
    vsa_sparsity = attention_backend_config.get(
        "VSA_sparsity", attention_backend_config.get("sparsity", 0.0)
    )
    req = Req(
        sampling_params=sampling_params,
        VSA_sparsity=vsa_sparsity,
    )
    sampling_params.apply_request_extra(req)
    if getattr(sampling_params, "max_sequence_length", None) is not None:
        req.max_sequence_length = sampling_params.max_sequence_length

    diffusers_kwargs = getattr(sampling_params, "diffusers_kwargs", None)
    if diffusers_kwargs and "max_sequence_length" in diffusers_kwargs:
        req.max_sequence_length = diffusers_kwargs["max_sequence_length"]

    action_prompt = (
        req.data_type == DataType.ACTION
        and isinstance(req.prompt, list)
        and bool(req.prompt)
        and all(isinstance(item, str) for item in req.prompt)
    )
    if not isinstance(req.prompt, str) and not action_prompt:
        raise TypeError(
            "`prompt` must be a string, or a non-empty list of strings for "
            f"batched action requests, but got {type(req.prompt)}"
        )

    req_width = getattr(req, "width", None)
    req_height = getattr(req, "height", None)
    if (req_width is not None and req_width <= 0) or (
        req_height is not None and req_height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req_height}, width={req_width}"
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
        videos = sample.permute(1, 2, 3, 0).contiguous()
        if videos.device.type == "cuda":
            try:
                host_videos = torch.empty(
                    videos.shape,
                    dtype=videos.dtype,
                    device="cpu",
                    pin_memory=True,
                )
            except RuntimeError:
                videos = videos.cpu().numpy()
            else:
                host_videos.copy_(videos, non_blocking=True)
                torch.cuda.current_stream(videos.device).synchronize()
                videos = host_videos.numpy()
        else:
            videos = videos.cpu().numpy()
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
) -> None:
    if not save_output:
        return
    if not save_file_path:
        logger.info("No output path provided, output not saved")
        return

    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    if data_type == DataType.VIDEO:
        quality = output_compression / 10 if output_compression is not None else 5
        output_format = data_type.get_default_extension()
        saved_with_audio = _try_save_video_with_audio(
            save_file_path=save_file_path,
            frames=materialized.frames,
            fps=materialized.fps,
            audio=materialized.audio,
            audio_sample_rate=audio_sample_rate,
            output_format=output_format,
            quality=quality,
        )
        if not saved_with_audio:
            imageio.mimsave(
                save_file_path,
                materialized.frames,
                fps=materialized.fps,
                format=output_format,
                codec="libx264",
                quality=quality,
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
    samples = (
        [
            attach_audio_to_video_sample(sample, audio, idx)
            for idx, sample in enumerate(outputs)
        ]
        if data_type == DataType.VIDEO
        else outputs
    )
    save_file_paths = [build_output_path(idx) for idx in range(len(outputs))]
    parallel_results = None
    if (
        data_type == DataType.VIDEO
        and len(outputs) > 1
        and save_output
        and frames_out is None
        and not enable_frame_interpolation
        and not enable_upscaling
    ):
        for path in save_file_paths:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        parallel_results = _try_save_cuda_videos_direct(
            samples,
            save_file_paths,
            fps=fps,
            audio_sample_rate=audio_sample_rate,
            output_compression=output_compression,
        )

    for idx, (sample, save_file_path) in enumerate(zip(samples, save_file_paths)):
        if data_type == DataType.ACTION:
            if samples_out is not None:
                samples_out.append(sample)
            if audios_out is not None:
                audios_out.append(None)
            if frames_out is not None:
                frames_out.append([])
            if save_output and save_file_path:
                os.makedirs(os.path.dirname(save_file_path) or ".", exist_ok=True)
                with open(save_file_path, "w", encoding="utf-8") as f:
                    json.dump(sample, f, ensure_ascii=False)
                logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
            output_paths.append(save_file_path)
            continue

        if data_type == DataType.VIDEO:
            if (
                save_output
                and save_file_path
                and frames_out is None
                and not enable_frame_interpolation
                and not enable_upscaling
            ):
                os.makedirs(os.path.dirname(save_file_path) or ".", exist_ok=True)
                direct_saved = parallel_results is not None and parallel_results[idx]
                if not direct_saved:
                    direct_saved = _try_save_cuda_video_direct(
                        save_file_path=save_file_path,
                        sample=sample,
                        fps=fps,
                        audio_sample_rate=audio_sample_rate,
                        output_compression=output_compression,
                    )
                if direct_saved:
                    if samples_out is not None:
                        samples_out.append(sample)
                    if audios_out is not None:
                        audios_out.append(select_output_audio(audio, idx))
                    output_paths.append(save_file_path)
                    logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
                    continue

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
    if data_type == DataType.ACTION:
        return []

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
    save_materialized_output(
        materialized,
        data_type,
        save_file_path,
        save_output=save_output,
        audio_sample_rate=audio_sample_rate,
        output_compression=output_compression,
    )
    return materialized.frames
