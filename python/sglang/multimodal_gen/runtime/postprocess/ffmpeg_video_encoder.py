# SPDX-License-Identifier: Apache-2.0
"""FFmpeg/libx264 output for raw video frames."""

import os
import subprocess
import tempfile
from typing import Optional

import imageio_ffmpeg
import numpy as np

from sglang.multimodal_gen.runtime.utils.process import local_media_subprocess_env


def resolve_ffmpeg_exe() -> str:
    # keep the pinned encoder binary: x264 upgrades change lossy video pixels
    return imageio_ffmpeg.get_ffmpeg_exe()


# ffmpeg's implicit libx264 default is `medium`. On diffusion output `fast` is
# both quicker and measurably closer to the frames the model produced.
X264_PRESET = "fast"


def x264_auto_thread_count(height: int) -> int:
    """Match x264's auto frame-thread count for progressive video."""
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1
    cpu_limit = max(1, cpu_count * 3 // 2)
    macroblock_rows = max(1, (height + 15) // 16)
    row_limit = max(1, macroblock_rows // 2)
    return min(cpu_limit, row_limit, 128)


def cuda_video_ffmpeg_command(
    save_file_path: str,
    *,
    width: int,
    height: int,
    fps: int,
    crf: int,
    x264_preset: Optional[str],
    wav_path: Optional[str],
) -> list[str]:
    command = [
        resolve_ffmpeg_exe(),
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
    if wav_path is None:
        command += ["-an"]
    else:
        command += ["-i", wav_path]
    command += [
        "-vcodec",
        "libx264",
        "-preset",
        x264_preset or X264_PRESET,
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

    command += ["-threads", str(x264_auto_thread_count(height))]
    if wav_path is not None:
        command += [
            "-acodec",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
        ]
    command += ["-v", "warning", save_file_path]
    return command


def save_video_ffmpeg(
    save_file_path: str,
    frames: list,
    *,
    fps: int,
    quality: float,
    x264_preset: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> None:
    if not frames:
        raise ValueError("video output requires at least one frame")
    if not 1 <= quality <= 10:
        raise ValueError("video quality must be between 1 and 10")
    first_frame = np.asarray(frames[0])
    height, width = first_frame.shape[:2]
    channels = first_frame.shape[2] if first_frame.ndim == 3 else 1
    pixel_format = {1: "gray", 3: "rgb24", 4: "rgba"}[channels]
    command = [
        resolve_ffmpeg_exe(),
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        pixel_format,
        "-r",
        f"{fps:.02f}",
        "-i",
        "pipe:0",
    ]
    if audio_path is None:
        command += ["-an"]
    else:
        command += ["-i", audio_path, "-acodec", "aac"]
    command += [
        "-vcodec",
        "libx264",
        "-preset",
        x264_preset or X264_PRESET,
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(int((1 - quality / 10.0) * 51)),
    ]
    if width % 16 or height % 16:
        command += [
            "-vf",
            f"scale={(width + 15) // 16 * 16}:{(height + 15) // 16 * 16}",
        ]
    command += [save_file_path]
    with tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(
            command,
            env=local_media_subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=stderr,
        ) as process:
            try:
                for frame in frames:
                    frame = np.asarray(frame)
                    if frame.shape != first_frame.shape or frame.dtype != np.uint8:
                        raise ValueError(
                            "video frames must have matching shapes and uint8 dtype"
                        )
                    process.stdin.write(memoryview(np.ascontiguousarray(frame)))
            finally:
                process.stdin.close()
            if process.wait() != 0:
                stderr.seek(0)
                raise RuntimeError(
                    f"ffmpeg video encoding failed: {stderr.read().decode(errors='replace')}"
                )
