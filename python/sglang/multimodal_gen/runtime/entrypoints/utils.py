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
import wave

import imageio
import numpy as np
import torch

from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

logger = init_logger(__name__)


def prepare_request(
    server_args: ServerArgs,
    sampling_params: SamplingParams,
) -> Req:
    """
    Create a Req object with sampling_params as a parameter.
    """
    req = Req(sampling_params=sampling_params, VSA_sparsity=server_args.VSA_sparsity)
    diffusers_kwargs = getattr(sampling_params, "diffusers_kwargs", None)
    if diffusers_kwargs:
        req.extra["diffusers_kwargs"] = diffusers_kwargs

    req.adjust_size(server_args)

    if (req.width is not None and req.width <= 0) or (
        req.height is not None and req.height <= 0
    ):
        raise ValueError(
            f"Height and width must be positive, got height={req.height}, width={req.width}"
        )

    return req


def _write_wav(
    audio: torch.Tensor | np.ndarray,
    save_path: str,
    sample_rate: int,
):
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().float().numpy()

    if audio.ndim == 1:
        audio = audio[None, :]
    elif audio.ndim == 3:
        audio = audio[0]

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)

    channels, samples = audio_int16.shape
    interleaved = audio_int16.T.reshape(-1)

    with wave.open(save_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())


def _save_video_with_audio(
    frames: list,
    audio: torch.Tensor | np.ndarray,
    save_path: str,
    fps: int,
    sample_rate: int,
    quality: int = 5,
):
    with tempfile.TemporaryDirectory(prefix="sgl_vwa_") as tmp_dir:
        tmp_video = os.path.join(tmp_dir, "video.mp4")
        tmp_audio = os.path.join(tmp_dir, "audio.wav")

        imageio.mimsave(
            tmp_video,
            frames,
            fps=fps,
            format=DataType.VIDEO.get_default_extension(),
            codec="libx264",
            quality=quality,
        )
        _write_wav(audio, tmp_audio, sample_rate)

        try:
            ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            tmp_video,
            "-i",
            tmp_audio,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            save_path,
        ]
        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "ffmpeg mux failed, saving video without audio. Error: %s",
                exc.stderr.decode(errors="ignore")[:300],
            )
            shutil.copyfile(tmp_video, save_path)


def post_process_sample(
    sample: torch.Tensor,
    data_type: DataType,
    fps: int,
    save_output: bool = True,
    save_file_path: str = None,
    audio: torch.Tensor | np.ndarray | None = None,
    audio_sample_rate: int | None = None,
):
    """
    Process sample output and save video if necessary
    """
    # 1. Vectorized processing on GPU/CPU tensor
    if sample.dim() == 3:
        # for images, dim t is missing
        sample = sample.unsqueeze(1)

    # Convert to uint8 and move to CPU in bulk
    # Shape: [C, T, H, W] -> [T, H, W, C]
    sample = (sample * 255).clamp(0, 255).to(torch.uint8)
    videos = sample.permute(1, 2, 3, 0).cpu().numpy()

    # Convert to list of frames for imageio
    frames = list(videos)

    # 2. Save outputs if requested
    if save_output:
        if save_file_path:
            os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
            if data_type == DataType.VIDEO:
                # TODO: make this configurable
                quality = 5
                if audio is not None and audio_sample_rate is not None:
                    _save_video_with_audio(
                        frames,
                        audio,
                        save_file_path,
                        fps=fps,
                        sample_rate=audio_sample_rate,
                        quality=quality,
                    )
                else:
                    imageio.mimsave(
                        save_file_path,
                        frames,
                        fps=fps,
                        format=data_type.get_default_extension(),
                        codec="libx264",
                        quality=quality,
                    )
            else:
                quality = 75
                if len(frames) > 1:
                    for i, image in enumerate(frames):
                        parts = save_file_path.rsplit(".", 1)
                        if len(parts) == 2:
                            indexed_path = f"{parts[0]}_{i}.{parts[1]}"
                        else:
                            indexed_path = f"{save_file_path}_{i}"
                        imageio.imwrite(indexed_path, image, quality=quality)
                else:
                    imageio.imwrite(save_file_path, frames[0], quality=quality)
            logger.info(f"Output saved to {CYAN}{save_file_path}{RESET}")
        else:
            logger.info(f"No output path provided, output not saved")

    return frames
