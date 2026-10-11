# SPDX-License-Identifier: Apache-2.0
"""Encode a video request's MP4 while its decoding stage is still producing frames."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.distributed import (
    get_replica_group,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import CYAN, RESET, init_logger

logger = init_logger(__name__)

# Converted frames the encoder may hold on the GPU before the decode waits for it.
_STREAMED_VIDEO_QUEUE_BYTES = 256 * 1024 * 1024


def streamed_video_output_path(
    batch: Req, server_args: ServerArgs, batch_size: int
) -> str | None:
    """Where the output rank can write the MP4 during the decode, if anywhere."""
    if not (
        batch.save_output
        and batch.return_file_paths_only
        and not batch.return_raw_frames
        and batch.data_type == DataType.VIDEO
        and not batch.enable_frame_interpolation
        and not batch.enable_upscaling
        and batch_size == 1
        and not (batch.extra or {}).get("dynamic_batch_output_paths")
        and not server_args.enable_torch_compile
        and type(server_args.pipeline_config).post_decoding
        is PipelineConfig.post_decoding
        and current_platform.is_cuda()
    ):
        return None
    if model_parallel_is_initialized() and get_replica_group().rank_in_group != 0:
        return None
    return batch.output_file_path(1, 0)


class EncodeWhileDecoding:
    """Feed each run of finished frames to the MP4 encoder while the VAE decodes the rest."""

    def __init__(
        self,
        save_file_path: str,
        batch: Req,
        finish_frames: Callable[[torch.Tensor], torch.Tensor] | None = None,
        audio=None,
        audio_sample_rate: int | None = None,
    ):
        self.save_file_path = save_file_path
        self._batch = batch
        self._finish_frames = finish_frames
        self._audio = audio
        self._audio_sample_rate = audio_sample_rate
        self._encoder = None
        self._failed = False

    def __call__(self, frames: torch.Tensor) -> None:
        """Encode frames that are [1, 3, t, H, W] in [0, 1] after ``finish_frames``."""
        if self._failed:
            return
        try:
            if self._finish_frames is not None:
                frames = self._finish_frames(frames)
            frames = frames[0]
            if self._encoder is None:
                self._encoder = self._open(frames)
                if self._encoder is None:
                    self._failed = True
                    return
            self._encoder.write(frames)
        except Exception:
            self._fail()

    def _open(self, frames: torch.Tensor):
        from sglang.multimodal_gen.runtime.entrypoints.utils import CudaVideoEncoder

        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        os.makedirs(os.path.dirname(self.save_file_path) or ".", exist_ok=True)
        return CudaVideoEncoder.open(
            self.save_file_path,
            device=frames.device,
            height=height,
            width=width,
            fps=self._batch.fps,
            num_frames=self._batch.num_frames or int(frames.shape[1]),
            audio=self._audio,
            audio_sample_rate=self._audio_sample_rate,
            output_compression=self._batch.output_compression,
            x264_preset=self._batch.x264_preset,
            max_queued_frames=_STREAMED_VIDEO_QUEUE_BYTES // (3 * height * width),
        )

    def finish(self) -> list[str] | None:
        """The saved path, or None when the worker has to save the frames instead."""
        if self._failed or self._encoder is None:
            return None
        try:
            self._encoder.close()
        except Exception:
            self._fail()
            return None
        logger.info(f"Output saved to {CYAN}{self.save_file_path}{RESET}")
        return [self.save_file_path]

    def abort(self) -> None:
        if self._encoder is not None:
            self._encoder.abort()

    def _fail(self) -> None:
        self._failed = True
        self.abort()
        logger.warning_once(
            "Encoding the video while it decodes failed; saving it after the "
            "decode instead. Enable debug logging for exception details."
        )
        logger.debug("Streamed video encode failure", exc_info=True)
