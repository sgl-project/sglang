# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import random
from collections.abc import Callable
from typing import cast

import numpy as np
import torch
import torchvision
from einops import rearrange
from torchvision import transforms

from sglang.multimodal_gen.configs.configs import VideoLoaderType
from sglang.multimodal_gen.dataset.transform import (
    CenterCropResizeVideo,
    TemporalRandomCrop,
)
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import (
    PreprocessBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs, WorkloadType


class VideoTransformStage(PipelineStage):
    """
    Crop a video in temporal dimension.
    """

    def __init__(
        self,
        train_fps: int,
        num_frames: int,
        max_height: int,
        max_width: int,
        do_temporal_sample: bool,
    ) -> None:
        self.train_fps = train_fps
        self.num_frames = num_frames
        if do_temporal_sample:
            self.temporal_sample_fn: Callable | None = TemporalRandomCrop(num_frames)
        else:
            self.temporal_sample_fn = None

        self.video_transform = transforms.Compose(
            [
                CenterCropResizeVideo((max_height, max_width)),
            ]
        )

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch = cast(PreprocessBatch, batch)
        assert isinstance(batch.fps, list)
        assert isinstance(batch.num_frames, list)

        if batch.data_type != "video":
            return batch

        if len(batch.video_loader) == 0:
            raise ValueError("Video loader is not set")

        video_pixel_batch = []

        for i in range(len(batch.video_loader)):
            frame_interval = batch.fps[i] / self.train_fps
            start_frame_idx = 0
            frame_indices = np.arange(
                start_frame_idx, batch.num_frames[i], frame_interval
            ).astype(int)
            if len(frame_indices) > self.num_frames:
                if self.temporal_sample_fn is not None:
                    begin_index, end_index = self.temporal_sample_fn(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                else:
                    frame_indices = frame_indices[: self.num_frames]

            if (
                server_args.preprocess_config.video_loader_type
                == VideoLoaderType.TORCHCODEC
            ):
                video = batch.video_loader[i].get_frames_at(frame_indices).data
            elif (
                server_args.preprocess_config.video_loader_type
                == VideoLoaderType.TORCHVISION
            ):
                video, _, _ = torchvision.io.read_video(
                    batch.video_loader[i], output_format="TCHW"
                )
                video = video[frame_indices]
            else:
                raise ValueError(
                    f"Invalid video loader type: {server_args.preprocess_config.video_loader_type}"
                )
            video = self.video_transform(video)
            video_pixel_batch.append(video)

        video_pixel_values = torch.stack(video_pixel_batch)
        video_pixel_values = rearrange(video_pixel_values, "b t c h w -> b c t h w")
        video_pixel_values = video_pixel_values.to(torch.uint8)

        if server_args.workload_type == WorkloadType.I2V:
            batch.pil_image = video_pixel_values[:, :, 0, :, :]

        video_pixel_values = video_pixel_values.float() / 255.0
        batch.latents = video_pixel_values
        batch.num_frames = [video_pixel_values.shape[2]] * len(batch.video_loader)
        batch.height = [video_pixel_values.shape[3]] * len(batch.video_loader)
        batch.width = [video_pixel_values.shape[4]] * len(batch.video_loader)
        return cast(Req, batch)


class TextTransformStage(PipelineStage):
    """
    Process text data according to the cfg rate.
    """

    def __init__(self, cfg_uncondition_drop_rate: float, seed: int) -> None:
        self.cfg_rate = cfg_uncondition_drop_rate
        self.rng = random.Random(seed)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch = cast(PreprocessBatch, batch)

        prompts = []
        for prompt in batch.prompt:
            if not isinstance(prompt, list):
                prompt = [prompt]
            prompt = self.rng.choice(prompt)
            prompt = prompt if self.rng.random() > self.cfg_rate else ""
            prompts.append(prompt)

        batch.prompt = prompts
        return cast(Req, batch)
