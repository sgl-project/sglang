# SPDX-License-Identifier: Apache-2.0
"""
MoVA-specific pipeline stages.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _crop_and_resize(image: Image.Image, height: int, width: int) -> Image.Image:
    image_np = np.array(image)
    image_height, image_width, _ = image_np.shape
    if image_height / image_width < height / width:
        cropped_width = int(image_height / height * width)
        left = (image_width - cropped_width) // 2
        image_np = image_np[:, left : left + cropped_width]
        return Image.fromarray(image_np).resize((width, height))
    cropped_height = int(image_width / width * height)
    top = (image_height - cropped_height) // 2
    image_np = image_np[top : top + cropped_height, :]
    return Image.fromarray(image_np).resize((width, height))


def _video_to_tensor(frames: list[Image.Image]) -> torch.Tensor:
    if not frames:
        raise ValueError("MoVA returned empty video frames")
    tensor_frames = []
    for frame in frames:
        if isinstance(frame, Image.Image):
            arr = np.array(frame)
        elif isinstance(frame, np.ndarray):
            arr = frame
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}")
        tensor_frames.append(torch.from_numpy(arr).permute(2, 0, 1))
    video = torch.stack(tensor_frames, dim=1).float() / 255.0
    return video.unsqueeze(0)


class MovaPreprocessStage(PipelineStage):
    """Prepare reference image for MoVA."""

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if batch.condition_image is None:
            raise ValueError("MoVA requires an input reference image")

        image = batch.condition_image
        if isinstance(image, list):
            image = image[0]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image for MoVA, got {type(image)}")

        if batch.height is None or batch.width is None:
            batch.height = image.height
            batch.width = image.width

        batch.condition_image = _crop_and_resize(image, batch.height, batch.width)
        return batch


class MovaInferenceStage(PipelineStage):
    """Run MoVA inference and return OutputBatch with audio."""

    def __init__(self, mova_pipeline):
        super().__init__()
        self.mova = mova_pipeline

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        pipe = self.mova
        pipe.eval()

        prompt = batch.prompt[0] if isinstance(batch.prompt, list) else batch.prompt
        video, audio = pipe(
            prompt=prompt,
            input_image=batch.condition_image,
            negative_prompt=batch.negative_prompt or "",
            seed=batch.seed,
            height=batch.height,
            width=batch.width,
            num_frames=batch.num_frames,
            video_fps=batch.fps,
            num_inference_steps=batch.num_inference_steps,
            sigma_shift=getattr(batch, "sigma_shift", 5.0),
            cfg_scale=batch.guidance_scale,
            visual_shift=getattr(batch, "visual_shift", 5.0),
            audio_shift=getattr(batch, "audio_shift", 5.0),
            cp_mesh=None,
        )

        video_tensor = _video_to_tensor(video)
        return OutputBatch(
            output=video_tensor,
            audio=audio,
            audio_sample_rate=getattr(pipe, "sample_rate", None),
            timings=batch.timings,
        )
