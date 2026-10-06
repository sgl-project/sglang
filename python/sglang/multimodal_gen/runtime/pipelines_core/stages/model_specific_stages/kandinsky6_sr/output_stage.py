# SPDX-License-Identifier: Apache-2.0
"""Stitch / crop / resize / output stage of Kandinsky 6 video SR."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch, Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.run_spec import (
    SR_PLAN_KEY,
    SR_REQUESTED_HW_KEY,
    SR_TILES_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiled import (
    stitch_tiles,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    crop_to_hw,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_io import (
    to_output_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.video_utils import (
    resize_to_target,
    resolve_target_hw,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Kandinsky6SROutputStage(PipelineStage):
    """Hann-blends the decoded tiles, restores the requested output size, applies the
    optional delivery resize and returns ``output = [1, 3, T, H, W]`` (fp16 in [0, 1])
    plus the mono source audio.

    CPU only: every model phase already ran.  ``batch.height`` / ``batch.width`` are set
    to the final size, ``batch.fps`` was set by the input stage (and now also carried on
    the returned ``OutputBatch``, so a worker-resampled fps reaches the client-side save
    path unchanged).
    """

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        tiles = batch.extra.pop(SR_TILES_KEY)
        plan = batch.extra.pop(SR_PLAN_KEY)
        video = stitch_tiles(tiles, plan)
        del tiles

        requested_hw = batch.extra.pop(SR_REQUESTED_HW_KEY, None)
        if requested_hw is not None:
            requested_h, requested_w = requested_hw
            video = crop_to_hw(
                video,
                (requested_h * plan.tiling_scale, requested_w * plan.tiling_scale),
            )

        target_hw = resolve_target_hw(
            batch.sr_target_resolution,
            tuple(video.shape[-2:]),
            batch.sr_target_resize_mode,
        )
        if target_hw is not None:
            video = resize_to_target(video, target_hw)
        batch.height, batch.width = video.shape[-2:]
        return OutputBatch(
            output=to_output_video(video),
            audio=batch.audio,
            audio_sample_rate=batch.audio_sample_rate,
            fps=batch.fps,
            metrics=batch.metrics,
            usage=batch.usage,
        )
