from sglang.multimodal_gen.runtime.models.upsamplers.ltx_2_upsampler import (
    upsample_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LTX2UpsamplingStage(PipelineStage):
    def __init__(self, upsampler, video_encoder_stats):
        super().__init__()
        self.upsampler = upsampler
        self.video_encoder_stats = video_encoder_stats

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        if hasattr(batch, "latents") and batch.latents is not None:
            batch.latents = upsample_video(
                batch.latents, self.video_encoder_stats, self.upsampler
            )
        return batch
