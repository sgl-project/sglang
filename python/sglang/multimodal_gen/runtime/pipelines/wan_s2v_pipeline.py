# SPDX-License-Identifier: Apache-2.0
"""Wan2.2 Speech-to-Video pipeline."""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _prepare_wan_s2v_shift(batch, server_args: ServerArgs):
    del batch
    return "shift", float(server_args.pipeline_config.flow_shift or 1.0)


class WanSpeechToVideoPipeline(ComposedPipelineBase):
    pipeline_name = "WanSpeechToVideoPipeline"
    is_video_pipeline = True
    _required_config_modules = [
        "transformer",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "vae",
        "audio_encoder",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        scheduler = self.modules.get("scheduler")
        if scheduler is not None and server_args.pipeline_config.flow_shift is not None:
            scheduler.set_shift(server_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        transformer = self.get_module("transformer")
        self.add_stage(
            WanS2VBeforeDenoisingStage(
                transformer=transformer,
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                audio_encoder=self.get_module("audio_encoder"),
            )
        )
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_prepare_wan_s2v_shift]
        )
        self.add_stage(
            DenoisingStage(
                transformer=transformer,
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            )
        )
        self.add_stage(
            DecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = WanSpeechToVideoPipeline
