# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _flow_shift_kwarg(batch, server_args: ServerArgs) -> tuple[str, float | None]:
    shift = (
        batch.flow_shift
        if batch.flow_shift is not None
        else server_args.pipeline_config.flow_shift
    )
    return ("shift", shift)


class LingBotVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LingBotVideoPipeline"
    is_video_pipeline = True

    _required_config_modules = (
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_stage(
            LingBotVideoTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("processor")],
                transformer=self.get_module("transformer"),
            ),
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_flow_shift_kwarg],
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_standard_decoding_stage()


EntryClass = [LingBotVideoPipeline]
