# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# Adapted from: https://github.com/Robbyant/lingbot-world

# SPDX-License-Identifier: Apache-2.0
"""
LingBot-World realtime causal DMD pipeline.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    AuxiliaryConditionEncodingStage,
    DMDTimestepPreparationStage,
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world.lingbot_world_causal_denoising import (
    LingBotWorldCausalDMDDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    CausalVaeDecodingStage,
    RealtimeChunkLatentPreparationStage,
    RealtimeImageVAEEncodingStage,
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotWorldCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LingBotWorldCausalDMDPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "image_encoder",
        "image_processor",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    def create_pipeline_stages(self, server_args) -> None:
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )

        image_encoder = self.get_module("image_encoder", None)
        image_processor = self.get_module("image_processor", None)
        self.add_stage_if(
            image_encoder is not None and image_processor is not None,
            ImageEncodingStage(
                image_encoder=image_encoder,
                image_processor=image_processor,
            ),
        )

        self.add_stage(AuxiliaryConditionEncodingStage())
        self.add_stage(
            RealtimeImageVAEEncodingStage(
                vae=self.get_module("vae"),
            )
        )
        self.add_stage(DMDTimestepPreparationStage(self.get_module("scheduler")))
        self.add_stage(
            RealtimeChunkLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            )
        )
        self.add_stage(
            LingBotWorldCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            CausalVaeDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = LingBotWorldCausalDMDPipeline
