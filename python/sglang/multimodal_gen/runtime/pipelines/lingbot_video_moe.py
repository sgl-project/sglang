# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoDenoisingStage,
    LingBotVideoImageConditioningStage,
    LingBotVideoPromptRewriteStage,
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
        config = server_args.pipeline_config
        self.add_stage_if(
            config.rewriter_url is not None,
            LingBotVideoPromptRewriteStage(
                url=config.rewriter_url or "",
                expand_model=config.rewriter_expand_model,
                map_model=config.rewriter_map_model,
                timeout=config.rewriter_timeout,
            ),
        )
        # Must precede text encoding, which reads the condition frame.
        self.add_stage(
            LingBotVideoImageConditioningStage(vae=self.get_module("vae")),
        )
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
            LingBotVideoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self._maybe_add_refiner_stages(server_args)
        self.add_standard_decoding_stage()

    def _maybe_add_refiner_stages(self, server_args: ServerArgs) -> None:
        """Hook for the two-pass subclass; the single-pass pipeline adds nothing."""
        return None


EntryClass = [LingBotVideoPipeline]
