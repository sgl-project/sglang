from sglang.multimodal_gen.runtime.models.model_stages.glm_image import (
    GlmImageBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class GlmImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "GlmImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "vision_language_encoder",
        "processor",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            stage_name="GlmImageBeforeDenoisingStage",
            stage=GlmImageBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vision_language_encoder=self.get_module("vision_language_encoder"),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage", stage=DecodingStage(vae=self.get_module("vae"))
        )


EntryClass = [GlmImagePipeline]
