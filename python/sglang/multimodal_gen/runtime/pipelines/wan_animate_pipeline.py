from os import sched_yield

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ImageEncodingStage,
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.conditioning import (
    ConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.wan_animate_conditioning import (
    WanAnimateConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.wan_video_processing import (
    VideoProcessingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanAnimatePipeline(ComposedPipelineBase):
    pipeline_name = "WanAnimatePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "image_encoder",
        "image_processor",
        "vae",
        "transformer",
    ]

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )

        self.add_stage(
            stage_name="image_encoding_stage",
            stage=ImageEncodingStage(
                image_encoder=self.get_module("image_encoder"),
                image_processor=self.get_module("image_processor"),
            ),
        )

        self.add_stage(
            stage_name="image_latent_preparation_stage",
            stage=ImageVAEEncodingStage(vae=self.get_module("vae")),
        )

        self.add_stage(
            stage_name="video_processing_stage",
            stage=VideoProcessingStage(),
        )

        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())

        for _ in range(2):
            self.add_stage(
                stage_name="wan_animate_conditioning_stage",
                stage=WanAnimateConditioningStage(vae=self.get_module("vae")),
            )

            self.add_stage(
                stage_name="timestep_preparation_stage",
                stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
            )

            self.add_stage(
                stage_name="latent_preparation_stage",
                stage=LatentPreparationStage(
                    scheduler=self.get_module("scheduler"),
                    transformer=self.get_module("transformer"),
                ),
            )

            self.add_stage(
                stage_name="denoising_stage",
                stage=DenoisingStage(
                    transformer=self.get_module("transformer"),
                    transformer_2=self.get_module("transformer_2"),
                    scheduler=self.get_module("scheduler"),
                    vae=self.get_module("vae"),
                ),
            )

            self.add_stage(
                stage_name="deconding_stage",
                stage=DecodingStage(vae=self.get_module("vae")),
            )


EntryClass = WanAnimatePipeline
