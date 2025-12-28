from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ImageEncodingStage,
    InputValidationStage,
    SegmentLoopStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.conditioning import (
    ConditioningStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.data_preprocessing import (
    WanDataPreprocessingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
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
            stage_name="data_preprocess_stage",
            stage=WanDataPreprocessingStage(
                preprocess_model_path=(
                    server_args.preprocess_model_path
                    if hasattr(server_args, "preprocess_model_path")
                    else None
                )
            ),
        )
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

        self.add_stage(
            stage_name="segment_loop_stage",
            stage=SegmentLoopStage(
                vae=self.get_module("vae"),
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
                transformer_2=self.get_module("transformer_2"),
            ),
        )


EntryClass = WanAnimatePipeline
