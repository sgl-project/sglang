from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class JoyImageEditPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "JoyImageEditPipeline"

    _required_config_modules = [
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):

        self.add_standard_ti2i_stages(
            vae_image_processor=None,
            prompt_encoding="image_encoding",
            image_processor_key="processor",
            prompt_text_encoder_key="text_encoder",
        )


EntryClass = JoyImageEditPipeline
