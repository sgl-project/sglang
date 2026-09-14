# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    prepare_qwen21_mu,
)


class QwenImage21Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImage21Pipeline"
    _required_config_modules = [
        "processor",
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: QwenImage21EncodingStage(
                self.get_module("text_encoder"),
                self.get_module("processor"),
                self.get_module("vae"),
                self.get_module("scheduler"),
            ),
            "conditioning_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_qwen21_mu]
        )
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = QwenImage21Pipeline
