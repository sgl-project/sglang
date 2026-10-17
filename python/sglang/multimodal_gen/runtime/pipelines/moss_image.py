# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.moss_image import (
    MOSSImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.moss_image import (
    MOSSImageBeforeDenoisingStage,
    MOSSImageDenoisingStage,
)


class MOSSImagePipeline(ComposedPipelineBase):
    pipeline_name = "MOSSImagePipeline"
    pipeline_config_cls = MOSSImagePipelineConfig
    sampling_params_cls = MOSSImageSamplingParams
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def _load_config(self):
        config = super()._load_config()
        # Route the export's custom classes through the standard SGLang loaders.
        for name, architecture in (
            ("transformer", "MOSSImageTransformer"),
            ("scheduler", "MOSSImageScheduler"),
        ):
            if config.get(name) == ["pipeline_moss_image", architecture]:
                config[name] = ["diffusers", architecture]
        return config

    def create_pipeline_stages(self, server_args):
        self.add_stage(InputValidationStage())
        self.add_stage(
            MOSSImageBeforeDenoisingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                scheduler=self.get_module("scheduler"),
            )
        )
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: MOSSImageDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
            "denoising_stage",
        )
        self.add_standard_decoding_stage()


EntryClass = MOSSImagePipeline
