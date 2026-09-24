# SPDX-License-Identifier: Apache-2.0
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.loader.component_loaders.ming_image import (
    MingImageEncoderLoader,
    MingImageTokenizerLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ming_image import (
    MingImageDecodingStage,
    MingImageEncodingStage,
    MingImageReferenceStage,
)


def prepare_mu(batch, server_args):
    tokens = (batch.height // 16) * (batch.width // 16)
    return "mu", 1.35 if tokens >= 4096 else 0.5 + (tokens - 256) * (1.15 - 0.5) / (
        4096 - 256
    )


class MingImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MingImagePipeline"
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
        "scheduler",
    ]
    component_loaders = {
        "text_encoder": MingImageEncoderLoader,
        "tokenizer": MingImageTokenizerLoader,
    }

    def _load_config(self):
        return {
            "_class_name": self.pipeline_name,
            "text_encoder": ["transformers", "MingImageEncoder"],
            "tokenizer": ["transformers", "PreTrainedTokenizerFast"],
            "transformer": ["diffusers", "DiffusionTransformer"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        }

    def _resolve_component_path(self, server_args, module_name, load_module_name):
        if module_name == "text_encoder":
            load_module_name = ""
        elif module_name == "tokenizer":
            load_module_name = "mllm"
        return super()._resolve_component_path(
            server_args, module_name, load_module_name
        )

    def create_pipeline_stages(self, server_args):
        self.get_module("scheduler").register_to_config(use_dynamic_shifting=True)
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: MingImageEncodingStage(
                [self.get_module("text_encoder")], [self.get_module("tokenizer")]
            ),
        )
        self.add_stage_factory(
            RoleType.ENCODER, lambda: MingImageReferenceStage(self.get_module("vae"))
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_mu])
        self.add_standard_denoising_stage()
        self.add_stage_factory(
            RoleType.DECODER,
            lambda: MingImageDecodingStage(self.get_module("vae"), pipeline=self),
        )


EntryClass = MingImagePipeline
