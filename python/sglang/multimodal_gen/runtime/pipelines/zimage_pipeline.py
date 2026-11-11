# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.zimage import ZImagePipelineConfig
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline, Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.denoising import (
    ProgressiveDenoisingStageRouter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.zimage import (
    ZImageProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.zimage_denoising import (
    ZImageDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = ((int(height) // vae_scale_factor) // 2) * (
        (int(width) // vae_scale_factor) // 2
    )
    return "mu", calculate_linear_shift(image_seq_len)


class ZImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "ZImagePipeline"

    # Used when the checkpoint is a single safetensors file with no
    # model_index.json to derive these from.
    pipeline_config_cls = ZImagePipelineConfig
    sampling_params_cls = ZImageSamplingParams

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def add_progressive_denoising_stage(
        self,
        progressive_stage_cls,
        transformer_key="transformer",
        transformer_2_key="transformer_2",
        scheduler_key="scheduler",
        vae_key="vae",
        stage_name="denoising_stage",
    ):
        def create_stage():
            kwargs = {
                "transformer": self.get_module(transformer_key),
                "scheduler": self.get_module(scheduler_key),
                "pipeline": self,
            }
            if transformer_2_key:
                transformer_2 = self.get_module(transformer_2_key, None)
                if transformer_2 is not None:
                    kwargs["transformer_2"] = transformer_2
            if vae_key:
                kwargs["vae"] = self.get_module(vae_key, None)
            return ProgressiveDenoisingStageRouter(
                standard_stage=ZImageDenoisingStage(**kwargs),
                progressive_stage_factory=lambda: progressive_stage_cls(**kwargs),
            )

        return self.add_stage_factory(RoleType.DENOISER, create_stage, stage_name)

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(
            prepare_extra_timestep_kwargs=[prepare_mu],
            progressive_denoising_stage_cls=ZImageProgressiveDenoisingStage,
        )


EntryClass = ZImagePipeline
