# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
    get_sp_parallel_rank,
    get_sp_world_size,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    SP_STOCHASTIC_NOISE_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.llada_image_source import (
    LLaDAImageSourceImageConditioningStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LLaDAImageLatentPreparationStage(LatentPreparationStage):
    def forward(self, batch, server_args):
        if batch.latents is None:
            shape = server_args.pipeline_config.prepare_latent_shape(
                batch, batch.batch_size, batch.num_frames
            )
            model_dtype = next(self.transformer.parameters()).dtype
            batch.latents = randn_tensor(
                shape,
                generator=batch.generator,
                device=get_local_torch_device(),
                dtype=model_dtype,
            ).float()
        batch = super().forward(batch, server_args)
        batch.extra.pop(SP_STOCHASTIC_NOISE_KEY, None)

        sp_size = get_sp_world_size()
        if self.scheduler.config.stochastic_sampling and sp_size > 1:
            full_shape = tuple(int(value) for value in batch.raw_latent_shape)
            if len(full_shape) != 4:
                raise ValueError(
                    "LLaDA-Image SP noise requires four-dimensional latents"
                )
            latent_height = full_shape[2]
            if latent_height % sp_size != 0:
                raise ValueError(
                    "LLaDA-Image SP noise requires height divisible by SP size"
                )
            local_height = latent_height // sp_size
            sp_rank = get_sp_parallel_rank()
            batch.extra[SP_STOCHASTIC_NOISE_KEY] = {
                "full_shape": full_shape,
                "dim": 2,
                "start": sp_rank * local_height,
                "length": local_height,
            }

        return batch


class LLaDAImagePipeline(ComposedPipelineBase):
    pipeline_name = "LLaDAImagePipeline"

    _required_config_modules: ClassVar[list[str]] = [
        "queryformer",
        "text_projection",
        "sigvq",
        "tokenizer",
        "transformer",
        "scheduler",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        text_runner = LLaDAImageTextEncoderRunner(
            model_root=self.model_path,
            queryformer=self.get_module("queryformer"),
            text_projection=self.get_module("text_projection"),
            tokenizer=self.get_module("tokenizer"),
            server_args=server_args,
        )
        self.add_module("text_encoder", text_runner)

        self.add_stage(InputValidationStage(), "input_validation_stage")
        self.add_stage(
            LLaDAImageTextConditioningStage(text_runner),
            "text_conditioning_stage",
        )
        self.add_stage(
            LLaDAImageSourceImageConditioningStage(
                sigvq=self.get_module("sigvq"),
                vae=self.get_module("vae"),
                image_processor=VaeImageProcessor(
                    vae_scale_factor=server_args.pipeline_config.latent_scale_factor
                ),
            ),
            "source_image_conditioning_stage",
        )
        self.add_stage(
            TimestepPreparationStage(self.get_module("scheduler")),
            "timestep_preparation_stage",
        )
        self.add_stage(
            LLaDAImageLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            ),
            "latent_preparation_stage",
        )
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                pipeline=self,
            ),
            "denoising_stage",
        )
        self.add_stage(
            DecodingStage(
                vae=self.get_module("vae"), pipeline=self, component_name="vae"
            ),
            "decoding_stage",
        )


EntryClass = [LLaDAImagePipeline]
