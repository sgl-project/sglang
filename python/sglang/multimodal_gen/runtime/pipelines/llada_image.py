# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import ClassVar

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.runtime.distributed import (
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    PlainStateDictComponentLoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
    ComponentResidencyError,
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    LLaDAImageTextConditioningStage,
    LLaDAImageTextEncoderRunner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.source import (
    LLaDAImageSourceImageConditioningStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
        return batch


def _check_fp8_transformer(server_args: ServerArgs, path: str) -> None:
    """Serve an FP8 transformer only for Turbo, resident on the GPU."""
    with open(os.path.join(path, "config.json")) as config_file:
        if "quantization_config" not in json.load(config_file):
            return
    from sglang.multimodal_gen.configs.sample.llada_image import (
        LLaDAImageTurboSamplingParams,
    )
    from sglang.multimodal_gen.registry import get_model_info

    # The same lookup picks the Base or Turbo sampling defaults.
    model_info = get_model_info(
        server_args.model_path, server_args.backend, server_args.model_id
    )
    if (
        model_info is None
        or model_info.sampling_param_cls is not LLaDAImageTurboSamplingParams
    ):
        raise ComponentCheckpointUnsupportedError(
            "LLaDA-Image serves FP8 weights only for the Turbo checkpoint "
            "inclusionAI/LLaDA-Image-Turbo-FP8. The FP8 Base checkpoint loses about "
            "0.03 of CVTG-2K word accuracy against BF16 Base, so serve "
            "inclusionAI/LLaDA-Image for Base."
        )
    explicit_mode = server_args.explicit_residency_mode("transformer")
    if explicit_mode not in (None, RESIDENT):
        raise ComponentResidencyError(
            "The LLaDA-Image FP8 transformer runs resident on the GPU, but this "
            f"launch selects {explicit_mode} for it. Remove --dit-cpu-offload, "
            "--dit-layerwise-offload, and transformer entries of "
            "--component-residency, --cpu-offload-components, and "
            "--layerwise-offload-components."
        )
    automatic_mode = server_args.residency_mode("transformer")
    if automatic_mode != RESIDENT:
        logger.warning(
            "Keeping the LLaDA-Image FP8 transformer resident on the GPU instead "
            "of the automatic %s placement.",
            automatic_mode,
        )
    server_args.require_component_resident(
        "transformer", feature_name="The LLaDA-Image FP8 transformer"
    )


class LLaDAImagePipeline(ComposedPipelineBase):
    pipeline_name = "LLaDAImagePipeline"
    component_loaders = {
        "queryformer": PlainStateDictComponentLoader,
        "text_projection": PlainStateDictComponentLoader,
        "sigvq": PlainStateDictComponentLoader,
    }

    _required_config_modules: ClassVar[list[str]] = [
        "queryformer",
        "text_projection",
        "sigvq",
        "tokenizer",
        "transformer",
        "scheduler",
        "vae",
    ]

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        path = super()._resolve_component_path(
            server_args, module_name, load_module_name
        )
        # Placement is decided before the transformer is built, so pin it here.
        if module_name == "transformer":
            _check_fp8_transformer(server_args, path)
        return path

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
