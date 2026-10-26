# SPDX-License-Identifier: Apache-2.0
"""BRIA Fibo 2.0 pipeline.

Thin assembly on purpose: the model-specific behaviour lives in
`configs/pipeline_configs/bria_fibo2.py` (caption handling, the six text bundles,
the FLUX shift schedule) and in the transformer/encoder classes. What is left
here is the stage order.

One checkpoint serves both generation and editing, and sglang picks the pipeline
class from the checkpoint's `model_index.json` (`BriaFibo2Pipeline`), so that
class has to carry both paths. The stages below therefore always run the
image-aware text encoding and the reference-image latent stage; both fall back to
the generation path when the request carries no image.
"""

from sglang.multimodal_gen.configs.pipeline_configs.bria_fibo2 import (
    BriaFibo2PipelineConfig,
    prepare_mu,
)
from sglang.multimodal_gen.configs.sample.bria_fibo2 import (
    BriaFibo2TurboSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class _Fibo2TokenizerFromProcessor:
    """Expose the Qwen3-VL processor's tokenizer in the `tokenizer` slot.

    The released Fibo 2.0 checkpoints declare a `processor` component and no
    separate `tokenizer`, while the standard text-encoding stages read the
    tokenizer from the module slot named `tokenizer`. The processor carries the
    very tokenizer the reference pipeline uses, so alias it before the stages are
    built. Nothing is copied: both names refer to the same object.
    """

    def __post_init__(self) -> None:
        if self.get_module("tokenizer") is None:
            processor = self.get_module("processor")
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise ValueError(
                    "BriaFibo2 needs a tokenizer: the checkpoint declares no "
                    "`tokenizer` component and its processor carries none."
                )
            self.add_module("tokenizer", tokenizer)
        super().__post_init__()


def add_bria_fibo2_stages(
    pipeline: ComposedPipelineBase, server_args: ServerArgs
) -> None:
    """Build the stage chain shared by generation and editing."""
    from diffusers.image_processor import VaeImageProcessor

    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
        InputValidationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bria_fibo2.condition_encoding import (
        BriaFibo2ConditionEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.bria_fibo2.source import (
        BriaFibo2SourceLatentsStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
        TimestepPreparationStage,
    )

    pipeline_config = server_args.pipeline_config
    # Output sizes and editing canvases are aligned to 16 pixels: the 8x FLUX.2
    # VAE times the 2x2 packing the transformer patchifies with.
    vae_image_processor = VaeImageProcessor(
        vae_scale_factor=pipeline_config.get_vae_scale_factor() * 2
    )

    pipeline.add_stage(
        InputValidationStage(vae_image_processor=vae_image_processor),
        "input_validation_stage",
    )
    # Qwen3-VL reads the prompt, with a vision marker and the images to edit when
    # the request has them; without images this is plain text encoding, because
    # one checkpoint serves both paths and the pipeline class is fixed.
    pipeline.add_stage(
        BriaFibo2ConditionEncodingStage(
            image_processor=pipeline.get_module("processor"),
            text_encoder=pipeline.get_module("text_encoder"),
            tokenizer=pipeline.get_module("tokenizer"),
        ),
        "image_encoding_stage",
    )
    # Reference images become one clean latent per image in
    # `batch.source_latents`; a generation request leaves it None.
    pipeline.add_stage(
        BriaFibo2SourceLatentsStage(
            vae=pipeline.get_module("vae"),
            image_processor=vae_image_processor,
        ),
        "source_latents_stage",
    )
    pipeline.add_stage(
        LatentPreparationStage(
            scheduler=pipeline.get_module("scheduler"),
            transformer=pipeline.get_module("transformer"),
        ),
        "latent_preparation_stage",
    )
    # `prepare_mu` supplies the FLUX linear shift computed from the image token
    # count; the transformer owns its RoPE table, so nothing else is threaded in.
    pipeline.add_stage(
        TimestepPreparationStage(
            pipeline.get_module("scheduler"),
            prepare_extra_set_timesteps_kwargs=[prepare_mu],
        ),
        "timestep_preparation_stage",
    )
    pipeline.add_stage(
        DenoisingStage(
            transformer=pipeline.get_module("transformer"),
            scheduler=pipeline.get_module("scheduler"),
            vae=pipeline.get_module("vae"),
            pipeline=pipeline,
        ),
        "denoising_stage",
    )
    pipeline.add_stage(
        DecodingStage(
            vae=pipeline.get_module("vae"), pipeline=pipeline, component_name="vae"
        ),
        "decoding_stage",
    )


_REQUIRED_MODULES = [
    "text_encoder",
    # The released checkpoints ship the processor, which also carries the
    # tokenizer that `__post_init__` aliases into the `tokenizer` slot.
    "processor",
    "vae",
    "transformer",
    "scheduler",
]


class BriaFibo2Pipeline(
    _Fibo2TokenizerFromProcessor, LoRAPipeline, ComposedPipelineBase
):
    """Generation and editing for the released Fibo 2.0 checkpoints."""

    pipeline_name = "BriaFibo2Pipeline"

    # Used when the checkpoint is a single safetensors file with no
    # model_index.json to derive these from.
    pipeline_config_cls = BriaFibo2PipelineConfig
    sampling_params_cls = BriaFibo2TurboSamplingParams

    _required_config_modules = list(_REQUIRED_MODULES)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        add_bria_fibo2_stages(self, server_args)


class BriaFibo2EditPipeline(
    _Fibo2TokenizerFromProcessor, LoRAPipeline, ComposedPipelineBase
):
    """Editing entry point under its own name, for task-based dispatch.

    The stage chain is the same as `BriaFibo2Pipeline`: a single checkpoint
    serves generation and editing, so neither class needs a different chain.
    """

    pipeline_name = "BriaFibo2EditPipeline"

    pipeline_config_cls = BriaFibo2PipelineConfig
    sampling_params_cls = BriaFibo2TurboSamplingParams

    _required_config_modules = list(_REQUIRED_MODULES)

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        add_bria_fibo2_stages(self, server_args)


EntryClass = [BriaFibo2Pipeline, BriaFibo2EditPipeline]
