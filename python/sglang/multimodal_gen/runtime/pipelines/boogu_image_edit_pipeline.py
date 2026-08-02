# SPDX-License-Identifier: Apache-2.0
"""Boogu-Image edit (reference-image / TI2I) pipeline.

Additive sibling of :class:`BooguImagePipeline` (PR #33182). It shares the same
checkpoint, DiT, VAE, encoder, and scheduler; the only structural difference is
the encoding stage, which routes an optional reference image to both the
Qwen3-VL encoder and the VAE (see :class:`BooguImageEditEncodingStage`). With no
reference image the pipeline degrades to the text-to-image behaviour.
"""

from sglang.multimodal_gen.configs.pipeline_configs.boogu_image_edit import (
    BooguImageEditPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.boogu_image_edit import (
    BooguImageEditSamplingParams,
)
from sglang.multimodal_gen.runtime.pipelines.boogu_image_pipeline import (
    BooguImagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.boogu_image_edit import (
    BooguImageEditEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class BooguImageEditPipeline(BooguImagePipeline):
    pipeline_name = "BooguImageEditPipeline"

    # Selected via `--pipeline-class-name BooguImageEditPipeline`. The Boogu
    # checkpoint serves both T2I and edit, so the path-based registry resolves
    # to the T2I config by default; these class attributes let
    # `get_pipeline_config_classes` override it with the edit config + the
    # second (image) guidance-scale sampling param (mirrors krea2.py).
    pipeline_config_cls = BooguImageEditPipelineConfig
    sampling_params_cls = BooguImageEditSamplingParams

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_stage(
            BooguImageEditEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
                vae=self.get_module("vae"),
            )
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = BooguImageEditPipeline
