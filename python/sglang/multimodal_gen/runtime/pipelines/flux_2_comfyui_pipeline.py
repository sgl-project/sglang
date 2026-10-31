# SPDX-License-Identifier: Apache-2.0
"""FLUX.2 pipelines for ComfyUI integrated mode.

ComfyUI keeps the sampler, text encoder and VAE and sends one DiT forward per
step, so these pipelines only load the transformer. One class per FLUX.2
family because the checkpoint spec and the pipeline config are selected by
pipeline name.
"""

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinBasePipelineConfig,
    Flux2KleinPipelineConfig,
    Flux2PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.flux import (
    Flux2KleinBaseSamplingParams,
    Flux2KleinSamplingParams,
    Flux2SamplingParams,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.flux2 import (
    FLUX2_KLEIN_BASE_PIPELINE,
    FLUX2_KLEIN_PIPELINE,
    FLUX2_PIPELINE,
)
from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline


class Flux2ComfyUIPipeline(Flux2Pipeline):
    pipeline_name = FLUX2_PIPELINE
    pipeline_config_cls = Flux2PipelineConfig
    sampling_params_cls = Flux2SamplingParams


class Flux2KleinComfyUIPipeline(Flux2Pipeline):
    pipeline_name = FLUX2_KLEIN_PIPELINE
    pipeline_config_cls = Flux2KleinPipelineConfig
    sampling_params_cls = Flux2KleinSamplingParams


class Flux2KleinBaseComfyUIPipeline(Flux2Pipeline):
    pipeline_name = FLUX2_KLEIN_BASE_PIPELINE
    pipeline_config_cls = Flux2KleinBasePipelineConfig
    sampling_params_cls = Flux2KleinBaseSamplingParams


EntryClass = [
    Flux2ComfyUIPipeline,
    Flux2KleinComfyUIPipeline,
    Flux2KleinBaseComfyUIPipeline,
]
