# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for sglang.multimodal_gen.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines.pipeline_batch_info import Req
from sglang.multimodal_gen.runtime.pipelines.pipeline_registry import (
    PipelineType,
    get_pipeline_registry,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class PipelineWithLoRA(LoRAPipeline, ComposedPipelineBase):
    """Type for a pipeline that has both ComposedPipelineBase and LoRAPipeline functionality."""

    pass


def build_pipeline(
    server_args: ServerArgs,
    pipeline_type: PipelineType | str = PipelineType.BASIC,
) -> PipelineWithLoRA:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class
    """
    # Get pipeline type
    model_path = server_args.model_path
    model_path = maybe_download_model(model_path)
    # server_args.downloaded_model_path = model_path
    logger.info("Model path: %s", model_path)

    config = verify_model_config_and_directory(model_path)
    pipeline_name = config.get("_class_name")
    if pipeline_name is None:
        raise ValueError(
            "Model config does not contain a _class_name attribute. "
            "Only diffusers format is supported."
        )

    # Get the appropriate pipeline registry based on pipeline_type
    logger.info(
        "Building pipeline of type: %s",
        (
            pipeline_type.value
            if isinstance(pipeline_type, PipelineType)
            else pipeline_type
        ),
    )
    pipeline_registry = get_pipeline_registry(pipeline_type)

    if isinstance(pipeline_type, str):
        pipeline_type = PipelineType.from_string(pipeline_type)

    pipeline_cls = pipeline_registry.resolve_pipeline_cls(
        pipeline_name, pipeline_type, server_args.workload_type
    )

    # instantiate the pipelines
    pipeline = pipeline_cls(model_path, server_args)

    logger.info("Pipelines instantiated")

    return cast(PipelineWithLoRA, pipeline)


__all__ = [
    "build_pipeline",
    "ComposedPipelineBase",
    "Req",
    "LoRAPipeline",
]
