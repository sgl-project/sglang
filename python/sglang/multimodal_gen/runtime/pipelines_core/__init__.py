# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for sglang.multimodal_gen.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
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
) -> PipelineWithLoRA:
    """
    Only works with valid hf diffusers configs. (model_index.json)
    We want to build a pipeline based on the inference args mode_path:
    1. download the model from the hub if it's not already downloaded
    2. verify the model config and directory
    3. based on the config, determine the pipeline class
    """
    model_path = server_args.model_path

    # Check if pipeline class is explicitly specified
    if server_args.pipeline_class_name:
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
        )

        _discover_and_register_pipelines()
        logger.info(f"Requested pipeline_class_name: {server_args.pipeline_class_name}")
        logger.info(
            f"Available pipelines in registry: {list(_PIPELINE_REGISTRY.keys())}"
        )
        pipeline_cls = _PIPELINE_REGISTRY.get(server_args.pipeline_class_name)
        if pipeline_cls is None:
            raise ValueError(
                f"Pipeline class '{server_args.pipeline_class_name}' not found in registry. "
                f"Available pipelines: {list(_PIPELINE_REGISTRY.keys())}"
            )
        logger.info(
            f"✓ Using explicitly specified pipeline: {server_args.pipeline_class_name} (class: {pipeline_cls.__name__})"
        )
    else:
        logger.info("No pipeline_class_name specified, using model_index.json")
        model_info = get_model_info(model_path, backend=server_args.backend)
        pipeline_cls = model_info.pipeline_cls
        logger.info(f"Using pipeline from model_index.json: {pipeline_cls.__name__}")

    # instantiate the pipelines
    pipeline = pipeline_cls(model_path, server_args)

    logger.info("Pipeline instantiated")

    return cast(PipelineWithLoRA, pipeline)


__all__ = [
    "build_pipeline",
    "ComposedPipelineBase",
    "Req",
    "LoRAPipeline",
]
