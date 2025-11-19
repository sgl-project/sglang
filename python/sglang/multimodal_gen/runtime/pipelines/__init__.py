# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Diffusion pipelines for sglang.multimodal_gen.

This package contains diffusion pipelines for generating videos and images.
"""

from typing import cast

from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
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
    model_info = get_model_info(model_path)
    if model_info is None:
        raise ValueError(f"Unsupported model: {model_path}")

    pipeline_cls = model_info.pipeline_cls

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
