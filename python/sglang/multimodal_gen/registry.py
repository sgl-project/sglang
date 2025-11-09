# SPDX-License-Identifier: Apache-2.0
"""
Central registry for multimodal models.

This module provides a centralized registry for multimodal models, including pipelines
and sampling parameters. It allows for easy registration and retrieval of model
information based on model paths or other identifiers.
"""

import dataclasses
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from sglang.multimodal_gen.configs.pipelines import (
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig

# Model-specific imports
from sglang.multimodal_gen.configs.pipelines.flux import FluxPipelineConfig
from sglang.multimodal_gen.configs.pipelines.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)
from sglang.multimodal_gen.configs.pipelines.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipelines.stepvideo import StepVideoT2VConfig
from sglang.multimodal_gen.configs.pipelines.wan import (
    FastWan2_1_T2V_480P_Config,
    Wan2_2_I2V_A14B_Config,
    Wan2_2_T2V_A14B_Config,
    Wan2_2_TI2V_5B_Config,
)
from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.configs.sample.hunyuan import (
    FastHunyuanSamplingParam,
    HunyuanSamplingParams,
)
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.configs.sample.stepvideo import StepVideoT2VSamplingParams
from sglang.multimodal_gen.configs.sample.wan import (
    FastWanT2V480PConfig,
    Wan2_1_Fun_1_3B_InP_SamplingParams,
    Wan2_2_I2V_A14B_SamplingParam,
    Wan2_2_T2V_A14B_SamplingParam,
    Wan2_2_TI2V_5B_SamplingParam,
    WanI2V_14B_480P_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
    WanT2V_1_3B_SamplingParams,
    WanT2V_14B_SamplingParams,
)
from sglang.multimodal_gen.runtime.architectures.basic.flux.flux import FluxPipeline
from sglang.multimodal_gen.runtime.architectures.basic.hunyuan.hunyuan_pipeline import (
    HunyuanVideoPipeline,
)
from sglang.multimodal_gen.runtime.architectures.basic.qwen_image.qwen_image import (
    QwenImageEditPipeline,
    QwenImagePipeline,
)
from sglang.multimodal_gen.runtime.architectures.basic.stepvideo.stepvideo_pipeline import (
    StepVideoPipeline,
)
from sglang.multimodal_gen.runtime.architectures.basic.wan.wan_i2v_pipeline import (
    WanImageToVideoPipeline,
)
from sglang.multimodal_gen.runtime.architectures.basic.wan.wan_pipeline import (
    WanPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model_index,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ModelInfo:
    """
    Encapsulates all configuration information required to register a
    diffusers model within this framework.
    """

    pipeline_cls: Type[ComposedPipelineBase]
    sampling_param_cls: Any
    pipeline_config_cls: Type[PipelineConfig]


# The central registry mapping a model name to its information
_MODEL_REGISTRY: Dict[str, ModelInfo] = {}

# Mappings from Hugging Face model paths to our internal model names
_MODEL_PATH_TO_NAME: Dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: List[Tuple[str, Callable[[str], bool]]] = []


def register_model(
    model_name: str,
    pipeline_cls: Type[ComposedPipelineBase],
    sampling_param_cls: Any,
    pipeline_config_cls: Type[PipelineConfig],
    model_path_to_name_mappings: Optional[Dict[str, str]] = None,
    model_name_detectors: Optional[List[Tuple[str, Callable[[str], bool]]]] = None,
):
    """
    Registers a new model family with its pipeline and sampling parameters.

    Args:
        model_name (str): The internal name for the model family.
        pipeline_cls (Type[ComposedPipelineBase]): The pipeline class for the model.
        sampling_param_cls (Any): The sampling parameters class for the model.
        model_path_to_name_mappings (Optional[Dict[str, str]]): A dictionary mapping
            Hugging Face model paths to the internal model name.
        model_name_detectors (Optional[List[Tuple[str, Callable[[str], bool]]]]): A list
            of tuples containing a model name and a detector function to identify the
            model family from a path or class name.
    """
    if model_name in _MODEL_REGISTRY:
        logger.warning(
            f"Model '{model_name}' is already registered and will be overwritten."
        )

    _MODEL_REGISTRY[model_name] = ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )
    if model_path_to_name_mappings:
        for path, name in model_path_to_name_mappings.items():
            if path in _MODEL_PATH_TO_NAME:
                logger.warning(
                    f"Model path '{path}' is already mapped to '{_MODEL_PATH_TO_NAME[path]}' and will be overwritten by '{name}'."
                )
            _MODEL_PATH_TO_NAME[path] = name

    if model_name_detectors:
        _MODEL_NAME_DETECTORS.extend(model_name_detectors)


def get_model_info(model_path: str) -> Optional[ModelInfo]:
    """
    Gets the ModelInfo for a given model path.

    This function resolves a model path to a registered model name and returns the
    corresponding ModelInfo. The resolution process is as follows:
    1. Check for an exact match in the model path to name mappings.
    2. Check for partial matches in the model path to name mappings.
    3. Use the detector functions to identify the model family.

    Args:
        model_path (str): The path to the model (local or on Hugging Face Hub).

    Returns:
        Optional[ModelInfo]: The ModelInfo for the given model path, or None if no
            match is found.
    """
    # 1. Exact match
    if model_path in _MODEL_PATH_TO_NAME:
        model_name = _MODEL_PATH_TO_NAME[model_path]
        return _MODEL_REGISTRY.get(model_name)

    # 2. Partial match
    for registered_id, model_name in _MODEL_PATH_TO_NAME.items():
        if registered_id in model_path:
            return _MODEL_REGISTRY.get(model_name)

    # 3. Use detectors
    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()

    for model_name, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            return _MODEL_REGISTRY.get(model_name)

    logger.warning(f"No model info found for '{model_path}'.")
    return None


# Registration of models
def _register_models():
    # Hunyuan
    register_model(
        model_name="hunyuan",
        pipeline_cls=HunyuanVideoPipeline,
        sampling_param_cls=HunyuanSamplingParams,
        pipeline_config_cls=HunyuanConfig,
        model_path_to_name_mappings={
            "hunyuanvideo-community/HunyuanVideo": "hunyuan",
        },
        model_name_detectors=[("hunyuan", lambda id: "hunyuan" in id.lower())],
    )
    register_model(
        model_name="fasthunyuan",
        pipeline_cls=HunyuanVideoPipeline,
        sampling_param_cls=FastHunyuanSamplingParam,
        pipeline_config_cls=FastHunyuanConfig,
        model_path_to_name_mappings={
            "FastVideo/FastHunyuan-diffusers": "fasthunyuan",
        },
    )

    # StepVideo
    register_model(
        model_name="stepvideo",
        pipeline_cls=StepVideoPipeline,
        sampling_param_cls=StepVideoT2VSamplingParams,
        pipeline_config_cls=StepVideoT2VConfig,
        model_path_to_name_mappings={
            "FastVideo/stepvideo-t2v-diffusers": "stepvideo",
        },
        model_name_detectors=[("stepvideo", lambda id: "stepvideo" in id.lower())],
    )

    # Wan
    register_model(
        model_name="wan-t2v-1.3b",
        pipeline_cls=WanPipeline,
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=WanT2V480PConfig,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": "wan-t2v-1.3b",
        },
        model_name_detectors=[("wanpipeline", lambda id: "wanpipeline" in id.lower())],
    )
    register_model(
        model_name="wan-t2v-14b",
        pipeline_cls=WanPipeline,
        sampling_param_cls=WanT2V_14B_SamplingParams,
        pipeline_config_cls=WanT2V720PConfig,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.1-T2V-14B-Diffusers": "wan-t2v-14b",
        },
    )
    register_model(
        model_name="wan-i2v-14b-480p",
        pipeline_cls=WanImageToVideoPipeline,
        sampling_param_cls=WanI2V_14B_480P_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": "wan-i2v-14b-480p",
        },
        model_name_detectors=[
            ("wanimagetovideo", lambda id: "wanimagetovideo" in id.lower())
        ],
    )
    register_model(
        model_name="wan-i2v-14b-720p",
        pipeline_cls=WanImageToVideoPipeline,
        sampling_param_cls=WanI2V_14B_720P_SamplingParam,
        pipeline_config_cls=WanI2V720PConfig,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": "wan-i2v-14b-720p",
        },
    )
    register_model(
        model_name="wan-fun-1.3b-inp",
        pipeline_cls=WanPipeline,
        sampling_param_cls=Wan2_1_Fun_1_3B_InP_SamplingParams,
        pipeline_config_cls=WanI2V480PConfig,
        model_path_to_name_mappings={
            "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers": "wan-fun-1.3b-inp",
        },
    )
    register_model(
        model_name="wan-ti2v-5b",
        pipeline_cls=WanImageToVideoPipeline,
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers": "wan-ti2v-5b",
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers": "wan-ti2v-5b",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers": "wan-ti2v-5b",
        },
    )
    register_model(
        model_name="wan-t2v-a14b",
        pipeline_cls=WanPipeline,
        sampling_param_cls=Wan2_2_T2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers": "wan-t2v-a14b",
        },
    )
    register_model(
        model_name="wan-i2v-a14b",
        pipeline_cls=WanImageToVideoPipeline,
        sampling_param_cls=Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        model_path_to_name_mappings={
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers": "wan-i2v-a14b",
        },
    )
    register_model(
        model_name="fast-wan-t2v-1.3b",
        pipeline_cls=WanPipeline,
        sampling_param_cls=FastWanT2V480PConfig,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        model_path_to_name_mappings={
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": "fast-wan-t2v-1.3b",
        },
    )

    # FLUX
    register_model(
        model_name="flux",
        pipeline_cls=FluxPipeline,
        sampling_param_cls=FluxSamplingParams,
        pipeline_config_cls=FluxPipelineConfig,
        model_path_to_name_mappings={
            "black-forest-labs/FLUX.1-dev": "flux",
        },
        model_name_detectors=[("flux", lambda id: "flux" in id.lower())],
    )

    # Qwen-Image
    register_model(
        model_name="qwen-image",
        pipeline_cls=QwenImagePipeline,
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImagePipelineConfig,
        model_path_to_name_mappings={
            "Qwen/Qwen-Image": "qwen-image",
        },
    )
    register_model(
        model_name="qwen-image-edit",
        pipeline_cls=QwenImageEditPipeline,
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImageEditPipelineConfig,
        model_path_to_name_mappings={
            "Qwen/Qwen-Image-Edit": "qwen-image-edit",
        },
    )


_register_models()
