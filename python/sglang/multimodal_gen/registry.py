# SPDX-License-Identifier: Apache-2.0
"""
Central registry for multimodal models.

This module provides a centralized registry for multimodal models, including pipelines
and sampling parameters. It allows for easy registration and retrieval of model
information based on model paths or other identifiers.
"""

import dataclasses
import importlib
import os
import pkgutil
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from sglang.multimodal_gen.configs.pipeline_configs import (
    FastHunyuanConfig,
    FluxPipelineConfig,
    HunyuanConfig,
    StepVideoT2VConfig,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
    ZImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.flux import Flux2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    FastWan2_1_T2V_480P_Config,
    FastWan2_2_TI2V_5B_Config,
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
from sglang.multimodal_gen.configs.sample.zimage import ZImageSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model_index,
    verify_model_config_and_directory,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# --- Part 1: Pipeline Discovery ---

_PIPELINE_REGISTRY: Dict[str, Type[ComposedPipelineBase]] = {}


def _discover_and_register_pipelines():
    """
    Automatically discover and register all ComposedPipelineBase subclasses.
    This function scans the 'sglang.multimodal_gen.runtime.pipelines' package,
    finds modules with an 'EntryClass' attribute, and maps the class's 'pipeline_name'
    to the class itself in a global registry.
    """
    if _PIPELINE_REGISTRY:  # run only once
        return

    package_name = "sglang.multimodal_gen.runtime.pipelines"
    package = importlib.import_module(package_name)

    for _, module_name, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        if not ispkg:
            pipeline_module = importlib.import_module(module_name)
            if hasattr(pipeline_module, "EntryClass"):
                entry_cls = pipeline_module.EntryClass
                entry_cls_list = (
                    [entry_cls] if not isinstance(entry_cls, list) else entry_cls
                )

                for cls in entry_cls_list:
                    if hasattr(cls, "pipeline_name"):
                        if cls.pipeline_name in _PIPELINE_REGISTRY:
                            logger.warning(
                                f"Duplicate pipeline name '{cls.pipeline_name}' found. Overwriting."
                            )
                        _PIPELINE_REGISTRY[cls.pipeline_name] = cls
    logger.debug(
        f"Registering pipelines complete, {len(_PIPELINE_REGISTRY)} pipelines registered"
    )


# --- Part 2: Config Registration ---
@dataclasses.dataclass
class ConfigInfo:
    """Encapsulates all configuration information required to register a
    diffusers model within this framework."""

    sampling_param_cls: Any
    pipeline_config_cls: Type[PipelineConfig]


# The central registry mapping a model name to its configuration information
_CONFIG_REGISTRY: Dict[str, ConfigInfo] = {}

# Mappings from Hugging Face model paths to our internal model names
_MODEL_HF_PATH_TO_NAME: Dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: List[Tuple[str, Callable[[str], bool]]] = []


def register_configs(
    sampling_param_cls: Any,
    pipeline_config_cls: Type[PipelineConfig],
    hf_model_paths: Optional[List[str]] = None,
    model_detectors: Optional[List[Callable[[str], bool]]] = None,
):
    """
    Registers configuration classes for a new model family.
    """
    model_id = str(len(_CONFIG_REGISTRY))

    _CONFIG_REGISTRY[model_id] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )
    if hf_model_paths:
        for path in hf_model_paths:
            if path in _MODEL_HF_PATH_TO_NAME:
                logger.warning(
                    f"Model path '{path}' is already mapped to '{_MODEL_HF_PATH_TO_NAME[path]}' and will be overwritten by '{model_id}'."
                )
            _MODEL_HF_PATH_TO_NAME[path] = model_id

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_id, detector))


def get_model_short_name(model_id: str) -> str:
    if "/" in model_id:
        return model_id.split("/")[-1]
    else:
        return model_id


def _get_config_info(model_path: str) -> Optional[ConfigInfo]:
    """
    Gets the ConfigInfo for a given model path using mappings and detectors.
    """
    # 1. Exact match
    if model_path in _MODEL_HF_PATH_TO_NAME:
        model_id = _MODEL_HF_PATH_TO_NAME[model_path]
        logger.debug(f"Resolved model path '{model_path}' from exact path match.")
        return _CONFIG_REGISTRY.get(model_id)

    # 2. Partial match: find the best (longest) match against all registered model hf paths.
    model_name = get_model_short_name(model_path.lower())
    all_model_hf_paths = sorted(_MODEL_HF_PATH_TO_NAME.keys(), key=len, reverse=True)
    for registered_model_hf_id in all_model_hf_paths:
        registered_model_name = get_model_short_name(registered_model_hf_id.lower())

        if registered_model_name == model_name:
            logger.debug(
                f"Resolved model name '{registered_model_hf_id}' from partial path match."
            )
            model_id = _MODEL_HF_PATH_TO_NAME[registered_model_hf_id]
            return _CONFIG_REGISTRY.get(model_id)

    # 3. Use detectors
    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()

    matched_model_names = []
    for model_id, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            logger.debug(
                f"Matched model name '{model_id}' using a registered detector."
            )
            matched_model_names += [model_id]

    if len(matched_model_names) >= 1:
        if len(matched_model_names) > 1:
            logger.warning(
                f"More than one model name is matched, using the first matched"
            )
        model_id = matched_model_names[0]
        return _CONFIG_REGISTRY.get(model_id)
    else:
        raise RuntimeError(f"No model info found for model path: {model_path}")


# --- Part 3: Main Resolver ---


@dataclasses.dataclass
class ModelInfo:
    """
    Encapsulates all configuration information required to register a
    diffusers model within this framework.
    """

    pipeline_cls: Type[ComposedPipelineBase]
    sampling_param_cls: Any
    pipeline_config_cls: Type[PipelineConfig]


@lru_cache(maxsize=1)
def get_model_info(model_path: str) -> Optional[ModelInfo]:
    """
    Resolves all necessary classes (pipeline, sampling, config) for a given model path.

    This function serves as the main entry point for model resolution. It performs two main tasks:
    1. Dynamically resolves the pipeline class by reading 'model_index.json' and matching
       '_class_name' against an auto-discovered registry of pipeline implementations.
    2. Resolves the associated configuration classes (for sampling and pipeline) using a
       manually registered mapping based on the model path.
    """
    # 1. Discover all available pipeline classes and cache them
    _discover_and_register_pipelines()

    # 2. Get pipeline class from model's model_index.json
    try:
        if os.path.exists(model_path):
            config = verify_model_config_and_directory(model_path)
        else:
            config = maybe_download_model_index(model_path)
    except Exception as e:
        logger.error(f"Could not read model config for '{model_path}': {e}")
        return None

    pipeline_class_name = config.get("_class_name")
    if not pipeline_class_name:
        logger.error(f"'_class_name' not found in model_index.json for '{model_path}'")
        return None

    pipeline_cls = _PIPELINE_REGISTRY.get(pipeline_class_name)
    if not pipeline_cls:
        logger.error(
            f"Pipeline class '{pipeline_class_name}' specified in '{model_path}' is not a registered EntryClass in the framework. "
            f"Available pipelines: {list(_PIPELINE_REGISTRY.keys())}"
        )
        return None

    # 3. Get configuration classes (sampling, pipeline config)
    config_info = _get_config_info(model_path)
    if not config_info:
        logger.error(
            f"Could not resolve configuration for model '{model_path}'. "
            "It is not a registered model path or detected by any registered model family detectors. "
            f"Known model paths: {list(_MODEL_HF_PATH_TO_NAME.keys())}"
        )
        return None

    # 4. Combine the complete model info
    model_info = ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=config_info.sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )
    logger.info(f"Found model info: {model_info}")

    return model_info


# Registration of model configs
def _register_configs():
    # Hunyuan
    register_configs(
        sampling_param_cls=HunyuanSamplingParams,
        pipeline_config_cls=HunyuanConfig,
        hf_model_paths=[
            "hunyuanvideo-community/HunyuanVideo",
        ],
        model_detectors=[lambda hf_id: "hunyuan" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=FastHunyuanSamplingParam,
        pipeline_config_cls=FastHunyuanConfig,
        hf_model_paths=[
            "FastVideo/FastHunyuan-diffusers",
        ],
    )

    # StepVideo
    register_configs(
        sampling_param_cls=StepVideoT2VSamplingParams,
        pipeline_config_cls=StepVideoT2VConfig,
        hf_model_paths=[
            "FastVideo/stepvideo-t2v-diffusers",
        ],
        model_detectors=[lambda hf_id: "stepvideo" in hf_id.lower()],
    )

    # Wan
    register_configs(
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=WanT2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda hf_id: "wanpipeline" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=WanT2V_14B_SamplingParams,
        pipeline_config_cls=WanT2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_480P_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda hf_id: "wanimagetovideo" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_720P_SamplingParam,
        pipeline_config_cls=WanI2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_1_Fun_1_3B_InP_SamplingParams,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ],
    )

    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=FastWan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        ],
    )

    register_configs(
        sampling_param_cls=Wan2_2_T2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        hf_model_paths=["Wan-AI/Wan2.2-T2V-A14B-Diffusers"],
    )
    register_configs(
        sampling_param_cls=Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        hf_model_paths=["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
    )
    register_configs(
        sampling_param_cls=FastWanT2V480PConfig,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        ],
    )

    # FLUX
    register_configs(
        sampling_param_cls=FluxSamplingParams,
        pipeline_config_cls=FluxPipelineConfig,
        hf_model_paths=[
            "black-forest-labs/FLUX.1-dev",
        ],
        model_detectors=[lambda hf_id: "flux.1" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=FluxSamplingParams,
        pipeline_config_cls=Flux2PipelineConfig,
        hf_model_paths=[
            "black-forest-labs/FLUX.2-dev",
        ],
        model_detectors=[lambda hf_id: "flux.2" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=ZImageSamplingParams,
        pipeline_config_cls=ZImagePipelineConfig,
        hf_model_paths=[
            "Tongyi-MAI/Z-Image-Turbo",
        ],
        model_detectors=[lambda hf_id: "z-image" in hf_id.lower()],
    )

    # Qwen-Image
    register_configs(
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImagePipelineConfig,
        hf_model_paths=["Qwen/Qwen-Image"],
    )
    register_configs(
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImageEditPipelineConfig,
        hf_model_paths=["Qwen/Qwen-Image-Edit"],
    )


_register_configs()
