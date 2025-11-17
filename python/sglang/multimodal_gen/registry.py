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
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from sglang.multimodal_gen.configs.pipelines import (
    FastHunyuanConfig,
    FluxPipelineConfig,
    HunyuanConfig,
    StepVideoT2VConfig,
    WanI2V480PConfig,
    WanI2V720PConfig,
    WanT2V480PConfig,
    WanT2V720PConfig,
)
from sglang.multimodal_gen.configs.pipelines.base import PipelineConfig
from sglang.multimodal_gen.configs.pipelines.qwen_image import (
    QwenImageEditPipelineConfig,
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipelines.wan import (
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
from sglang.multimodal_gen.runtime.pipelines.composed_pipeline_base import (
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
    This function scans the 'sglang.multimodal_gen.runtime.architectures' package,
    finds modules with an 'EntryClass' attribute, and maps the class's 'pipeline_name'
    to the class itself in a global registry.
    """
    if _PIPELINE_REGISTRY:  # E-run only once
        return

    package_name = "sglang.multimodal_gen.runtime.architectures"
    package = importlib.import_module(package_name)

    for _, pipeline_type_str, ispkg in pkgutil.iter_modules(package.__path__):
        if not ispkg:
            continue
        pipeline_type_package_name = f"{package_name}.{pipeline_type_str}"
        pipeline_type_package = importlib.import_module(pipeline_type_package_name)
        for _, arch, ispkg_arch in pkgutil.iter_modules(pipeline_type_package.__path__):
            if not ispkg_arch:
                continue
            arch_package_name = f"{pipeline_type_package_name}.{arch}"
            arch_package = importlib.import_module(arch_package_name)
            for _, module_name, ispkg_module in pkgutil.walk_packages(
                arch_package.__path__, arch_package.__name__ + "."
            ):
                if not ispkg_module:
                    pipeline_module = importlib.import_module(module_name)
                    if hasattr(pipeline_module, "EntryClass"):
                        entry_cls = pipeline_module.EntryClass
                        if not isinstance(entry_cls, list):
                            entry_cls_list = [entry_cls]
                        else:
                            entry_cls_list = entry_cls

                        for cls in entry_cls_list:
                            if hasattr(cls, "pipeline_name"):
                                if cls.pipeline_name in _PIPELINE_REGISTRY:
                                    logger.warning(
                                        f"Duplicate pipeline name '{cls.pipeline_name}' found. Overwriting."
                                    )
                                _PIPELINE_REGISTRY[cls.pipeline_name] = cls
                            # else:
                            #     logger.warning(
                            #         f"Pipeline class {cls.__name__} does not have a 'pipeline_name' attribute."
                            #     )


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
_MODEL_PATH_TO_NAME: Dict[str, str] = {}

# Detectors to identify model families from paths or class names
_MODEL_NAME_DETECTORS: List[Tuple[str, Callable[[str], bool]]] = []


def register_configs(
    model_name: str,
    sampling_param_cls: Any,
    pipeline_config_cls: Type[PipelineConfig],
    model_paths: Optional[List[str]] = None,
    model_detectors: Optional[List[Callable[[str], bool]]] = None,
):
    """
    Registers configuration classes for a new model family.
    """
    if model_name in _CONFIG_REGISTRY:
        logger.warning(
            f"Config for model '{model_name}' is already registered and will be overwritten."
        )

    _CONFIG_REGISTRY[model_name] = ConfigInfo(
        sampling_param_cls=sampling_param_cls,
        pipeline_config_cls=pipeline_config_cls,
    )
    if model_paths:
        for path in model_paths:
            if path in _MODEL_PATH_TO_NAME:
                logger.warning(
                    f"Model path '{path}' is already mapped to '{_MODEL_PATH_TO_NAME[path]}' and will be overwritten by '{model_name}'."
                )
            _MODEL_PATH_TO_NAME[path] = model_name

    if model_detectors:
        for detector in model_detectors:
            _MODEL_NAME_DETECTORS.append((model_name, detector))


def _get_config_info(model_path: str) -> Optional[ConfigInfo]:
    """
    Gets the ConfigInfo for a given model path using mappings and detectors.
    """
    # 1. Exact match
    if model_path in _MODEL_PATH_TO_NAME:
        model_name = _MODEL_PATH_TO_NAME[model_path]
        return _CONFIG_REGISTRY.get(model_name)

    # 2. Partial match: find the best (longest) match to avoid conflicts
    #    like "Qwen-Image" and "Qwen-Image-Edit".
    cleaned_model_path = re.sub(r"--", "/", model_path.lower())

    best_match_name = None
    best_match_len = -1

    # Check mappings, prioritizing longer keys to resolve ambiguity
    sorted_mappings = sorted(
        _MODEL_PATH_TO_NAME.items(), key=lambda item: len(item[0]), reverse=True
    )

    for registered_id, model_name in sorted_mappings:
        normalized_registered_id = registered_id.lower()
        if normalized_registered_id in cleaned_model_path:
            # Find the best match based on the longest key
            if len(normalized_registered_id) > best_match_len:
                best_match_len = len(normalized_registered_id)
                best_match_name = model_name

    if best_match_name:
        return _CONFIG_REGISTRY.get(best_match_name)

    # 3. Use detectors
    if os.path.exists(model_path):
        config = verify_model_config_and_directory(model_path)
    else:
        config = maybe_download_model_index(model_path)

    pipeline_name = config.get("_class_name", "").lower()

    for model_name, detector in _MODEL_NAME_DETECTORS:
        if detector(model_path.lower()) or detector(pipeline_name):
            return _CONFIG_REGISTRY.get(model_name)

    return None


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
        logger.warning(
            f"No specific configuration registered for '{model_path}'. "
            f"Falling back to default SamplingParams and PipelineConfig."
        )
        # Fallback to defaults if no specific config is found
        from sglang.multimodal_gen.configs.sample.base import SamplingParams

        config_info = ConfigInfo(
            sampling_param_cls=SamplingParams, pipeline_config_cls=PipelineConfig
        )

    # 4. Combine and return the complete model info
    return ModelInfo(
        pipeline_cls=pipeline_cls,
        sampling_param_cls=config_info.sampling_param_cls,
        pipeline_config_cls=config_info.pipeline_config_cls,
    )


# Registration of model configs
def _register_configs():
    # Hunyuan
    register_configs(
        model_name="hunyuan",
        sampling_param_cls=HunyuanSamplingParams,
        pipeline_config_cls=HunyuanConfig,
        model_paths=[
            "hunyuanvideo-community/HunyuanVideo",
        ],
        model_detectors=[lambda id: "hunyuan" in id.lower()],
    )
    register_configs(
        model_name="fasthunyuan",
        sampling_param_cls=FastHunyuanSamplingParam,
        pipeline_config_cls=FastHunyuanConfig,
        model_paths=[
            "FastVideo/FastHunyuan-diffusers",
        ],
    )

    # StepVideo
    register_configs(
        model_name="stepvideo",
        sampling_param_cls=StepVideoT2VSamplingParams,
        pipeline_config_cls=StepVideoT2VConfig,
        model_paths=[
            "FastVideo/stepvideo-t2v-diffusers",
        ],
        model_detectors=[lambda id: "stepvideo" in id.lower()],
    )

    # Wan
    register_configs(
        model_name="wan-t2v-1.3b",
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=WanT2V480PConfig,
        model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda id: "wanpipeline" in id.lower()],
    )
    register_configs(
        model_name="wan-t2v-14b",
        sampling_param_cls=WanT2V_14B_SamplingParams,
        pipeline_config_cls=WanT2V720PConfig,
        model_paths=[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
    )
    register_configs(
        model_name="wan-i2v-14b-480p",
        sampling_param_cls=WanI2V_14B_480P_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda id: "wanimagetovideo" in id.lower()],
    )
    register_configs(
        model_name="wan-i2v-14b-720p",
        sampling_param_cls=WanI2V_14B_720P_SamplingParam,
        pipeline_config_cls=WanI2V720PConfig,
        model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        model_name="wan-fun-1.3b-inp",
        sampling_param_cls=Wan2_1_Fun_1_3B_InP_SamplingParams,
        pipeline_config_cls=WanI2V480PConfig,
        model_paths=[
            "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        ],
    )
    register_configs(
        model_name="wan-ti2v-5b",
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        model_paths=[
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ],
    )

    register_configs(
        model_name="fastwan-ti2v-5b",
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=FastWan2_2_TI2V_5B_Config,
        model_paths=[
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        ],
    )

    register_configs(
        model_name="wan-t2v-a14b",
        sampling_param_cls=Wan2_2_T2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        model_paths=[
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        ],
    )
    register_configs(
        model_name="wan-i2v-a14b",
        sampling_param_cls=Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        model_paths=[
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        ],
    )
    register_configs(
        model_name="fast-wan-t2v-1.3b",
        sampling_param_cls=FastWanT2V480PConfig,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        model_paths=[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        ],
    )

    # FLUX
    register_configs(
        model_name="flux",
        sampling_param_cls=FluxSamplingParams,
        pipeline_config_cls=FluxPipelineConfig,
        model_paths=[
            "black-forest-labs/FLUX.1-dev",
        ],
        model_detectors=[lambda id: "flux" in id.lower()],
    )

    # Qwen-Image
    register_configs(
        model_name="qwen-image",
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImagePipelineConfig,
        model_paths=[
            "Qwen/Qwen-Image",
        ],
        model_detectors=[lambda id: "qwen-image" in id.lower()],
    )
    register_configs(
        model_name="qwen-image-edit",
        sampling_param_cls=QwenImageSamplingParams,
        pipeline_config_cls=QwenImageEditPipelineConfig,
        model_paths=[
            "Qwen/Qwen-Image-Edit",
        ],
    )


_register_configs()
