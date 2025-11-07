# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Callable
from typing import Any

from sglang.multimodal_gen.configs.sample.flux import FluxSamplingParams
from sglang.multimodal_gen.configs.sample.hunyuan import (
    FastHunyuanSamplingParam,
    HunyuanSamplingParams,
)
from sglang.multimodal_gen.configs.sample.qwenimage import QwenImageSamplingParams
from sglang.multimodal_gen.configs.sample.stepvideo import StepVideoT2VSamplingParams

# isort: off
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
    SelfForcingWanT2V480PConfig,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    maybe_download_model_index,
    verify_model_config_and_directory,
)

# isort: on
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
# Registry maps specific model weights to their config classes
SAMPLING_PARAM_REGISTRY: dict[str, Any] = {
    "FastVideo/FastHunyuan-diffusers": FastHunyuanSamplingParam,
    "hunyuanvideo-community/HunyuanVideo": HunyuanSamplingParams,
    "FastVideo/stepvideo-t2v-diffusers": StepVideoT2VSamplingParams,
    # Wan2.1
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": WanT2V_1_3B_SamplingParams,
    "Wan-AI/Wan2.1-T2V-14B-Diffusers": WanT2V_14B_SamplingParams,
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers": WanI2V_14B_480P_SamplingParam,
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers": WanI2V_14B_720P_SamplingParam,
    "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers": Wan2_1_Fun_1_3B_InP_SamplingParams,
    # Wan2.2
    "Wan-AI/Wan2.2-TI2V-5B-Diffusers": Wan2_2_TI2V_5B_SamplingParam,
    "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers": Wan2_2_TI2V_5B_SamplingParam,
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers": Wan2_2_T2V_A14B_SamplingParam,
    "Wan-AI/Wan2.2-I2V-A14B-Diffusers": Wan2_2_I2V_A14B_SamplingParam,
    # FastWan2.1
    "FastVideo/FastWan2.1-T2V-1.3B-Diffusers": FastWanT2V480PConfig,
    # FastWan2.2
    "FastVideo/FastWan2.2-TI2V-5B-Diffusers": Wan2_2_TI2V_5B_SamplingParam,
    # Causal Self-Forcing Wan2.1
    "wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers": SelfForcingWanT2V480PConfig,
    # Add other specific weight variants
    "black-forest-labs/FLUX.1-dev": FluxSamplingParams,
    "Qwen/Qwen-Image": QwenImageSamplingParams,
    "Qwen/Qwen-Image-Edit": QwenImageSamplingParams,
}

# For determining pipeline type from model ID
SAMPLING_PARAM_DETECTOR: dict[str, Callable[[str], bool]] = {
    "hunyuan": lambda id: "hunyuan" in id.lower(),
    "wanpipeline": lambda id: "wanpipeline" in id.lower(),
    "wanimagetovideo": lambda id: "wanimagetovideo" in id.lower(),
    "stepvideo": lambda id: "stepvideo" in id.lower(),
    # Add other pipeline architecture detectors
    "flux": lambda id: "flux" in id.lower(),
}

# Fallback configs when exact match isn't found but architecture is detected
SAMPLING_FALLBACK_PARAM: dict[str, Any] = {
    "hunyuan": HunyuanSamplingParams,  # Base Hunyuan config as fallback for any Hunyuan variant
    "wanpipeline": WanT2V_1_3B_SamplingParams,  # Base Wan config as fallback for any Wan variant
    "wanimagetovideo": WanI2V_14B_480P_SamplingParam,
    "stepvideo": StepVideoT2VSamplingParams,
    # Other fallbacks by architecture
    "flux": FluxSamplingParams,
}


def get_sampling_param_cls_for_name(pipeline_name_or_path: str) -> Any | None:
    """Get the appropriate sampling param for specific pretrained weights."""

    if os.path.exists(pipeline_name_or_path):
        config = verify_model_config_and_directory(pipeline_name_or_path)
        logger.warning(
            "sgl-diffusion may not correctly identify the optimal sampling param for this model, as the local directory may have been renamed."
        )
    else:
        config = maybe_download_model_index(pipeline_name_or_path)

    pipeline_name = config["_class_name"]

    # First try exact match for specific weights
    if pipeline_name_or_path in SAMPLING_PARAM_REGISTRY:
        return SAMPLING_PARAM_REGISTRY[pipeline_name_or_path]

    # Try partial matches (for local paths that might include the weight ID)
    for registered_id, config_class in SAMPLING_PARAM_REGISTRY.items():
        if registered_id in pipeline_name_or_path:
            return config_class

    # If no match, try to use the fallback config
    fallback_config = None
    # Try to determine pipeline architecture for fallback
    for pipeline_type, detector in SAMPLING_PARAM_DETECTOR.items():
        if detector(pipeline_name.lower()):
            fallback_config = SAMPLING_FALLBACK_PARAM.get(pipeline_type)
            break

    logger.warning(
        "No match found for pipeline %s, using fallback sampling param %s.",
        pipeline_name_or_path,
        fallback_config,
    )
    return fallback_config
