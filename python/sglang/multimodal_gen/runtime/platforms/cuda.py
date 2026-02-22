# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/cuda.py
"""
CUDA platform implementation for multimodal_gen.

This module re-exports the unified CudaPlatform from sglang.platforms.cuda.
For backward compatibility, all existing class names and functions are aliased
from the unified module.

New code should import directly from sglang.platforms.cuda.
"""

# Keep Platform import for backward compatibility
from sglang.multimodal_gen.runtime.platforms.interface import Platform

# Re-export everything from unified CUDA platform
from sglang.platforms.cuda import (
    CudaPlatform,
    CudaPlatformBase,
    NonNvmlCudaPlatform,
    NvmlCudaPlatform,
    device_id_to_physical_device_id,
    with_nvml_context,
)

# Re-export interface types for backward compatibility
from sglang.platforms.interface import (
    AttentionBackendEnum,
    DeviceCapability,
    PlatformEnum,
)

__all__ = [
    "CudaPlatform",
    "CudaPlatformBase",
    "NonNvmlCudaPlatform",
    "NvmlCudaPlatform",
    "Platform",
    "PlatformEnum",
    "DeviceCapability",
    "AttentionBackendEnum",
    "device_id_to_physical_device_id",
    "with_nvml_context",
]
