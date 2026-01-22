# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Mooncake-specific utilities for custom memory pool management."""

import logging
from typing import Any, Optional, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Global constants for custom memory pool types
SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES = ["NVLINK", "BAREX"]


def init_mooncake_custom_mem_pool(
    device: str,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Initialize custom memory pool based on environment variable.

    Args:
        device: The device to allocate memory on

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool, custom_mem_pool_type)
    """
    enable_custom_mem_pool, custom_mem_pool_type = (
        check_mooncake_custom_mem_pool_enabled()
    )

    custom_mem_pool = None

    if enable_custom_mem_pool:
        try:
            if custom_mem_pool_type == "NVLINK":
                from mooncake.allocator import NVLinkAllocator, MemoryBackend
                mem_backend = NVLinkAllocator.detect_mem_backend()
                if mem_backend == MemoryBackend.USE_CUMEMCREATE:
                    logger.info("I support fabric mem, using NVLink memory pool")
                    allocator = NVLinkAllocator.get_allocator(device)
                    custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
                    logger.debug(
                        f"Initialized NVLink memory pool on device {device}"
                    )
                    return True, custom_mem_pool, custom_mem_pool_type
                elif  mem_backend == MemoryBackend.USE_CUDAMALLOC:
                    logger.info("Fabric memory not supported, falling back to default cudaMalloc")
                    return False, None, None
                else:
                    logger.info("Memory Backend Unknown or UnSupported")
                    return False, None, None

            elif custom_mem_pool_type == "BAREX":
                from mooncake.allocator import BarexAllocator
                allocator = BarexAllocator.get_allocator(device)
                custom_mem_pool = torch.cuda.MemPool(allocator.allocator())
                logger.debug(
                    f"Initialized BAREX memory pool on device {device}"
                )
                return True, custom_mem_pool, custom_mem_pool_type

            else:
                logger.error(f"Unsupported custom mem pool type: {custom_mem_pool_type}")
                return False, None, None

        except ImportError as e:
            logger.warning(
                f"Failed to import mooncake allocator for {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            return False, None, None

        except Exception as e:
            logger.error(
                f"Failed to initialize custom memory pool {custom_mem_pool_type}: {e}. "
                f"Falling back to default memory pool."
            )
            return False, None, None
    else:
        return False, None, None



def check_mooncake_custom_mem_pool_enabled() -> Tuple[bool, Optional[str]]:
    """
    Check if custom memory pool is enabled without importing allocators.

    Returns:
        Tuple of (enable_custom_mem_pool, custom_mem_pool_type)
    """
    custom_mem_pool_type = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get()

    if custom_mem_pool_type is not None:
        # Handle boolean True as NVLINK
        if custom_mem_pool_type.lower() == "true":
            custom_mem_pool_type = "NVLINK"
        enable_custom_mem_pool = (
            custom_mem_pool_type in SUPPORTED_MOONCAKE_CUSTOM_MEM_POOL_TYPES
        )
    else:
        enable_custom_mem_pool = False
        custom_mem_pool_type = None

    return enable_custom_mem_pool, custom_mem_pool_type
