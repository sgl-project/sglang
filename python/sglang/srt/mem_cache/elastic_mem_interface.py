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

"""
ElasticMem interface for SGLang integration with kvcached backend.
Migrated from elasticmem/kvcached/integration/sglang/interfaces.py

This file contains the SGLang-specific logic for integrating with kvcached,
removing the need for the complex autopatch mechanism.
"""

import math
from typing import List, Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var

try:
    from kvcached.kv_cache_manager import KVCacheManager
    from kvcached.tp_ipc_util import start_worker_listener_thread
    from kvcached.utils import CONTIGUOUS_LAYOUT, PAGE_SIZE, get_kvcached_logger
    from kvcached.vmm_ops import (
        create_kv_tensors,
        init_kvcached as _init_kvcached_impl,
        shutdown_kvcached as _shutdown_kvcached_impl,
    )
    KVCACHED_AVAILABLE = True
except ImportError:
    KVCACHED_AVAILABLE = False

if KVCACHED_AVAILABLE:
    logger = get_kvcached_logger()
else:
    import logging
    logger = logging.getLogger(__name__)

# Global state for elasticmem
_elasticmem_initialized: bool = False
_elasticmem_device = None
_async_sched = False
_contiguous_layout = CONTIGUOUS_LAYOUT if KVCACHED_AVAILABLE else True


def is_elasticmem_available() -> bool:
    """Check if elasticmem (kvcached) backend is available."""
    return KVCACHED_AVAILABLE and get_bool_env_var("ENABLE_KVCACHED", False)


def init_elasticmem(
    tp_rank: int = 0,
    tp_size: int = 1,
    device: Optional[str] = None,
    async_sched: bool = False,
) -> None:
    """Initialize elasticmem backend."""
    if not KVCACHED_AVAILABLE:
        raise ImportError(
            "kvcached is not available. Please install kvcached with "
            "`pip install kvcached --no-build-isolation` to use elastic KV cache."
        )
    
    global _elasticmem_initialized, _elasticmem_device, _async_sched
    if _elasticmem_initialized:
        return

    if device is None:
        device = f"cuda:{torch.cuda.current_device()}"

    _init_kvcached_impl(device, PAGE_SIZE, _contiguous_layout)
    _elasticmem_initialized = True
    _elasticmem_device = device
    _async_sched = async_sched

    if tp_size > 1:
        # start the listener thread for tensor parallel kv cache management
        start_worker_listener_thread(torch.cuda.current_device())

    logger.info(f"ElasticMem initialized on device {device}")


def shutdown_elasticmem() -> None:
    """Shutdown elasticmem backend."""
    if not KVCACHED_AVAILABLE:
        return
        
    global _elasticmem_initialized, _elasticmem_device, _async_sched
    if not _elasticmem_initialized:
        return

    _shutdown_kvcached_impl()
    _elasticmem_initialized = False
    _elasticmem_device = None
    _async_sched = False
    
    logger.info("ElasticMem shutdown completed")


def create_elastic_kv_tensors(
    kvcache_shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: str,
    num_layers: int,
    page_size: int = 1,
    attention_type: str = "MHA",  # TODO: support MLA
    kv_layout: str = "NHD",  # NHD: (num_tokens, head_num, head_dim)
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create elastic KV cache tensors."""
    if not KVCACHED_AVAILABLE:
        raise ImportError("kvcached is not available")
        
    if not _elasticmem_initialized:
        raise RuntimeError(
            "ElasticMem is not initialized. Please call init_elasticmem() first."
        )

    if attention_type != "MHA":
        raise ValueError(f"Attention type {attention_type} is not supported.")

    if kv_layout != "NHD":
        raise ValueError(f"KV layout {kv_layout} is not supported.")

    if len(kvcache_shape) <= 2:
        raise ValueError(f"Unsupported kv cache shape: {kvcache_shape}")

    assert torch.cuda.is_available(), "CUDA is not available."
    if page_size != 1:
        logger.warning("ElasticMem is only tested with page_size=1 for SGLang.")

    # SGLang named it "page" to be consistent with PagedAttention. But we call
    # it "block" to distinguish a KV cache block and a physical memory page.
    block_size = page_size
    block_mem_size = math.prod(kvcache_shape[1:]) * dtype.itemsize
    blocks_per_page = PAGE_SIZE // block_mem_size

    gpu_mem_size = torch.cuda.get_device_properties(device).total_memory

    # Calculate virtual memory size based on layout
    # For contiguous layout, C++ will handle num_layers multiplication
    # So we still calculate per-layer size and let C++ multiply
    num_pages = gpu_mem_size // num_layers // 2 // PAGE_SIZE
    virtual_mem_size = num_pages * PAGE_SIZE * 2

    raw_kv_tensors = create_kv_tensors(virtual_mem_size, dtype.itemsize,
                                       device, num_layers)

    # Fix: Derive num_tokens from actual tensor capacity instead of GPU memory estimation
    raw_tensor = raw_kv_tensors[0]
    elem_per_token = num_layers * 2 * math.prod(kvcache_shape[1:])  # 2 for K/V
    actual_num_tokens = raw_tensor.numel() // elem_per_token
    
    # Align to page boundaries to avoid partial pages
    tokens_per_page = block_size * blocks_per_page
    actual_num_tokens = (actual_num_tokens // tokens_per_page) * tokens_per_page
    
    logger.info(f"[ElasticMem] Actual allocated tensor size: {raw_tensor.numel() * raw_tensor.element_size()} bytes")
    logger.info(f"[ElasticMem] Derived actual_num_tokens: {actual_num_tokens} (original estimate: {block_size * blocks_per_page * num_pages})")
    
    actual_kvcache_shape: List[int] = list(kvcache_shape)
    actual_kvcache_shape[0] = actual_num_tokens

    k_tensors, v_tensors = [], []

    if not _contiguous_layout:
        for t in raw_kv_tensors:
            t = t.view(2, *actual_kvcache_shape).view(dtype=dtype)
            k_tensors.append(t.narrow(0, 0, 1).view(actual_kvcache_shape))
            v_tensors.append(t.narrow(0, 1, 1).view(actual_kvcache_shape))
    else:
        contiguous_tensor = raw_kv_tensors[0].view(
            actual_num_tokens, num_layers, 2,
            *actual_kvcache_shape[1:]).view(dtype=dtype)
        for i in range(num_layers):
            k_tensors.append(contiguous_tensor[:, i, 0, :, :])
            v_tensors.append(contiguous_tensor[:, i, 1, :, :])

    return k_tensors, v_tensors


def get_elasticmem_cache_manager(
    num_blocks: int,
    block_size: int,
    cell_size: int,
    num_layers: int,
    reserve_null_block: bool = True
) -> "KVCacheManager":
    """Get elastic memory cache manager."""
    if not KVCACHED_AVAILABLE:
        raise ImportError("kvcached is not available")
        
    if not _elasticmem_initialized:
        raise RuntimeError(
            "ElasticMem is not initialized. Please call init_elasticmem() first."
        )

    return KVCacheManager(
        num_blocks,
        block_size,
        cell_size,
        num_layers,
        async_sched=_async_sched,
        reserve_null_block=reserve_null_block,
    ) 