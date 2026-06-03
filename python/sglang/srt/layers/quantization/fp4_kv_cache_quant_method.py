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
"""Legacy import path for FP4 KV cache quantization methods.

Runtime code should import from ``kv_cache_quant_method.py``. This module keeps
older imports working while making the ownership clear: recipe selection, buffer
creation, and quant/dequant contracts live in ``kv_cache_quant_method.py``;
low-level tensor operations live in ``kvfp4_tensor.py``.
"""

from sglang.srt.layers.quantization.kv_cache_quant_method import (
    KV_CACHE_QUANT_REGISTRY,
    KVCacheQuantMethod,
    MXFP4Method,
    NVFP4Method,
)

FP4KVCacheQuantMethod = KVCacheQuantMethod


class NVFP4KVMethod(NVFP4Method):
    """Compatibility alias for ``NVFP4Method``."""

    SCALE_BLOCK_SIZE = NVFP4Method.scale_block_size


class MXFP4KVMethod(MXFP4Method):
    """Compatibility alias for ``MXFP4Method``."""

    SCALE_BLOCK_SIZE = MXFP4Method.scale_block_size


class BlockFP4KVMethod(MXFP4Method):
    """Compatibility name for the old block-FP4 method.

    The canonical recipe name is ``mxfp4``. The ``blockfp4`` name is retained
    only for older tests/imports that predate the explicit recipe naming.
    """

    name = "blockfp4"
    SCALE_BLOCK_SIZE = MXFP4Method.scale_block_size


FP4_KV_CACHE_QUANT_REGISTRY = {
    "nvfp4": NVFP4KVMethod,
    "mxfp4": MXFP4KVMethod,
    "blockfp4": BlockFP4KVMethod,
}


def get_fp4_kv_cache_quant_method(name: str, **kwargs) -> KVCacheQuantMethod:
    """Instantiate a legacy FP4 KV cache quantization method by name."""
    if name not in FP4_KV_CACHE_QUANT_REGISTRY:
        raise ValueError(
            f"Unknown fp4_kv_cache_recipe: '{name}'. "
            f"Available: {list(FP4_KV_CACHE_QUANT_REGISTRY)}"
        )
    return FP4_KV_CACHE_QUANT_REGISTRY[name](**kwargs)


__all__ = [
    "BlockFP4KVMethod",
    "FP4KVCacheQuantMethod",
    "FP4_KV_CACHE_QUANT_REGISTRY",
    "KV_CACHE_QUANT_REGISTRY",
    "MXFP4KVMethod",
    "NVFP4KVMethod",
    "get_fp4_kv_cache_quant_method",
]
