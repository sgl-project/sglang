# SPDX-License-Identifier: Apache-2.0
"""Unit tests for KVarN memory pool integration.

Tests the KVarNTokenToKVPool class directly, without launching a full
sglang server.  Requires GPU for tensor allocation.

Run:
    KVARN_POOL_GPU=1 python -m pytest tests/kvarn/test_kvarn_pool_gpu.py -v
"""

import os

import pytest
import torch

if not os.environ.get("KVARN_POOL_GPU", "0") == "1":
    pytest.skip("Set KVARN_POOL_GPU=1 to run KVarN pool tests", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("No CUDA GPU available", allow_module_level=True)


def test_kvarn_pool_basic():
    """Test that KVarNTokenToKVPool can be instantiated and stores fp16."""
    from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
    from sglang.srt.mem_cache.memory_pool import KVarNTokenToKVPool

    cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
    pool = KVarNTokenToKVPool(
        size=1024,
        page_size=128,
        dtype=torch.uint8,  # marker dtype
        head_num=8,
        head_dim=128,
        layer_num=4,
        device="cuda",
        enable_memory_saver=False,
        kvarn_config=cfg,
        v_head_dim=128,
    )
    assert pool.head_num == 8
    assert pool.head_dim == 128
    assert pool.dtype == torch.float16  # overridden from uint8
    assert pool.store_dtype == torch.float16

    # Check buffers are fp16
    for buf in pool.k_buffer:
        assert buf.dtype == torch.float16
        assert buf.shape == (1024 + 128, 8, 128)
    for buf in pool.v_buffer:
        assert buf.dtype == torch.float16
        assert buf.shape == (1024 + 128, 8, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
