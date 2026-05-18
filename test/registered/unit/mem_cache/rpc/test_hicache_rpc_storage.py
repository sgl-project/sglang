# SPDX-License-Identifier: Apache-2.0
"""HiCacheRpcStorage and allocator unit tests."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.rpc.hicache_rpc_storage import (
    HiCacheRpcStorage,
    MemfdTensorAllocator,
)


def test_allocator_bump_and_fd():
    a = MemfdTensorAllocator()
    t = a.allocate((64, 64), dtype=torch.bfloat16, device="cpu")
    assert t.shape == (64, 64)
    assert a.fd >= 0
    assert getattr(t, "memfd_offset") == 0
    assert a.base_ptr is not None


def test_storage_requires_socket():
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unit",
        extra_config=None,
    )
    with pytest.raises(ValueError, match="mempool_socket"):
        HiCacheRpcStorage(cfg, None)


def test_construct_grpc_stub():
    cfg = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=False,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name="unit",
        extra_config={
            "mempool_socket": "unix:///tmp/nonexistent_mempool.sock",
            "grpc_socket": "unix:///tmp/nonexistent_grpc.sock",
        },
    )
    s = HiCacheRpcStorage(cfg, None)
    assert s._stub is not None
    s.close()
