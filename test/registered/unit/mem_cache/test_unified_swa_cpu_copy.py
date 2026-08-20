"""Regression tests for sync-free UnifiedSWAKVPool SWA validity filtering."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.mem_cache.unified_memory_pool import UnifiedSWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_pool(full_v2p, swa_v2p, page_size=1):
    pool = object.__new__(UnifiedSWAKVPool)
    pool._full_allocator = SimpleNamespace(
        page_size=page_size, virtual_to_physical=torch.tensor(full_v2p)
    )
    pool._swa_allocator = SimpleNamespace(
        page_size=page_size, virtual_to_physical=torch.tensor(swa_v2p)
    )
    pool.full_kv_pool = Mock()
    pool.swa_kv_pool = Mock()
    pool.full_kv_pool.get_cpu_copy.side_effect = lambda x: x.clone()
    pool.swa_kv_pool.get_cpu_copy.side_effect = lambda x: x.clone()
    return pool


def test_get_cpu_copy_filters_tombstones_without_scalar_item():
    pool = _make_pool([0, 11, 12, 13], [0, 21, -1, 23])
    indices = torch.tensor([1, 2, 3])

    with patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")):
        result = pool.get_cpu_copy(indices)

    assert torch.equal(result["full"], torch.tensor([11, 12, 13]))
    assert torch.equal(result["swa"], torch.tensor([21, 23]))


def test_empty_swa_copy_and_load_remain_noops_without_scalar_item():
    pool = _make_pool([0, 11, 12], [0, -1, -1])
    indices = torch.tensor([1, 2])

    with patch.object(torch.Tensor, "item", side_effect=AssertionError("sync")):
        result = pool.get_cpu_copy(indices)
        pool.load_cpu_copy(result, indices)

    saved = pool.swa_kv_pool.get_cpu_copy.call_args.args[0]
    restored = pool.swa_kv_pool.load_cpu_copy.call_args.args[1]
    assert saved.numel() == 0
    assert restored.numel() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
