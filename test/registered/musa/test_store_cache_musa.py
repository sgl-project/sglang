"""MUSA tests for the autograd-opaque KV-cache store."""

import pytest
import torch

from sglang.kernels.ops.kvcache.kvcache import store_cache_index_copy
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=30, suite="nightly-musa-1-gpu", nightly=True)


def test_store_cache_index_copy_is_autograd_opaque() -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    k = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    k_cache.requires_grad_(True)
    v_cache.requires_grad_(True)
    indices = torch.arange(8, device="musa", dtype=torch.int32)

    store_cache_index_copy(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices.long()], k)
    assert torch.equal(v_cache[indices.long()], v)
    assert k_cache.grad_fn is None
    assert v_cache.grad_fn is None


def test_store_cache_index_copy_is_graph_safe() -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    k = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    indices = torch.arange(8, device="musa", dtype=torch.int64)

    for _ in range(2):
        store_cache_index_copy(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        store_cache_index_copy(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()
    graph.replay()
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices], k)
    assert torch.equal(v_cache[indices], v)


def test_memory_pool_routes_musa_store_through_opaque_op() -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    row_dim = 64
    k = torch.randn((8, row_dim), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, row_dim), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, row_dim), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, row_dim), dtype=torch.bfloat16, device="musa")
    k_cache.requires_grad_(True)
    v_cache.requires_grad_(True)
    indices = torch.arange(8, device="musa", dtype=torch.int32)

    _set_kv_buffer_impl(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        row_dim=row_dim,
        store_dtype=torch.bfloat16,
        device_module=torch.musa,
        size_limit=k_cache.shape[0],
    )
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices.long()], k)
    assert torch.equal(v_cache[indices.long()], v)
    assert k_cache.grad_fn is None
    assert v_cache.grad_fn is None
