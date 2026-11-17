"""MUSA tests for the public Triton KV-cache store."""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_impl
from sglang.srt.utils import is_musa
from sglang.test.ci.ci_register import register_musa_ci

register_musa_ci(est_time=30, suite="nightly-musa-1-gpu", nightly=True)


def store(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    _set_kv_buffer_impl(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        row_dim=k.shape[-1],
        store_dtype=k.dtype,
        device_module=torch.musa,
        size_limit=k_cache.shape[0],
    )


def test_musa_cache_store_is_autograd_opaque() -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    k = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    k_cache.requires_grad_(True)
    v_cache.requires_grad_(True)
    indices = torch.arange(8, device="musa", dtype=torch.int32)

    store(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices.long()], k)
    assert torch.equal(v_cache[indices.long()], v)
    assert k_cache.grad_fn is None
    assert v_cache.grad_fn is None


def test_musa_cache_store_is_graph_safe() -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    k = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, 16), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, 16), dtype=torch.bfloat16, device="musa")
    indices = torch.arange(8, device="musa", dtype=torch.int64)

    for _ in range(2):
        store(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        store(k, v, k_cache, v_cache, indices)
    torch.musa.synchronize()
    graph.replay()
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices], k)
    assert torch.equal(v_cache[indices], v)


def test_musa_cache_store_keeps_asymmetric_kv_on_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not is_musa():
        pytest.skip("MUSA is required")

    row_dim = 16
    v_row_dim = 8
    k = torch.randn((8, row_dim), dtype=torch.bfloat16, device="musa")
    v = torch.randn((8, v_row_dim), dtype=torch.bfloat16, device="musa")
    k_cache = torch.zeros((32, row_dim), dtype=torch.bfloat16, device="musa")
    v_cache = torch.zeros((32, v_row_dim), dtype=torch.bfloat16, device="musa")
    indices = torch.tensor(
        [3, 7, 11, 15, 19, 23, 27, 31], device="musa", dtype=torch.int32
    )

    def fail_triton_store(*args: object, **kwargs: object) -> None:
        raise AssertionError("asymmetric MUSA KV rows must use the fallback")

    monkeypatch.setattr(
        "sglang.srt.mem_cache.memory_pool._set_kv_buffer_prefix_valid_impl",
        fail_triton_store,
    )

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
        alt_stream=None,
        v_row_dim=v_row_dim,
    )
    torch.musa.synchronize()

    assert torch.equal(k_cache[indices.long()], k)
    assert torch.equal(v_cache[indices.long()], v)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
