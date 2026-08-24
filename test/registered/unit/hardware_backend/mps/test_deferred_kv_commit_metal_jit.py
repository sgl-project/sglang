"""Small-memory validation for the two-launch deferred KV commit."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from sglang.kernels.ops.kvcache._deferred_kv_commit_metal_jit import (
    commit_deferred_kv,
)
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=3, suite="stage-a-unit-test-mps")

_HAS_MPS_JIT = torch.backends.mps.is_available() and callable(
    getattr(torch.mps, "compile_shader", None)
)


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
def test_deferred_kv_commit_updates_prefill_rows_in_two_async_launches():
    torch.manual_seed(101)
    layers, num_rows, slots_count, kv_heads, head_dim = 28, 4, 16, 8, 128
    new_k = torch.randn(
        layers, num_rows, kv_heads, head_dim, device="mps", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    slots = torch.tensor([3, 11, 5, 14], device="mps", dtype=torch.int64)
    k_pools = [
        torch.zeros(slots_count, kv_heads, head_dim, device="mps", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v_pools = [torch.zeros_like(pool) for pool in k_pools]

    with mock.patch.object(
        torch.mps, "synchronize", wraps=torch.mps.synchronize
    ) as synchronize:
        commit_deferred_kv(
            new_k, new_v, slots, k_pools, v_pools, num_kv_heads=8, head_dim=128
        )
    assert synchronize.call_count == 0

    torch.mps.synchronize()
    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots].cpu(), new_k[layer].cpu())
        torch.testing.assert_close(v_pools[layer][slots].cpu(), new_v[layer].cpu())
        untouched = torch.tensor(
            [0, 1, 2, 4, 6, 7, 8, 9, 10, 12, 13, 15],
            device="mps",
            dtype=torch.int64,
        )
        assert torch.count_nonzero(k_pools[layer][untouched]).item() == 0
        assert torch.count_nonzero(v_pools[layer][untouched]).item() == 0


@pytest.mark.skipif(not _HAS_MPS_JIT, reason="requires Torch 2.13 MPS Metal JIT")
@pytest.mark.parametrize("layers", [3, 15])
def test_generic_commit_supports_non_qwen_shapes_and_layer_chunks(layers):
    rows, slots_count, kv_heads, head_dim = 2, 7, 2, 64
    new_k = torch.randn(
        layers, rows, kv_heads, head_dim, device="mps", dtype=torch.bfloat16
    )
    new_v = torch.randn_like(new_k)
    slots = torch.tensor([1, 5], device="mps", dtype=torch.int64)
    k_pools = [
        torch.zeros(slots_count, kv_heads, head_dim, device="mps", dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v_pools = [torch.zeros_like(pool) for pool in k_pools]

    commit_deferred_kv(
        new_k,
        new_v,
        slots,
        k_pools,
        v_pools,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
    )
    torch.mps.synchronize()

    for layer in range(layers):
        torch.testing.assert_close(k_pools[layer][slots].cpu(), new_k[layer].cpu())
        torch.testing.assert_close(v_pools[layer][slots].cpu(), new_v[layer].cpu())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
