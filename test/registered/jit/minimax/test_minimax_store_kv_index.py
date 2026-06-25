"""Correctness for the fused MiniMax-M3 KV + index cache store kernel.

Verifies the single fused launch writes the main K/V, the index K, and the
optional index V into their pools at out_cache_loc rows exactly as the separate
index_put_ stores would, for both value modes and int32/int64 indices.
"""

import pytest
import torch

from sglang.jit_kernel.minimax_store_kv_index import store_kv_index
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")
register_amd_ci(est_time=10, suite="nightly-amd-kernel-1-gpu", nightly=True)

dev = "cuda"
HEAD_DIM = 128


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("has_v", [False, True])
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("T", [1, 7, 128, 513])
def test_store_kv_index(dtype, num_kv_heads, has_v, idx_dtype, T):
    torch.manual_seed(T * 17 + num_kv_heads)
    head_bytes = HEAD_DIM * dtype.itemsize
    N = 4096

    def rnd(*shape):
        return torch.randn(*shape, dtype=dtype, device=dev)

    k = rnd(T, num_kv_heads * HEAD_DIM)
    v = rnd(T, num_kv_heads * HEAD_DIM)
    idx_k = rnd(T, HEAD_DIM)
    idx_v = rnd(T, HEAD_DIM) if has_v else None

    k_cache = torch.zeros(N, num_kv_heads * HEAD_DIM, dtype=dtype, device=dev)
    v_cache = torch.zeros_like(k_cache)
    idx_k_cache = torch.zeros(N, HEAD_DIM, dtype=dtype, device=dev)
    idx_v_cache = torch.zeros_like(idx_k_cache) if has_v else None

    loc = torch.randperm(N, device=dev)[:T].to(idx_dtype)

    store_kv_index(
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        loc,
        num_kv_heads=num_kv_heads,
        head_bytes=head_bytes,
    )

    ll = loc.long()
    k_ref = torch.zeros_like(k_cache)
    v_ref = torch.zeros_like(v_cache)
    ik_ref = torch.zeros_like(idx_k_cache)
    k_ref[ll], v_ref[ll], ik_ref[ll] = k, v, idx_k

    assert torch.equal(k_cache, k_ref)
    assert torch.equal(v_cache, v_ref)
    assert torch.equal(idx_k_cache, ik_ref)
    if has_v:
        iv_ref = torch.zeros_like(idx_v_cache)
        iv_ref[ll] = idx_v
        assert torch.equal(idx_v_cache, iv_ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
