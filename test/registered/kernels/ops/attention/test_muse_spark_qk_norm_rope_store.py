"""Muse Spark QK norm (+ interleaved RoPE, + KV store): Triton against the torch reference."""

import pytest
import torch

from sglang.kernels.ops.attention.muse_spark_qk_norm_rope_store import FusedQKNorm
from sglang.kernels.spec import KernelBackend
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_HEAD_DIM, _NUM_Q, _NUM_KV, _MAX_POS, _SLOTS = 64, 8, 2, 4096, 1024


def _cos_sin_cache():
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, _HEAD_DIM, 2, device="cuda") / _HEAD_DIM)
    )
    freqs = torch.outer(torch.arange(_MAX_POS, device="cuda").float(), inv_freq)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)


@pytest.mark.parametrize("num_tokens", [1, 17, 700])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("cache_dtype", [None, torch.bfloat16, torch.float8_e4m3fn])
def test_qk_norm_matches_native(num_tokens, rope, cache_dtype):
    torch.manual_seed(num_tokens)
    op = FusedQKNorm(
        head_dim=_HEAD_DIM, num_q_heads=_NUM_Q, num_kv_heads=_NUM_KV, eps=1e-6
    )
    width = (_NUM_Q + 2 * _NUM_KV) * _HEAD_DIM
    # A row-strided view, as the fused QKV-O projection output is.
    qkv = torch.randn(num_tokens, width + 64, device="cuda", dtype=torch.bfloat16)[
        :, :width
    ]
    positions = torch.randint(0, _MAX_POS, (num_tokens,), device="cuda")
    args = (positions, _cos_sin_cache()) if rope else (None, None)

    def kv_store():
        if cache_dtype is None:
            return None
        torch.manual_seed(0)
        shape = (_SLOTS, _NUM_KV * _HEAD_DIM)
        k_cache = torch.randn(shape, device="cuda").to(cache_dtype)
        v_cache = torch.randn(shape, device="cuda").to(cache_dtype)
        # Slot 0 is the CUDA-graph padding slot and is never written.
        cache_loc = torch.randperm(_SLOTS, device="cuda")[:num_tokens]
        return k_cache, v_cache, cache_loc

    ref_store, store = kv_store(), kv_store()
    ref_q, ref_k = op.forward_native(qkv, *args, kv_store=ref_store)
    q, k = op(qkv, *args, kv_store=store, backend=KernelBackend.TRITON)

    assert q.dtype == ref_q.dtype and k.dtype == ref_k.dtype
    _assert_close(q, ref_q)
    _assert_close(k, ref_k)
    if cache_dtype is not None:
        for out, ref in zip(store[:2], ref_store[:2]):
            _assert_close(out, ref)


@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("all_padding", [False, True])
def test_qk_norm_kv_store_write_contract(rope, cache_dtype, all_padding):
    torch.manual_seed(813)
    op = FusedQKNorm(
        head_dim=_HEAD_DIM, num_q_heads=_NUM_Q, num_kv_heads=_NUM_KV, eps=1e-6
    )
    num_tokens, capacity = 5, 19
    width = (_NUM_Q + 2 * _NUM_KV) * _HEAD_DIM
    storage = torch.randn(num_tokens, width + 64, device="cuda", dtype=torch.bfloat16)
    qkv = storage[:, :width]
    # V is copied, not normalized or rotated. Include signed zeros, E4M3
    # midpoints (round-to-even in both directions), and saturation endpoints.
    values = torch.tensor(
        [0.0, -0.0, 1.0625, 1.1875, -1.0625, -1.1875, 448.0, -448.0, 512.0, -512.0],
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = qkv[:, (_NUM_Q + _NUM_KV) * _HEAD_DIM :]
    v.copy_(values[torch.arange(v.numel(), device="cuda") % values.numel()].view_as(v))
    storage_before = storage.clone()
    positions = torch.tensor([0, 1, 63, 64, _MAX_POS - 1], device="cuda")
    args = (positions, _cos_sin_cache()) if rope else (None, None)
    # Repeated slot zero is legal padding. Live slots are unique and include
    # both ends of the writable pool; most rows must remain untouched.
    slots = torch.tensor(
        [0, 0, 0, 0, 0] if all_padding else [0, capacity - 1, 1, 0, 9],
        device="cuda",
    )
    shape = (capacity, _NUM_KV * _HEAD_DIM)
    k_cache = torch.full(shape, 11.0, device="cuda").to(cache_dtype)
    v_cache = torch.full(shape, -13.0, device="cuda").to(cache_dtype)
    k_before, v_before = k_cache.clone(), v_cache.clone()
    store = (k_cache, v_cache, slots)
    ref_store = (k_before.clone(), v_before.clone(), slots)
    ref_q, ref_k = op.forward_native(qkv, *args, kv_store=ref_store)
    q, k = op(qkv, *args, kv_store=store, backend=KernelBackend.TRITON)
    assert q.dtype == ref_q.dtype and k.dtype == ref_k.dtype
    _assert_close(q, ref_q)
    _assert_close(k, ref_k)

    live = slots != 0
    untouched = torch.ones(capacity, device="cuda", dtype=torch.bool)
    untouched[slots[live]] = False
    for cache, before in ((k_cache, k_before), (v_cache, v_before)):
        torch.testing.assert_close(
            cache.view(torch.uint8)[untouched],
            before.view(torch.uint8)[untouched],
            rtol=0,
            atol=0,
        )
    # Compare stored K with the returned BF16 K from the same execution:
    # reduction/RoPE differences versus torch need tolerance, a store does not.
    expected_k, expected_v = k[live], v[live]
    if cache_dtype == torch.float8_e4m3fn:
        expected_k = expected_k.float().clamp(-448.0, 448.0)
        expected_v = expected_v.float().clamp(-448.0, 448.0)
    for cache, expected in ((k_cache, expected_k), (v_cache, expected_v)):
        torch.testing.assert_close(
            cache.view(torch.uint8)[slots[live]],
            expected.to(cache_dtype).contiguous().view(torch.uint8),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        storage.view(torch.uint8), storage_before.view(torch.uint8), rtol=0, atol=0
    )


def _assert_close(out, ref):
    # One e4m3 step is 2^-3 relative; Triton and torch may round a tie apart.
    tol = 0.125 if out.dtype == torch.float8_e4m3fn else 2e-2
    torch.testing.assert_close(out.float(), ref.float(), rtol=tol, atol=tol)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
