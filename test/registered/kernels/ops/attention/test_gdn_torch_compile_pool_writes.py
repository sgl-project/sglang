"""torch.compile writes one layer of a cache pool in place, without cloning the pool.

Regression test for #51651: Inductor cloned the all-layers pool from its start through the written layer.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkvzba_causal_conv1d_update_contiguous,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-small")

LAYERS, LAYER = 4, 2  # a clone through layer 2 spans three layers
BATCH, SLOTS = 4, 257  # enough slots that one layer dwarfs the step's own outputs


def _run_twice(fn, *args):
    """Run `fn` compiled twice; return the second result and the memory it added at peak."""
    torch._dynamo.reset()
    compiled = torch.compile(fn)
    compiled(*args)  # compile and autotune outside the measurement
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    result = compiled(*args)
    torch.cuda.synchronize()
    return result, torch.cuda.max_memory_allocated() - before


def _layer_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def test_conv_update_writes_one_layer_in_place():
    torch.manual_seed(0)
    qkv_dim, v_dim, num_v_heads, head_v_dim = 128, 64, 2, 32
    dtype = torch.bfloat16
    qkvz = torch.randn(BATCH, qkv_dim + v_dim, device="cuda", dtype=dtype)
    ba = torch.randn(BATCH, 2 * num_v_heads, device="cuda", dtype=dtype)
    weight = torch.randn(qkv_dim, 4, device="cuda", dtype=dtype)
    pool = torch.randn(LAYERS, SLOTS, qkv_dim, 3, device="cuda", dtype=dtype)
    indices = torch.arange(BATCH, device="cuda", dtype=torch.int32)

    def step(pool):
        return fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            pool[LAYER],
            weight,
            None,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation="silu",
        )

    expected_pool = pool.clone()
    step(expected_pool)
    expected = step(expected_pool)
    actual, extra = _run_twice(step, pool)

    assert extra < _layer_bytes(pool[0])
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


def test_packed_decode_writes_one_layer_in_place():
    torch.manual_seed(0)
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = 2, 4, 32, 32
    qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
    mixed_qkv = torch.randn(BATCH, qkv_dim, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(BATCH, num_v_heads, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(BATCH, num_v_heads, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(num_v_heads, device="cuda")
    dt_bias = torch.randn(num_v_heads, device="cuda")
    shape = (LAYERS, SLOTS, num_v_heads, head_v_dim, head_k_dim)
    pool = 0.1 * torch.randn(shape, device="cuda")
    indices = torch.arange(BATCH, device="cuda", dtype=torch.int32)
    kernel = TritonGDNKernel()

    def step(pool):
        return kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=head_k_dim**-0.5,
            ssm_states=pool[LAYER],
            cache_indices=indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
        )

    expected_pool = pool.clone()
    step(expected_pool)
    expected = step(expected_pool)
    actual, extra = _run_twice(step, pool)

    assert extra < _layer_bytes(pool[0])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


def test_kv_store_writes_k_and_v_in_place():
    torch.manual_seed(0)
    head_num, head_dim = 2, 64
    kv_pool = MHATokenToKVPool(
        size=4096,
        page_size=1,
        head_num=head_num,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        layer_num=LAYERS,
        device="cuda",
        enable_memory_saver=False,
    )
    k_buf, v_buf = kv_pool.k_buffer[LAYER], kv_pool.v_buffer[LAYER]
    # The layout that triggers the clone: K and V are views of one allocation.
    assert k_buf.untyped_storage().data_ptr() == v_buf.untyped_storage().data_ptr()
    loc = torch.arange(1, BATCH + 1, device="cuda")
    k = torch.randn(BATCH, head_num, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(BATCH, head_num, head_dim, device="cuda", dtype=torch.bfloat16)

    def step(k, v):
        kv_pool.set_kv_buffer(None, loc, k, v, layer_id_override=LAYER)

    # Build the store kernel eagerly, as serving does before capture: its first
    # call logs, which splits the compiled graph between the slicing and the write.
    step(torch.zeros_like(k), torch.zeros_like(v))
    _, extra = _run_twice(step, k, v)

    assert extra < _layer_bytes(k_buf)
    torch.testing.assert_close(k_buf[loc], k, rtol=0, atol=0)
    torch.testing.assert_close(v_buf[loc], v, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
