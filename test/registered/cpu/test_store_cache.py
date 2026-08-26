import sys
from types import SimpleNamespace

import pytest
import sgl_kernel  # noqa: F401
import torch

from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
    CPUFP8KVCacheMethod,
)
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=14, suite="base-b-test-cpu")

torch.manual_seed(42)

DEVICE = "cpu"
CACHE_SIZE = 4096

# for fp8 KV stored as uint8, e.g. float8_e4m3fn and float8_e5m2
DTYPES = [torch.float16, torch.bfloat16, torch.uint8]
DTYPE_IDS = ["float16", "bfloat16", "uint8"]


def _store_cache_cpu(k, v, k_cache, v_cache, indices):
    row_dim = k.size(1) * k.size(2)
    torch.ops.sgl_kernel.store_cache_cpu(k, v, k_cache, v_cache, indices, row_dim)


def _random_tensor(shape, dtype):
    """FP8 KV is stored as uint8; randn is not implemented for Byte."""
    if dtype == torch.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=DEVICE)
    return torch.randn(shape, dtype=dtype, device=DEVICE)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8, 16, 32])
@pytest.mark.parametrize("batch_size", [1, 7, 133])
def test_store_cache(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[:batch_size]

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices] = k
    v_cache_ref[indices] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


@pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize("batch_size", [11])
def test_store_cache_int32_indices(batch_size, num_heads, head_dim, dtype):
    shape = (batch_size, num_heads, head_dim)
    cache_shape = (CACHE_SIZE, num_heads, head_dim)
    k = _random_tensor(shape, dtype)
    v = _random_tensor(shape, dtype)
    k_cache = _random_tensor(cache_shape, dtype)
    v_cache = _random_tensor(cache_shape, dtype)
    indices = torch.randperm(CACHE_SIZE, device=DEVICE, dtype=torch.int64)[
        :batch_size
    ].to(torch.int32)

    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    k_cache_ref[indices.long()] = k
    v_cache_ref[indices.long()] = v

    _store_cache_cpu(k, v, k_cache, v_cache, indices)

    assert torch.equal(k_cache, k_cache_ref)
    assert torch.equal(v_cache, v_cache_ref)


def test_mha_fp8_e4m3_kv_pool_uses_static_scales():
    pool = MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=32,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        quant_method=CPUFP8KVCacheMethod(),
    )
    loc = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
    layer = SimpleNamespace(layer_id=0)
    cache_k = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)
    cache_v = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)

    pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale=0.5, v_scale=0.25)

    assert pool.k_scale_buffer is None
    assert pool.v_scale_buffer is None
    assert isinstance(pool.get_key_buffer(0), torch.Tensor)
    assert isinstance(pool.get_value_buffer(0), torch.Tensor)

    k_ref = pool.k_buffer[0][loc].clone()
    v_ref = pool.v_buffer[0][loc].clone()
    kv_cpu = pool.get_cpu_copy(loc)

    pool.k_buffer[0][loc].zero_()
    pool.v_buffer[0][loc].zero_()
    pool.load_cpu_copy(kv_cpu, loc)

    assert torch.equal(pool.k_buffer[0][loc], k_ref)
    assert torch.equal(pool.v_buffer[0][loc], v_ref)


@pytest.mark.parametrize(
    ("k_scale", "v_scale"),
    [(None, None), (0.5, 0.25)],
    ids=["unit-scale-default", "non-unit-static-scale"],
)
def test_mha_fp8_e4m3_pool_decode_numerics(k_scale, v_scale):
    seq_len = 16
    num_heads = 2
    head_dim = 64
    num_kv_splits = 8
    sm_scale = head_dim**-0.5
    pool = MHATokenToKVPool(
        size=seq_len,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=num_heads,
        head_dim=head_dim,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
        quant_method=CPUFP8KVCacheMethod(),
    )
    layer = SimpleNamespace(layer_id=0)
    loc = torch.arange(seq_len, dtype=torch.int64, device=DEVICE)
    cache_k = torch.randn(
        (seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    cache_v = torch.randn(
        (seq_len, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE
    )
    pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale=k_scale, v_scale=v_scale)

    effective_k_scale = 1.0 if k_scale is None else k_scale
    effective_v_scale = 1.0 if v_scale is None else v_scale
    k_dequant = (pool.get_key_buffer(0).float() * effective_k_scale).to(torch.bfloat16)
    v_dequant = (pool.get_value_buffer(0).float() * effective_v_scale).to(
        torch.bfloat16
    )
    query = torch.randn((1, num_heads, head_dim), dtype=torch.bfloat16, device=DEVICE)
    output = torch.empty_like(query)
    req_to_token = loc.to(torch.int32).unsqueeze(0)
    req_pool_indices = torch.zeros(1, dtype=torch.int64, device=DEVICE)
    seq_lens = torch.full((1,), seq_len, dtype=torch.int64, device=DEVICE)
    attn_logits = torch.empty(
        (1, num_heads, num_kv_splits, head_dim + 1),
        dtype=torch.float32,
        device=DEVICE,
    )

    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        pool.get_key_buffer(0),
        pool.get_value_buffer(0),
        effective_k_scale,
        effective_v_scale,
        output,
        None,
        None,
        None,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        0.0,
        False,
        0,
        None,
        None,
    )

    output_ref = (
        torch.nn.functional.scaled_dot_product_attention(
            query.movedim(0, 1).unsqueeze(0),
            k_dequant[:seq_len].movedim(0, 1).unsqueeze(0),
            v_dequant[:seq_len].movedim(0, 1).unsqueeze(0),
            scale=sm_scale,
        )
        .squeeze(0)
        .movedim(1, 0)
    )
    torch.testing.assert_close(output, output_ref, atol=3e-2, rtol=1e-6)


@pytest.mark.skipif(
    not cpu_has_amx_support(), reason="FP8 E4M3 MLA KV cache requires AMX"
)
def test_mla_fp8_e4m3_kv_pool_uses_static_scale():
    pool = MLATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        kv_lora_rank=16,
        qk_rope_head_dim=16,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
    )
    loc = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
    target_loc = torch.tensor([11, 13], dtype=torch.int64, device=DEVICE)
    layer = SimpleNamespace(layer_id=0)
    cache_k = torch.randn((2, 1, 32), dtype=torch.bfloat16, device=DEVICE)
    cache_v = torch.empty((2, 1, 16), dtype=torch.bfloat16, device=DEVICE)

    pool.set_kv_buffer(layer, loc, cache_k, cache_v, k_scale=0.5, v_scale=0.5)

    expected_bytes = sum(t.numel() * t.element_size() for t in pool.kv_buffer)
    assert pool.get_kv_size_bytes() == expected_bytes

    kv_ref = pool.kv_buffer[0][loc].clone()
    pool.move_kv_cache(target_loc, loc)

    assert torch.equal(pool.kv_buffer[0][target_loc], kv_ref)

    kv_cpu = pool.get_cpu_copy(target_loc)
    pool.kv_buffer[0][target_loc].zero_()
    pool.load_cpu_copy(kv_cpu, target_loc)

    assert torch.equal(pool.kv_buffer[0][target_loc], kv_ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
