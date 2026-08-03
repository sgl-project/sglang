import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.utils import cpu_has_amx_support
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-b-test-cpu")

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


def _cpu_has_amx():
    return cpu_has_amx_support()


def _import_mha_pool():
    pytest.importorskip(
        "xgrammar.structural_tag",
        reason="local xgrammar is too old for workspace sglang imports",
    )
    try:
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    except ImportError as exc:
        if "AnyTokensFormat" in str(exc):
            pytest.skip("local xgrammar is too old for workspace sglang imports")
        raise
    return MHATokenToKVPool


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


@pytest.mark.skipif(not _cpu_has_amx(), reason="FP8 E4M3 KV cache requires AMX")
def test_mha_fp8_e4m3_kv_pool_updates_scales():
    MHATokenToKVPool = _import_mha_pool()

    pool = MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=32,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
    )
    loc = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
    layer = SimpleNamespace(layer_id=0)
    cache_k = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)
    cache_v = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)

    pool.set_kv_buffer(layer, loc, cache_k, cache_v)

    assert pool.k_scale_buffer is not None
    assert pool.v_scale_buffer is not None
    assert torch.all(pool.k_scale_buffer[0][loc] > 0)
    assert torch.all(pool.v_scale_buffer[0][loc] > 0)


@pytest.mark.skipif(not _cpu_has_amx(), reason="FP8 E4M3 KV cache requires AMX")
def test_mha_fp8_e4m3_kv_pool_rejects_hnd(monkeypatch):
    MHATokenToKVPool = _import_mha_pool()

    monkeypatch.setattr(envs.SGLANG_USE_HND_KVCACHE, "get", lambda: True)
    with pytest.raises(NotImplementedError, match="SGLANG_USE_HND_KVCACHE"):
        MHATokenToKVPool(
            size=32,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            head_num=2,
            head_dim=32,
            layer_num=1,
            device=DEVICE,
            enable_memory_saver=False,
        )


@pytest.mark.skipif(not _cpu_has_amx(), reason="FP8 E4M3 KV cache requires AMX")
def test_mha_fp8_e4m3_kv_pool_rejects_dcp_mask():
    MHATokenToKVPool = _import_mha_pool()

    pool = MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.float8_e4m3fn,
        head_num=2,
        head_dim=32,
        layer_num=1,
        device=DEVICE,
        enable_memory_saver=False,
    )
    loc = torch.tensor([3, 7], dtype=torch.int64, device=DEVICE)
    layer = SimpleNamespace(layer_id=0)
    cache_k = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)
    cache_v = torch.randn((2, 2, 32), dtype=torch.bfloat16, device=DEVICE)
    dcp_kv_mask = torch.ones((2,), dtype=torch.bool, device=DEVICE)

    with pytest.raises(NotImplementedError, match="DCP KV masks"):
        pool.set_kv_buffer(layer, loc, cache_k, cache_v, dcp_kv_mask=dcp_kv_mask)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
