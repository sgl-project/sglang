"""NVFP4 KV cache pools: native FP4 data and block scales per sub-pool."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15,
    stage="base-b-kernel-unit",
    runner_config="4-gpu-b200",
)

pytestmark = pytest.mark.skipif(
    not is_sm100_supported(), reason="NVFP4 KV cache requires SM100"
)


@torch.inference_mode()
def test_hybrid_swa_pool_stores_native_fp4_per_sub_pool():
    """A hybrid SWA pool built with the NVFP4 method owns native FP4 scales in
    both sub-pools, routes each global layer to its sub-pool, and indexes the
    per-layer global scale by global layer id."""
    from sglang.srt.layers.attention.trtllm_mha_backend import (
        _trtllm_native_nvfp4_kv_buffer,
    )
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        NVFP4KVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    page_size, heads, head_dim, size = 16, 2, 128, 64
    full_ids, swa_ids = [1, 3], [0, 2]
    method = NVFP4KVCacheMethod(num_layers=4, device="cuda", page_size=page_size)
    method.configure_attention_backends("trtllm_mha", "trtllm_mha")
    method.k_scales_gpu[3] = 4.0  # global layer 3 = full sub-pool layer 1
    pool = SWAKVPool(
        size=size,
        size_swa=size,
        page_size=page_size,
        dtype=torch.float4_e2m1fn_x2,
        head_num=heads,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_ids,
        full_attention_layer_ids=full_ids,
        device="cuda",
        quant_method=method,
    )

    torch.manual_seed(2)
    k = torch.randn(8, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    loc = torch.arange(page_size, page_size + 8, device="cuda")
    write_loc = KVWriteLoc(
        loc, window_locs={SWAKVPool.WINDOW_SUB_POOL: loc}, physical=True
    )
    for layer_id in (*full_ids, *swa_ids):
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer_id), write_loc, k, v, None, None
        )

    native = {i: _trtllm_native_nvfp4_kv_buffer(pool, i) for i in (*full_ids, *swa_ids)}
    for layer_id, (k_fp4, _, k_scale, _) in native.items():
        assert k_fp4[loc].any(), layer_id
        assert k_scale.view(torch.uint8).any(), layer_id
    # Same input, global scale 4x larger: every block scale is ~4x smaller.
    k_scale_1 = native[1][2].float()
    k_scale_3 = native[3][2].float()
    nonzero = k_scale_1 > 0
    ratio = (k_scale_1[nonzero] / k_scale_3[nonzero]).median()
    assert 3.5 < float(ratio) < 4.5


@torch.inference_mode()
def test_hybrid_swa_pool_raw_buffers_route_per_sub_pool():
    """With the linear (SM120 XQA) scale layout, get_raw_kv_buffer returns each
    global layer's packed data and scales from its own sub-pool."""
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        NVFP4KVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    page_size, heads, head_dim, size = 16, 2, 128, 64
    full_ids, swa_ids = [1, 3], [0, 2]
    method = NVFP4KVCacheMethod(
        num_layers=4, device="cuda", page_size=page_size, native_scale_layout=False
    )
    method.configure_attention_backends("trtllm_mha", "trtllm_mha")
    pool = SWAKVPool(
        size=size,
        size_swa=size,
        page_size=page_size,
        dtype=torch.float4_e2m1fn_x2,
        head_num=heads,
        head_dim=head_dim,
        swa_attention_layer_ids=swa_ids,
        full_attention_layer_ids=full_ids,
        device="cuda",
        quant_method=method,
    )

    torch.manual_seed(3)
    k = torch.randn(8, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    full_loc = torch.arange(page_size, page_size + 8, device="cuda")
    swa_loc = full_loc + page_size
    write_loc = KVWriteLoc(
        full_loc, window_locs={SWAKVPool.WINDOW_SUB_POOL: swa_loc}, physical=True
    )
    # Write only layers 1 (full) and 2 (sliding window).
    for layer_id in (1, 2):
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=layer_id), write_loc, k, v, None, None
        )

    for layer_id, sub_pool, index, loc, written in (
        (1, pool.full_kv_pool, 0, full_loc, True),
        (3, pool.full_kv_pool, 1, full_loc, False),
        (0, pool.swa_kv_pool, 0, swa_loc, False),
        (2, pool.swa_kv_pool, 1, swa_loc, True),
    ):
        k_fp4, v_fp4, k_scale, v_scale = pool.get_raw_kv_buffer(layer_id)
        expected = sub_pool.get_raw_kv_buffer(sub_pool.start_layer + index)
        assert k_fp4.data_ptr() == expected[0].data_ptr(), layer_id
        assert k_scale.data_ptr() == expected[2].data_ptr(), layer_id
        assert bool(k_fp4[loc].any()) == written, layer_id
        assert bool(v_scale.view(torch.uint8)[loc].any()) == written, layer_id
