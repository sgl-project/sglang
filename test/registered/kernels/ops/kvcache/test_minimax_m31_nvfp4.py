"""Packed M3.1 KV writes and relocations preserve payloads and block scales."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.minimax_m3_nvfp4_pool import MiniMaxM3Nvfp4KVPool
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

LAYER = SimpleNamespace(layer_id=0)


def _sparse_pool(heads, page_size=128, pages=8):
    return MiniMaxM3Nvfp4KVPool(
        size=pages * page_size - page_size,
        page_size=page_size,
        dtype=torch.float8_e4m3fn,
        head_num=heads,
        head_dim=128,
        idx_head_dim=128,
        dense_layer_ids=[],
        sparse_layer_ids=[0],
        disable_value_sparse_layer_ids=[0],
        device="cuda",
        start_layer=0,
        end_layer=1,
    )


def _split_qkv(qkv, heads):
    # Projection splits retain the wider QKV row stride, as the model supplies.
    return qkv[:, heads : 2 * heads], qkv[:, 2 * heads : 3 * heads], qkv[:, 0]


def _randn(shape):
    return torch.randn(*shape, device="cuda").to(torch.bfloat16)


def _zeros(shape):
    return torch.zeros(*shape, dtype=torch.bfloat16, device="cuda")


def _tiny(shape):
    # Below the E4M3 scale underflow point, where the reference writes a zero
    # scale and a saturated payload.
    return torch.full(shape, 1e-4, dtype=torch.bfloat16, device="cuda")


def _huge(shape):
    return (torch.randn(*shape, device="cuda") * 3e4).to(torch.bfloat16)


def _e2m1_grid(shape):
    # Values on the e2m1 round-to-nearest midpoints after scaling.
    grid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0], device="cuda")
    count = shape[0] * shape[1] * shape[2]
    return grid.repeat(count // 8 + 1)[:count].reshape(shape).to(torch.bfloat16)


@pytest.mark.parametrize(
    "gen, heads, loc_dtype, n",
    [
        (_zeros, 1, torch.int32, 64),
        (_tiny, 4, torch.int64, 64),
        (_huge, 4, torch.int32, 17),
        (_e2m1_grid, 1, torch.int64, 64),
    ],
)
def test_fused_store_matches_separate_stores(gen, heads, loc_dtype, n):
    # The eager reference takes the flashinfer quantizer only on SM100/SM103;
    # elsewhere it is a different quantizer, so the comparison means nothing.
    if not get_platform().is_sm100:
        pytest.skip("fused NVFP4 store is accepted on SM100/SM103 only")
    torch.manual_seed(9)
    reference, subject = _sparse_pool(heads), _sparse_pool(heads)
    subject.use_nvfp4_fused_quantize_store = True
    k, v, ik = _split_qkv(gen((n, 4 * heads, 128)), heads)
    slots = torch.randperm(8 * 128, device="cuda")[:n].to(loc_dtype)

    reference.set_kv_buffer(LAYER, slots, k, v)
    reference.set_index_k_buffer(LAYER, slots, ik)
    subject.set_fused_kv_index_buffer(LAYER, slots, k, v, ik, None)

    for a, b in zip(
        reference.main_pool.get_raw_kv_buffers(0),
        subject.main_pool.get_raw_kv_buffers(0),
    ):
        assert torch.equal(a, b)
    for a, b in zip(
        reference.index_k_pool.get_raw_k_buffers(0),
        subject.index_k_pool.get_raw_k_buffers(0),
    ):
        assert torch.equal(a, b)


def test_move_slots_carries_every_buffer():
    torch.manual_seed(42)
    pool = _sparse_pool(heads=2)
    main, index = pool.main_pool, pool.index_k_pool
    k, v, ik = _split_qkv(_randn((17, 8, 128)), 2)
    loc = torch.arange(120, 137, device="cuda")
    pool.set_kv_buffer(LAYER, loc, k, v)
    pool.set_index_k_buffer(LAYER, loc, ik)
    before = main.get_quantized_rows(0, loc) + index.get_quantized_rows(0, loc)

    main.move_slots(loc + 1, loc)
    index.move_slots(loc + 1, loc)
    after = main.get_quantized_rows(0, loc + 1) + index.get_quantized_rows(0, loc + 1)
    for a, b in zip(before, after):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def test_fused_store_selected_exactly_on_sm100():
    pool = _sparse_pool(heads=2)
    assert pool.use_nvfp4_fused_quantize_store == get_platform().is_sm100
    # The base byte-copy fast path never applies to a quantizing cache.
    assert pool.use_minimax_fused_kv_index_store is False


def test_disabled_fused_store_builds_no_kernel(monkeypatch):
    import sglang.kernels.ops.kvcache.nvfp4_store_kv_index as fused_store

    def fail(*args, **kwargs):
        raise AssertionError("the NVFP4 fused store JIT must not build when disabled")

    monkeypatch.setattr(fused_store, "_jit_module", fail)
    pool = _sparse_pool(heads=2)
    pool.use_nvfp4_fused_quantize_store = False
    k, v, ik = _split_qkv(_randn((9, 8, 128)), 2)
    slots = torch.arange(9, device="cuda", dtype=torch.int32)

    pool.set_fused_kv_index_buffer(LAYER, slots, k, v, ik, None)

    payload, scales = pool.index_k_pool.get_raw_k_buffers(0)
    assert payload[0, :9].any() and scales[0, :9].any()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
