"""MXFP8 KV cache must never write the reserved CUDA-graph padding slot.

Padding lanes carry undefined activations that quantize to NaN payload and
0xFF e8m0 scales; attention reads slot 0 back for padded page-table entries,
so a poisoned slot 0 defeats probability masking (0 * NaN = NaN in PV).
Asserts require slot 0 to stay exactly zero, so finite-garbage writes fail too.
"""

from dataclasses import replace

import pytest
import torch

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_sm() < 100,
    reason="MXFP8 KV cache requires SM100+",
)

DEV, HD, PS, NHKV = "cuda", 128, 128, 2


def _make_pool(**kwargs):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8

    return MHATokenToKVPoolMXFP8(
        size=kwargs.pop("size", 4 * PS),
        page_size=PS,
        dtype=torch.float8_e4m3fn,
        head_num=NHKV,
        head_dim=HD,
        layer_num=kwargs.pop("layer_num", 1),
        device=DEV,
        enable_memory_saver=False,
        **kwargs,
    )


class _Layer:
    layer_id = 0


def _quantize(k, v):
    from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8

    km, vm = to_mxfp8(k), to_mxfp8(v)
    return (
        km.data,
        vm.data,
        km.scale.view(torch.float8_e8m0fnu),
        vm.scale.view(torch.float8_e8m0fnu),
    )


def _assert_slot0_zero(pool):
    kc, vc = pool.get_kv_buffer(0)
    ksf, vsf = pool.get_kv_scale_buffer(0)
    s0k = kc.view(-1, PS, NHKV, HD)[0, 0].view(torch.uint8)
    s0v = vc.view(-1, PS, NHKV, HD)[0, 0].view(torch.uint8)
    assert int(s0k.sum()) == 0, "reserved slot K payload written"
    assert int(s0v.sum()) == 0, "reserved slot V payload written"
    zero_loc = torch.zeros(1, dtype=torch.int64, device=DEV)
    s0_ksf = pool._read_sf_interleaved(ksf, zero_loc).view(torch.uint8)
    s0_vsf = pool._read_sf_interleaved(vsf, zero_loc).view(torch.uint8)
    assert int(s0_ksf.sum()) == 0, "reserved slot K scales written"
    assert int(s0_vsf.sum()) == 0, "reserved slot V scales written"


@requires_sm100
def test_direct_path_skips_reserved_slot():
    torch.manual_seed(0)
    pool = _make_pool()
    k = torch.randn(4, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v = torch.randn(4, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    k[[0, 2]] = float("nan")
    v[[0, 2]] = float("nan")
    kq, vq, ks, vs = _quantize(k, v)
    loc = torch.tensor([0, 7, 0, 9], dtype=torch.int64, device=DEV)

    pool.set_kv_buffer(_Layer(), loc, kq, vq, ks, vs)

    _assert_slot0_zero(pool)
    kc, _ = pool.get_kv_buffer(0)
    got = kc.view(-1, PS, NHKV, HD)[0, 7].view(torch.uint8)
    assert torch.equal(got, kq[1].view(torch.uint8)), "non-reserved write corrupted"


@requires_sm100
def test_fused_quant_store_path_skips_reserved_slot():
    """k_scale=None routes to the fused quant_store_kv_mxfp8 kernel."""
    torch.manual_seed(2)
    pool = _make_pool()
    k = torch.randn(4, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v = torch.randn(4, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    k[[1, 3]] = float("nan")
    v[[1, 3]] = float("nan")
    loc = torch.tensor([5, 0, 9, 0], dtype=torch.int64, device=DEV)

    pool.set_kv_buffer(_Layer(), loc, k, v)

    _assert_slot0_zero(pool)
    kc, _ = pool.get_kv_buffer(0)
    valid = kc.view(-1, PS, NHKV, HD)[0, 5].float()
    assert not torch.isnan(valid).any() and valid.abs().sum() > 0, (
        "fused valid write lost"
    )


@requires_sm100
def test_unified_fused_quant_store_matches_packed_pool():
    """BF16 without descales must write strided payload and interleaved scales."""
    from sglang.srt.mem_cache.unified_memory_pool import (
        MHASubPoolSpec,
        UnifiedKVPool,
        build_unified_mha_pool,
    )

    torch.manual_seed(7)
    spec = MHASubPoolSpec(
        name="full",
        layer_num=2,
        head_num=NHKV,
        head_dim=HD,
        store_dtype=torch.uint8,
        kv_cache_dtype=torch.float8_e4m3fn,
        scale_block_size=32,
        grow_direction="down",
    )
    buffer = UnifiedKVPool(
        total_bytes=8 * PS * spec.entry_bytes(),
        sub_pool_specs=[spec, replace(spec, name="swa", grow_direction="up")],
        device=DEV,
        enable_memory_saver=False,
        page_size=PS,
    )
    unified = build_unified_mha_pool(
        unified_buffer=buffer,
        sub_pool_name="full",
        page_size=PS,
        enable_alt_stream=False,
    )
    packed = _make_pool(size=unified.size, layer_num=2, enable_alt_stream=False)
    assert not unified.k_buffer[0].is_contiguous()
    assert not unified.v_buffer[0].is_contiguous()
    # Compare every byte, including untouched slots and neighboring layers.
    for pool in (unified, packed):
        for buf in (
            pool.k_buffer + pool.v_buffer + pool.k_scale_buffer + pool.v_scale_buffer
        ):
            buf.view(torch.uint8).zero_()

    loc = torch.tensor([0, PS, PS + 31, PS + 32, 2 * PS - 1, 2 * PS], device=DEV)
    for layer_id in range(2):
        # Model K/V may themselves be strided views of a shared projection.
        kv = torch.randn(len(loc), 2, NHKV, HD, dtype=torch.bfloat16, device=DEV)
        k, v = kv.unbind(1)
        k[0] = float("nan")
        v[0] = float("nan")
        layer = _Layer()
        layer.layer_id = layer_id
        unified.set_kv_buffer(layer, loc, k, v)
        # Use independent quantization + store as the reference, rather than
        # the same fused kernel, so quantization and scale writes are checked.
        packed.set_kv_buffer(layer, loc, *_quantize(k, v))

        for name in ("k_buffer", "v_buffer", "k_scale_buffer", "v_scale_buffer"):
            for actual, expected in zip(getattr(unified, name), getattr(packed, name)):
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    _assert_slot0_zero(unified)


@requires_sm100
def test_set_kv_buffer_is_cuda_graph_capture_safe():
    torch.manual_seed(5)
    pool = _make_pool(enable_alt_stream=False)
    k = torch.randn(2, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v = torch.randn(2, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    kq, vq, ks, vs = _quantize(k, v)
    loc = torch.tensor([0, 7], dtype=torch.int64, device=DEV)

    for _ in range(2):  # warmup
        pool.set_kv_buffer(_Layer(), loc, kq, vq, ks, vs)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        pool.set_kv_buffer(_Layer(), loc, kq, vq, ks, vs)
    g.replay()
    torch.cuda.synchronize()

    _assert_slot0_zero(pool)
    kc, _ = pool.get_kv_buffer(0)
    got = kc.view(-1, PS, NHKV, HD)[0, 7].view(torch.uint8)
    assert torch.equal(got, kq[1].view(torch.uint8)), "captured valid write lost"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
