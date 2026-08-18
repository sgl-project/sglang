"""MXFP8 KV cache must never write the reserved CUDA-graph padding slot.

Slot 0 is the reserved padding sink; padded page-table entries read its page
back. CUDA-graph padding lanes carry undefined activations, which quantize to
NaN payload and 0xFF e8m0 scales. The bf16 store_cache path skips writes to
the reserved slot (reserved_skip_index); the MXFP8 pool paths must do the
same, or the poisoned slot is read back by attention (0 * NaN = NaN in PV)
and detonates the run.

Every test asserts the strong invariant -- slot 0 payload bytes and scale
entries stay exactly zero -- so any write to the reserved slot fails the test,
not just a NaN one. Guard safety is covered too: neighbors sharing page 0
(payload and interleaved per-slot scales) must be byte-identical after writes
that target the reserved slot, and set_kv_buffer must capture and replay
inside a CUDA graph.
"""

import pytest
import torch

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="4-gpu-b200")

requires_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_device_sm() < 100,
    reason="MXFP8 KV cache requires SM100+",
)

DEV, HD, PS, NHKV = "cuda", 128, 128, 2


def _make_pool(**kwargs):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8

    return MHATokenToKVPoolMXFP8(
        size=4 * PS,
        page_size=PS,
        dtype=torch.float8_e4m3fn,
        head_num=NHKV,
        head_dim=HD,
        layer_num=1,
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
    # Rows 0 and 2 are padding lanes with undefined (NaN) activations.
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
def test_skip_does_not_clobber_page0_neighbors():
    """Slots sharing page 0 (payload and the interleaved scale chunk) must
    survive later writes that target the reserved slot."""
    torch.manual_seed(1)
    pool = _make_pool()
    k = torch.randn(5, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v = torch.randn(5, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    kq, vq, ks, vs = _quantize(k, v)
    neighbors = torch.tensor([1, 2, 3, 64, 127], dtype=torch.int64, device=DEV)
    pool.set_kv_buffer(_Layer(), neighbors, kq, vq, ks, vs)

    kc, vc = pool.get_kv_buffer(0)
    ksf, vsf = pool.get_kv_scale_buffer(0)
    k_bytes = kc.view(-1, PS, NHKV, HD)[0, neighbors.cpu()].view(torch.uint8).clone()
    v_bytes = vc.view(-1, PS, NHKV, HD)[0, neighbors.cpu()].view(torch.uint8).clone()
    k_sf = pool._read_sf_interleaved(ksf, neighbors).clone()
    v_sf = pool._read_sf_interleaved(vsf, neighbors).clone()

    k2 = torch.randn(2, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v2 = torch.randn(2, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    k2[0] = float("nan")
    v2[0] = float("nan")
    kq2, vq2, ks2, vs2 = _quantize(k2, v2)
    pool.set_kv_buffer(
        _Layer(),
        torch.tensor([0, 200], dtype=torch.int64, device=DEV),
        kq2,
        vq2,
        ks2,
        vs2,
    )

    _assert_slot0_zero(pool)
    kc, vc = pool.get_kv_buffer(0)
    ksf, vsf = pool.get_kv_scale_buffer(0)
    assert torch.equal(
        kc.view(-1, PS, NHKV, HD)[0, neighbors.cpu()].view(torch.uint8), k_bytes
    ), "page-0 neighbor K payload clobbered"
    assert torch.equal(
        vc.view(-1, PS, NHKV, HD)[0, neighbors.cpu()].view(torch.uint8), v_bytes
    ), "page-0 neighbor V payload clobbered"
    assert torch.equal(
        pool._read_sf_interleaved(ksf, neighbors).view(torch.uint8),
        k_sf.view(torch.uint8),
    ), "page-0 neighbor K scales clobbered"
    assert torch.equal(
        pool._read_sf_interleaved(vsf, neighbors).view(torch.uint8),
        v_sf.view(torch.uint8),
    ), "page-0 neighbor V scales clobbered"


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
    assert (
        not torch.isnan(valid).any() and valid.abs().sum() > 0
    ), "fused valid write lost"


@requires_sm100
def test_capture_mode_flag_skips_reserved_slot():
    from sglang.srt.model_executor.runner_utils import capture_mode

    torch.manual_seed(3)
    pool = _make_pool()
    k = torch.randn(3, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    v = torch.randn(3, NHKV, HD, dtype=torch.bfloat16, device=DEV) * 0.5
    k[0] = float("nan")
    v[0] = float("nan")
    kq, vq, ks, vs = _quantize(k, v)
    loc = torch.tensor([0, 7, 9], dtype=torch.int64, device=DEV)

    capture_mode.is_capture_mode = True
    try:
        pool.set_kv_buffer(_Layer(), loc, kq, vq, ks, vs)
    finally:
        capture_mode.is_capture_mode = False
    torch.cuda.synchronize()

    _assert_slot0_zero(pool)
    kc, _ = pool.get_kv_buffer(0)
    got = kc.view(-1, PS, NHKV, HD)[0, 7].view(torch.uint8)
    assert torch.equal(got, kq[1].view(torch.uint8)), "capture-mode valid write lost"


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
