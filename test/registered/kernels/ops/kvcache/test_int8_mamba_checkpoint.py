# SPDX-License-Identifier: Apache-2.0


import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.mamba_checkpoint_pool import Int8CheckpointStore
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="extra-a", runner_config="1-gpu-small")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def enable_fused_codec():
    with envs.SGLANG_EXPERIMENTAL_FUSED_INT8_CHECKPOINT.override(True):
        yield


def exact(a, b):
    torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)


def pool(num_layers, s, h, v, k, dtype, device, strided=False):
    shape = (num_layers, s, h, v, k * (2 if strided else 1))
    a = torch.randn(shape, device=device, dtype=dtype) * 0.06
    return a[..., ::2] if strided else a


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize(
    "geometry", [(2, 3, 17, 33), (3, 4, 128, 128), (2, 3, 127, 129)]
)
@pytest.mark.parametrize("n", [0, 1, 4])
@pytest.mark.parametrize("strided", [False, True])
def test_codec_exact(dtype, geometry, n, strided):
    torch.manual_seed(123)
    num_layers, h, v, k = geometry
    device = "cuda:0"
    s = 9
    active = pool(num_layers, s, h, v, k, dtype, device, strided)
    src = torch.tensor([7, 1, 7, 3], device=device, dtype=torch.int32)[:n]
    dst = torch.tensor([2, 6, 1, 8], device=device, dtype=torch.int64)[:n]
    # Non-contiguous GPU slot-index views.
    src = src.repeat_interleave(2)[::2]
    dst = dst.repeat_interleave(2)[::2]
    store = Int8CheckpointStore(
        num_layers=num_layers,
        num_slots=s,
        num_heads=h,
        head_v_dim=v,
        head_k_dim=k,
        device=device,
        scale_dtype=dtype,
    )
    store.qdata.fill_(-99)
    store.scale.fill_(-7)
    expected_q = store.qdata.clone()
    expected_s = store.scale.clone()
    q, scale = Int8CheckpointStore.quantize(active[:, src])
    expected_q[:, dst] = q
    expected_s[:, dst] = scale
    store.store_from_pool(active, src, dst)
    if n:
        assert store._fused_store_calls == 1
    exact(store.qdata, expected_q)
    exact(store.scale, expected_s)
    output = torch.full_like(active, 5)
    expected = output.clone()
    expected[:, src.unique()] = (
        5  # sentinel preservation is checked across the whole tensor.
    )
    # Duplicate read source indices are legal; output destinations remain unique.
    expected[:, dst] = store.load(dst, dtype)
    store.copy_to_pool(output, dst, dst)
    if n:
        assert store._fused_restore_calls == 1
    exact(output, expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_edges(dtype):
    device = "cuda"
    num_layers, h, v, k = 1, 1, 128, 16
    a = torch.zeros((num_layers, 4, h, v, k), device=device, dtype=dtype)
    # Scale=1 in the first column; half-way rounding cases elsewhere.
    a[0, 1, 0, 0, :] = 127
    vals = torch.tensor(
        [0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5, -126.5], device=device, dtype=dtype
    )
    a[0, 1, 0, 1:9, :] = vals[:, None]
    a[0, 1, 0, :, 1] *= 1e-7
    a[0, 1, 0, 0, 2] = float("nan")
    a[0, 1, 0, 0, 3] = float("inf")
    a[0, 1, 0, 0, 4] = -float("inf")
    a[0, 1, 0, :, 5] = torch.linspace(-1e-10, 1e-10, 128, device=device, dtype=dtype)
    idx = torch.tensor([1, 2], device=device)
    st = Int8CheckpointStore(
        num_layers=num_layers,
        num_slots=4,
        num_heads=h,
        head_v_dim=v,
        head_k_dim=k,
        device=device,
        scale_dtype=dtype,
    )
    q, s = st.quantize(a[:, idx])
    st.store_from_pool(a, idx, idx)
    exact(st.qdata[:, idx], q)
    exact(st.scale[:, idx], s)
    out = torch.zeros_like(a)
    st.copy_to_pool(out, idx, idx)
    exact(out[:, idx], st.load(idx, dtype))


@pytest.mark.parametrize("v,dtype", [(257, torch.bfloat16), (128, torch.float64)])
def test_unsupported_cuda_input_uses_exact_eager_fallback(v, dtype):
    """Opting in must preserve valid states outside the Triton shape/dtype domain."""
    device = "cuda"
    active = pool(2, 5, 3, v, 33, dtype, device)
    src = torch.tensor([4, 1], device=device)
    dst = torch.tensor([2, 3], device=device)
    store = Int8CheckpointStore(
        num_layers=2,
        num_slots=5,
        num_heads=3,
        head_v_dim=v,
        head_k_dim=33,
        device=device,
        scale_dtype=dtype,
    )
    assert store._fused_codec is not None
    store.qdata.fill_(-99)
    store.scale.fill_(-7)
    expected_q, expected_scale = store.qdata.clone(), store.scale.clone()
    q, scale = store.quantize(active[:, src])
    expected_q[:, dst], expected_scale[:, dst] = q, scale
    store.store_from_pool(active, src, dst)
    exact(store.qdata, expected_q)
    exact(store.scale, expected_scale)
    output = torch.full_like(active, 5)
    expected = output.clone()
    expected[:, src] = store.dequantize(q, scale, dtype)
    store.copy_to_pool(output, dst, src)
    exact(output, expected)
    assert store._fused_store_calls == store._fused_restore_calls == 0


@pytest.mark.parametrize("device", ["cuda:0", "cuda:1"])
def test_model_geometry(device):
    if int(device.rsplit(":", 1)[1]) >= torch.cuda.device_count():
        pytest.skip("second CUDA device unavailable")
    torch.manual_seed(321)
    a = pool(48, 8, 24, 128, 128, torch.bfloat16, device)
    idx = torch.tensor([7, 1, 3, 4, 2, 6, 5], device=device)
    dst = torch.arange(1, 8, device=device)
    st = Int8CheckpointStore(
        num_layers=48,
        num_slots=8,
        num_heads=24,
        head_v_dim=128,
        head_k_dim=128,
        device=device,
    )
    q, s = st.quantize(a[:, idx])
    st.store_from_pool(a, idx, dst)
    exact(st.qdata[:, dst], q)
    exact(st.scale[:, dst], s)
    output = torch.zeros_like(a)
    st.copy_to_pool(output, dst, idx)
    exact(output[:, idx], st.load(dst, a.dtype))


@pytest.mark.parametrize(
    "source_dtype,scale_dtype",
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
def test_mixed_scale_and_pool_strides(source_dtype, scale_dtype):
    num_layers, h, v, k = 2, 3, 37, 65
    device = "cuda"
    a = pool(num_layers, 6, h, v, k, source_dtype, device, True)
    st = Int8CheckpointStore(
        num_layers=num_layers,
        num_slots=6,
        num_heads=h,
        head_v_dim=v,
        head_k_dim=k,
        device=device,
        scale_dtype=scale_dtype,
    )
    st.qdata = torch.empty(
        (num_layers, 6, h, v, k * 2), device=device, dtype=torch.int8
    )[..., ::2]
    st.scale = torch.empty(
        (num_layers, 6, h, 1, k * 2), device=device, dtype=scale_dtype
    )[..., ::2]
    idx = torch.tensor([4, 1, 3], device=device)
    dst = torch.tensor([1, 5, 2], device=device)
    q, s = st.quantize(a[:, idx])
    st.store_from_pool(a, idx, dst)
    exact(st.qdata[:, dst], q)
    exact(st.scale[:, dst], s.to(scale_dtype))
    b = pool(num_layers, 6, h, v, k, source_dtype, device, True)
    expected = b.clone()
    expected[:, idx] = st.load(dst, source_dtype)
    st.copy_to_pool(b, dst, idx)
    exact(b, expected)


def test_cuda_graph_and_slot_updates():
    device = "cuda"
    a = pool(2, 5, 3, 128, 128, torch.bfloat16, device)
    b = torch.zeros_like(a)
    st = Int8CheckpointStore(
        num_layers=2,
        num_slots=5,
        num_heads=3,
        head_v_dim=128,
        head_k_dim=128,
        device=device,
    )
    src = torch.tensor([3, 1], device=device)
    dst = torch.tensor([1, 4], device=device)
    st.store_from_pool(a, src, dst)
    st.copy_to_pool(b, dst, src)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        st.store_from_pool(a, src, dst)
        st.copy_to_pool(b, dst, src)
    a.mul_(0.75)
    src.copy_(torch.tensor([2, 4], device=device))
    dst.copy_(torch.tensor([3, 0], device=device))
    graph.replay()
    torch.cuda.synchronize()
    q, s = st.quantize(a[:, src])
    exact(st.qdata[:, dst], q)
    exact(st.scale[:, dst], s)
    exact(b[:, src], st.dequantize(q, s, a.dtype))


def test_checkpoint_offsets_above_two_gib():
    """Real final pool layout crosses the signed 32-bit byte-offset boundary."""
    torch.manual_seed(456)
    device = "cuda:0"
    a = pool(48, 3, 24, 128, 128, torch.bfloat16, device)
    st = Int8CheckpointStore(
        num_layers=48,
        num_slots=137,
        num_heads=24,
        head_v_dim=128,
        head_k_dim=128,
        device=device,
    )
    assert st.qdata.numel() > 2**31
    src = torch.tensor([2], device=device)
    dst = torch.tensor([136], device=device)
    q, scale = st.quantize(a[:, src])
    st.store_from_pool(a, src, dst)
    exact(st.qdata[:, dst], q)
    exact(st.scale[:, dst], scale)
    out = torch.zeros_like(a)
    st.copy_to_pool(out, dst, src)
    exact(out[:, src], st.dequantize(q, scale, a.dtype))


if __name__ == "__main__":
    import sys

    args = [arg for arg in sys.argv[1:] if arg != "-f"]
    raise SystemExit(pytest.main([__file__, "-v", *args]))
