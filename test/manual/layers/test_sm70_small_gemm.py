import pytest
import torch

from sglang.kernels.ops.gemm.sm70_small_gemm import _CONFIGS

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="Small GEMM kernels require V100",
)


@pytest.mark.parametrize("rows,n,k", list(_CONFIGS))
def test_small_gemm_reference_and_graph_replay(rows, n, k, monkeypatch):
    from sglang.kernels.ops.gemm.sm70_dense_gemv import linear, supported
    from sglang.kernels.ops.gemm.sm70_qwen_fusions import gate_up_supported, qkv_ba_supported

    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "1")
    monkeypatch.setenv("SGLANG_SM70_MTP_SMALL_GEMM", "1")
    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "1")
    torch.manual_seed(n + k + rows)
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16) * 0.01
    assert supported(x, weight)
    assert not gate_up_supported(x, weight)
    assert not qkv_ba_supported(x, weight, weight[:24])
    result = linear(x, weight)
    reference = (x.float() @ weight.float().T).half()
    torch.testing.assert_close(result, reference, rtol=0.002, atol=0.001)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = linear(x, weight)
    for scale in [-0.5, 2.0, 0.0]:
        x.mul_(scale)
        graph.replay()
        reference = (x.float() @ weight.float().T).half()
        torch.testing.assert_close(result, reference, rtol=0.002, atol=0.001)
    monkeypatch.setenv("SGLANG_SM70_MTP_SMALL_GEMM", "0")
    assert not supported(x, weight)


def test_hc_batch_dispatch_falls_back_for_disabled_or_unaligned(monkeypatch):
    from sglang.kernels.ops.gemm.hc_mix import (
        sm70_hc_down_gemv_silu,
        sm70_hc_down_gemv_silu_supported,
    )

    x = torch.randn(4, 10240, device="cuda", dtype=torch.float16)
    wd = torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.01
    wu = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.01
    assert sm70_hc_down_gemv_silu_supported(x, wd, wu)
    assert not sm70_hc_down_gemv_silu_supported(x.repeat(2, 1), wd, wu)
    # Rows 2/4 stay on the sm70 dispatch; the native knobs and the 16B
    # alignment only select the Triton rows kernel, whose result the
    # fallback must match bit-exactly.
    monkeypatch.setenv("SGLANG_SM70_HC_NATIVE", "0")
    reference = sm70_hc_down_gemv_silu(x, wd, 4)
    shifted = torch.randn(x.numel() + 1, device="cuda", dtype=torch.float16)[
        1:
    ].view_as(x)
    shifted_reference = sm70_hc_down_gemv_silu(shifted, wd, 4)
    monkeypatch.setenv("SGLANG_SM70_HC_NATIVE", "1")
    monkeypatch.setenv("SGLANG_SM70_MTP_HC", "0")
    torch.testing.assert_close(
        sm70_hc_down_gemv_silu(x, wd, 4), reference, rtol=0, atol=0
    )
    monkeypatch.setenv("SGLANG_SM70_MTP_HC", "1")
    torch.testing.assert_close(
        sm70_hc_down_gemv_silu(shifted, wd, 4), shifted_reference, rtol=0, atol=0
    )
