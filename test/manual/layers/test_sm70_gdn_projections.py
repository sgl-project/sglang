import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="GDN projection fusion requires V100",
)


@pytest.mark.parametrize("rows", [2, 4, 8])
@pytest.mark.parametrize("seed", [7, 31])
def test_projection_layout_matches_separate_kernels_and_replays(
    rows, seed, monkeypatch
):
    from sglang.kernels.ops.gemm.sm70_dense_gemv import linear
    from sglang.kernels.ops.gemm.sm70_qwen_fusions import qkvzba, qkvzba_supported

    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "1")
    monkeypatch.setenv("SGLANG_SM70_QWEN_FUSIONS", "1")
    monkeypatch.setenv("SGLANG_SM70_MTP_QKVZBA", "1")
    monkeypatch.setenv("SGLANG_SM70_BATCHED_FUSIONS", "0")
    torch.manual_seed(seed)
    x = torch.randn(rows, 2560, device="cuda", dtype=torch.float16)
    w = torch.randn(4096, 2560, device="cuda", dtype=torch.float16) * 0.01
    tail = torch.randn(24, 2560, device="cuda", dtype=torch.float16) * 0.01
    if rows == 8:
        assert not qkvzba_supported(x, w, tail)
    monkeypatch.setenv("SGLANG_SM70_BATCHED_FUSIONS", "1")
    assert qkvzba_supported(x, w, tail)
    assert not qkvzba_supported(x[:1], w, tail)
    assert not qkvzba_supported(x, None, tail)
    assert not qkvzba_supported(x, w, tail[:, 1:])
    shifted = torch.empty(x.numel() + 1, device=x.device, dtype=x.dtype)[1:].view_as(x)
    assert not qkvzba_supported(shifted, w, tail)
    qkvzba(x, w, tail)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = qkvzba(x, w, tail)
    for scale in [1.0, -0.5, 0.0]:
        x.mul_(scale)
        graph.replay()
        # Existing two/four-row kernels are an independent layout reference
        # for the eight-row fusion, with the same per-row reduction order.
        projected = torch.cat([linear(part, w) for part in x.split(4)])
        ba = torch.cat([linear(part, tail) for part in x.split(4)])
        expected = (
            projected[:, :2560],
            projected[:, 2560:].reshape(rows, 12, 128),
            ba[:, :12],
            ba[:, 12:],
        )
        for actual, reference in zip(result, expected):
            assert actual.is_contiguous()
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        fp32 = (x.float() @ w.float().T).half()
        torch.testing.assert_close(result[0], fp32[:, :2560], rtol=0.002, atol=0.001)
        torch.testing.assert_close(
            result[1].flatten(1), fp32[:, 2560:], rtol=0.002, atol=0.001
        )
        torch.testing.assert_close(
            torch.cat(result[2:], dim=1),
            (x.float() @ tail.float().T).half(),
            rtol=0.002,
            atol=0.001,
        )
    monkeypatch.setenv("SGLANG_SM70_MTP_QKVZBA", "0")
    assert not qkvzba_supported(x, w, tail)
