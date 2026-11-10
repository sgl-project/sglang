import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="SM70 GEMV kernels require a V100",
)


@pytest.mark.parametrize(
    "n,k",
    [
        (4096, 2560),
        (3584, 2560),
        (2560, 1536),
        (320, 2560),
        (2560, 160),
        (512, 2560),
        (640, 2560),
        (24, 2560),
        (1, 2560),
        (10240, 2560),
        (2560, 2560),
        (62080, 2560),
    ],
)
def test_gemv_fp32_reference_and_graph_replay(n, k):
    from sglang.kernels.ops.gemm.sm70_dense_gemv import linear

    torch.manual_seed(n + k)
    x = torch.randn((1, k), device="cuda", dtype=torch.float16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.01
    output = linear(x, weight)
    expected = (x.float() @ weight.float().T).half()
    torch.testing.assert_close(output, expected, rtol=0.002, atol=0.001)
    assert (
        torch.linalg.vector_norm(output.float() - expected.float())
        / torch.linalg.vector_norm(expected.float())
        < 0.0005
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = linear(x, weight)
    for scale in [0.5, -2.0, 0.0]:
        x.mul_(scale)
        graph.replay()
        expected = (x.float() @ weight.float().T).half()
        torch.testing.assert_close(captured, expected, rtol=0.002, atol=0.001)


def test_dispatch_rejects_unaligned_or_unsupported_operands(monkeypatch):
    from sglang.kernels.ops.gemm.sm70_dense_gemv import supported

    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "1")
    x = torch.zeros((1, 2560), device="cuda", dtype=torch.float16)
    weight = torch.zeros((512, 2560), device="cuda", dtype=torch.float16)
    assert supported(x, weight)
    assert not supported(x.expand(2, -1), weight)
    assert not supported(x.float(), weight.float())
    assert not supported(x, weight, torch.zeros(512, device="cuda"))
    shifted = torch.zeros(2561, device="cuda", dtype=torch.float16)[1:].view(1, 2560)
    assert not supported(shifted, weight)
    monkeypatch.setenv("SGLANG_SM70_DENSE_GEMV", "0")
    assert not supported(x, weight)
