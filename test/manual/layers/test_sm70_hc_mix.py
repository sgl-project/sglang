import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="The native hyperconnection kernels require V100",
)


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_native_hc_mix_matches_fp32_reference(seed, rows):
    from sglang.kernels.ops.elementwise.sm70_hc_mix import hc_down, hc_up

    torch.manual_seed(seed)
    x = torch.randn(rows, 10240, device="cuda", dtype=torch.float16)
    down_weight = torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.01
    up_weight = torch.randn(10240, 320, device="cuda", dtype=torch.float16) * 0.01
    down = hc_down(x, down_weight)
    projected = (x.float() @ down_weight.float().T).half().float() / 4
    expected_down = torch.nn.functional.silu(projected).half()
    torch.testing.assert_close(down, expected_down, rtol=0.001, atol=0.0002)

    output = hc_up(down, x, up_weight)
    up = (down.float() @ up_weight.float().T).half().float()
    expected = (torch.sigmoid(up) * x.float()).reshape(rows, 4, 2560).mean(1).half()
    torch.testing.assert_close(output, expected, rtol=0.001, atol=0.0002)

    # Replays use the current stream and fresh inputs, without stale scratch.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = hc_up(hc_down(x, down_weight), x, up_weight)
    for _ in range(3):
        x.mul_(0.5)
        graph.replay()
        eager = hc_up(hc_down(x, down_weight), x, up_weight)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)


def test_hc_unaligned_input_uses_triton_fallback(monkeypatch):
    from sglang.kernels.ops.gemm.hc_mix import sm70_hc_down_gemv_silu

    x = torch.randn(10241, device="cuda", dtype=torch.float16)[1:].view(1, 10240)
    weight = torch.randn(320, 10240, device="cuda", dtype=torch.float16) * 0.01
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    monkeypatch.setenv("SGLANG_SM70_HC_NATIVE", "0")
    reference = sm70_hc_down_gemv_silu(x, weight, 4)
    monkeypatch.setenv("SGLANG_SM70_HC_NATIVE", "1")
    output = sm70_hc_down_gemv_silu(x, weight, 4)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
