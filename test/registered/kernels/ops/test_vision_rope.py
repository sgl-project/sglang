import pytest
import torch

from sglang.kernels.ops.attention.vision_rope import apply_fused_qk_complex_rope
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def native_complex_rope(q, k, freqs_cis):
    freqs_cis = freqs_cis.unsqueeze(-2)
    q_complex = torch.view_as_complex(q.float().view(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().view(*k.shape[:-1], -1, 2))
    q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


@pytest.mark.parametrize("tokens", [1, 480, 5660, 8360])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_qk_complex_rope_matches_native(tokens, dtype):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The fused vision RoPE path targets Hopper or newer NVIDIA GPUs")

    qkv = torch.randn(tokens, 3, 12, 128, dtype=dtype, device="cuda")
    q, k, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(tokens, 64, dtype=torch.float32, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    expected_q, expected_k = native_complex_rope(q, k, freqs_cis)
    actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs_cis)

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)
    assert actual_q.is_contiguous()
    assert actual_k.is_contiguous()


def test_fused_qk_complex_rope_cuda_graph_replay():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("The fused vision RoPE path targets Hopper or newer NVIDIA GPUs")

    qkv = torch.randn(480, 3, 12, 128, dtype=torch.bfloat16, device="cuda")
    q, k, _ = torch.unbind(qkv, dim=1)
    angles = torch.randn(480, 64, dtype=torch.float32, device="cuda")
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    expected_q, expected_k = native_complex_rope(q, k, freqs_cis)

    apply_fused_qk_complex_rope(q, k, freqs_cis)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs_cis)
    graph.replay()

    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_k, expected_k)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
