import sys

import pytest
import torch

from sglang.kernels.ops.activation.activation import silu_and_mul as jit_silu_and_mul
from sglang.kernels.ops.quantization.per_token_quant_fp8 import (
    per_token_quant_fp8,
    silu_and_mul_per_token_quant_fp8,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _run_fused(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k2 = x.shape
    k = k2 // 2
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    silu_and_mul_per_token_quant_fp8(x, q, s)
    return q, s


def _run_unfused(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, k2 = x.shape
    k = k2 // 2
    act = torch.empty((m, k), dtype=x.dtype, device=x.device)
    major, _ = torch.cuda.get_device_capability(x.device)
    max_vec = 16 if major >= 10 else 8
    if k % max_vec == 0:
        jit_silu_and_mul(x, act)
    else:
        gate, up = x.chunk(2, dim=-1)
        act = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    per_token_quant_fp8(act, q, s)
    return q, s


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def _warp_dispatch_num_tokens() -> int:
    return torch.cuda.get_device_properties(0).multi_processor_count * 16 + 3


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_dim",
    [
        # Small & medium decode CTA register-cached (256 threads)
        (1, 1024),
        (7, 4096),
        (32, 12288),
        # Large hidden_dim CTA register-cached (512 threads, e.g. Qwen3.5-27B / Llama-70B)
        (1, 17408),
        (16, 17408),
        (64, 28672),
        # Very large hidden_dim CTA 2-pass fallback (> 32768)
        (4, 36864),
        # Non-8/16-aligned hidden_dim (vec4 fallback)
        (13, 1076),
        (13, 1368),
    ],
)
def test_silu_and_mul_per_token_quant_fp8_parity(dtype, num_tokens, hidden_dim):
    """Fused kernel must match unfused silu_and_mul + per_token_quant_fp8."""
    torch.manual_seed(2026)
    x = torch.randn((num_tokens, hidden_dim * 2), dtype=dtype, device="cuda")

    actual_q, actual_s = _run_fused(x)
    expected_q, expected_s = _run_unfused(x)

    major, _ = torch.cuda.get_device_capability(x.device)
    max_vec = 16 if major >= 10 else 8
    if hidden_dim % max_vec == 0:
        _assert_bitwise_equal(actual_s, expected_s)
        _assert_bitwise_equal(actual_q, expected_q)
    else:
        # sgl_kernel fallback uses a slightly different exp approximation than JIT silu
        torch.testing.assert_close(actual_s, expected_s, rtol=1e-2, atol=1e-4)
        actual_dq = actual_q.float() * actual_s
        expected_dq = expected_q.float() * expected_s
        torch.testing.assert_close(actual_dq, expected_dq, rtol=1e-1, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_per_token_quant_fp8_warp_dispatch_bit_exact(dtype):
    """Large-M warp-per-token register-cached path must be bit-exact."""
    torch.manual_seed(2027)
    num_tokens = _warp_dispatch_num_tokens()
    x = torch.randn((num_tokens, 2048 * 2), dtype=dtype, device="cuda")

    actual_q, actual_s = _run_fused(x)
    expected_q, expected_s = _run_unfused(x)

    _assert_bitwise_equal(actual_s, expected_s)
    _assert_bitwise_equal(actual_q, expected_q)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dispatch", ["cta", "warp"])
def test_silu_and_mul_per_token_quant_fp8_zero_rows_no_nan(dtype, dispatch):
    """All-zero rows must produce finite zero output and zero scale (no NaNs)."""
    num_tokens = 4 if dispatch == "cta" else _warp_dispatch_num_tokens()
    x = torch.zeros((num_tokens, 1024 * 2), dtype=dtype, device="cuda")

    q, s = _run_fused(x)

    assert not torch.isnan(q.float()).any()
    assert not torch.isnan(s).any()
    assert torch.all(q.float() == 0.0)
    assert torch.all(s == 0.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_per_token_quant_fp8_cuda_graph_replay(dtype):
    """CUDA Graph capture and replay with varying amplitudes must match unfused."""
    torch.manual_seed(2028)
    x = torch.randn((16, 17408 * 2), dtype=dtype, device="cuda")
    q = torch.empty((16, 17408), dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty((16, 1), dtype=torch.float32, device="cuda")

    # Warmup JIT compilation outside capture
    silu_and_mul_per_token_quant_fp8(x, q, s)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        silu_and_mul_per_token_quant_fp8(x, q, s)

    for scale_factor in (0.1, 1.0, 5.0):
        x.normal_().mul_(scale_factor)
        graph.replay()
        expected_q, expected_s = _run_unfused(x)
        _assert_bitwise_equal(s, expected_s)
        _assert_bitwise_equal(q, expected_q)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
