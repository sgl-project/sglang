import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4 import silu_and_mul_clamp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="silu_and_mul_clamp requires CUDA",
)


def _reference(input: torch.Tensor, limit: float) -> torch.Tensor:
    gate, up = input.chunk(2, dim=-1)
    limit_tensor = torch.tensor(limit, dtype=input.dtype, device=input.device)
    gate = torch.minimum(gate, limit_tensor)
    up = torch.minimum(torch.maximum(up, -limit_tensor), limit_tensor)
    return (F.silu(gate.float()) * up.float()).to(input.dtype)


@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("hidden_size", [4096, 8192, 16384])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_clamp_matches_reference(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(hidden_size + num_tokens)
    input = (
        torch.randn(
            num_tokens,
            hidden_size * 2,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 8
    ).to(dtype)
    output = torch.full(
        (num_tokens, hidden_size),
        float("nan"),
        device="cuda",
        dtype=dtype,
    )

    silu_and_mul_clamp(input, output, 7.0)
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, _reference(input, 7.0), rtol=2e-2, atol=5e-2)


def test_silu_and_mul_clamp_rejects_mismatched_width() -> None:
    input = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 4104, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="input last dim must be 2"):
        silu_and_mul_clamp(input, output, 7.0)


def test_silu_and_mul_clamp_rejects_unaligned_width() -> None:
    input = torch.randn(1, 8200, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 4100, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="out_dim must be divisible by vector size"):
        silu_and_mul_clamp(input, output, 7.0)


def test_silu_and_mul_clamp_rejects_empty_width() -> None:
    input = torch.empty(1, 0, device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(input)

    with pytest.raises(RuntimeError, match="out_dim must be positive"):
        silu_and_mul_clamp(input, output, 7.0)


def test_silu_and_mul_clamp_rejects_output_dtype_mismatch() -> None:
    input = torch.randn(1, 8192, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(1, 4096, device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="Dtype value.*not in the allowed options"):
        silu_and_mul_clamp(input, output, 7.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
