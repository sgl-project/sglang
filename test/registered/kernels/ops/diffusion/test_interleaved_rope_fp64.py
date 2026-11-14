import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    fused_interleaved_rope_fp64,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def eager_interleaved_rope(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
    output[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
    return output


CASES = get_ci_test_range(
    [
        (1, 1, 1, 2),
        (1, 17, 3, 12),
        (2, 129, 5, 112),
        (2, 7800, 20, 112),
    ],
    [(1, 17, 3, 12), (2, 7800, 20, 112)],
)


@pytest.mark.parametrize("batch,seq_len,num_heads,head_dim", CASES)
def test_interleaved_rope_fp64_is_bit_exact(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn_like(q)
    cos = torch.randn(
        1,
        seq_len,
        1,
        head_dim,
        dtype=torch.float64,
        device="cuda",
        generator=generator,
    )
    sin = torch.randn_like(cos)

    q_out, k_out = fused_interleaved_rope_fp64(q, k, cos, sin)

    assert torch.equal(q_out, eager_interleaved_rope(q, cos, sin))
    assert torch.equal(k_out, eager_interleaved_rope(k, cos, sin))
    assert q_out.data_ptr() not in (q.data_ptr(), k.data_ptr())
    assert k_out.data_ptr() not in (q.data_ptr(), k.data_ptr())


def _assert_rope_bits(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> None:
    for actual, source in zip(fused_interleaved_rope_fp64(q, k, cos, sin), (q, k)):
        expected = eager_interleaved_rope(source, cos, sin)
        assert torch.equal(actual.isnan(), expected.isnan())
        non_nan = ~expected.isnan()
        assert torch.equal(
            actual.view(torch.int16)[non_nan],
            expected.view(torch.int16)[non_nan],
        )


@pytest.mark.parametrize("scale", [1.0, 1e-300, 1e300])
def test_interleaved_rope_fp64_special_values(scale: float) -> None:
    # All BF16 encodings exercise subnormals, signed zeros and nonfinite inputs.
    q = (
        torch.arange(65536, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 128, 1, 512)
    )
    k = q.flip(-1)
    generator = torch.Generator(device="cuda").manual_seed(42)
    angles = torch.randn(
        q.shape, device="cuda", dtype=torch.float64, generator=generator
    )
    _assert_rope_bits(q, k, angles.cos() * scale, angles.sin() * scale)


def test_interleaved_rope_fp64_cancellation_and_zero_signs() -> None:
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 2.0**-133, -(2.0**-133), 2.0**127],
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = values[:, None].expand(-1, 2).reshape(1, 7, 1, 2).contiguous()
    k = q.neg()
    for coefficient in [0.0, -0.0, 0.7, -0.7, float("inf"), float("nan")]:
        table = torch.full(q.shape, coefficient, device="cuda", dtype=torch.float64)
        _assert_rope_bits(q, k, table, table)


def test_interleaved_rope_fp64_rounds_through_fp32_at_bf16_ties() -> None:
    midpoints = 1 + (torch.arange(128, device="cuda", dtype=torch.float64) + 0.5) / 128
    table = midpoints[:, None].expand(-1, 6).clone()
    table[:, :2] = torch.nextafter(
        table[:, :2], torch.full_like(table[:, :2], -torch.inf)
    )
    table[:, 4:] = torch.nextafter(
        table[:, 4:], torch.full_like(table[:, 4:], torch.inf)
    )
    cos = table.reshape(1, 128, 1, 6)
    sin = torch.zeros_like(cos)
    q = torch.ones(cos.shape, device="cuda", dtype=torch.bfloat16)
    _assert_rope_bits(q, q.neg(), cos, sin)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
