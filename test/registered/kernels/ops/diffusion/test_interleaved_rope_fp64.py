import sys

import pytest
import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.diffusion import (
    can_use_interleaved_rope_fp64,
    fused_interleaved_rope_fp64,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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


def test_interleaved_rope_fp64_predicate_rejects_unsupported_inputs() -> None:
    q = torch.empty(1, 17, 3, 12, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    cos = torch.empty(1, 17, 1, 12, dtype=torch.float64, device="cuda")
    sin = torch.empty_like(cos)

    assert can_use_interleaved_rope_fp64(q, k, cos, sin)
    assert not can_use_interleaved_rope_fp64(q.flatten(), k, cos, sin)
    assert not can_use_interleaved_rope_fp64(q.float(), k, cos, sin)
    assert not can_use_interleaved_rope_fp64(q, k[..., ::2], cos, sin)
    assert not can_use_interleaved_rope_fp64(q, k, cos.float(), sin)
    assert not can_use_interleaved_rope_fp64(q, k, cos[..., :-2], sin[..., :-2])
    unaligned = torch.empty(q.numel() + 1, dtype=torch.bfloat16, device="cuda")[
        1:
    ].view_as(q)
    assert unaligned.is_contiguous()
    assert not can_use_interleaved_rope_fp64(unaligned, k, cos, sin)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
