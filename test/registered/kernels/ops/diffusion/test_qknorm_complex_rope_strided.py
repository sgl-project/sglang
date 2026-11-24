# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.diffusion.rope.qknorm_complex_rope_strided_triton import (
    can_use_qknorm_complex_rope_strided,
    qknorm_complex_rope_strided,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="base-a-jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _eager(x, weight, rope, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    normed = (x.float() * torch.rsqrt(variance + eps)).to(x.dtype) * weight
    z = torch.view_as_complex(normed.float().unflatten(-1, (-1, 2)))
    return torch.view_as_real(z * rope[None, :, None]).flatten(-2).to(x.dtype)


@pytest.mark.parametrize("seq_len,heads", [(2048, 32), (7, 4)])
def test_reads_heads_in_place_from_packed_projection(seq_len, heads):
    torch.manual_seed(0)
    qkv = torch.randn(1, seq_len, 3 * heads * 128, device="cuda", dtype=torch.bfloat16)
    q, k, _ = qkv.unflatten(-1, (3, heads, 128)).unbind(2)
    weight = torch.rand(128, device="cuda", dtype=torch.bfloat16) + 0.5
    angle = torch.rand(seq_len, 64, device="cuda") * 6.28
    rope = torch.polar(torch.ones_like(angle), angle)
    for x in (q, k):
        assert not x.is_contiguous()
        assert can_use_qknorm_complex_rope_strided(x, weight, rope)
        actual = qknorm_complex_rope_strided(x, weight, rope, 1e-6)
        assert actual.is_contiguous()
        # only the variance reduction order differs from the eager chain
        torch.testing.assert_close(
            actual, _eager(x.contiguous(), weight, rope, 1e-6), atol=3e-2, rtol=2e-2
        )


def test_rejects_unsupported_layouts():
    x = torch.randn(2, 7, 4, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(128, device="cuda", dtype=torch.bfloat16)
    rope = torch.ones(7, 64, device="cuda", dtype=torch.complex64)
    assert not can_use_qknorm_complex_rope_strided(x, weight, rope)
    assert not can_use_qknorm_complex_rope_strided(x[:1].transpose(2, 3), weight, rope)
    assert not can_use_qknorm_complex_rope_strided(x[:1], weight, rope[:6])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
