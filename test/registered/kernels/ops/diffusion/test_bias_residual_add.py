# SPDX-License-Identifier: Apache-2.0
"""Fused conv bias + residual add must equal aten's two bf16 adds bit for bit.

Regressions caught: a channel index that drifts in either layout (channels_last
4D/5D and NCHW/NCDHW), a single fp32 add without the intermediate bf16 rounding
(differs from the eager chain at the last bit), and predicates admitting strided
or misaligned operands.
"""

import pytest
import torch

from sglang.kernels.ops.diffusion import bias_residual_add, can_use_bias_residual_add
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def reference(y, bias, h):
    return (y + bias.view(1, -1, *([1] * (y.ndim - 2)))) + h


def to_layout(x, layout):
    if layout == "nchw":
        return x.contiguous()
    fmt = torch.channels_last if x.ndim == 4 else torch.channels_last_3d
    return x.contiguous(memory_format=fmt)


@pytest.mark.parametrize(
    "shape", [(1, 144, 1, 32, 32), (2, 288, 16, 16), (1, 8, 1, 8, 8), (1, 1152, 4, 4)]
)
@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("amplitude", [1e-3, 1.0, 100.0])
@torch.no_grad()
def test_bit_exact_and_layout_preserved(shape, layout, amplitude):
    torch.manual_seed(0)
    y = to_layout(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude, layout
    )
    h = to_layout(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * amplitude, layout
    )
    bias = (torch.randn(shape[1], device="cuda") * amplitude).to(torch.bfloat16)
    assert can_use_bias_residual_add(y, bias, h)
    out = bias_residual_add(y, bias, h)
    assert out.stride() == y.stride()
    assert torch.equal(out, reference(y, bias, h))


@torch.no_grad()
def test_predicates():
    y = torch.randn(1, 16, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    h = torch.randn_like(y)
    bias = torch.randn(16, device="cuda", dtype=torch.bfloat16)
    assert can_use_bias_residual_add(y, bias, h)
    yl = y.contiguous(memory_format=torch.channels_last_3d)
    assert not can_use_bias_residual_add(yl, bias, h)  # strides differ
    assert can_use_bias_residual_add(
        yl, bias, h.contiguous(memory_format=torch.channels_last_3d)
    )
    assert not can_use_bias_residual_add(y.float(), bias.float(), h.float())
    assert not can_use_bias_residual_add(y, bias[:8], h)
    odd = torch.randn(1, 6, 1, 8, 8, device="cuda", dtype=torch.bfloat16)
    assert can_use_bias_residual_add(
        odd, bias[:6], torch.randn_like(odd)
    )  # spatial % 8 == 0
    oddl = odd.contiguous(memory_format=torch.channels_last_3d)
    assert not can_use_bias_residual_add(oddl, bias[:6], oddl.clone())  # C % 8 != 0
    narrow = torch.randn(1, 16, 1, 3, 3, device="cuda", dtype=torch.bfloat16)
    assert not can_use_bias_residual_add(narrow, bias, torch.randn_like(narrow))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
