"""The swizzled MXFP8 quantizer against ``flashinfer.mxfp8_quantize``: payload and
the swizzled E8M0 scale buffer, padding included, byte for byte."""

import sys

import pytest
import torch

from sglang.kernels.ops.quantization.mxfp8_swizzled_triton import (
    can_use_mxfp8_swizzled,
    mxfp8_quantize_swizzled,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.mxfp8 import assert_matches_flashinfer_mxfp8

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

ROWS, HIDDEN = 333, 5376  # rows and scale columns both need padding


def test_quantize_swizzled_is_byte_exact() -> None:
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn((ROWS, HIDDEN), device="cuda", generator=g) * 3).to(torch.bfloat16)
    x[0, :32] = 0  # an all-zero block takes the minimum exponent
    assert can_use_mxfp8_swizzled(x)
    assert_matches_flashinfer_mxfp8(mxfp8_quantize_swizzled(x), x)


def test_predicates_reject_unsupported_input() -> None:
    assert not can_use_mxfp8_swizzled(
        torch.randn(4, 64, device="cuda", dtype=torch.float16)
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
