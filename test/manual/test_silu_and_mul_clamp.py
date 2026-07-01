"Tests for silu_and_mul_clamp"

import pytest
import torch

from sglang.jit_kernel.dsv4.moe import silu_and_mul_clamp, silu_and_mul_clamp_torch
from sglang.srt.utils import get_device


@pytest.mark.parametrize("M", [16, 128])
@pytest.mark.parametrize("H", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul_clamp(M, H, dtype):
    "Test silu_and_mul_clamp against reference"
    torch.manual_seed(42)
    swiglu_limit = 10.0

    inp = torch.randn((M, 2 * H), dtype=dtype, device=get_device())
    out = torch.randn((M, H), dtype=dtype, device=get_device())
    ref_out = torch.zeros_like(out)

    silu_and_mul_clamp(inp, out, swiglu_limit)
    silu_and_mul_clamp_torch(inp, ref_out, swiglu_limit)

    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
