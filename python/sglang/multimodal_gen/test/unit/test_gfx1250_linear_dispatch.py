# SPDX-License-Identifier: Apache-2.0
"""On gfx1250, bf16 linears whose output passes 2**31 bytes must stay correct.

aiter's tgemm writes wrong values, without faulting, to output rows past 2**31 bytes;
at Wan2.2 720p that turned the last decoded video frame into noise."""

import unittest

import pytest
import torch

from sglang.multimodal_gen.runtime.layers import linear
from sglang.srt.utils import is_gfx1250_supported

# Wan2.2 A14B 720p FFN up-projection: the output is 2.49 GB, past 2**31 bytes.
M, N, K = 90000, 13824, 5120


class TestGfx1250LargeOutputLinear(unittest.TestCase):
    def test_rows_past_2gib_output_match_fp32_reference(self):
        if not (linear._USE_AITER_TGEMM and is_gfx1250_supported()):
            pytest.skip("needs gfx1250 with SGLANG_USE_ROCM_AITER_TGEMM=1")
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / K**0.5
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        out = linear.apply_unquantized_linear(x, w, b)
        # Row 77672 is the first whose output byte offset crosses 2**31.
        rows = torch.cat([torch.arange(0, 128), torch.arange(M - 12000, M - 11872)])
        rows = rows.cuda()
        ref = x[rows].float() @ w.float().t() + b.float()
        torch.testing.assert_close(out[rows].float(), ref, atol=0.05, rtol=0.02)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
