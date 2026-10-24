"""Diffusion linears must not reach aiter tgemm past its gfx1250 int32 byte-offset limit.

On gfx1250, tgemm writes wrong values, without faulting, to output rows past 2**31
bytes; at Wan2.2 720p that turned the last decoded video frame into noise. Off
gfx1250 the size bound is unchanged."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.layers import linear
from sglang.test.test_utils import CustomTestCase

# Wan2.2 A14B 720p FFN up-projection: 1.24e9 bf16 outputs = 2.49 GB.
M, N, K = 90000, 13824, 5120


def _route(is_gfx1250: bool, m: int) -> MagicMock:
    tgemm = MagicMock()
    x = torch.empty(m, K, dtype=torch.bfloat16, device="meta")
    w = torch.empty(N, K, dtype=torch.bfloat16, device="meta")
    with (
        patch.object(linear, "_USE_AITER_TGEMM", True),
        patch.object(linear, "_IS_GFX1250_TGEMM", is_gfx1250),
        # Isolate the tgemm bound from the large-M FlyDSL route.
        patch.object(linear, "_USE_FLYDSL_GFX1250_GEMM", False),
        patch.object(linear, "tgemm", tgemm, create=True),
    ):
        linear.apply_unquantized_linear(x, w)
    return tgemm


class TestTgemmSizeBound(CustomTestCase):
    def test_gfx1250_output_past_2gib_skips_tgemm(self):
        self.assertFalse(_route(is_gfx1250=True, m=M).mm.called)

    def test_gfx1250_small_output_keeps_tgemm(self):
        self.assertTrue(_route(is_gfx1250=True, m=512).mm.called)

    def test_other_archs_keep_tgemm_for_same_shape(self):
        self.assertTrue(_route(is_gfx1250=False, m=M).mm.called)


if __name__ == "__main__":
    unittest.main()
