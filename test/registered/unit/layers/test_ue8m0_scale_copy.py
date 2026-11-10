"""UE8M0 scale bytes must be copied as bits, not as uint8 magnitudes.

Official DSV4.1-Flash stores MXFP8 scales as ``float8_e8m0fnu``. SM70
allocates those params as ``uint8``. ``Tensor.copy_`` between those dtypes
is a numeric cast (2^(e-127) → 0 or 1), which unpacks to all-zero weights.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    copy_with_check,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    prepare_mxfp8_layer_for_sm70_marlin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUe8m0ScaleCopy(CustomTestCase):
    def test_naive_copy_collapses_ue8m0_bytes(self):
        raw = torch.tensor([113, 127, 140], dtype=torch.uint8)
        src = raw.view(torch.float8_e8m0fnu)
        dst = torch.zeros(3, dtype=torch.uint8)
        dst.copy_(src)
        self.assertFalse(torch.equal(dst, raw))
        self.assertEqual(int(dst[1]), 1)

    def test_copy_with_check_preserves_ue8m0_bytes(self):
        raw = torch.tensor([113, 127, 140], dtype=torch.uint8)
        src = raw.view(torch.float8_e8m0fnu)
        dst = torch.zeros(3, dtype=torch.uint8)
        copy_with_check(dst, src)
        self.assertTrue(torch.equal(dst, raw))

    def test_row_parallel_scale_load_preserves_ue8m0(self):
        full = torch.tensor([[113, 127, 140, 141]], dtype=torch.uint8).view(
            torch.float8_e8m0fnu
        )
        param = BlockQuantScaleParameter(
            data=torch.zeros(1, 2, dtype=torch.uint8),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *a, **k: None,
        )
        param.load_row_parallel_weight(full, tp_rank=1, use_presharded_weights=False)
        self.assertTrue(
            torch.equal(param.data, torch.tensor([[140, 141]], dtype=torch.uint8))
        )

    def test_numeric_scale_copy_unpack_is_refused(self):
        n, k = 32, 64
        weight = torch.full((n, k), 2.0).to(torch.float8_e4m3fn)
        raw = torch.full((n, k // 32), 120, dtype=torch.uint8)
        dst = torch.zeros_like(raw)
        dst.copy_(raw.view(torch.float8_e8m0fnu))
        self.assertTrue(torch.equal(dst, torch.zeros_like(raw)))
        layer = torch.nn.Module()
        layer.weight_block_size = [1, 32]
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale_inv = torch.nn.Parameter(dst, requires_grad=False)
        layer.orig_dtype = torch.float16
        with self.assertRaisesRegex(RuntimeError, "all-zero"):
            prepare_mxfp8_layer_for_sm70_marlin(layer)


if __name__ == "__main__":
    unittest.main()
