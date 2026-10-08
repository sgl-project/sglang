# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.kda_kernels.residual_gate_add_jit import (
    can_use_residual_gate_add_cuda,
    residual_gate_add,
)


class TestResidualGateAddHip(unittest.TestCase):
    def test_hip_skips_cuda_ptx_fast_path(self):
        with patch(
            "sglang.kernels.kda_kernels.residual_gate_add_jit.is_hip",
            return_value=True,
        ):
            self.assertFalse(can_use_residual_gate_add_cuda(None, None, None))

    def test_hip_residual_gate_add_matches_eager(self):
        residual = torch.randn(2, 8, dtype=torch.float16)
        update = torch.randn(2, 8, dtype=torch.float16)
        gate = torch.randn(1, 8, dtype=torch.float16)
        with patch(
            "sglang.kernels.kda_kernels.residual_gate_add_jit.is_hip",
            return_value=True,
        ):
            out = residual_gate_add(residual, update, gate)
        torch.testing.assert_close(out, residual + update * gate)


if __name__ == "__main__":
    unittest.main()
