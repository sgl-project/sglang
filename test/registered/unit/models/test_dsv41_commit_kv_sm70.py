"""SM70 DSpark commit-KV must not call a missing w8a8 kernel."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.speculative.dspark.dspark_draft_model import (
    _block_quant_stack_applies,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsv41CommitKvSm70(CustomTestCase):
    def test_block_quant_stack_requires_callable_w8a8(self):
        quant_method = SimpleNamespace(
            block_quant=True,
            w8a8_block_fp8_linear=None,
            quant_config=SimpleNamespace(weight_block_size=[128, 128]),
        )
        linear = SimpleNamespace(
            quant_method=quant_method,
            weight=torch.empty(128, 128, dtype=torch.float8_e4m3fn),
        )
        self.assertFalse(_block_quant_stack_applies(wkv_linears=[linear]))

        quant_method.w8a8_block_fp8_linear = lambda **kwargs: None
        self.assertTrue(_block_quant_stack_applies(wkv_linears=[linear]))


if __name__ == "__main__":
    unittest.main()
