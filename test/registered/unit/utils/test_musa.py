"""Regression tests for MUSA backend detection."""

import importlib.util
import unittest

import torch

from sglang.srt.utils.common import is_musa
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@unittest.skipIf(
    importlib.util.find_spec("torchada") is not None,
    "requires torchada to be unavailable",
)
class TestIsMusaTorchCompile(CustomTestCase):
    def test_fullgraph_compile_without_torchada(self):
        """Backend detection must not break a full graph without torchada."""

        def fn(x):
            if is_musa():
                return x + 1
            return x - 1

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        result = compiled(torch.tensor(1))
        self.assertEqual(result.item(), 0)


if __name__ == "__main__":
    unittest.main()
