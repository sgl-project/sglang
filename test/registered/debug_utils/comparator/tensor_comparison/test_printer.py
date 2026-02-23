import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.tensor_comparison.compare import compare_tensors
from sglang.srt.debug_utils.comparator.tensor_comparison.printer import (
    print_comparison,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestPrintComparison:
    def test_no_error(self):
        x = torch.randn(5, 5)
        y = x + torch.randn(5, 5) * 0.01
        info = compare_tensors(x_baseline=x, x_target=y, name="printer_test")

        print_comparison(info=info, diff_threshold=1e-3)

    def test_shape_mismatch(self):
        x = torch.randn(3, 4)
        y = torch.randn(5, 6)
        info = compare_tensors(x_baseline=x, x_target=y, name="mismatch_print")

        print_comparison(info=info, diff_threshold=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
