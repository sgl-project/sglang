import unittest

from sglang.srt.true_on_policy import log_softmax
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_ROW_KERNELS = (log_softmax._tile_stats, log_softmax._write_logprobs)


class TestRowCountIsRuntime(unittest.TestCase):
    def test_row_count_and_stride_are_runtime_arguments(self):
        """A constexpr row count compiles both kernels again for every new decode batch or response chunk."""
        for kernel in _ROW_KERNELS:
            params = {param.name: param for param in kernel.params}
            with self.subTest(kernel.__name__):
                self.assertFalse(params["M"].is_constexpr)
                self.assertTrue(params["M"].do_not_specialize)
                self.assertFalse(params["STRIDE"].is_constexpr)


if __name__ == "__main__":
    unittest.main()
