"""CPU-only tests for safe DFlash CUDA graph batch-size selection."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

try:
    from sglang.srt.model_executor.cuda_graph_runner import (
        _get_safe_dflash_capture_bs,
    )
except ImportError:  # the runner/backend refactor removed this module
    raise unittest.SkipTest(
        "safe DFlash capture sizing is not provided by this tree"
    ) from None

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _spec_algorithm(is_dflash):
    return SimpleNamespace(is_dflash=lambda: is_dflash)


class TestDFlashCudaGraphCapture(unittest.TestCase):
    def test_hybrid_dflash_uses_every_exact_batch_size(self):
        hybrid_backend = SimpleNamespace(linear_attn_backend=object())

        capture_bs = _get_safe_dflash_capture_bs(
            [1, 2, 4], _spec_algorithm(True), hybrid_backend
        )

        self.assertEqual(capture_bs, [1, 2, 3, 4])

    def test_contiguous_hybrid_dflash_sizes_are_unchanged(self):
        hybrid_backend = SimpleNamespace(linear_attn_backend=object())

        capture_bs = _get_safe_dflash_capture_bs(
            [1, 2, 3, 4], _spec_algorithm(True), hybrid_backend
        )

        self.assertEqual(capture_bs, [1, 2, 3, 4])

    def test_non_dflash_hybrid_backend_keeps_requested_sizes(self):
        hybrid_backend = SimpleNamespace(linear_attn_backend=object())

        capture_bs = _get_safe_dflash_capture_bs(
            [1, 2, 4], _spec_algorithm(False), hybrid_backend
        )

        self.assertEqual(capture_bs, [1, 2, 4])

    def test_full_attention_dflash_keeps_requested_sizes(self):
        full_attention_backend = SimpleNamespace()

        capture_bs = _get_safe_dflash_capture_bs(
            [1, 2, 4], _spec_algorithm(True), full_attention_backend
        )

        self.assertEqual(capture_bs, [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
