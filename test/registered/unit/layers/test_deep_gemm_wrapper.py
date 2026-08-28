from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.deep_gemm_wrapper import entrypoint


class _FakeDeepGemm:
    def __init__(self):
        self.alignment = 128
        self.set_values = []

    def get_mk_alignment_for_contiguous_layout(self):
        return self.alignment

    def set_mk_alignment_for_contiguous_layout(self, value):
        self.alignment = value
        self.set_values.append(value)


class TestDeepGemmWrapper(unittest.TestCase):
    def test_contiguous_alignment_scope_restores_after_error(self):
        fake_deep_gemm = _FakeDeepGemm()
        with patch.object(entrypoint, "deep_gemm", fake_deep_gemm, create=True):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                with entrypoint.contiguous_layout_alignment_scope(32):
                    self.assertEqual(fake_deep_gemm.alignment, 32)
                    raise RuntimeError("failed")

        self.assertEqual(fake_deep_gemm.alignment, 128)
        self.assertEqual(fake_deep_gemm.set_values, [32, 128])


if __name__ == "__main__":
    unittest.main()
