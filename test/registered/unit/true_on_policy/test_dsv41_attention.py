"""TOP DeepSeek-V4.1 attention: what the single program refuses instead of running another one."""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.true_on_policy import dsv41_attention

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDsv41Attention(CustomTestCase):
    def test_compressor_refuses_the_fused_projection(self):
        """The fused wkv_gate layout belongs to the SM100 compress kernels, not the TOP projection."""
        compressor = SimpleNamespace(use_fused_gate=True, compress_ratio=2)
        with self.assertRaisesRegex(NotImplementedError, "separate wkv/wgate"):
            dsv41_attention.compressor_project(compressor, None)


if __name__ == "__main__":
    unittest.main()
