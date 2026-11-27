"""TOP DeepSeek-V4.1 Engram: what the single program refuses instead of running another one."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.true_on_policy import dsv41_engram

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _gate_inputs(weight_dtype=torch.bfloat16):
    x = torch.zeros(2, 4, 8, dtype=torch.bfloat16)
    kv = torch.zeros(2, 40, dtype=torch.bfloat16)
    w = torch.ones(4, 8, dtype=weight_dtype)
    return x, kv, w, w


class TestDsv41Engram(CustomTestCase):
    def test_gate_refuses_the_torch_fallback(self):
        """Off CUDA, serving would take the torch gate, whose reduction order differs from the fused kernel's."""
        with self.assertRaisesRegex(NotImplementedError, "fused kernel"):
            dsv41_engram.gate(*_gate_inputs(), 1e-20, 1e-6)

    def test_gate_refuses_fp32_q_k_weights(self):
        """Serving holds BF16 q/k weights; an FP32 master would gate with other values."""
        x, kv, q, k = _gate_inputs(torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "BF16"):
            dsv41_engram.gate(x, kv, q, k, 1e-20, 1e-6)

    def test_packed_hash_refuses_batched_ids(self):
        with self.assertRaisesRegex(NotImplementedError, "packed 1-D"):
            dsv41_engram.hash_packed(None, torch.zeros(2, 3, dtype=torch.int64), [3, 3])


if __name__ == "__main__":
    unittest.main()
