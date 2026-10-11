"""LM-head dispatch to the BF16 skinny GEMM, without a GPU.

bf16_lm_head_matmul and both production LM-head callers (target logits and the
DSpark draft projection) must call the kernel only when
SGLANG_ENABLE_BF16_SKINNY_LM_HEAD is set and bf16_skinny_supported accepts the
operands, and use torch.matmul otherwise. The kernel and its predicate are
stubbed; their GPU behavior is tested in
test/registered/kernels/ops/gemm/test_bf16_skinny_gemm.py.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.gemm import bf16_skinny_gemm as skinny
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsProcessor, bf16_lm_head_matmul
from sglang.srt.models.dspark import project_through_lm_head
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _target_logits(hidden, weight):
    processor = SimpleNamespace(use_fp32_lm_head=False, rl_on_policy_target=None)
    return LogitsProcessor._compute_lm_head(
        processor, hidden, SimpleNamespace(weight=weight)
    )


def _draft_logits(hidden, weight):
    return project_through_lm_head(
        hidden, SimpleNamespace(weight=weight, quant_method=None)
    )


CALLERS = {
    "bf16_lm_head_matmul": bf16_lm_head_matmul,
    "target logits": _target_logits,
    "draft projection": _draft_logits,
}


class TestBf16LmHeadDispatch(CustomTestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        # FP32 hidden states: the callers cast them to the weight dtype first.
        self.hidden = torch.randn(6, 256, generator=g)
        self.weight = torch.randn(1000, 256, generator=g).bfloat16()
        self.kernel_out = torch.full((6, 1000), 7.0, dtype=torch.bfloat16)

    def _call(self, caller, enabled, supported):
        with (
            envs.SGLANG_ENABLE_BF16_SKINNY_LM_HEAD.override(enabled),
            mock.patch.object(
                skinny, "bf16_skinny_supported", return_value=supported
            ) as predicate,
            mock.patch.object(
                skinny, "bf16_skinny_gemm", return_value=self.kernel_out
            ) as kernel,
        ):
            out = caller(self.hidden, self.weight)
        return out, predicate, kernel

    def test_enabled_and_supported_calls_the_kernel(self):
        for name, caller in CALLERS.items():
            with self.subTest(name):
                out, predicate, kernel = self._call(caller, True, True)
                kernel.assert_called_once()
                x, w = kernel.call_args.args
                self.assertTrue(torch.equal(x, self.hidden.bfloat16()))
                self.assertIs(w, self.weight)
                self.assertIs(out, self.kernel_out)

    def test_otherwise_uses_matmul(self):
        want = self.hidden.bfloat16() @ self.weight.T
        for name, caller in CALLERS.items():
            for enabled, supported in ((True, False), (False, True)):
                with self.subTest(name, enabled=enabled, supported=supported):
                    out, predicate, kernel = self._call(caller, enabled, supported)
                    kernel.assert_not_called()
                    self.assertEqual(predicate.called, enabled)
                    self.assertTrue(torch.equal(out, want))

    def test_cpu_operands_are_not_supported(self):
        self.assertFalse(
            skinny.bf16_skinny_supported(self.hidden.bfloat16(), self.weight)
        )


if __name__ == "__main__":
    unittest.main()
