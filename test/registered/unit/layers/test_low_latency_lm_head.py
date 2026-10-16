"""DSV4.1 low-latency LM head: which calls take FlashInfer's CuTe DSL mm_bf16, with
what operands, and that the logits processor dispatches to it."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.models.dsv41_low_latency import lm_head as ll
from sglang.srt.models.dsv41_low_latency import low_m_gemm
from sglang.test.test_utils import CustomTestCase


def _head(dtype=torch.bfloat16, quant_method=None):
    head = nn.Module()
    head.weight = nn.Parameter(torch.randn(64, 32).to(dtype))
    head.quant_method = quant_method or UnquantizedEmbeddingMethod()
    return head


class TestLowLatencyLmHead(CustomTestCase):
    def setUp(self):
        self.mm_bf16 = MagicMock(return_value="ll")
        fake = types.ModuleType("flashinfer")
        fake.mm_bf16 = self.mm_bf16
        for patcher in (
            patch.dict(sys.modules, {"flashinfer": fake}),
            patch.object(low_m_gemm, "_use_pdl", return_value=True),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_eligible_call_passes_column_major_weight(self):
        head = _head()
        x = torch.zeros(6, 32, dtype=torch.bfloat16)
        self.assertEqual(ll.LowLatencyLmHeadMethod().apply(head, x), "ll")
        args, kwargs = self.mm_bf16.call_args
        # Detached views of the operands (DLPack refuses tensors that require grad).
        self.assertEqual(args[0].data_ptr(), x.data_ptr())
        self.assertFalse(args[1].requires_grad)
        # mm_bf16 wants B as (K, N) column-major: the (N, K) weight transposed, not copied.
        self.assertEqual(args[1].shape, (32, 64))
        self.assertEqual(args[1].data_ptr(), head.weight.data_ptr())
        self.assertTrue(args[1].T.is_contiguous())
        self.assertEqual(kwargs, {"pdl": True, "backend": "cute-dsl"})

    def test_ineligible_calls_take_the_stock_linear(self):
        bf16 = torch.bfloat16
        head = _head()
        cases = {
            "17 rows": (torch.randn(17, 32, dtype=bf16), None),
            "fp32 hidden": (torch.randn(6, 32), None),
            "non-contiguous": (torch.randn(32, 6, dtype=bf16).T, None),
            "3-D": (torch.randn(1, 6, 32, dtype=bf16), None),
            "bias": (torch.randn(6, 32, dtype=bf16), torch.randn(64, dtype=bf16)),
        }
        for name, (x, bias) in cases.items():
            with self.subTest(name):
                got = ll.LowLatencyLmHeadMethod().apply(head, x, bias)
                want = F.linear(x.to(bf16), head.weight, bias)
                self.assertTrue(torch.equal(got, want))
        self.mm_bf16.assert_not_called()

    def test_autotune_covers_every_bucket_up_to_16(self):
        head = _head(quant_method=ll.LowLatencyLmHeadMethod())
        ll.autotune(head)
        rows = [call.args[0].shape[0] for call in self.mm_bf16.call_args_list]
        self.assertEqual(rows, [1, 2, 4, 8, 16])

    def test_autotune_skips_an_uninstalled_head(self):
        ll.autotune(_head())
        self.mm_bf16.assert_not_called()


class TestInstall(CustomTestCase):
    def test_installs_on_an_unquantized_bf16_head(self):
        head = _head()
        with patch.object(ll, "cutedsl_low_m_supported", return_value=True):
            ll.install(head)
        self.assertIsInstance(head.quant_method, ll.LowLatencyLmHeadMethod)

    def test_keeps_the_head_otherwise(self):
        class _QuantizedHead(UnquantizedEmbeddingMethod):
            pass

        cases = {
            "fp32 weight": (_head(dtype=torch.float32), True),
            "other quant method": (_head(quant_method=_QuantizedHead()), True),
            "no CuTe DSL mm_bf16": (_head(), False),
        }
        for name, (head, supported) in cases.items():
            with self.subTest(name):
                method = head.quant_method
                with patch.object(
                    ll, "cutedsl_low_m_supported", return_value=supported
                ):
                    ll.install(head)
                self.assertIs(head.quant_method, method)

    def test_logits_processor_dispatches_to_it(self):
        # The stock unquantized head bypasses quant_method.apply; this one must not.
        head = _head(quant_method=ll.LowLatencyLmHeadMethod())
        self.assertTrue(should_apply_lm_head_quant_method(head, head.quant_method))


if __name__ == "__main__":
    unittest.main()
