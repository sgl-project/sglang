import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import (
    lm_head_logits_dtype,
    project_through_lm_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeFp8LinearMethod:
    def apply(self, layer, x, bias):
        return torch.matmul(x, layer.weight.to(x.dtype).T)


class TestLmHeadLogitsDtype(CustomTestCase):
    def test_declared_dtype_matches_projection(self):
        hidden = torch.randn(3, 8).to(torch.bfloat16)
        heads = {
            "bf16": SimpleNamespace(
                weight=torch.randn(16, 8).to(torch.bfloat16), quant_method=None
            ),
            "fp32": SimpleNamespace(weight=torch.randn(16, 8), quant_method=None),
            # A float-typed packed weight: the quant kernel emits the hidden's dtype.
            "fp8": SimpleNamespace(
                weight=torch.randn(16, 8).to(torch.float8_e4m3fn),
                quant_method=_FakeFp8LinearMethod(),
            ),
        }
        for name, head in heads.items():
            with self.subTest(head=name):
                self.assertEqual(
                    project_through_lm_head(hidden, head).dtype,
                    lm_head_logits_dtype(head, hidden.dtype),
                )


if __name__ == "__main__":
    unittest.main()
