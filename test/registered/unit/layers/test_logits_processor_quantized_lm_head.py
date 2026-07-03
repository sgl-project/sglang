from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestQuantizedLmHead(CustomTestCase):
    def test_quantized_lm_head_uses_quant_method(self):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        hidden_states = torch.randn(2, 8)
        expected = torch.randn(2, 16)
        quant_method = MagicMock()
        quant_method.apply.return_value = expected
        lm_head = SimpleNamespace(quant_method=quant_method)

        actual = processor._compute_lm_head(hidden_states, lm_head)

        self.assertIs(actual, expected)
        quant_method.apply.assert_called_once_with(lm_head, hidden_states, None)


if __name__ == "__main__":
    import unittest

    unittest.main()
