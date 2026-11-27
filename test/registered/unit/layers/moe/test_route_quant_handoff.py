"""route_quant_handoff under torch.compile."""

import unittest

import torch

from sglang.srt.layers.moe import route_quant_handoff
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTakePaddedRowsMasked(CustomTestCase):
    def tearDown(self):
        route_quant_handoff.clear()

    def test_unset_flag_is_traceable(self):
        """Every biased grouped top-k call reads the flag; fullgraph capture
        failed on the msgspec.Struct write that reset an unset flag."""
        torch._dynamo.reset()

        def mask_rows(topk_ids: torch.Tensor) -> torch.Tensor:
            if route_quant_handoff.take_padded_rows_masked():
                return topk_ids
            return topk_ids + 1

        compiled = torch.compile(mask_rows, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(torch.zeros(3)), torch.ones(3))

    def test_set_flag_is_consumed_once(self):
        route_quant_handoff._handoff.padded_rows_masked = True
        self.assertTrue(route_quant_handoff.take_padded_rows_masked())
        self.assertFalse(route_quant_handoff.take_padded_rows_masked())


if __name__ == "__main__":
    unittest.main()
