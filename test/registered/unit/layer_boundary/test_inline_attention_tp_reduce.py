"""An inline attention TP sum must not be repeated by the next boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.layer_boundary.contracts import BatchVariant, StageKind
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.layers.layer_boundary.stage import StageBoundary
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestInlineAttentionTpReduce(unittest.TestCase):
    def _boundary(self, *, group=SumGroup.ATTN_TP, variant=BatchVariant.ORDINARY):
        produced = SimpleNamespace(always_partial=True, group=group, update=PLAIN_ADD)
        plan = SimpleNamespace(
            finishes_directly=True,
            path_for=lambda _: SimpleNamespace(output=produced),
            variant_for=lambda _: variant,
        )
        return StageBoundary(
            plan, declaration=SimpleNamespace(kind=StageKind.ATTENTION)
        )

    def test_completed_sum_is_not_declared_again(self):
        boundary = self._boundary()
        fb = SimpleNamespace(residual_stream=ResidualStream(torch.zeros(2, 4)))
        output = torch.ones(2, 4)
        with (
            patch(
                "sglang.srt.layers.layer_boundary.stage.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1, attn_cp_size=1),
            ),
            patch(
                "sglang.srt.layers.layer_boundary.stage.get_forward",
                return_value=SimpleNamespace(attn_input_scattered=False),
            ),
        ):
            self.assertTrue(boundary.can_inline_attn_tp_reduce(fb))
            self.assertIs(boundary.finish(output, fb, already_reduced=True), output)
        self.assertIsNone(fb.residual_stream.pending.owed)

    def test_partial_sum_is_still_declared(self):
        boundary = self._boundary()
        fb = SimpleNamespace(residual_stream=ResidualStream(torch.zeros(2, 4)))
        boundary.finish(torch.ones(2, 4), fb)
        self.assertIsNotNone(fb.residual_stream.pending.owed)

    def test_other_paths_reject_inline_sum(self):
        fb = SimpleNamespace(residual_stream=ResidualStream(torch.zeros(2, 4)))
        with (
            patch(
                "sglang.srt.layers.layer_boundary.stage.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1, attn_cp_size=1),
            ),
            patch(
                "sglang.srt.layers.layer_boundary.stage.get_forward",
                return_value=SimpleNamespace(attn_input_scattered=False),
            ),
        ):
            for boundary in (
                self._boundary(group=SumGroup.TP),
                self._boundary(variant=BatchVariant.INPUT_SCATTERED),
            ):
                with self.assertRaisesRegex(RuntimeError, "cannot be reduced inline"):
                    boundary.finish(torch.ones(2, 4), fb, already_reduced=True)


if __name__ == "__main__":
    unittest.main()
