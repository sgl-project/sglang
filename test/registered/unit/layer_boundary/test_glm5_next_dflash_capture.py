"""Regression for DFLASH aux-hidden capture on mHC models.

GLM-5.3-Flash runs with mhc=True. Its MHC boundary folds the residual
into the widened hidden state and returns residual=None, so CUDA-graph
capture used to crash on `hidden_states + residual`. DFLASH also has to
contract that widened state back to the draft hidden size; skipping the
contract is a silent shape/quality bug the crash-guard alone would miss.

Under a MoE that runs on each rank's attention-TP rows, the capture is gathered
over attention TP only for a batch whose captured rows are on those slices.
"""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.layer_boundary import (
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.glm5_next import Glm5NextModel
from sglang.test.boundary_fixtures import Norm, parallel_of, planning
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGlm5NextDflashCapture(CustomTestCase):
    def test_dflash_contracts_mhc_hidden_state_without_residual(self):
        model = Glm5NextModel.__new__(Glm5NextModel)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(mhc=True, hc_mult=4)
        model.dflash_capture = True

        hidden_states = torch.arange(24, dtype=torch.float32).reshape(2, 12)

        captured = ResidualStream().snapshot(hidden_states)
        actual = model._prepare_aux_hidden_state(captured)
        expected = hidden_states.unflatten(-1, (4, -1)).mean(dim=-2)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(tuple(actual.shape), (2, 3))

    def test_capture_after_a_rank_row_moe_gathers_only_split_rows(self):
        def batch(rows):
            return SimpleNamespace(
                input_ids=torch.ones(rows),
                forward_mode=ForwardMode.EXTEND,
                attn_cp_metadata=None,
            )

        # rows -> whether the second layer's input capture is on attention-TP
        # slices; rows that do not divide over attention TP stay whole only
        # when --disable-attn-tp-gather lets such a batch through.
        cases = {True: {5: False, 8: True, 13: False}, False: {5: True, 8: True}}
        for disabled, expected in cases.items():
            parallel = parallel_of(
                attn_dp=1, attn_tp=4, disable_attn_tp_gather=disabled
            )
            with planning(parallel, a2a=True, boundary_reduction="ar"):
                with layer_stack():
                    stages = [
                        stage
                        for _ in range(2)
                        for stage in append_stages(
                            (declare_attn(), Norm()),
                            (declare_ffn(sparse=True), Norm()),
                        )
                    ]
                layer = SimpleNamespace(attn_boundary=stages[2])
                for rows, on_slices in expected.items():
                    with self.subTest(disable_attn_tp_gather=disabled, rows=rows):
                        self.assertIs(
                            Glm5NextModel._captures_attn_tp_slices(
                                layer=layer, forward_batch=batch(rows)
                            ),
                            on_slices,
                        )


if __name__ == "__main__":
    unittest.main()
