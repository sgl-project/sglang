import unittest

import test_declared_decoder_boundary as fixture

from sglang.srt.layers.layer_boundary import (
    TokenAxis,
)
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.ops import moe_cp_take_back_output
from sglang.test.boundary_fixtures import make_test_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDenseMlpUnderPrefillCP(CustomTestCase):
    """A TP-sharded dense MLP under prefill CP must gather tokens across CP ranks
    before its all-reduce, or CP pairs sum partial outputs of different tokens
    (issue #38019: Qwen3-32B emitted garbage that never reached EOS)."""

    def test_dense_mlp_gathers_across_cp(self):
        parallel = fixture.parallel_of(
            attn_dp=1, attn_tp=2, attn_cp=2, enable_prefill_cp=True
        )
        with fixture.planning(parallel):
            stages = make_test_stages(
                attention_norm=fixture.Norm(), ffn_norm=fixture.Norm()
            )
        attention = stages.attn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL).entry
        steps = stages.ffn.plan.paths.get(BatchVariant.CONTEXT_PARALLEL)
        self.assertIn(TokenAxis.ATTN_CP, attention.input_rows.sharded)
        self.assertNotIn(TokenAxis.ATTN_CP, steps.entry.input_rows.sharded)
        self.assertIs(
            steps.entry.prepare.keywords["step"].func, comm_ops._then_moe_cp_gather
        )
        self.assertIs(
            steps.output_move,
            moe_cp_take_back_output,
        )


if __name__ == "__main__":
    unittest.main()
