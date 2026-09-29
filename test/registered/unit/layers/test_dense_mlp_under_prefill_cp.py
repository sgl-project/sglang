import unittest

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    SumGroup,
    TokenAxis,
    decoder_layer_edges,
    decoder_layer_sides,
    make_boundary,
    make_output_boundary,
)
from sglang.srt.layers.communicator import ops as comm_ops
from sglang.srt.layers.communicator.boundary import CpMoves
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDenseMlpUnderPrefillCP(CustomTestCase):
    """A TP-sharded dense MLP under prefill CP must gather tokens across CP ranks
    before its all-reduce, or CP pairs sum partial outputs of different tokens
    (issue #38019: Qwen3-32B emitted garbage that never reached EOS)."""

    def test_dense_mlp_gathers_across_cp(self):
        sides = decoder_layer_sides(
            axis_sizes={
                TokenAxis.ATTN_DP: 1,
                TokenAxis.ATTN_CP: 2,
                TokenAxis.ATTN_TP_SCATTER: 2,
            },
            ffn_on_local_rows=False,
            previous_on_local_rows=False,
            is_last_layer=False,
            attention_gathers_local_rows=False,
            ffn_group=SumGroup.TP,
            leaves_for_next_layer=False,
            leaves_for_reduce_scatter=False,
            leaves_for_reduce_scatterv=False,
        )
        # The TP group spans every CP rank, so the MLP takes every row.
        self.assertIn(TokenAxis.ATTN_CP, sides.attention.layout.sharded)
        self.assertNotIn(TokenAxis.ATTN_CP, sides.ffn.layout.sharded)
        moves = CpMoves(
            gather=comm_ops._mlp_input_gather_moe_cp,
            take_back=comm.CommunicateSummableTensorPairFn._scatter_hidden_states_moe,
        )
        edges = decoder_layer_edges(sides)
        step = make_boundary(edges.into_ffn, cp_moves=moves).prepare.keywords["step"]
        self.assertIs(step.func, comm_ops._mlp_input_gather_moe_cp)
        out = make_output_boundary(edges.out_of_ffn, cp_moves=moves)
        self.assertIs(out.output_move, moves.take_back)


if __name__ == "__main__":
    unittest.main()
