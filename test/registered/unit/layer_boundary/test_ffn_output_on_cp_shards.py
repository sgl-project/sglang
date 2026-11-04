"""An FFN declaring output_on_cp_shards (Kimi-K3.1's latent MoE under CP-TP group
sharing): on a CP extend whose input it takes gathered over attention CP, its
output lands on this rank's CP rows, the residual's, so the exit keeps it as is
instead of taking a block back. Every other path keeps its contract."""

import unittest

import test_declared_decoder_boundary as fixture

from sglang.srt.layers.layer_boundary import TokenAxis
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.factories import declare_attn, declare_ffn
from sglang.srt.layers.layer_boundary.ops import attn_cp_take_back_output, keep_output
from sglang.test.boundary_fixtures import build_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CP = BatchVariant.CONTEXT_PARALLEL
ORDINARY = BatchVariant.ORDINARY


def _ffn(parallel, *, on_cp_shards=True, a2a=False):
    with fixture.planning(parallel, dsa_cp=True, a2a=a2a):
        _, ffn = build_stages(
            (declare_attn(), fixture.Norm()),
            (
                declare_ffn(
                    sparse=True,
                    output_complete=True,
                    output_on_cp_shards=on_cp_shards,
                ),
                fixture.Norm(),
            ),
            previous=declare_ffn(sparse=True),
        )
    return ffn


def _sharing(**overrides):
    fields = dict(
        attn_dp=1,
        attn_tp=1,
        attn_cp=2,
        enable_prefill_cp=True,
        enable_cp_tp_group_sharing=True,
    )
    return fixture.parallel_of(**(fields | overrides))


class TestFfnOutputOnCpShards(CustomTestCase):
    def test_cp_extend_output_stays_on_the_residual_rows(self):
        ffn = _ffn(_sharing())
        edges = ffn.plan.edges[CP]
        self.assertNotIn(TokenAxis.ATTN_CP, edges.incoming.need.layout.sharded)
        self.assertEqual(edges.outgoing.produced.layout, edges.outgoing.residual_to)
        self.assertIs(ffn.plan.paths[CP].output_move, keep_output)
        self.assertTrue(ffn.plan.outputs_on_cp_shards(CP))

    def test_without_the_declaration_the_exit_takes_the_block_back(self):
        ffn = _ffn(_sharing(), on_cp_shards=False)
        self.assertIs(ffn.plan.paths[CP].output_move, attn_cp_take_back_output)
        self.assertFalse(ffn.plan.outputs_on_cp_shards(CP))

    def test_an_ordinary_batch_keeps_its_contract(self):
        declared, plain = _ffn(_sharing()), _ffn(_sharing(), on_cp_shards=False)
        self.assertEqual(
            declared.plan.edges[ORDINARY].outgoing.produced,
            plain.plan.edges[ORDINARY].outgoing.produced,
        )
        self.assertIs(
            declared.plan.paths[ORDINARY].output_move,
            plain.plan.paths[ORDINARY].output_move,
        )
        self.assertFalse(declared.plan.outputs_on_cp_shards(ORDINARY))

    def test_a_moe_on_its_own_rows_keeps_its_contract(self):
        # An a2a MoE computes on this rank's rows already: nothing to put back.
        parallel = _sharing(attn_dp=2)
        declared = _ffn(parallel, a2a=True)
        plain = _ffn(parallel, a2a=True, on_cp_shards=False)
        self.assertEqual(
            declared.plan.edges[CP].outgoing.produced,
            plain.plan.edges[CP].outgoing.produced,
        )
        self.assertFalse(declared.plan.outputs_on_cp_shards(CP))

    def test_requires_a_complete_sparse_output(self):
        for kwargs in (
            dict(sparse=True),
            dict(sparse=False, output_complete=True),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                declare_ffn(output_on_cp_shards=True, **kwargs)


if __name__ == "__main__":
    unittest.main()
