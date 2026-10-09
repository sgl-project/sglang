"""A Qwen4-exp PLE layer reads its input on full attention rows."""

import unittest
from types import SimpleNamespace

import test_declared_decoder_boundary as fixture

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.contracts import BatchVariant
from sglang.srt.layers.layer_boundary.layout import TokenAxis
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.models.qwen4_exp import _build_qwen4_exp_stages, _has_ple
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _residual_ops(*, attn_reads_every_row):
    def unused(*args, **kwargs):
        raise AssertionError("construction only")

    return GatedResidualState(
        expand=unused,
        attn_mix=unused,
        ffn_mix=unused,
        attn_combine=unused,
        ffn_combine=unused,
        attn_reads_every_row=attn_reads_every_row,
    ).residual_ops()


def _sliced_over_attention_tp(layout):
    return TokenAxis.ATTN_TP in layout.sharded


class TestQwen4ExpPleRows(CustomTestCase):
    def test_the_ffn_before_a_ple_layer_hands_on_full_rows(self):
        # An all-to-all MoE with attention TP keeps the residual on this
        # rank's slice of the rows between layers; the PLE embedding is
        # computed for every row, so its layer must read the full rows.
        config = SimpleNamespace(num_hidden_layers=4, ple_layer_ids=[3])
        with (
            fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=2), a2a=True),
            layer_stack(),
        ):
            layers = [
                _build_qwen4_exp_stages(
                    _residual_ops(attn_reads_every_row=_has_ple(layer_id, config)),
                    sparse=True,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        ordinary = BatchVariant.ORDINARY
        ffn_hands_on = [
            _sliced_over_attention_tp(ffn.plan.edges[ordinary].outgoing.residual_to)
            for _, ffn in layers
        ]
        attn_reads = [
            _sliced_over_attention_tp(attn.plan.edges[ordinary].incoming.residual)
            for attn, _ in layers
        ]
        # Layer 2 has the PLE: layer 1's FFN gathers the rows back, the last
        # layer's FFN ends the stack on attention rows anyway.
        self.assertEqual(ffn_hands_on, [True, False, True, False])
        self.assertEqual(attn_reads, [False, True, False, True])


if __name__ == "__main__":
    unittest.main()
