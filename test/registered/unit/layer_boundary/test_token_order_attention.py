"""Linear attention under CP-TP group sharing (Qwen4-Exp): on a CP extend the
boundary gathers its input in token order, and the FFN's input completes its sum
over the TP group and takes this rank's shard back."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import (
    EdgeContract,
    InputContract,
    Layout,
    OutputContract,
    ProducerReduction,
    TokenAxis,
    bind_entry,
    declare_attn,
    layer_stack,
    ops,
    prepare,
)
from sglang.srt.layers.layer_boundary.contracts import BatchVariant, CpMoves
from sglang.srt.layers.layer_boundary.layout import SumGroup
from sglang.srt.layers.layer_boundary.ops import (
    cp_gather_in_token_order,
    cp_take_back_in_token_order,
)
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.models import qwen4_exp
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CP = BatchVariant.CONTEXT_PARALLEL
ORDINARY = BatchVariant.ORDINARY


def _residual_ops():
    def unused(*args, **kwargs):
        raise AssertionError("construction only")

    return GatedResidualState(
        expand=unused,
        attn_mix=unused,
        ffn_mix=unused,
        attn_combine=unused,
        ffn_combine=unused,
    ).residual_ops()


def _step(stage, variant):
    return stage.plan.paths[variant].entry.prepare.keywords["step"]


def _build_layers(parallel):
    """A linear-attention layer, then a sparse-attention layer."""
    config = SimpleNamespace(
        model_type="qwen4_exp_text", num_hidden_layers=2, ple_layer_ids=[]
    )
    with (
        fixture.planning(parallel),
        get_parallel().override(
            enable_cp_tp_group_sharing=parallel.enable_cp_tp_group_sharing
        ),
        layer_stack(),
    ):
        linear = qwen4_exp._build_qwen4_exp_stages(
            config, 0, _residual_ops(), linear_attention=True
        )
        sparse = qwen4_exp._build_qwen4_exp_stages(config, 1, _residual_ops())
    return linear, sparse


class TestTokenOrderAttention(CustomTestCase):
    def setUp(self):
        parallel = fixture.parallel_of(
            attn_dp=1,
            attn_tp=1,
            attn_cp=4,
            enable_prefill_cp=True,
            enable_cp_tp_group_sharing=True,
        )
        self.linear, self.sparse = _build_layers(parallel)

    def test_linear_attention_reads_the_whole_sequence_in_token_order(self):
        attn, ffn = self.linear
        entry = attn.plan.paths[CP].entry
        self.assertIs(_step(attn, CP).func, prepare._update_read)
        self.assertIs(entry.input_move, cp_gather_in_token_order)
        # The FFN's input sums over the TP group (the attention-CP group under
        # sharing), takes this rank's shard back, writes it into the residual
        # and reads there, then gathers for the MoE.
        step = _step(ffn, CP)
        self.assertIs(step.func, prepare._move_before_read)
        take_back = step.keywords["move"]
        self.assertIs(take_back.func, cp_take_back_in_token_order)
        self.assertIs(take_back.keywords["group"], SumGroup.ATTN_CP)
        read = step.keywords["read_step"]
        self.assertIs(read.func, prepare._then_moe_cp_gather)
        self.assertIs(read.keywords["gather"].func, prepare._update_read)
        entry = ffn.plan.paths[CP].entry
        self.assertIs(entry.declared_sum, SumGroup.ATTN_CP)
        # A sum already completed for this batch only takes the shard back.
        completed = entry.prepare.keywords["completed_step"].keywords["move"]
        self.assertIsNone(completed.keywords["group"])

    def test_an_ordinary_batch_completes_the_tp_sum(self):
        _, ffn = self.linear
        step = _step(ffn, ORDINARY)
        self.assertIs(step.func, prepare._reduce_update_read)
        # The attention-CP group is the TP group under sharing.
        self.assertIs(step.keywords["group"], SumGroup.ATTN_CP)

    def test_sparse_attention_keeps_its_cp_shard(self):
        attn, ffn = self.sparse
        self.assertIsNone(attn.plan.paths[CP].entry.input_move)
        self.assertIsNone(ffn.plan.paths[CP].entry.declared_sum)
        self.assertIs(_step(ffn, CP).keywords["gather"].func, prepare._update_read)

    def test_without_sharing_linear_attention_binds_like_attention(self):
        parallel = fixture.parallel_of(attn_dp=1, attn_tp=2)
        linear, sparse = _build_layers(parallel)

        def steps(stage):
            entry = stage.plan.paths[ORDINARY].entry
            return entry.prepare.keywords["step"].func, entry.declared_sum

        self.assertEqual(
            [steps(stage) for stage in linear], [steps(stage) for stage in sparse]
        )

    def test_token_order_needs_an_always_partial_attention(self):
        with self.assertRaises(ValueError):
            declare_attn(in_token_order=True, reduction=ProducerReduction.EXIT_SCOPED)

    def test_only_a_token_order_output_is_taken_back(self):
        shard, full = Layout(frozenset({TokenAxis.ATTN_CP})), Layout(frozenset())
        moves = CpMoves(gather=prepare._then_moe_cp_gather, take_back=None)

        def edge(produced, need):
            return EdgeContract(
                produced=produced, need=need, residual=shard, residual_to=shard
            )

        token_order = OutputContract(
            full, group=SumGroup.TP, always_partial=True, in_token_order=True
        )
        # No model runs two token-order stages back to back.
        with self.assertRaises(NotImplementedError):
            bind_entry(
                edge(token_order, InputContract(full, in_token_order=True)),
                cp_moves=moves,
            )
        # Rows not in token order are taken back by the CP moves, which here
        # cannot complete a TP sum.
        rank_major = OutputContract(full, group=SumGroup.TP, always_partial=True)
        with self.assertRaises(NotImplementedError):
            bind_entry(edge(rank_major, InputContract(shard)), cp_moves=moves)

    def test_the_take_back_sums_then_shards(self):
        calls = []

        def sum_group(group):
            def all_reduce(hidden):
                calls.append(("sum", group))
                return hidden * 2

            return SimpleNamespace(all_reduce=all_reduce)

        def shard(hidden, forward_batch):
            calls.append(("shard",))
            return hidden[:2]

        residual = torch.ones(2, 3)
        with (
            patch.object(ops, "_sum_group", sum_group),
            patch.object(ops, "cp_shard_hidden_states", shard),
        ):
            hidden, kept = cp_take_back_in_token_order(
                torch.ones(4, 3), residual, None, group=SumGroup.ATTN_CP
            )
        self.assertEqual(calls, [("sum", SumGroup.ATTN_CP), ("shard",)])
        torch.testing.assert_close(hidden, torch.full((2, 3), 2.0))
        self.assertIs(kept, residual)


if __name__ == "__main__":
    unittest.main()
