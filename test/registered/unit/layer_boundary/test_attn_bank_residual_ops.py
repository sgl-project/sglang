"""A stack read through the attention-residual bank's stage reads computes
what the decoder computed when its MLP folded the pending add itself."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.layer_boundary import (
    BatchVariant,
    ExitRows,
    append_stages,
    declare_attn,
    declare_ffn,
    layer_stack,
)
from sglang.srt.layers.layer_boundary import prepare as comm_ops
from sglang.srt.layers.layer_boundary.ops import update_attn_tp_gather_output
from sglang.srt.layers.layer_boundary.residual import attn_bank
from sglang.srt.layers.layer_boundary.residual.add_norm import REPLACE_AT_EXIT
from sglang.srt.layers.layer_boundary.residual.attn_bank import (
    AttnBank,
    AttnBankOutputRead,
    AttnBankState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

TOKENS = 3
HIDDEN = 8
BLOCK = 2
LAYERS = 5


class _Norm(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.scale = 0.5 + seed / 7
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.scale


class _Proj(torch.nn.Module):
    def __init__(self, seed):
        super().__init__()
        self.weight = torch.randn(
            1, HIDDEN, generator=torch.Generator().manual_seed(seed)
        )

    def forward(self, x):
        return x @ self.weight.t(), None


def _mix(prefix_sum, bank, nvb, score_proj, score_norm):
    # The torch reference for the triton score + combine pair.
    return attn_residual.aggregate_stream_torch(
        prefix_sum, bank, nvb, score_proj, score_norm
    )


class _Layer:
    """One layer's parameters and stand-ins for its attention and its MLP."""

    def __init__(self, idx):
        self.writes_block = idx % BLOCK == 0
        self.attn_proj, self.attn_score_norm = _Proj(4 * idx), _Norm(4 * idx)
        self.ffn_proj, self.ffn_score_norm = _Proj(4 * idx + 1), _Norm(4 * idx + 1)
        self.input_norm, self.post_norm = _Norm(4 * idx + 2), _Norm(4 * idx + 3)
        self.attn_weight = 0.3 + 0.1 * idx
        self.mlp_weight = -0.2 + 0.05 * idx

    def attention(self, x):
        return torch.tanh(x) * self.attn_weight

    def mlp(self, x):
        return torch.sin(x) * self.mlp_weight

    def ops(self, bank):
        return AttnBankState(
            bank,
            self.attn_proj,
            self.attn_score_norm,
            self.ffn_proj,
            self.ffn_score_norm,
            writes_block=self.writes_block,
        ).residual_ops()


LAYER_LIST = [_Layer(i) for i in range(LAYERS)]
OUT_PROJ, OUT_SCORE_NORM, FINAL_NORM = _Proj(99), _Norm(99), _Norm(100)


def _legacy(hidden, layers, bank_rows=None, *, final=True):
    """The decoder's own order: the MLP adds the pending prefix to its output,
    so each layer hands on the whole head."""
    bank = attn_residual.AttnResidual(
        hidden, -(-(layers.stop) // BLOCK), block_residual=bank_rows
    )
    for layer in LAYER_LIST[layers]:
        hidden, prefix = bank.forward(
            hidden,
            None,
            layer.attn_proj,
            layer.attn_score_norm,
            layer.input_norm,
            write=layer.writes_block,
        )
        if layer.writes_block:
            prefix = None
        hidden, prefix = bank.forward(
            layer.attention(hidden),
            prefix,
            layer.ffn_proj,
            layer.ffn_score_norm,
            layer.post_norm,
        )
        hidden = layer.mlp(hidden) + prefix
    if not final:
        return hidden, bank.block_residual
    hidden, _ = bank.forward(hidden, None, OUT_PROJ, OUT_SCORE_NORM, FINAL_NORM)
    return hidden


def _staged(
    hidden, layers, bank_rows=None, *, read_after_write, writes_stream=(), final=True
):
    """The stage boundaries' order: each layer's output and the residual stay
    apart, and the next read folds the add into its aggregation. A layer in
    ``writes_stream`` adds the residual into its own output instead (a latent
    MoE's tail add) and hands on the written stream."""
    holder = AttnBank()
    holder.open(hidden, -(-(layers.stop) // BLOCK), bank_rows)
    # A stack's first stage, or a pipeline rank handed the whole head: the
    # stream is written and the read takes it alone.
    residual, contribution = hidden, None
    for idx in range(layers.start, layers.stop):
        layer = LAYER_LIST[idx]
        ops = layer.ops(holder)
        if contribution is None:
            hidden, residual = ops.attn_readout.read(residual, layer.input_norm)
        else:
            hidden, residual = ops.attn_readout.update_and_read(
                ops.attn_update, contribution, residual, layer.input_norm
            )
        attn_out = layer.attention(hidden)
        if residual is None and read_after_write:
            # What a stream written by the attention's read hands the FFN.
            hidden, residual = ops.ffn_readout.read(attn_out, layer.post_norm)
        else:
            hidden, residual = ops.ffn_readout.update_and_read(
                ops.attn_update, attn_out, residual, layer.post_norm
            )
        if idx in writes_stream:
            residual, contribution = layer.mlp(hidden) + residual, None
        else:
            contribution = layer.mlp(hidden)
    if contribution is None:
        if not final:
            return residual, holder.require().block_residual
        return AttnBankOutputRead(holder, OUT_PROJ, OUT_SCORE_NORM, FINAL_NORM)(
            residual
        )
    if not final:
        return contribution + residual, holder.require().block_residual
    # Called as the final norm of an output and its residual: it returns the
    # normalized output and the head it aggregated.
    hidden, head = AttnBankOutputRead(holder, OUT_PROJ, OUT_SCORE_NORM, FINAL_NORM)(
        contribution, residual
    )
    torch.testing.assert_close(head, contribution + residual, rtol=0, atol=0)
    return hidden


class TestAttnBankResidualOps(CustomTestCase):
    def setUp(self):
        patcher = patch.object(attn_residual, "_mix_fused", _mix)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.hidden = torch.randn(
            TOKENS, HIDDEN, generator=torch.Generator().manual_seed(7)
        )

    def assert_identical(self, got, want):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_the_stack_matches_the_decoders_own_order(self):
        want = _legacy(self.hidden, slice(0, LAYERS))
        for read_after_write in (False, True):
            for writes_stream in ((), (1, 3), (LAYERS - 1,)):
                with self.subTest(
                    read_after_write=read_after_write, writes_stream=writes_stream
                ):
                    got = _staged(
                        self.hidden,
                        slice(0, LAYERS),
                        read_after_write=read_after_write,
                        writes_stream=writes_stream,
                    )
                    self.assert_identical(got, want)

    def test_a_pipeline_rank_continues_from_the_head_and_the_bank(self):
        want = _legacy(self.hidden, slice(0, LAYERS))
        # Split before a layer that writes a block and before one that reads,
        # after a layer that adds the residual itself and after one that
        # leaves the add to the next read.
        for split in (2, 3):
            for writes_stream in ((), (split - 1,)):
                with self.subTest(split=split, writes_stream=writes_stream):
                    head, bank_rows = _staged(
                        self.hidden,
                        slice(0, split),
                        read_after_write=False,
                        writes_stream=writes_stream,
                        final=False,
                    )
                    legacy_head, legacy_rows = _legacy(
                        self.hidden, slice(0, split), final=False
                    )
                    # The wire carries what the decoder's own order sent.
                    self.assert_identical(head, legacy_head)
                    self.assert_identical(bank_rows, legacy_rows)
                    got = _staged(
                        head,
                        slice(split, LAYERS),
                        bank_rows,
                        read_after_write=False,
                    )
                    self.assert_identical(got, want)

    def test_a_read_on_a_shard_aggregates_its_rows_of_the_bank(self):
        """After a reduce-scatter over attention TP each rank reads its
        contiguous shard of the rows, against the same shard of the bank."""
        generator = torch.Generator().manual_seed(3)
        hidden = torch.randn(4, HIDDEN, generator=generator)
        attn_out = torch.randn(4, HIDDEN, generator=generator)
        layer = LAYER_LIST[0]
        holder = AttnBank()
        holder.open(hidden, 1)
        ops = layer.ops(holder)
        # The write layer's attention read banks every token's row.
        ops.attn_readout.read(hidden, layer.input_norm)
        want, _ = ops.ffn_readout.read(attn_out, layer.post_norm)

        shard = slice(2, 4)
        second_of_two = SimpleNamespace(attn_tp_rank=1, attn_tp_size=2)
        with patch.object(attn_bank, "get_parallel", lambda: second_of_two):
            got, residual = ops.ffn_readout.read(attn_out[shard], layer.post_norm)
            self.assert_identical(got, want[shard])
            self.assert_identical(residual, attn_out[shard])
            with self.assertRaisesRegex(RuntimeError, "attention-TP shard"):
                ops.ffn_readout.read(attn_out[:3], layer.post_norm)

    def test_declared_capabilities(self):
        ops = LAYER_LIST[1].ops(AttnBank())
        # The updates are ordinary adds, so they may cross a pipeline boundary
        # and defer the sum they complete to the next read.
        for update in (ops.attn_update, ops.ffn_update):
            self.assertTrue(update.is_plain_add)
            self.assertFalse(update.applied_at_exit)
            self.assertTrue(update.outlives_layer)
        # The reads aggregate the bank, which no add+norm kernel computes, on
        # this rank's own rows.
        for readout in (ops.attn_readout, ops.ffn_readout):
            self.assertFalse(readout.is_plain_norm)
            self.assertTrue(readout.reads_before_dp_gather)


class TestAttnBankSpMoeStages(CustomTestCase):
    """A latent MoE dispatched over an a2a backend with attention TP runs on
    this rank's shard of the rows: the FFN's entry reduce-scatters the
    attention output and slices the residual, and each MoE layer's exit
    gathers its stream back to every row, as the decoder did."""

    def build(self, exit_rows):
        holder = AttnBank()
        with (
            fixture.planning(
                fixture.parallel_of(attn_dp=1, attn_tp=2),
                a2a=True,
                boundary_reduction="ar",
            ),
            layer_stack(),
        ):
            return [
                append_stages(
                    (
                        declare_attn(read=ops.attn_readout, update=ops.attn_update),
                        fixture.Norm(),
                    ),
                    (
                        declare_ffn(
                            read=ops.ffn_readout,
                            update=REPLACE_AT_EXIT,
                            sparse=True,
                            next_layer_sparse=True,
                            output_complete=True,
                            exit_rows=exit_rows,
                        ),
                        fixture.Norm(),
                    ),
                )[1]
                for ops in (LAYER_LIST[i].ops(holder) for i in range(3))
            ]

    def entry_step(self, ffn):
        prepare = ffn.plan.paths[BatchVariant.ORDINARY].entry.prepare
        return prepare.keywords["step"]

    def test_each_moe_layer_returns_to_every_row(self):
        for ffn in self.build(ExitRows.ATTENTION):
            step = self.entry_step(ffn)
            self.assertIs(step.func, comm_ops._attn_tp_reduce_scatter_update_read)
            self.assertTrue(step.keywords["scatters_residual"])
            self.assertIs(step.keywords["read"], ffn.declaration.read)
            move = ffn.plan.paths[BatchVariant.ORDINARY].output_move
            self.assertIs(move.func, update_attn_tp_gather_output)


if __name__ == "__main__":
    unittest.main()
