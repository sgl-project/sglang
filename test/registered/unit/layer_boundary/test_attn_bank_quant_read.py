"""An attention-residual bank read in a quant_format aggregates without the
norm and runs the norm fused with that quantization, where the aggregation can
leave the norm out; elsewhere it reads plainly."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import attn_residual
from sglang.srt.layers.layer_boundary.residual import attn_bank
from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBankState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Bank:
    """Records the out_norm each aggregation was asked for."""

    def __init__(self, unnormed: bool):
        self.unnormed = unnormed
        self.out_norms = []
        self.block_residual = torch.zeros(2, 1, 4)

    def aggregates_unnormed(self, hidden_size):
        return self.unnormed

    def forward(self, hidden, residual, score_proj, score_norm, out_norm, **kw):
        self.out_norms.append(out_norm)
        mixed = hidden + 1
        return (mixed if out_norm is None else out_norm(mixed)), hidden


def _state(bank, *, reads_slices=False):
    return AttnBankState(
        bank=SimpleNamespace(require=lambda: bank),
        attn_score_proj=None,
        attn_score_norm=None,
        ffn_score_proj=None,
        ffn_score_norm=None,
        reads_slices=reads_slices,
    )


def _quant_kernel(quant_format):
    def run(norm, hidden, residual, post_residual_addition):
        assert residual is None and post_residual_addition is None
        return (quant_format, norm(hidden)), hidden

    return run


class TestAttnBankQuantRead(CustomTestCase):
    def setUp(self):
        self.norm = lambda x: x * 2
        self.x = torch.ones(2, 4)

    def test_a_quantized_read_fuses_the_norm_with_the_quant(self):
        bank = _Bank(unnormed=True)
        readout = _state(bank).residual_ops().attn_readout
        with patch.object(attn_bank, "_norm_quant_kernel", _quant_kernel):
            out, _ = readout.read(self.x, self.norm, quant_format="fp8")
        self.assertEqual(bank.out_norms, [None])
        self.assertEqual(out[0], "fp8")
        torch.testing.assert_close(out[1], (self.x + 1) * 2)

    def test_a_bank_that_must_norm_reads_plainly(self):
        bank = _Bank(unnormed=False)
        readout = _state(bank).residual_ops().attn_readout
        with patch.object(attn_bank, "_norm_quant_kernel", _quant_kernel):
            out, _ = readout.read(self.x, self.norm, quant_format="fp8")
        self.assertEqual(bank.out_norms, [self.norm])
        torch.testing.assert_close(out, (self.x + 1) * 2)

    def test_a_read_gathered_afterwards_reads_plainly(self):
        bank = _Bank(unnormed=True)
        readout = _state(bank, reads_slices=True).residual_ops().attn_readout
        with patch.object(attn_bank, "_norm_quant_kernel", _quant_kernel):
            out, _ = readout.read(self.x, self.norm, quant_format="fp8")
        self.assertEqual(bank.out_norms, [self.norm])
        self.assertIsInstance(out, torch.Tensor)

    def test_the_ffn_read_still_refuses_a_quant_format(self):
        readout = _state(_Bank(unnormed=True)).residual_ops().ffn_readout
        with self.assertRaises(NotImplementedError):
            readout.read(self.x, self.norm, quant_format="fp8")


class TestAggregatesUnnormed(CustomTestCase):
    def test_an_empty_bank_can_always_leave_the_norm_out(self):
        bank = attn_residual.AttnResidual(torch.zeros(2, 4), block_num=2)
        with patch.object(attn_residual, "_use_hip_fused", return_value=False):
            self.assertTrue(bank.aggregates_unnormed(4))

    def test_a_filled_bank_needs_the_rocm_kernel(self):
        bank = attn_residual.AttnResidual(torch.zeros(2, 4), block_num=2)
        bank.write(torch.zeros(2, 4))
        with patch.object(attn_residual, "_use_hip_fused", return_value=False):
            self.assertFalse(bank.aggregates_unnormed(4))
        with patch.object(attn_residual, "_use_hip_fused", return_value=True):
            self.assertTrue(bank.aggregates_unnormed(4))


if __name__ == "__main__":
    unittest.main()
