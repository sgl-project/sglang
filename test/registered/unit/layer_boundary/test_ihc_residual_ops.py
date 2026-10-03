import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.layer_boundary.residual.ihc import IHCState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC_MULT = 3
HIDDEN = 4


def _expand(hidden_states):
    """The layer stack's 2-D input widened into streams; already widened passes
    through, as the gated write-back returns streams."""
    if hidden_states.ndim == 3:
        return hidden_states
    return hidden_states.unsqueeze(1).repeat(1, HC_MULT, 1)


def _gate(residual, seed):
    """A per-stream coefficient standing in for the layer's learned gates."""
    rows = residual.shape[0]
    return torch.arange(rows * HC_MULT, dtype=torch.float32).reshape(
        rows, HC_MULT
    ) * 0.1 + float(seed)


def _pre(seed):
    def pre(residual, out_norm=None):
        gate = _gate(residual, seed)
        mixed = (gate.unsqueeze(-1) * residual).sum(dim=1)
        if out_norm is not None:
            mixed = out_norm(mixed)
        return mixed, gate, residual

    return pre


def _post(output, residual, gate):
    return gate.unsqueeze(-1) * output.unsqueeze(1) + residual


class _Norm:
    """A stand-in whose effect is visible in the output."""

    def __call__(self, hidden_states):
        return hidden_states * 2.0


def _state(post_pre=None):
    return IHCState(
        hc_mult=HC_MULT,
        expand=_expand,
        attn_pre=_pre(1),
        ffn_pre=_pre(2),
        attn_post=_post,
        ffn_post=_post,
        post_pre=post_pre,
    )


class TestIHCResidualOps(CustomTestCase):
    def setUp(self):
        self.hidden = torch.arange(2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)

    def test_entry_widens_the_stack_input(self):
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        self.assertEqual(residual.shape, (2, HC_MULT, HIDDEN))
        # Every stream starts from the same input row.
        for stream in range(HC_MULT):
            torch.testing.assert_close(residual[:, stream], self.hidden)

    def test_read_coefficient_is_consumed_by_the_same_stage(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)

        self.assertIsNone(state.post_gate)
        _, residual = ops.attn_readout.read(residual, None)
        gate = state.post_gate
        self.assertIsNotNone(gate)

        output = torch.full((2, HIDDEN), 5.0)
        updated = ops.attn_update.update(output, residual)
        # The write-back used the gate the read produced, not a fresh one.
        torch.testing.assert_close(updated, _post(output, residual, gate))

    def test_ffn_write_back_clears_the_coefficient(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        ops.attn_readout.read(residual, None)
        attn_out = torch.full((2, HIDDEN), 5.0)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, attn_out, residual, None
        )
        self.assertIsNotNone(state.post_gate)

        ffn_out = torch.full((2, HIDDEN), 7.0)
        ops.ffn_update.update(ffn_out, residual)
        self.assertIsNone(state.post_gate)

    def test_fused_post_pre_matches_the_unfused_chain(self):
        unfused = _state().residual_ops()
        residual = unfused.attn_readout.init_residual(self.hidden)
        unfused.attn_readout.read(residual, None)
        attn_out = torch.full((2, HIDDEN), 5.0)
        want_input, want_residual = unfused.ffn_readout.update_and_read(
            unfused.attn_update, attn_out, residual, _Norm()
        )

        def post_pre(hidden_states, res, gate, out_norm):
            res = _post(hidden_states, res, gate)
            mixed, next_gate, res = _pre(2)(res, out_norm)
            return mixed, next_gate, res

        fused_state = _state(post_pre=post_pre)
        fused = fused_state.residual_ops()
        residual = fused.attn_readout.init_residual(self.hidden)
        fused.attn_readout.read(residual, None)
        got_input, got_residual = fused.ffn_readout.update_and_read(
            fused.attn_update, attn_out, residual, _Norm()
        )
        torch.testing.assert_close(got_input, want_input)
        torch.testing.assert_close(got_residual, want_residual)

    def test_the_terminal_write_back_leaves_the_streams_widened(self):
        """Unlike MHC, the stack's terminal read is a learned head, so the FFN
        write-back must not contract."""
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        ops.attn_readout.read(residual, None)
        attn_out = torch.full((2, HIDDEN), 5.0)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, attn_out, residual, None
        )
        written = ops.ffn_update.update(torch.full((2, HIDDEN), 7.0), residual)
        self.assertEqual(written.shape, (2, HC_MULT, HIDDEN))

    def test_declared_capabilities(self):
        ops = _state().residual_ops()
        # The gated write-back is not an add, so a sum it writes into must be
        # complete first, and no add+norm fusion may claim it.
        self.assertFalse(ops.attn_update.is_plain_add)
        self.assertFalse(ops.ffn_update.is_plain_add)
        self.assertFalse(ops.attn_readout.is_plain_norm)
        self.assertFalse(ops.ffn_readout.is_plain_norm)
        # The FFN writes at its exit; the attention leaves its update to the
        # FFN input's read.
        self.assertTrue(ops.ffn_update.applied_at_exit)
        self.assertFalse(ops.attn_update.applied_at_exit)

    def test_rejects_reads_it_does_not_implement(self):
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        with self.assertRaises(NotImplementedError):
            ops.attn_readout.read(residual, None, quant_format="fp8")
        with self.assertRaises(NotImplementedError):
            ops.attn_readout.update_and_read(
                ops.ffn_update, self.hidden, residual, None
            )
        with self.assertRaises(NotImplementedError):
            ops.ffn_readout.read(residual, None)

    def test_ffn_read_rejects_another_layers_update(self):
        ops = _state().residual_ops()
        other = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        ops.attn_readout.read(residual, None)
        with self.assertRaises(NotImplementedError):
            ops.ffn_readout.update_and_read(
                other.attn_update, self.hidden, residual, None
            )

    def test_attn_tp_slice_moves_the_coefficient_with_the_streams(self):
        state = _state()
        ops = state.residual_ops()
        residual = torch.arange(4 * HC_MULT * HIDDEN, dtype=torch.float32).reshape(
            4, HC_MULT, HIDDEN
        )
        state.post_gate = _gate(residual, 1)
        with patch(
            "sglang.srt.layers.layer_boundary.residual.ihc.get_parallel"
        ) as parallel:
            parallel.return_value.attn_tp_rank = 1
            parallel.return_value.attn_tp_size = 2
            sliced = ops.attn_update.slice_residual_attn_tp(residual)
        torch.testing.assert_close(sliced, residual[2:])
        # The gate rows must follow the residual rows they scale.
        torch.testing.assert_close(state.post_gate, _gate(residual, 1)[2:])

    def test_attn_tp_gather_is_rejected(self):
        ops = _state().residual_ops()
        with self.assertRaises(NotImplementedError):
            ops.attn_update.gather_residual_attn_tp(self.hidden)


if __name__ == "__main__":
    unittest.main()
