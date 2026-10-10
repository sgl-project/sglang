import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_declared_decoder_boundary as fixture
import torch

from sglang.srt.layers.layer_boundary import layer_stack
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen4_exp import _build_qwen4_exp_stages
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC_COUNT = 3
HIDDEN = 4
WIDE = HC_COUNT * HIDDEN


def _expand(hidden_states):
    """The layer stack's input widened into streams; already widened passes
    through, as the write-back returns streams."""
    if hidden_states.shape[-1] == WIDE:
        return hidden_states
    return torch.cat([hidden_states for _ in range(HC_COUNT)], dim=-1)


def _normalize(residual, seed):
    """A stand-in for the per-branch norm each stage's mix applies."""
    return residual * (0.5 + seed)


def _mix(seed):
    """Stands in for GatedResidual.mix: returns the mixed input and the streams,
    their normalization, and optional gate partials consumed at write-back."""

    def mix(hyper_input):
        if hyper_input.shape[0] == 0:
            empty = hyper_input.new_empty((0, HIDDEN))
            # An empty batch puts the un-normalized streams in the second slot.
            return empty, (hyper_input, hyper_input, None)
        normed = _normalize(hyper_input, seed)
        mixed = normed.unflatten(-1, (HC_COUNT, HIDDEN)).mean(dim=-2)
        return mixed, (hyper_input, normed, None)

    return mix


def _combine(block_output, residuals):
    """Stands in for GatedResidual.combine: the injection coefficient is
    computed at write time from the normalized residual the read produced."""
    hyper_input, normed, _ = residuals
    if block_output.shape[0] == 0:
        return hyper_input
    coefficient = 2 * torch.sigmoid(
        normed.unflatten(-1, (HC_COUNT, HIDDEN)).mean(dim=-1)
    )
    injected = block_output.unsqueeze(-2) * coefficient.unsqueeze(-1)
    return (hyper_input.unflatten(-1, (HC_COUNT, HIDDEN)) + injected).flatten(-2)


def _state(attn_mix=None):
    return GatedResidualState(
        expand=_expand,
        attn_mix=attn_mix if attn_mix is not None else _mix(1),
        ffn_mix=_mix(2),
        attn_combine=_combine,
        ffn_combine=_combine,
    )


class TestGatedResidualOps(CustomTestCase):
    def setUp(self):
        self.hidden = torch.arange(2 * HIDDEN, dtype=torch.float32).reshape(2, HIDDEN)
        self.output = torch.full((2, HIDDEN), 5.0)

    def test_entry_widens_the_stack_input(self):
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        self.assertEqual(residual.shape, (2, WIDE))
        for stream in range(HC_COUNT):
            torch.testing.assert_close(
                residual[:, stream * HIDDEN : (stream + 1) * HIDDEN], self.hidden
            )

    def test_entry_passes_already_widened_streams_through(self):
        """An MTP draft is handed streams that are already widened."""
        ops = _state().residual_ops()
        widened = torch.ones(2, WIDE)
        self.assertIs(ops.attn_readout.init_residual(widened), widened)

    def test_the_write_back_uses_the_residual_its_own_read_normalized(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)

        self.assertIsNone(state.normed)
        _, residual = ops.attn_readout.read(residual, None)
        normed = state.normed
        self.assertIsNotNone(normed)
        torch.testing.assert_close(normed, _normalize(residual, 1))

        torch.testing.assert_close(
            ops.attn_update.update(self.output, residual),
            _combine(self.output, (residual, normed, None)),
        )

    def test_the_write_back_reads_the_carried_value_not_a_fresh_one(self):
        """The coefficient is computed at write time, so what crosses from the
        read is the normalized residual itself. Replacing it must move the
        write-back; recomputing from the streams would not."""
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        _, residual = ops.attn_readout.read(residual, None)

        state.normed = torch.full_like(state.normed, -3.0)
        torch.testing.assert_close(
            ops.attn_update.update(self.output, residual),
            _combine(self.output, (residual, state.normed, None)),
        )

    def test_ffn_write_back_clears_the_carried_value(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        ops.attn_readout.read(residual, None)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, self.output, residual, None
        )
        self.assertIsNotNone(state.normed)

        ops.ffn_update.update(torch.full((2, HIDDEN), 7.0), residual)
        # Nothing survives the layer: the next layer's read produces its own.
        self.assertIsNone(state.normed)

    def test_gate_partials_follow_their_read_and_are_cleared_after_write_back(self):
        def mix(seed):
            def read(residual):
                hidden, (residual, normed, _) = _mix(seed)(residual)
                return hidden, (residual, normed, normed.mean(dim=-1, keepdim=True))

            return read

        def combine(hidden, residuals):
            residual, normed, partials = residuals
            return _combine(hidden * partials, (residual, normed, None))

        state = GatedResidualState(_expand, mix(1), mix(2), combine, combine)
        ops = state.residual_ops()
        residual = _expand(self.hidden)
        ops.attn_readout.read(residual, None)
        _, attn_residuals = mix(1)(residual)
        expected = combine(self.output, attn_residuals)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, self.output, residual, None
        )
        torch.testing.assert_close(residual, expected)
        _, ffn_residuals = mix(2)(expected)
        result = ops.ffn_update.update(self.output, residual)
        torch.testing.assert_close(result, combine(self.output, ffn_residuals))
        self.assertIsNone(state.normed)
        self.assertIsNone(state.gate_partials)

    def test_reduced_ffn_output_still_writes_the_gated_residual_once(self):
        for already_reduced in (False, True):
            with self.subTest(already_reduced=already_reduced):
                state = _state()
                residual = _expand(self.hidden)
                state.normed = _normalize(residual, 2)
                expected = _combine(self.output * 4, (residual, state.normed, None))
                batch = SimpleNamespace(
                    residual_stream=ResidualStream(residual),
                    forward_mode=ForwardMode.DECODE,
                )
                with (
                    fixture.planning(fixture.parallel_of(attn_dp=1, attn_tp=4)),
                    layer_stack(),
                ):
                    _, ffn = _build_qwen4_exp_stages(
                        state.residual_ops(),
                        sparse=True,
                    )
                with patch(
                    "sglang.srt.layers.layer_boundary.exit.sum_output",
                    side_effect=lambda value, *args, **kwargs: value * 4,
                ) as reduce:
                    result = ffn.complete_now(
                        self.output * (4 if already_reduced else 1),
                        batch,
                        already_reduced=already_reduced,
                    )
                self.assertEqual(reduce.call_count, int(not already_reduced))
                torch.testing.assert_close(result, expected)
                self.assertIs(batch.residual_stream.residual, result)
                self.assertIsNone(batch.residual_stream.pending)
                self.assertIsNone(state.normed)

    def test_the_ffn_input_is_read_from_the_updated_streams(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        _, residual = ops.attn_readout.read(residual, None)
        attn_normed = state.normed

        got_input, got_residual = ops.ffn_readout.update_and_read(
            ops.attn_update, self.output, residual, None
        )
        want_residual = _combine(self.output, (residual, attn_normed, None))
        want_input, (_, want_normed, _) = _mix(2)(want_residual)
        torch.testing.assert_close(got_residual, want_residual)
        torch.testing.assert_close(got_input, want_input)
        torch.testing.assert_close(state.normed, want_normed)

    def test_a_read_may_contribute_to_the_streams_before_mixing(self):
        """A layer whose own contribution joins the streams ahead of its read
        (the Qwen4 PLE embedding) folds it into the mix it hands the state, so
        the read returns streams this layer has added to."""
        contribution = torch.full((2, WIDE), 0.25)
        state = _state(attn_mix=lambda residual: _mix(1)(residual + contribution))
        ops = state.residual_ops()
        entered = ops.attn_readout.init_residual(self.hidden)

        _, residual = ops.attn_readout.read(entered, None)
        torch.testing.assert_close(residual, entered + contribution)
        # The write-back lands on the contributed streams, once.
        torch.testing.assert_close(
            ops.attn_update.update(self.output, residual),
            _combine(self.output, (entered + contribution, state.normed, None)),
        )

    def test_an_empty_batch_keeps_the_streams_and_their_width(self):
        state = _state()
        ops = state.residual_ops()
        residual = ops.attn_readout.init_residual(torch.empty(0, HIDDEN))
        self.assertEqual(residual.shape, (0, WIDE))

        hidden_states, residual = ops.attn_readout.read(residual, None)
        self.assertEqual(hidden_states.shape, (0, HIDDEN))
        # The mix leaves the un-normalized streams behind on an empty batch.
        torch.testing.assert_close(state.normed, residual)

        written = ops.attn_update.update(torch.empty(0, HIDDEN), residual)
        torch.testing.assert_close(written, residual)

    def test_the_terminal_write_back_leaves_the_streams_widened(self):
        """The stack's terminal read mixes the streams down, so the FFN
        write-back must not contract them."""
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        ops.attn_readout.read(residual, None)
        _, residual = ops.ffn_readout.update_and_read(
            ops.attn_update, self.output, residual, None
        )
        written = ops.ffn_update.update(torch.full((2, HIDDEN), 7.0), residual)
        self.assertEqual(written.shape, (2, WIDE))

    def test_declared_capabilities(self):
        ops = _state().residual_ops()
        # A gated injection is not an add, so the sum it writes into must be
        # complete first and no add+norm fusion may claim it.
        self.assertFalse(ops.attn_update.is_plain_add)
        self.assertFalse(ops.ffn_update.is_plain_add)
        # Each read normalizes the streams itself.
        self.assertFalse(ops.attn_readout.is_plain_norm)
        self.assertFalse(ops.ffn_readout.is_plain_norm)
        # The FFN writes at its exit; the attention leaves its write-back to
        # the FFN input's read, inside the same layer.
        self.assertTrue(ops.ffn_update.applied_at_exit)
        self.assertFalse(ops.attn_update.applied_at_exit)
        self.assertFalse(ops.attn_update.outlives_layer)
        self.assertFalse(ops.ffn_update.outlives_layer)

    def test_rejects_reads_it_does_not_implement(self):
        ops = _state().residual_ops()
        residual = ops.attn_readout.init_residual(self.hidden)
        with self.assertRaises(NotImplementedError):
            ops.attn_readout.read(residual, None, quant_format="fp8")
        # The mix normalizes the streams, so a separate norm has no place.
        with self.assertRaises(NotImplementedError):
            ops.attn_readout.read(residual, lambda hidden_states: hidden_states)
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

    def test_attn_tp_slice_moves_the_carried_value_with_the_streams(self):
        state = _state()
        ops = state.residual_ops()
        residual = torch.arange(4 * WIDE, dtype=torch.float32).reshape(4, WIDE)
        state.normed = _normalize(residual, 1)
        partials = torch.arange(4 * HC_COUNT).reshape(4, HC_COUNT)
        state.gate_partials = partials
        with patch(
            "sglang.srt.layers.layer_boundary.residual.gated.get_parallel"
        ) as parallel:
            parallel.return_value.attn_tp_rank = 1
            parallel.return_value.attn_tp_size = 2
            sliced = ops.attn_update.slice_residual_attn_tp(residual)
        torch.testing.assert_close(sliced, residual[2:])
        # The normalized rows must follow the stream rows they scale.
        torch.testing.assert_close(state.normed, _normalize(residual, 1)[2:])
        torch.testing.assert_close(state.gate_partials, partials[2:])

    def test_attn_tp_gather_is_rejected(self):
        ops = _state().residual_ops()
        with self.assertRaises(NotImplementedError):
            ops.attn_update.gather_residual_attn_tp(self.hidden)


if __name__ == "__main__":
    unittest.main()
