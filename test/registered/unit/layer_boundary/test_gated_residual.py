"""Gated hyper-connection state follows boundary residual rows and ordering."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.layer_boundary.adapters import attention
from sglang.srt.layers.layer_boundary.residual import gated
from sglang.srt.layers.layer_boundary.residual.gated import GatedResidualState
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class ToyGatedResidual:
    """A nonlinear mix/combine that consumes both raw and normalized input."""

    hc_count = 2
    hidden_size = 3

    def __init__(self, scale):
        self.scale = scale

    def mix(self, hyper_input):
        normalized = hyper_input / (1 + hyper_input.square().mean(-1, keepdim=True))
        mixed = normalized.reshape(-1, self.hc_count, self.hidden_size).sum(1)
        return mixed * self.scale, (hyper_input, normalized)

    def combine(self, hidden_states, residuals):
        residual, normalized = residuals
        injected = hidden_states.repeat(1, self.hc_count)
        return residual + self.scale * injected * torch.sigmoid(normalized)


class TestGatedResidual(CustomTestCase):
    def setUp(self):
        self.attention = ToyGatedResidual(0.5)
        self.ffn = ToyGatedResidual(0.75)
        self.state = GatedResidualState(self.attention, self.ffn)
        self.residual = self.state.layer_residual()
        self.hidden = torch.arange(24, dtype=torch.float32).reshape(4, 6) / 10

    def test_preserves_attention_then_ffn_mix_combine_order(self):
        expected_attn, expected_residual = self.attention.mix(self.hidden)
        expected_after_attn = self.attention.combine(
            expected_attn.square(), expected_residual
        )
        expected_ffn, expected_residual = self.ffn.mix(expected_after_attn)
        expected = self.ffn.combine(expected_ffn.sin(), expected_residual)

        attn, residual = self.residual.attention_read.read(self.hidden, None)
        ffn, residual = self.residual.ffn_read.update_and_read(
            self.residual.attention_update, attn.square(), residual, None
        )
        actual = self.residual.ffn_update.update(ffn.sin(), residual)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertIsNone(self.residual.attention_update.state.normalized)
        self.assertIsNone(self.residual.ffn_update.state.normalized)
        self.assertFalse(self.residual.attention_update.adds_plainly)
        self.assertFalse(self.residual.attention_update.at_producer)
        self.assertTrue(self.residual.ffn_update.at_producer)
        self.assertFalse(self.residual.ffn_update.can_defer_across_layers)

    def test_first_attention_expands_embeddings_only_once(self):
        embedding = self.hidden[:, : self.attention.hidden_size]
        expanded = self.residual.attention_read.enter(embedding)
        torch.testing.assert_close(expanded, embedding.repeat(1, 2))
        self.assertIs(self.residual.attention_read.enter(expanded), expanded)
        with self.assertRaisesRegex(ValueError, "width"):
            self.residual.attention_read.enter(torch.empty(4, 5))

    def test_shard_moves_normalized_state_with_raw_residual(self):
        mixed, residual = self.residual.attention_read.read(self.hidden, None)
        expected = self.attention.combine(mixed, self.attention.mix(self.hidden)[1])
        update = self.residual.attention_update
        with patch.object(
            attention,
            "get_parallel",
            lambda: SimpleNamespace(attn_tp_size=2, attn_tp_rank=1),
        ):
            shard = update.residual_to_attn_tp_shard(residual)
        actual = update.update(mixed[2:], shard)
        torch.testing.assert_close(actual, expected[2:], rtol=0, atol=0)

    def test_gather_moves_both_states_without_shared_scratch_aliasing(self):
        full_mixed, full_residuals = self.attention.mix(self.hidden)
        _, local_residual = self.residual.attention_read.read(self.hidden[2:], None)
        update = self.residual.attention_update

        def gather_pair(pair, forward_batch):
            self.assertIsNone(forward_batch)
            torch.testing.assert_close(pair[0], self.hidden[2:])
            torch.testing.assert_close(pair[1], full_residuals[1][2:])
            return tuple(t.clone() for t in full_residuals)

        with patch.object(gated, "gather_attention_tp", side_effect=gather_pair) as ag:
            gathered = update.residual_from_attn_tp_shards(local_residual)
        ag.assert_called_once()
        actual = update.update(full_mixed, gathered)
        expected = self.attention.combine(full_mixed, full_residuals)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_stage_preserves_widened_residual_shape(self):
        hidden = torch.empty(0, self.attention.hidden_size)
        expanded = self.residual.attention_read.enter(hidden)
        attn, residual = self.residual.attention_read.read(expanded, None)
        ffn, residual = self.residual.ffn_read.update_and_read(
            self.residual.attention_update, attn, residual, None
        )
        output = self.residual.ffn_update.update(ffn, residual)
        self.assertEqual(output.shape, (0, 6))
        self.assertIsNone(self.residual.ffn_update.state.normalized)

    def test_update_uses_actual_producer_and_addition_precedes_mix(self):
        updated = self.hidden + 2
        producer = SimpleNamespace(update=Mock(return_value=updated))
        addition = torch.full_like(self.hidden, 0.5)
        actual, residual = self.residual.ffn_read.update_and_read(
            producer,
            torch.empty(4, 3),
            self.hidden,
            None,
            post_residual_addition=addition,
        )
        producer.update.assert_called_once()
        expected, _ = self.ffn.mix(updated + addition)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(residual, updated + addition)

    def test_update_cannot_reuse_auxiliary_state_after_consumption(self):
        mixed, residual = self.residual.attention_read.read(self.hidden, None)
        self.residual.attention_update.update(mixed, residual)
        with self.assertRaisesRegex(RuntimeError, "requires its stage's read"):
            self.residual.attention_update.update(mixed, residual)


if __name__ == "__main__":
    unittest.main()
