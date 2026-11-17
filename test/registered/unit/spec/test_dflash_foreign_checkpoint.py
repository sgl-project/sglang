"""The DFLASH draft must refuse a checkpoint that belongs to another model.

A speculative deployment serves two models, and a from-disk weight update
names one checkpoint. When the target's weights land in the draft they fit
(the hidden sizes often match), so nothing errors and speculation silently
breaks. The draft loader therefore has to notice a weight set that cannot be
its own.
"""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from sglang.srt.models.dflash import DFlashDraftModel, find_foreign_weight_reason
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN = 32


def _draft_config(num_hidden_layers=2):
    return Qwen3Config(
        architectures=["DFlashDraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        max_position_embeddings=4096,
        dflash_config={"block_size": 4, "target_layer_ids": [0]},
    )


class TestDFlashForeignCheckpoint(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True))
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def _draft(self, num_hidden_layers=2):
        return DFlashDraftModel(_draft_config(num_hidden_layers))

    def _draft_weights(self, model):
        return [
            (name, torch.randn_like(param))
            for name, param in model.named_parameters()
        ]

    def test_genuine_draft_checkpoint_loads(self):
        """The draft's own weights must keep loading, unchanged."""
        model = self._draft()
        weights = self._draft_weights(model)

        model.load_weights(weights)

        loaded = dict(weights)
        for name, param in model.named_parameters():
            torch.testing.assert_close(param, loaded[name], rtol=0, atol=0)

    def test_target_head_weight_is_rejected(self):
        """The target owns an embedding the draft does not have."""
        model = self._draft()
        weights = self._draft_weights(model)
        weights.append(("model.embed_tokens.weight", torch.zeros(64, HIDDEN)))

        with self.assertRaisesRegex(ValueError, "does not match the draft model"):
            model.load_weights(weights)

    def test_target_lm_head_is_rejected(self):
        """A vocabulary head belongs to the target, never to the draft."""
        model = self._draft()
        weights = self._draft_weights(model)
        weights.append(("lm_head.weight", torch.zeros(64, HIDDEN)))

        with self.assertRaisesRegex(ValueError, "does not match the draft model"):
            model.load_weights(weights)

    def test_layer_beyond_the_draft_is_rejected(self):
        """The draft has fewer layers than the target, so a later layer is foreign."""
        model = self._draft(num_hidden_layers=2)
        weights = self._draft_weights(model)
        weights.append(("model.layers.7.self_attn.o_proj.weight", torch.zeros(HIDDEN)))

        with self.assertRaisesRegex(ValueError, "only has 2 layers"):
            model.load_weights(weights)

    def test_layer_inside_the_draft_is_not_rejected(self):
        """The draft's own layers share their index with the target's."""
        model = self._draft(num_hidden_layers=2)
        weights = self._draft_weights(model)
        weights.append(("model.layers.1.input_layernorm.weight", torch.zeros(HIDDEN)))

        model.load_weights(weights)

    def test_extra_names_are_tolerated(self):
        """HF rotary caches and similar extras must not fail the load."""
        model = self._draft()
        weights = self._draft_weights(model)
        weights.append(("model.rotary_emb.inv_freq", torch.zeros(4)))
        weights.append(("model.layers.0.self_attn.rotary_emb.inv_freq", torch.zeros(4)))

        model.load_weights(weights)

    def test_list_input_is_validated_before_any_weight_is_written(self):
        """A materialized list is checked up front, so nothing lands in the draft."""
        model = self._draft()
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(0.5)
        before = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        weights = [
            (name, torch.zeros_like(param)) for name, param in model.named_parameters()
        ]
        # The foreign name comes last: a streaming check would already have
        # written the earlier weights.
        weights.append(("model.embed_tokens.weight", torch.zeros(64, HIDDEN)))

        with self.assertRaisesRegex(ValueError, "does not match the draft model"):
            model.load_weights(weights)

        for name, param in model.named_parameters():
            torch.testing.assert_close(param, before[name], rtol=0, atol=0)

    def test_lazy_iterator_is_checked_without_draining_it(self):
        """The disk loader streams a checkpoint; materializing it would blow up
        host memory on a large MoE checkpoint."""
        model = self._draft()
        produced = []

        def weights():
            for name in (
                "model.embed_tokens.weight",
                "layers.0.input_layernorm.weight",
            ):
                produced.append(name)
                yield name, torch.zeros(HIDDEN)

        with self.assertRaisesRegex(ValueError, "does not match the draft model"):
            model.load_weights(weights())

        self.assertEqual(produced, ["model.embed_tokens.weight"])

    def test_mostly_unresolved_checkpoint_warns(self):
        """A checkpoint the draft barely recognizes is worth a warning."""
        model = self._draft()
        weights = [
            ("layers.0.input_layernorm.weight", torch.zeros(HIDDEN)),
            ("some.other.weight", torch.zeros(1)),
            ("yet.another.weight", torch.zeros(1)),
        ]

        with self.assertLogs("sglang.srt.models.dflash", level="WARNING") as logs:
            model.load_weights(weights)

        self.assertTrue(
            any("belongs to another model" in line for line in logs.output),
            logs.output,
        )

    def test_mostly_resolved_checkpoint_does_not_warn(self):
        """A draft checkpoint with a few extras must stay quiet."""
        model = self._draft()
        weights = self._draft_weights(model)
        weights.append(("model.rotary_emb.inv_freq", torch.zeros(4)))

        with self.assertNoLogs("sglang.srt.models.dflash", level="WARNING"):
            model.load_weights(weights)


class TestFindForeignWeightReason(CustomTestCase):
    def test_names_the_draft_owns_are_never_foreign(self):
        names = ["layers.0.input_layernorm.weight", "embed_tokens.weight"]
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(
                    find_foreign_weight_reason(
                        name, num_draft_layers=2, draft_param_names=names
                    )
                )

    def test_head_weights_are_foreign_unless_the_draft_owns_them(self):
        reason = find_foreign_weight_reason(
            "model.lm_head.weight", num_draft_layers=2, draft_param_names=[]
        )
        self.assertIn("lm_head.weight", reason)

        self.assertIsNone(
            find_foreign_weight_reason(
                "model.lm_head.weight",
                num_draft_layers=2,
                draft_param_names=["lm_head.weight"],
            )
        )

    def test_layer_index_beyond_the_draft_is_foreign(self):
        reason = find_foreign_weight_reason(
            "model.layers.5.mlp.down_proj.weight",
            num_draft_layers=2,
            draft_param_names=[],
        )
        self.assertIn("layer 5", reason)
        self.assertIn("2 layers", reason)

    def test_layer_check_is_skipped_without_a_layer_count(self):
        """A draft without layers cannot prove a layer index foreign."""
        self.assertIsNone(
            find_foreign_weight_reason(
                "model.layers.5.mlp.down_proj.weight",
                num_draft_layers=0,
                draft_param_names=[],
            )
        )

    def test_unknown_names_are_not_proven_foreign(self):
        self.assertIsNone(
            find_foreign_weight_reason(
                "model.rotary_emb.inv_freq", num_draft_layers=2, draft_param_names=[]
            )
        )


if __name__ == "__main__":
    unittest.main()
