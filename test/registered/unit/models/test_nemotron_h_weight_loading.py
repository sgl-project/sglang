"""
Unit tests for NemotronHForCausalLM.load_weights.

Regression test for Nemotron-H expert scale checkpoint tensors that map to
parameters absent from the current runtime model.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.nemotron_h import NemotronHForCausalLM


class _FakePPGroup:
    is_first_rank = True
    is_last_rank = True


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(
        self, param, loaded_weight, name, *, shard_id=None, expert_id=None
    ):
        self.loaded = (param, loaded_weight, name, shard_id, expert_id)


class _RecordingParam:
    """Param whose weight_loader matches the plain (non-expert) load path:
    weight_loader(param, loaded_weight)."""

    def __init__(self):
        self.loaded_weight = None

    def weight_loader(self, param, loaded_weight):
        self.loaded_weight = loaded_weight


class TestNemotronHWeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(NemotronHForCausalLM)
        model.config = SimpleNamespace(n_routed_experts=2, max_n_routed_experts=2)
        model.model = SimpleNamespace()
        model.pp_group = _FakePPGroup()
        model.remap_prefix = {}
        model.remap_substr = {}
        model.stacked_params_mapping = []
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_expert_input_scale_without_target_parameter_is_skipped(self):
        """Expert scale weights absent from params_dict should not raise KeyError."""
        model = self._make_minimal_model()
        weights = [
            (
                "model.layers.1.mixer.experts.0.down_proj.input_scale",
                torch.ones(1),
            )
        ]

        model.load_weights(weights)

    def test_expert_weight_with_target_parameter_is_loaded(self):
        param = _FakeParam()
        model = self._make_minimal_model(
            [("model.layers.1.mixer.experts.w2_weight", param)]
        )
        loaded_weight = torch.ones(1)
        weights = [
            (
                "model.layers.1.mixer.experts.0.down_proj.weight",
                loaded_weight,
            )
        ]

        model.load_weights(weights)

        self.assertEqual(
            param.loaded,
            (
                param,
                loaded_weight,
                "model.layers.1.mixer.experts.w2_weight",
                "w2",
                0,
            ),
        )

    def test_mtp_keeps_shared_embed_tokens_and_lm_head(self):
        """MTP draft load must keep the shared embed_tokens + lm_head, not only
        mtp.layers.*. Regression for issue #21138: dropping the shared embedding
        / head makes the draft accept zero tokens (accept rate 0.00). The remap
        rewrites "embeddings" -> "embed_tokens" before the MTP filter, so the
        whitelist must match the post-remap names."""
        embed = _RecordingParam()
        head = _RecordingParam()
        mtp_layer = _RecordingParam()
        skipped = _RecordingParam()
        model = self._make_minimal_model(
            [
                ("model.embed_tokens.weight", embed),
                ("lm_head.weight", head),
                ("model.layers.0.norm.weight", mtp_layer),
                ("model.layers.5.norm.weight", skipped),
            ]
        )
        # Production remap: backbone -> model, embeddings -> embed_tokens.
        model.remap_prefix = {"backbone": "model"}
        model.remap_substr = {"embeddings": "embed_tokens"}

        w_embed, w_head, w_mtp, w_skip = (torch.ones(1) for _ in range(4))
        weights = [
            ("backbone.embeddings.weight", w_embed),  # -> model.embed_tokens.weight
            ("lm_head.weight", w_head),  # -> lm_head.weight
            ("mtp.layers.0.norm.weight", w_mtp),  # -> model.layers.0.norm.weight
            ("backbone.layers.5.norm.weight", w_skip),  # non-MTP target -> skipped
        ]

        model.load_weights(weights, is_mtp=True)

        self.assertIs(embed.loaded_weight, w_embed, "shared embed_tokens dropped")
        self.assertIs(head.loaded_weight, w_head, "shared lm_head dropped")
        self.assertIs(
            mtp_layer.loaded_weight, w_mtp, "mtp.layers.* not remapped/loaded"
        )
        self.assertIsNone(
            skipped.loaded_weight, "non-MTP target weight should be skipped"
        )


if __name__ == "__main__":
    unittest.main()
