# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.multimodal_gen.runtime.loader.component_loaders.magi2_loader import (
    _is_ep_sharded,
)


class TestExpertAxisSharding(unittest.TestCase):
    def test_matches_the_checkpoint_spelling_of_every_sharded_tensor(self):
        prefix = "blocks.5.mlp"
        for suffix in (
            "moe_mlp.router.gate",
            "moe_mlp.router.expert_bias",
            "moe_mlp.W_gate",
            "moe_mlp.W_up",
            "moe_mlp.W_down",
        ):
            with self.subTest(suffix=suffix):
                self.assertTrue(_is_ep_sharded(f"{prefix}.{suffix}"))

    def test_model_side_expert_names_are_never_matched(self):
        # w13_weight and w2 exist only after _relayout_experts, which runs once
        # every rank has finished reading. Listing them as sharded suffixes reads
        # as coverage but can never fire, so the names must stay out.
        for suffix in ("moe_mlp.experts.w13_weight", "moe_mlp.experts.w2"):
            with self.subTest(suffix=suffix):
                self.assertFalse(_is_ep_sharded(f"blocks.5.mlp.{suffix}"))

    def test_replicated_tensors_are_not_sharded(self):
        # The dangerous direction: a false positive here row-slices a replicated
        # tensor, so a rank silently loads a fraction of its own weights.
        for name in (
            "blocks.5.mlp.split_linear.weight",
            "blocks.5.mlp.shared_expert_fc1.weight",
            "blocks.5.attention.linear_qkv.weight",
            "pre_adapter.video_embedder.weight",
        ):
            with self.subTest(name=name):
                self.assertFalse(_is_ep_sharded(name))


if __name__ == "__main__":
    unittest.main()
