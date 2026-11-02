"""Weight-loading regressions for Bolt E4B NVFP4 DFlash.

Checkpoint: nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4.
"""

import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashHeadWeights(CustomTestCase):
    def make_model(self):
        model = DFlashDraftModel.__new__(DFlashDraftModel)
        nn.Module.__init__(model)
        model.projector_type = None
        model.layers = nn.ModuleList()
        model.own_lm_head = nn.Linear(8, 16, bias=False)
        model._own_lm_head_loaded = False
        model.norm = nn.Linear(8, 8, bias=False)
        return model

    def test_checkpoint_head_is_loaded_instead_of_silently_discarded(self):
        for prefix in ("", "model."):
            with self.subTest(prefix=prefix):
                model = self.make_model()
                weight = torch.arange(128, dtype=torch.float32).reshape(16, 8)
                model.load_weights([(prefix + "lm_head.weight", weight)])
                torch.testing.assert_close(model.own_lm_head.weight, weight)

    def test_partial_updates_preserve_the_loaded_head(self):
        model = self.make_model()
        head = torch.arange(128, dtype=torch.float32).reshape(16, 8)
        model.load_weights([("lm_head.weight", head)])

        norm = torch.full((8, 8), 0.25)
        model.load_weights([("norm.weight", norm)])
        torch.testing.assert_close(model.norm.weight, norm)
        torch.testing.assert_close(model.own_lm_head.weight, head)

        updated_head = -head
        model.load_weights([("lm_head.weight", updated_head)])
        torch.testing.assert_close(model.own_lm_head.weight, updated_head)
        torch.testing.assert_close(model.norm.weight, norm)

    def test_partial_update_after_a_loader_bypasses_load_weights(self):
        model = self.make_model()
        head = torch.arange(128, dtype=torch.float32).reshape(16, 8)
        with torch.no_grad():
            model.own_lm_head.weight.copy_(head)
        model.post_load_weights()

        norm = torch.full((8, 8), 0.5)
        model.load_weights([("norm.weight", norm)])
        torch.testing.assert_close(model.norm.weight, norm)
        torch.testing.assert_close(model.own_lm_head.weight, head)

    def test_owned_head_rejects_incompatible_output_paths(self):
        for dflash_config in (
            {"selector_rank": 1, "selector_top_k": 1},
            {
                "projector_type": "domino",
                "shift_label": True,
                "pure_draft_prefix_len": 1,
                "gru_hidden_dim": 4,
                "emb_dim": 4,
            },
        ):
            with self.subTest(dflash_config=dflash_config):
                config = Qwen3Config(
                    hidden_size=8,
                    num_hidden_layers=1,
                    has_own_lm_head=True,
                    dflash_config=dflash_config,
                )
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    DFlashDraftModel(config)

    def test_declared_head_cannot_remain_uninitialized(self):
        with self.assertRaisesRegex(ValueError, "missing head weights"):
            self.make_model().load_weights([])


if __name__ == "__main__":
    unittest.main()
