"""
Unit tests for the missing-vision-weights check in the Qwen3-VL / Qwen3.5 loaders.

A text-only checkpoint whose config still declares a vision tower used to load
without error, leaving the tower uninitialized for the multimodal warmup request
and every image request. The initial load now raises instead; later load_weights
calls (partial updates such as RL weight sync) are not checked.

Related: https://github.com/sgl-project/sglang/issues/26745
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration

TEXT_PARAM = "model.embed_tokens.weight"
VISUAL_PARAM = "visual.patch_embed.proj.weight"
TEXT_CKPT = ("model.language_model.embed_tokens.weight", torch.ones(4, 2))
VISUAL_CKPT = ("model.visual.patch_embed.proj.weight", torch.ones(4, 2))


def _param(shape):
    param = nn.Parameter(torch.zeros(shape), requires_grad=False)
    param.weight_loader = lambda p, w, shard_id=None: p.data.copy_(w)
    return param


class _Holder(nn.Module):
    """A module tree whose parameter names match the Qwen3-VL layout."""

    _check_vision_weights_loaded = (
        Qwen3VLForConditionalGeneration._check_vision_weights_loaded
    )
    post_load_weights = Qwen3VLForConditionalGeneration.post_load_weights

    def __init__(self, param_names, **config):
        super().__init__()
        for full_name in param_names:
            *path, leaf = full_name.split(".")
            module = self
            for part in path:
                if not hasattr(module, part):
                    module.add_module(part, nn.Module())
                module = getattr(module, part)
            module.register_parameter(leaf, _param((4, 2)))
        if not hasattr(self, "visual"):
            self.visual = None  # language_model_only builds no tower
        self.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_experts=2,
            encoder_only=False,
            **config,
        )
        self.pp_group = SimpleNamespace(is_last_rank=True)
        self.pooler = None
        self.enable_shared_expert_fusion = False
        self.num_fused_shared_experts = 0
        self._initial_weights_loaded = False


LOADERS = {
    "qwen3_vl": Qwen3VLForConditionalGeneration.load_weights,
    "qwen3_5": Qwen3_5ForConditionalGeneration.load_weights,
    "qwen3_5_moe": Qwen3_5MoeForConditionalGeneration.load_weights,
}


class TestMissingVisionWeights(unittest.TestCase):
    def test_text_only_checkpoint_is_rejected(self):
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM, VISUAL_PARAM])
                with self.assertRaisesRegex(ValueError, "language_model_only"):
                    load_weights(model, [TEXT_CKPT])

    def test_full_checkpoint_loads(self):
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM, VISUAL_PARAM])
                load_weights(model, [TEXT_CKPT, VISUAL_CKPT])
                loaded = model.visual.patch_embed.proj.weight
                self.assertTrue(torch.equal(loaded, VISUAL_CKPT[1]))

    def test_text_only_update_after_full_load(self):
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM, VISUAL_PARAM])
                load_weights(model, [TEXT_CKPT, VISUAL_CKPT])
                load_weights(model, [TEXT_CKPT])

    def test_text_only_update_after_loader_without_load_weights(self):
        # Dummy / sharded-state / remote loaders call post_load_weights instead.
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM, VISUAL_PARAM])
                model.post_load_weights()
                load_weights(model, [TEXT_CKPT])

    def test_language_model_only_has_no_tower(self):
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM])
                load_weights(model, [TEXT_CKPT])

    def test_language_only_worker_is_not_checked(self):
        # Encoder disaggregation: the language worker gets embeddings from the
        # encoder server and may load only the language model.
        for label, load_weights in LOADERS.items():
            with self.subTest(label):
                model = _Holder([TEXT_PARAM, VISUAL_PARAM], language_only=True)
                load_weights(model, [TEXT_CKPT])


if __name__ == "__main__":
    unittest.main()
