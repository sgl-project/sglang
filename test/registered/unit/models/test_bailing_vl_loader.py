"""Regression tests for streaming Bailing multimodal weight dispatch."""

import unittest

import torch
import torch.nn as nn

from sglang.srt.configs.bailing_hybrid import BailingMoeV3VLConfig
from sglang.srt.models.bailing_mm import BailingMMNativeForConditionalGeneration
from sglang.srt.models.bailing_mm_v3 import (
    BailingMoeV3VLForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _OneShotWeights:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.values)


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.word_embeddings = nn.Embedding(3, 2)
        self.model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.model.norm = nn.LayerNorm(2, bias=False)
        self.lm_head = nn.Linear(2, 3, bias=False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            params[name].data.copy_(value)
            loaded.add(name)
        return loaded


class _PublicVisionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv_proj = nn.Linear(2, 6)
        self.attn.proj = nn.Linear(2, 2)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(2, 4)
        self.mlp.linear_fc2 = nn.Linear(4, 2)


class _PublicVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Linear(2, 2)
        self.pos_embed = nn.Embedding(3, 2)
        self.blocks = nn.ModuleList([_PublicVisionBlock()])
        self.merger = nn.Module()
        self.merger.norm = nn.LayerNorm(2)
        # Simulate modules created by the inherited Qwen default. Coverage
        # must ignore them when Bailing deepstack is disabled.
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(2, 2)])


class TestBailingVLWeightLoading(CustomTestCase):
    @staticmethod
    def _wrapper(wrapper_class):
        wrapper = wrapper_class.__new__(wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.model = _TextModel()
        wrapper._build_mm_encoders = True
        if wrapper_class is BailingMoeV3VLForConditionalGeneration:
            wrapper.visual = _PublicVision()
            wrapper.linear_proj = nn.Sequential(
                nn.Linear(2, 2), nn.GELU(), nn.Linear(2, 2)
            )
            wrapper.deepstack_visual_indexes = ()
        else:
            wrapper.vision = nn.Identity()
            wrapper.linear_proj = nn.Linear(2, 2)
        return wrapper

    @staticmethod
    def _filled_weights(wrapper, checkpoint_to_parameter):
        params = dict(wrapper.named_parameters())
        return [
            (checkpoint_name, torch.full_like(params[parameter_name], value))
            for checkpoint_name, parameter_name, value in checkpoint_to_parameter
        ]

    @classmethod
    def _public_weights(cls, wrapper):
        return cls._filled_weights(
            wrapper,
            [
                (
                    "model.word_embeddings.weight",
                    "model.model.word_embeddings.weight",
                    1,
                ),
                ("model.layers.0.weight", "model.model.layers.0.weight", 2),
                ("model.norm.weight", "model.model.norm.weight", 3),
                ("lm_head.weight", "model.lm_head.weight", 4),
                (
                    "model.visual.blocks.0.attn.qkv.weight",
                    "visual.blocks.0.attn.qkv_proj.weight",
                    5,
                ),
                (
                    "model.visual.blocks.0.attn.qkv.bias",
                    "visual.blocks.0.attn.qkv_proj.bias",
                    6,
                ),
                (
                    "model.visual.blocks.0.attn.proj.weight",
                    "visual.blocks.0.attn.proj.weight",
                    7,
                ),
                (
                    "model.visual.blocks.0.attn.proj.bias",
                    "visual.blocks.0.attn.proj.bias",
                    8,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc1.weight",
                    "visual.blocks.0.mlp.linear_fc1.weight",
                    9,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc1.bias",
                    "visual.blocks.0.mlp.linear_fc1.bias",
                    10,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc2.weight",
                    "visual.blocks.0.mlp.linear_fc2.weight",
                    11,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc2.bias",
                    "visual.blocks.0.mlp.linear_fc2.bias",
                    12,
                ),
                (
                    "model.visual.patch_embed.proj.weight",
                    "visual.patch_embed.proj.weight",
                    13,
                ),
                (
                    "model.visual.patch_embed.proj.bias",
                    "visual.patch_embed.proj.bias",
                    14,
                ),
                ("model.visual.pos_embed.weight", "visual.pos_embed.weight", 15),
                ("model.visual.merger.norm.weight", "visual.merger.norm.weight", 16),
                ("model.visual.merger.norm.bias", "visual.merger.norm.bias", 17),
                ("linear_proj.0.weight", "linear_proj.0.weight", 18),
                ("linear_proj.0.bias", "linear_proj.0.bias", 19),
                ("linear_proj.2.weight", "linear_proj.2.weight", 20),
                ("linear_proj.2.bias", "linear_proj.2.bias", 21),
            ],
        )

    def test_v3_loader_accepts_public_checkpoint_names_once(self):
        """The public checkpoint layout must load without a second iterator pass."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        weights = _OneShotWeights(self._public_weights(wrapper))

        wrapper.load_weights(weights)

        self.assertEqual(weights.iterations, 1)
        torch.testing.assert_close(
            wrapper.visual.blocks[0].attn.qkv_proj.weight,
            torch.full((6, 2), 5.0),
        )
        torch.testing.assert_close(wrapper.linear_proj[2].bias, torch.full((2,), 21.0))

    def test_legacy_loader_consumes_checkpoint_once(self):
        """The legacy nested checkpoint layout must remain single-pass."""
        wrapper = self._wrapper(BailingMMNativeForConditionalGeneration)
        weights = _OneShotWeights(
            self._filled_weights(
                wrapper,
                [
                    (
                        "model.model.word_embeddings.weight",
                        "model.model.word_embeddings.weight",
                        1,
                    ),
                    ("model.linear_proj.weight", "linear_proj.weight", 2),
                    ("model.linear_proj.bias", "linear_proj.bias", 3),
                ],
            )
        )

        wrapper.load_weights(weights)

        self.assertEqual(weights.iterations, 1)
        torch.testing.assert_close(wrapper.linear_proj.weight, torch.full((2, 2), 2.0))

    def test_public_config_does_not_enable_qwen_deepstack_defaults(self):
        """An omitted public deepstack field must not create random modules."""
        config = BailingMoeV3VLConfig(vision_config={"disable_merger_proj": True})

        self.assertEqual(config.vision_config.deepstack_visual_indexes, [])

    def test_required_multimodal_weight_coverage_is_enforced(self):
        """A truncated public checkpoint must not leave random projection bias."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        weights = _OneShotWeights(self._public_weights(wrapper)[:-1])

        with self.assertRaisesRegex(
            RuntimeError, "Missing required Bailing VL weights"
        ):
            wrapper.load_weights(weights)


if __name__ == "__main__":
    unittest.main()
