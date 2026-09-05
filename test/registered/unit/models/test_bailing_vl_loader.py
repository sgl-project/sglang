"""Regression tests for streaming Bailing multimodal weight dispatch."""

import unittest

import torch
import torch.nn as nn

from sglang.srt.configs.bailing_hybrid import (
    BailingHybridConfig,
    BailingMoeV3VLConfig,
)
from sglang.srt.models.bailing_mm import BailingMMNativeForConditionalGeneration
from sglang.srt.models.bailing_mm_v3 import (
    BailingMoeV3VLForConditionalGeneration,
)
from sglang.srt.models.bailing_moe_v3 import is_bailing_multi_gate_enabled
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


class _PublicRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3, 2))
        self.expert_bias = nn.Parameter(torch.zeros(3))


class _PublicTextLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = _PublicRouter()


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.word_embeddings = nn.Embedding(3, 2)
        self.model.layers = nn.ModuleList([_PublicTextLayer()])
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
            wrapper.multi_gate_enabled = False
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
                (
                    "model.layers.0.mlp.gate.weight",
                    "model.model.layers.0.mlp.gate.weight",
                    2,
                ),
                (
                    "model.layers.0.mlp.gate.expert_bias",
                    "model.model.layers.0.mlp.gate.expert_bias",
                    3,
                ),
                ("model.norm.weight", "model.model.norm.weight", 4),
                ("lm_head.weight", "model.lm_head.weight", 5),
                (
                    "model.visual.blocks.0.attn.qkv.weight",
                    "visual.blocks.0.attn.qkv_proj.weight",
                    6,
                ),
                (
                    "model.visual.blocks.0.attn.qkv.bias",
                    "visual.blocks.0.attn.qkv_proj.bias",
                    7,
                ),
                (
                    "model.visual.blocks.0.attn.proj.weight",
                    "visual.blocks.0.attn.proj.weight",
                    8,
                ),
                (
                    "model.visual.blocks.0.attn.proj.bias",
                    "visual.blocks.0.attn.proj.bias",
                    9,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc1.weight",
                    "visual.blocks.0.mlp.linear_fc1.weight",
                    10,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc1.bias",
                    "visual.blocks.0.mlp.linear_fc1.bias",
                    11,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc2.weight",
                    "visual.blocks.0.mlp.linear_fc2.weight",
                    12,
                ),
                (
                    "model.visual.blocks.0.mlp.linear_fc2.bias",
                    "visual.blocks.0.mlp.linear_fc2.bias",
                    13,
                ),
                (
                    "model.visual.patch_embed.proj.weight",
                    "visual.patch_embed.proj.weight",
                    14,
                ),
                (
                    "model.visual.patch_embed.proj.bias",
                    "visual.patch_embed.proj.bias",
                    15,
                ),
                ("model.visual.pos_embed.weight", "visual.pos_embed.weight", 16),
                ("model.visual.merger.norm.weight", "visual.merger.norm.weight", 17),
                ("model.visual.merger.norm.bias", "visual.merger.norm.bias", 18),
                ("linear_proj.0.weight", "linear_proj.0.weight", 19),
                ("linear_proj.0.bias", "linear_proj.0.bias", 20),
                ("linear_proj.2.weight", "linear_proj.2.weight", 21),
                ("linear_proj.2.bias", "linear_proj.2.bias", 22),
            ],
        )

    def test_v3_loader_accepts_public_checkpoint_names_once(self):
        """The public checkpoint layout must load without a second iterator pass."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        weights = _OneShotWeights(self._public_weights(wrapper))

        wrapper.load_weights(weights)

        self.assertEqual(weights.iterations, 1)
        torch.testing.assert_close(
            wrapper.model.model.layers[0].mlp.gate.expert_bias,
            torch.full((3,), 3.0),
        )
        torch.testing.assert_close(
            wrapper.visual.blocks[0].attn.qkv_proj.weight,
            torch.full((6, 2), 6.0),
        )
        torch.testing.assert_close(wrapper.linear_proj[2].bias, torch.full((2,), 22.0))

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

    def test_public_config_selects_standard_single_router(self):
        """Absent MultiRouter evidence must retain the public single gate and bias."""
        config = BailingMoeV3VLConfig(
            text_config={
                "score_function": "sigmoid",
                "moe_router_enable_expert_bias": True,
                "routed_scaling_factor": 2.5,
                "n_group": 8,
                "topk_group": 4,
                "num_experts": 512,
                "num_experts_per_tok": 8,
            }
        )

        self.assertFalse(is_bailing_multi_gate_enabled(config.text_config))
        self.assertTrue(config.text_config.moe_router_enable_expert_bias)
        self.assertEqual(config.text_config.score_function, "sigmoid")

    def test_multi_gate_requires_explicit_config_evidence(self):
        """Internal MultiRouter checkpoints remain reachable only by declaration."""
        for config in (
            BailingHybridConfig(multi_gate=True),
            BailingHybridConfig(router_type="MultiRouter"),
        ):
            with self.subTest(config=config):
                self.assertTrue(is_bailing_multi_gate_enabled(config))

    def test_required_multimodal_weight_coverage_is_enforced(self):
        """A truncated public checkpoint must not leave random projection bias."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        weights = _OneShotWeights(self._public_weights(wrapper)[:-1])

        with self.assertRaisesRegex(
            RuntimeError, "Missing required Bailing VL weights"
        ):
            wrapper.load_weights(weights)

    def test_required_single_router_weight_coverage_is_enforced(self):
        """Every public MoE layer must load its sole gate weight and expert bias."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        public_weights = self._public_weights(wrapper)

        for missing_name in (
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.gate.expert_bias",
        ):
            with self.subTest(missing_name=missing_name):
                weights = _OneShotWeights(
                    [item for item in public_weights if item[0] != missing_name]
                )
                with self.assertRaisesRegex(
                    RuntimeError, "Missing required Bailing VL router weights"
                ):
                    wrapper.load_weights(weights)


if __name__ == "__main__":
    unittest.main()
