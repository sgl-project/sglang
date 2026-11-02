"""Regression test for image-generation checkpoints that declare Qwen3-VL.

HiDream-ai/HiDream-O1-Image ships a stock ``qwen3_vl`` config and
``architectures=["Qwen3VLForConditionalGeneration"]``, so it resolves to this
class, but its checkpoint carries a flow-matching pixel head. The loader drops
weight names it has no parameter for, so the server used to come up healthy and
answer every request with an empty completion instead of refusing the model.
"""

import unittest

import torch

from sglang.srt.models.qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    check_not_image_generation_head,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=14, suite="base-a-test-cpu")

# Weight names taken from the HiDream-O1-Image safetensors index.
PIXEL_HEAD_NAMES = [
    "model.x_embedder.proj1.weight",
    "model.t_embedder1.mlp.0.bias",
    "model.final_layer2.linear.weight",
]

QWEN3_VL_NAMES = [
    "model.language_model.layers.0.self_attn.q_proj.weight",
    "model.language_model.norm.weight",
    "model.visual.patch_embed.proj.weight",
    "model.visual.pos_embed.weight",
    "model.visual.blocks.3.attn.qkv.weight",
    "model.visual.deepstack_merger_list.0.linear_fc1.weight",
    "lm_head.weight",
    # Near-misses that other checkpoints do use as real parameters.
    "model.final_layernorm.weight",
    "text_model.final_layer_norm.bias",
]


class TestQwen3VLImageGenerationReject(CustomTestCase):
    def test_load_weights_rejects_pixel_head(self):
        model = object.__new__(Qwen3VLForConditionalGeneration)
        model.named_parameters = lambda **kwargs: iter(())

        for name in PIXEL_HEAD_NAMES:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    model.load_weights([(name, torch.ones(1))])

    def test_qwen3_vl_names_are_accepted(self):
        for name in QWEN3_VL_NAMES:
            with self.subTest(name=name):
                check_not_image_generation_head(name)


if __name__ == "__main__":
    unittest.main()
