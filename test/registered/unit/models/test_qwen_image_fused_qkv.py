"""CPU-only coverage for Qwen-Image fused QKV projection wiring."""

import unittest
from unittest.mock import patch

from torch import nn

import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_image
from sglang.multimodal_gen.configs.models.dits.qwenimage import QwenImageArchConfig
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ModuleStub(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class TestQwenImageFusedQKV(CustomTestCase):
    def _make_attention(self, quant_config):
        with (
            patch.object(qwen_image, "get_tp_world_size", return_value=2),
            patch.object(qwen_image, "MergedColumnParallelLinear", _ModuleStub),
            patch.object(qwen_image, "ColumnParallelLinear", _ModuleStub),
            patch.object(qwen_image, "RowParallelLinear", _ModuleStub),
            patch.object(qwen_image, "RMSNorm", _ModuleStub),
            patch.object(qwen_image, "USPAttention", _ModuleStub),
        ):
            return qwen_image.QwenImageCrossAttention(
                dim=16,
                num_heads=4,
                head_dim=4,
                added_kv_proj_dim=16,
                context_pre_only=False,
                quant_config=quant_config,
                prefix="transformer_blocks.0.attn",
            )

    def test_unquantized_attention_merges_only_text_projection(self):
        attention = self._make_attention(quant_config=None)

        self.assertFalse(attention.use_fused_qkv)
        self.assertTrue(attention.use_fused_added_qkv)
        self.assertIsInstance(attention.to_q, _ModuleStub)
        self.assertIsInstance(attention.to_added_qkv, _ModuleStub)
        self.assertFalse(hasattr(attention, "to_qkv"))
        self.assertFalse(hasattr(attention, "add_q_proj"))

    def test_other_quantization_keeps_unfused_projections(self):
        attention = self._make_attention(quant_config=object())

        self.assertFalse(attention.use_fused_qkv)
        self.assertFalse(attention.use_fused_added_qkv)
        self.assertIsInstance(attention.to_q, _ModuleStub)
        self.assertIsInstance(attention.add_q_proj, _ModuleStub)
        self.assertFalse(hasattr(attention, "to_qkv"))
        self.assertFalse(hasattr(attention, "to_added_qkv"))

    def test_checkpoint_projection_rules_merge_in_qkv_order(self):
        mapping = get_param_names_mapping(QwenImageArchConfig().param_names_mapping)

        cases = (
            ("add_q_proj", "to_added_qkv", 0),
            ("add_k_proj", "to_added_qkv", 1),
            ("add_v_proj", "to_added_qkv", 2),
        )
        for source, target, merge_index in cases:
            with self.subTest(source=source):
                mapped, actual_index, merge_count = mapping(
                    f"transformer_blocks.0.attn.{source}.weight"
                )
                self.assertEqual(mapped, f"transformer_blocks.0.attn.{target}.weight")
                self.assertEqual(actual_index, merge_index)
                self.assertEqual(merge_count, 3)


if __name__ == "__main__":
    unittest.main()
