"""CPU regression tests for PEFT GDN packing."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.utils import get_normalized_target_modules, get_stacked_multiply
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen35PEFTLoRA(CustomTestCase):
    def adapter(self, rslora=False):
        cfg = LoRAConfig.from_dict(
            dict(
                peft_type="LORA",
                r=4,
                lora_alpha=8,
                use_rslora=rslora,
                target_modules=["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"],
            )
        )
        return LoRAAdapter(
            "test", cfg, SimpleNamespace(num_hidden_layers=1), None, None
        )

    def test_peft_gdn_packing_preserves_deltas(self):
        torch.manual_seed(42)
        rank, hidden = 4, 8
        sizes = {"in_proj_qkv": 16, "in_proj_z": 8, "in_proj_b": 2, "in_proj_a": 2}
        weights = {}
        originals = {}
        prefix = "base_model.model.model.layers.0.linear_attn."
        for module, out_dim in sizes.items():
            a, b = torch.randn(rank, hidden), torch.randn(out_dim, rank)
            originals[module] = (a, b)
            weights[prefix + module + ".lora_A.weight"] = a
            weights[prefix + module + ".lora_B.weight"] = b
        adapter = self.adapter()
        adapter.initialize_weights_from_tensors(weights)
        packed = adapter.layers[0].weights
        self.assertEqual(len(packed), 4)
        x = torch.randn(3, hidden)
        for packed_name, modules, split_sizes in (
            ("in_proj_qkvz", ["in_proj_qkv", "in_proj_z"], [4, 4, 8, 8]),
            ("in_proj_ba", ["in_proj_b", "in_proj_a"], [2, 2]),
        ):
            a = packed[prefix + packed_name + ".lora_A.weight"]
            b = packed[prefix + packed_name + ".lora_B.weight"]
            a_slices = a.split(rank)
            b_slices = b.split(split_sizes)
            result = torch.cat(
                [x @ ai.T @ bi.T for ai, bi in zip(a_slices, b_slices)], dim=-1
            )
            expected = torch.cat(
                [x @ originals[m][0].T @ originals[m][1].T for m in modules], dim=-1
            )
            torch.testing.assert_close(result, expected)
            self.assertEqual(len(a_slices), get_stacked_multiply(packed_name))
        self.assertEqual(
            get_normalized_target_modules(sizes), {"in_proj_qkvz", "in_proj_ba"}
        )

    def test_missing_gdn_partner_is_rejected(self):
        # Dropping a trained projection must not silently serve a different adapter.
        for name in ("in_proj_qkv", "in_proj_b"):
            with self.subTest(projection=name):
                weights = {
                    f"model.layers.0.linear_attn.{name}.lora_A.weight": torch.ones(
                        4, 8
                    ),
                }
                with self.assertRaisesRegex(ValueError, "requires"):
                    self.adapter().initialize_weights_from_tensors(weights)

    def test_existing_split_qkvz_layout_preserves_deltas(self):
        # Pre-existing four-projection checkpoints must not repeat A twice.
        adapter = self.adapter()
        prefix = "model.layers.0.linear_attn."
        weights = {}
        expected = []
        x = torch.arange(8, dtype=torch.float32).reshape(1, 8)
        for index, name in enumerate(
            ("in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z")
        ):
            a = torch.full((4, 8), index + 1.0)
            b = torch.full((2, 4), index + 0.5)
            weights[f"{prefix}{name}.lora_A.weight"] = a
            weights[f"{prefix}{name}.lora_B.weight"] = b
            expected.append(x @ a.T @ b.T)
        adapter.initialize_weights_from_tensors(weights)
        loaded = adapter.layers[0].weights
        a = loaded[f"{prefix}in_proj_qkvz.lora_A.weight"]
        b = loaded[f"{prefix}in_proj_qkvz.lora_B.weight"]
        self.assertEqual(a.shape, (16, 8))
        actual = torch.cat(
            [x @ ai.T @ bi.T for ai, bi in zip(a.split(4), b.split(2))], -1
        )
        torch.testing.assert_close(actual, torch.cat(expected, -1))


if __name__ == "__main__":
    unittest.main()
