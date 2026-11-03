# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Qwen3.5 GDN LoRA packed-projection contracts."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.utils import (
    get_normalized_target_modules,
    get_stacked_multiply,
)
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_PREFIX = "base_model.model.model.layers.0.linear_attn"


def _weight_name(module_name: str, factor: str) -> str:
    return f"{_PREFIX}.{module_name}.lora_{factor}.weight"


def _adapter() -> LoRAAdapter:
    return LoRAAdapter.__new__(LoRAAdapter)


class TestQwen3_5GDNTargetContracts(CustomTestCase):
    def test_split_checkpoint_targets_normalize_to_fused_runtime_modules(self):
        self.assertEqual(
            get_normalized_target_modules(
                [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ]
            ),
            {"in_proj_qkvz", "in_proj_ba", "out_proj"},
        )

    def test_fused_runtime_modules_are_declared_and_sized(self):
        self.assertIn("in_proj_qkvz", Qwen3_5ForCausalLM.supported_lora_modules)
        self.assertIn("in_proj_ba", Qwen3_5ForCausalLM.supported_lora_modules)
        self.assertEqual(get_stacked_multiply("in_proj_qkvz"), 4)
        self.assertEqual(get_stacked_multiply("in_proj_ba"), 2)

        for module_name in ("in_proj_qkvz", "in_proj_ba", "out_proj"):
            self.assertIn(module_name, SUPPORTED_LORA_TARGET_MODULES)

        model = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            hidden_size=4096,
            head_dim=128,
            num_attention_heads=32,
            linear_num_value_heads=32,
        )
        self.assertEqual(model.get_hidden_dim("in_proj_ba", 0), (4096, 64))


class TestQwen3_5GDNWeightNormalization(CustomTestCase):
    def test_grouped_qkv_and_z_weights_expand_to_four_runtime_slices(self):
        qkv_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        z_a = qkv_a + 100
        qkv_b = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        z_b = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 100
        weights = {
            _weight_name("in_proj_qkv", "A"): qkv_a,
            _weight_name("in_proj_z", "A"): z_a,
            _weight_name("in_proj_qkv", "B"): qkv_b,
            _weight_name("in_proj_z", "B"): z_b,
        }

        _adapter()._normalize_in_proj_qkvz(weights)

        fused_a = _weight_name("in_proj_qkvz", "A")
        fused_b = _weight_name("in_proj_qkvz", "B")
        self.assertEqual(set(weights), {fused_a, fused_b})
        torch.testing.assert_close(
            weights[fused_a],
            torch.cat((qkv_a, qkv_a, qkv_a, z_a), dim=0),
        )
        torch.testing.assert_close(weights[fused_b], torch.cat((qkv_b, z_b), dim=0))

    def test_split_b_and_a_weights_stack_into_runtime_module(self):
        b_lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        a_lora_a = b_lora_a + 100
        b_lora_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        a_lora_b = b_lora_b + 100
        weights = {
            _weight_name("in_proj_b", "A"): b_lora_a,
            _weight_name("in_proj_a", "A"): a_lora_a,
            _weight_name("in_proj_b", "B"): b_lora_b,
            _weight_name("in_proj_a", "B"): a_lora_b,
        }

        _adapter()._normalize_in_proj_ba(weights)

        fused_a = _weight_name("in_proj_ba", "A")
        fused_b = _weight_name("in_proj_ba", "B")
        self.assertEqual(set(weights), {fused_a, fused_b})
        torch.testing.assert_close(
            weights[fused_a], torch.cat((b_lora_a, a_lora_a), dim=0)
        )
        torch.testing.assert_close(
            weights[fused_b], torch.cat((b_lora_b, a_lora_b), dim=0)
        )

    def test_already_fused_ba_repeats_only_the_shared_a_factor(self):
        fused_a = _weight_name("in_proj_ba", "A")
        fused_b = _weight_name("in_proj_ba", "B")
        lora_a = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        lora_b = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        weights = {fused_a: lora_a.clone(), fused_b: lora_b.clone()}

        _adapter()._normalize_in_proj_ba(weights)

        torch.testing.assert_close(weights[fused_a], lora_a.repeat(2, 1))
        torch.testing.assert_close(weights[fused_b], lora_b)


if __name__ == "__main__":
    unittest.main()
