"""Step5 LoRA geometry: per-layer adapter widths and routed experts kept out of LoRA.

An MoE layer's gate_up_proj/down_proj is its shared expert (share_expert_dim wide),
a dense layer's is its MLP (intermediate_size wide); the generic fallback would size
every layer from intermediate_size. Routed experts never take an adapter, so their
FusedMoE stays unwrapped and the dense buffers are allocated in every layer.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.utils import lora_adapts_routed_experts
from sglang.srt.models.step5 import Step5ForCausalLM
from sglang.srt.models.step5_vl import Step5ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _step5():
    model = Step5ForCausalLM.__new__(Step5ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        hidden_size=4096,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=192,
        intermediate_size=13824,
        share_expert_dim=1536,
        moe_layer_list=[1, 2, 3],
    )
    return model


class TestStep5LoRAHiddenDim(unittest.TestCase):
    def test_attention_widths(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("qkv_proj", 0), (4096, 192 * (64 + 8)))
        self.assertEqual(model.get_hidden_dim("o_proj", 1), (192 * 64, 4096))

    def test_dense_layer_mlp(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 0), (4096, 2 * 13824))
        self.assertEqual(model.get_hidden_dim("down_proj", 4), (13824, 4096))

    def test_moe_layer_shared_expert(self):
        model = _step5()
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 2), (4096, 2 * 1536))
        self.assertEqual(model.get_hidden_dim("down_proj", 3), (1536, 4096))

    def test_frozen_projections_are_rejected(self):
        model = _step5()
        for module_name in ("g_proj", "gate_up_proj_moe", "down_proj_moe"):
            with self.assertRaises(ValueError):
                model.get_hidden_dim(module_name, 1)

    def test_vl_wrapper_delegates_to_the_language_model(self):
        wrapper = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
        torch.nn.Module.__init__(wrapper)
        wrapper.language_model = _step5()
        self.assertEqual(wrapper.get_hidden_dim("down_proj", 1), (1536, 4096))


class TestStep5RoutedExpertsStayFrozen(unittest.TestCase):
    def test_step5_models_opt_out(self):
        self.assertFalse(lora_adapts_routed_experts(_step5()))
        wrapper = Step5ForConditionalGeneration.__new__(Step5ForConditionalGeneration)
        self.assertFalse(lora_adapts_routed_experts(wrapper))

    def test_other_models_keep_routed_expert_lora(self):
        self.assertTrue(lora_adapts_routed_experts(torch.nn.Module()))


if __name__ == "__main__":
    unittest.main()
