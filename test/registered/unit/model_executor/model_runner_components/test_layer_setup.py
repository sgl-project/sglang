"""Unit tests for model-runner layer discovery."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.layer_setup import (
    compute_attention_and_moe_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestComputeAttentionAndMoeLayers(unittest.TestCase):
    def test_native_discovery_does_not_read_ffn(self):
        class NativeLayer:
            mlp = None

            @property
            def ffn(self):
                raise AssertionError("Native layer discovery must not read ffn")

        mlp = SimpleNamespace(experts=object())
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(mlp=mlp),
                SimpleNamespace(ffn=SimpleNamespace(experts=object())),
                NativeLayer(),
                SimpleNamespace(),
            ]
        )

        result = compute_attention_and_moe_layers(layer_model)

        self.assertEqual(result.moe_layers, [mlp.experts, None, None, None])
        self.assertEqual(result.moe_fusions, [mlp, None, None, None])
        self.assertEqual(result.attention_layers, [None] * 4)

    def test_explicit_moe_members_remain_supported(self):
        for member in ("block_sparse_moe", "moe", "mixer"):
            with self.subTest(member=member):
                moe = SimpleNamespace(experts=object())
                layer_model = SimpleNamespace(layers=[SimpleNamespace(**{member: moe})])
                result = compute_attention_and_moe_layers(layer_model)
                self.assertEqual(result.moe_layers, [moe.experts])
                self.assertEqual(result.moe_fusions, [moe])

    def test_model_runner_uses_transformers_adapter_without_reregistering(self):
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.models.transformers import TransformersForCausalLM

        class UpstreamFFN(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                return self.experts(x)

        class UpstreamLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = UpstreamFFN()

            def forward(self, x):
                return self.mlp(x)

        legacy = UpstreamLayer()
        modern = nn.Module()
        modern.ffn = UpstreamFFN()
        both = UpstreamLayer()
        both.ffn = UpstreamFFN()
        layer_model = nn.Module()
        layer_model.layers = nn.ModuleList([legacy, modern, both, nn.Module()])
        params_before = {
            name: param.data_ptr() for name, param in layer_model.named_parameters()
        }
        state_names_before = set(layer_model.state_dict())
        x = torch.ones(1, 2)
        output_before = legacy(x)

        # Native discovery reads the current SGLang member. Only the
        # Transformers adapter supplies a policy for third-party members.
        class NativeModel(nn.Module):
            @property
            def get_layer_ffn(self):
                raise AssertionError("ModelRunner must not probe optional FFN hooks")

        native_result = ModelRunner.get_cuda_graph_layers(
            SimpleNamespace(model=NativeModel()), layer_model
        )
        self.assertEqual(native_result.moe_fusions, [legacy.mlp, None, both.mlp, None])

        adapter = TransformersForCausalLM.__new__(TransformersForCausalLM)
        nn.Module.__init__(adapter)
        adapter.model = layer_model
        result = ModelRunner.get_cuda_graph_layers(
            SimpleNamespace(model=adapter), layer_model
        )
        self.assertEqual(result.moe_fusions, [legacy.mlp, modern.ffn, both.ffn, None])
        self.assertEqual(
            result.moe_layers,
            [legacy.mlp.experts, modern.ffn.experts, both.ffn.experts, None],
        )
        self.assertEqual(result.attention_layers, [None] * 4)
        self.assertEqual(
            params_before,
            {name: param.data_ptr() for name, param in layer_model.named_parameters()},
        )
        self.assertEqual(state_names_before, set(layer_model.state_dict()))
        self.assertFalse(hasattr(legacy, "ffn"))
        self.assertTrue(torch.equal(legacy(x), output_before))

    def test_deepseek_mla_registers_mha_companion(self):
        attn_mqa = SimpleNamespace()
        attn_mha = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    self_attn=SimpleNamespace(attn_mqa=attn_mqa, attn_mha=attn_mha)
                )
            ]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [attn_mqa])
        self.assertEqual(mha_companion_layers, [attn_mha])
        self.assertNotIn("_pcg_mha_companion", vars(attn_mqa))

    def test_pipeline_placeholders_preserve_global_layer_ids(self):
        local_attention = SimpleNamespace()
        layer_model = SimpleNamespace(
            layers=[SimpleNamespace(), SimpleNamespace()]
            + [SimpleNamespace(self_attn=SimpleNamespace(attn=local_attention))]
        )

        attention_layers, _, _, _, mha_companion_layers = (
            compute_attention_and_moe_layers(layer_model)
        )

        self.assertEqual(attention_layers, [None, None, local_attention])
        self.assertEqual(mha_companion_layers, [None, None, None])


if __name__ == "__main__":
    unittest.main()
