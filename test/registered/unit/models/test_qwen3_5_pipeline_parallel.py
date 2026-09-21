import inspect
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import Qwen3_5MoeForConditionalGeneration
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpModel,
    _is_weight_outside_pp_stage,
    _pack_qwen4_exp_pp_proxy,
    _unpack_qwen4_exp_pp_proxy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestQwen3_5PipelineParallel(CustomTestCase):
    @staticmethod
    def _make_mtp_weight_loader_stub():
        model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3)
        model.config = SimpleNamespace(num_experts=None)
        model.quant_config = None
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(torch.nan)
        return model

    @staticmethod
    def _get_num_fused_shared_experts(layers, start_layer, end_layer):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
        return Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)

    def test_get_num_fused_shared_experts_returns_zero_without_layers(self):
        model = SimpleNamespace(model=SimpleNamespace())

        num_fused_shared_experts = (
            Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_get_num_fused_shared_experts_uses_local_pp_layers(self):
        layers = [
            PPMissingLayer(),
            PPMissingLayer(),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=2,
            end_layer=4,
        )

        self.assertEqual(num_fused_shared_experts, 1)

    def test_get_num_fused_shared_experts_returns_zero_without_local_fusion(self):
        layers = [
            PPMissingLayer(),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=1,
            end_layer=2,
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_mtp_loads_vl_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights(
            [("model.language_model.embed_tokens.weight", expected)]
        )

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_mtp_loads_text_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights([("model.embed_tokens.weight", expected)])

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_qwen4_exp_exposes_pipeline_parallel_inputs(self):
        self.assertIn(
            "pp_proxy_tensors",
            inspect.signature(Qwen4ExpForConditionalGeneration.forward).parameters,
        )
        self.assertIn(
            "pp_proxy_tensors", inspect.signature(Qwen4ExpModel.forward).parameters
        )

    def test_qwen4_exp_pipeline_proxy_uses_flat_hidden_state(self):
        hidden_states = torch.zeros((2, 10240))

        proxy = _pack_qwen4_exp_pp_proxy(hidden_states)
        unpacked_hidden_states, residual = _unpack_qwen4_exp_pp_proxy(proxy)

        self.assertEqual(set(proxy.tensors), {"hidden_states"})
        self.assertIs(unpacked_hidden_states, hidden_states)
        self.assertIsNone(residual)

    def test_qwen4_exp_skips_ple_weights_outside_local_pp_stage(self):
        self.assertTrue(
            _is_weight_outside_pp_stage(
                "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
                start_layer=24,
                end_layer=48,
            )
        )
        self.assertFalse(
            _is_weight_outside_pp_stage(
                "model.layers.24.ple.ple_embedding.ngram_embedding.shard_0.weight",
                start_layer=24,
                end_layer=48,
            )
        )


if __name__ == "__main__":
    unittest.main()
