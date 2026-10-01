import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Linear(torch.nn.Module):
    def __init__(self, rows, hidden):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, hidden, dtype=torch.bfloat16), requires_grad=False
        )
        self.bias = None
        self.quant_method = UnquantizedLinearMethod()


class TestQwen3_5PackedGDNInProj(CustomTestCase):
    # Qwen3.5-4B widths with a small hidden size.
    QKVZ, BA, HIDDEN = 12288, 64, 256

    def setUp(self):
        lora = patch.object(
            qwen3_5,
            "get_lora",
            return_value=SimpleNamespace(enable_lora=False, lora_paths=None),
        )
        lora.start()
        self.addCleanup(lora.stop)

    def _make_model(self, wrapper_cls, model_type):
        gdn = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
        torch.nn.Module.__init__(gdn)
        gdn.in_proj_qkvz = _Linear(self.QKVZ, self.HIDDEN)
        gdn.in_proj_ba = _Linear(self.BA, self.HIDDEN)
        gdn._fused_in_proj_weight = None
        gdn._fused_in_proj_qkvz_width = 0
        backbone = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        torch.nn.Module.__init__(backbone)
        backbone.config = SimpleNamespace(model_type=model_type)
        backbone.gdn = gdn
        model = wrapper_cls.__new__(wrapper_cls)
        torch.nn.Module.__init__(model)
        model.model = backbone
        return model, gdn

    def test_dense_wrapper_packs_on_cuda(self):
        """The model runner calls the hook on the top-level model; the dense
        Qwen3.5 wrapper must reach the backbone, as the MoE wrapper does."""
        for wrapper_cls in (
            Qwen3_5ForConditionalGeneration,
            Qwen3_5MoeForConditionalGeneration,
        ):
            with self.subTest(wrapper=wrapper_cls.__name__):
                model, gdn = self._make_model(wrapper_cls, "qwen3_5_text")
                qkvz, ba = (
                    gdn.in_proj_qkvz.weight.clone(),
                    gdn.in_proj_ba.weight.clone(),
                )
                with patch.object(qwen3_5, "_is_cuda", True):
                    model.prepare_before_cuda_graph_capture(model_runner=None)

                self.assertIsNotNone(gdn._fused_in_proj_weight)
                x = torch.randn(64, self.HIDDEN, dtype=torch.bfloat16)
                # bf16_gemm_dispatch has no CPU kernel; the split is under test.
                with patch.object(
                    qwen3_5, "bf16_gemm_dispatch", torch.nn.functional.linear
                ):
                    got_qkvz, got_ba = gdn._forward_input_proj(x)
                torch.testing.assert_close(
                    got_qkvz, torch.nn.functional.linear(x, qkvz)
                )
                torch.testing.assert_close(got_ba, torch.nn.functional.linear(x, ba))

    def test_packing_needs_cuda_or_aiter_and_qwen3_5(self):
        for is_cuda, model_type in (
            (False, "qwen3_5_text"),
            (True, "qwen4_exp_text"),
        ):
            with self.subTest(is_cuda=is_cuda, model_type=model_type):
                model, gdn = self._make_model(
                    Qwen3_5ForConditionalGeneration, model_type
                )
                with (
                    patch.object(qwen3_5, "_is_cuda", is_cuda),
                    patch.object(qwen3_5, "_use_aiter", False),
                ):
                    model.prepare_before_cuda_graph_capture(model_runner=None)

                self.assertIsNone(gdn._fused_in_proj_weight)


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


if __name__ == "__main__":
    unittest.main()
