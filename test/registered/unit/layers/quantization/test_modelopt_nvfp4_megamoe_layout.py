"""CPU regressions for ModelOpt NVFP4 MegaMoE W13 weight layout."""

import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _expected_deinterleave(source: torch.Tensor, *, up_first: bool) -> torch.Tensor:
    gate = source[..., 0::2, :]
    up = source[..., 1::2, :]
    return torch.cat((up, gate) if up_first else (gate, up), dim=-2)


class TestModelOptNvFp4MegaMoeLayout(CustomTestCase):
    @staticmethod
    def _make_method(a2a_backend, runner_backend):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod

        config = modelopt_mod.ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        with (
            patch.object(
                modelopt_mod,
                "get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            patch.object(
                modelopt_mod,
                "get_moe_runner_backend",
                return_value=runner_backend,
            ),
            patch.object(
                modelopt_mod,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
            patch.object(modelopt_mod, "is_cuda", return_value=False),
        ):
            return modelopt_mod.ModelOptNvFp4FusedMoEMethod(config)

    @staticmethod
    def _make_loader_layer(method, *, nominal_trtllm_layout):
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        layer = object.__new__(FusedMoE)
        torch.nn.Module.__init__(layer)
        layer.moe_tp_rank = 0
        layer.moe_tp_size = 1
        layer.scheme = None
        layer.quant_method = method
        layer.use_flashinfer_trtllm_moe = nominal_trtllm_layout
        layer.use_triton_kernels = False
        layer.use_presharded_weights = False
        layer._has_fused_shared = False
        layer._num_local_routed = 1
        layer.quant_config = SimpleNamespace(get_name=lambda: "nvfp4")
        layer.moe_runner_config = SimpleNamespace(is_gated=True)
        layer.__dict__["use_padded_loading"] = False
        return layer

    @staticmethod
    def _load_w13_pair(layer, method):
        weight = torch.nn.Parameter(torch.full((1, 4, 2), -1.0), requires_grad=False)
        block_scale = torch.nn.Parameter(
            torch.full((1, 4, 1), -1.0), requires_grad=False
        )
        tensor_scale = torch.nn.Parameter(torch.full((1, 2), -1.0), requires_grad=False)

        with patch.object(
            type(method),
            "load_up_proj_weight_first",
            new_callable=PropertyMock,
            return_value=False,
        ):
            for shard_id, value in (("w1", 11.0), ("w3", 33.0)):
                layer._weight_loader_impl(
                    weight,
                    torch.full((2, 2), value),
                    "w13_weight",
                    shard_id,
                    expert_id=0,
                )
                layer._weight_loader_impl(
                    block_scale,
                    torch.full((2, 1), value + 1),
                    "w13_weight_scale",
                    shard_id,
                    expert_id=0,
                )
                layer._weight_loader_impl(
                    tensor_scale,
                    torch.tensor(value + 2),
                    "w13_weight_scale_2",
                    shard_id,
                    expert_id=0,
                )

        return weight, block_scale, tensor_scale

    def test_layout_decision_matrix(self):
        trtllm_runners = (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.EXPERIMENTAL_SGL_TRTLLM,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        )
        cases = []
        for runner_backend in trtllm_runners:
            cases.extend(
                (
                    (MoeA2ABackend.MEGAMOE, runner_backend, True, False),
                    (MoeA2ABackend.NONE, runner_backend, True, True),
                    (MoeA2ABackend.DEEPEP, runner_backend, True, True),
                )
            )
        cases.extend(
            (
                (
                    MoeA2ABackend.MEGAMOE,
                    MoeRunnerBackend.DEEP_GEMM,
                    False,
                    False,
                ),
                (
                    MoeA2ABackend.NONE,
                    MoeRunnerBackend.AUTO,
                    False,
                    False,
                ),
                (
                    MoeA2ABackend.FLASHINFER_MEGAMOE,
                    MoeRunnerBackend.FLASHINFER_MEGAMOE,
                    False,
                    False,
                ),
            )
        )

        for a2a_backend, runner_backend, nominal, layout in cases:
            with self.subTest(a2a=a2a_backend, runner=runner_backend):
                method = self._make_method(a2a_backend, runner_backend)
                self.assertIs(method.enable_flashinfer_trtllm_moe, nominal)
                self.assertIs(method.use_flashinfer_trtllm_weight_layout, layout)
                if not a2a_backend.is_megamoe():
                    self.assertEqual(layout, nominal)

    def test_modelopt_loader_keeps_weights_and_scales_in_w13_for_megamoe(self):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod

        method = self._make_method(
            MoeA2ABackend.MEGAMOE,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        )
        method.moe_runner_config = SimpleNamespace(is_gated=True)
        layer = self._make_loader_layer(method, nominal_trtllm_layout=True)

        with patch.object(
            modelopt_mod,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ):
            weight, block_scale, tensor_scale = self._load_w13_pair(layer, method)

        torch.testing.assert_close(weight[0, :2], torch.full((2, 2), 11.0))
        torch.testing.assert_close(weight[0, 2:], torch.full((2, 2), 33.0))
        torch.testing.assert_close(block_scale[0, :2], torch.full((2, 1), 12.0))
        torch.testing.assert_close(block_scale[0, 2:], torch.full((2, 1), 34.0))
        torch.testing.assert_close(tensor_scale[0], torch.tensor([13.0, 35.0]))

    def test_modelopt_loader_keeps_w31_for_real_trtllm_consumer(self):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod

        method = self._make_method(
            MoeA2ABackend.NONE,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        )
        method.moe_runner_config = SimpleNamespace(is_gated=True)
        layer = self._make_loader_layer(method, nominal_trtllm_layout=True)

        with patch.object(
            modelopt_mod,
            "get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ):
            weight, block_scale, tensor_scale = self._load_w13_pair(layer, method)

        torch.testing.assert_close(weight[0, :2], torch.full((2, 2), 33.0))
        torch.testing.assert_close(weight[0, 2:], torch.full((2, 2), 11.0))
        torch.testing.assert_close(block_scale[0, :2], torch.full((2, 1), 34.0))
        torch.testing.assert_close(block_scale[0, 2:], torch.full((2, 1), 12.0))
        torch.testing.assert_close(tensor_scale[0], torch.tensor([35.0, 13.0]))

    def test_non_modelopt_methods_retain_layer_owned_w31_layout(self):
        import sglang.srt.layers.moe.fused_moe_triton.layer as layer_mod
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsMxInt4MoE,
        )
        from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
        from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

        unquantized = UnquantizedFusedMoEMethod(use_flashinfer_trtllm_moe=True)
        unquantized.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load = (
            lambda **_kwargs: None
        )
        methods = (
            ("unquantized", unquantized),
            ("fp8", object.__new__(Fp8MoEMethod)),
            ("mxint4", object.__new__(CompressedTensorsMxInt4MoE)),
        )

        for name, method in methods:
            with self.subTest(method=name):
                layer = self._make_loader_layer(method, nominal_trtllm_layout=True)
                weight = torch.nn.Parameter(
                    torch.full((1, 4, 2), -1.0), requires_grad=False
                )
                with patch.object(
                    layer_mod,
                    "get_moe_runner_backend",
                    return_value=MoeRunnerBackend.AUTO,
                ):
                    for shard_id, value in (("w1", 11.0), ("w3", 33.0)):
                        layer._weight_loader_impl(
                            weight,
                            torch.full((2, 2), value),
                            "w13_weight",
                            shard_id,
                            expert_id=0,
                        )

                torch.testing.assert_close(weight[0, :2], torch.full((2, 2), 33.0))
                torch.testing.assert_close(weight[0, 2:], torch.full((2, 2), 11.0))

    def test_interleaved_megamoe_post_load_produces_canonical_w13_once(self):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod

        source_weight = torch.arange(2 * 8 * 4).reshape(2, 8, 4)
        source_scale = torch.arange(2 * 8 * 2).reshape(2, 8, 2)
        layer = SimpleNamespace(
            inference_moe_w13_interleaved=True,
            w13_weight=SimpleNamespace(data=source_weight.clone()),
            w13_weight_scale=SimpleNamespace(data=source_scale.clone()),
        )
        method = object.__new__(modelopt_mod.ModelOptNvFp4FusedMoEMethod)
        method.enable_flashinfer_trtllm_moe = True
        method.use_flashinfer_trtllm_weight_layout = False
        build_inputs = []
        method._build_mega_moe_weights = lambda current: build_inputs.append(
            (current.w13_weight.data.clone(), current.w13_weight_scale.data.clone())
        )

        with patch.object(
            modelopt_mod,
            "get_moe_a2a_backend",
            return_value=MoeA2ABackend.MEGAMOE,
        ):
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)

        expected_weight = _expected_deinterleave(source_weight, up_first=False)
        expected_scale = _expected_deinterleave(source_scale, up_first=False)
        torch.testing.assert_close(layer.w13_weight.data, expected_weight)
        torch.testing.assert_close(layer.w13_weight_scale.data, expected_scale)
        self.assertTrue(layer._w13_deinterleaved)
        self.assertEqual(len(build_inputs), 2)
        for built_weight, built_scale in build_inputs:
            torch.testing.assert_close(built_weight, expected_weight)
            torch.testing.assert_close(built_scale, expected_scale)

    def test_megamoe_builder_rejects_trtllm_w31_layout(self):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod

        method = object.__new__(modelopt_mod.ModelOptNvFp4FusedMoEMethod)
        method.use_flashinfer_trtllm_weight_layout = True

        with self.assertRaisesRegex(AssertionError, "canonical W13"):
            method._build_mega_moe_weights(SimpleNamespace())

    def test_online_modelopt_moe_inherits_layout_decision(self):
        import sglang.srt.layers.quantization.modelopt_quant as modelopt_mod
        import sglang.srt.layers.quantization.nvfp4_online as online_mod

        config = SimpleNamespace(use_per_token_activation=False)
        with (
            patch.object(
                modelopt_mod,
                "get_moe_a2a_backend",
                return_value=MoeA2ABackend.MEGAMOE,
            ),
            patch.object(
                modelopt_mod,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
            ),
            patch.object(
                modelopt_mod,
                "get_platform",
                return_value=SimpleNamespace(is_blackwell=True),
            ),
            patch.object(modelopt_mod, "is_cuda", return_value=False),
        ):
            method = online_mod.ModelOptNvFp4OnlineFusedMoEMethod(
                config,
                "model.layers.0.mlp.experts",
            )

        self.assertTrue(method.enable_flashinfer_trtllm_moe)
        self.assertFalse(method.use_flashinfer_trtllm_weight_layout)


if __name__ == "__main__":
    unittest.main()
