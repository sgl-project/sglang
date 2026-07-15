import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.ep_moe.layer import DeepEPMoE
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    resolve_cutedsl_standard_scales,
)
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    MoeA2ABackend,
    MoeRunnerBackend,
)
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
    NvFp4OnlineConfig,
)


class TestNvFp4OnlineCuteDsl(CustomTestCase):
    def _verify_quantization(
        self,
        requested: str,
        detected: str,
        *,
        is_draft_model: bool = True,
        architecture: str = "Qwen3_5ForCausalLMMTP",
        model_type: str = "qwen3_5_moe_text",
        quant_algo: str = "NVFP4",
    ) -> str:
        config = ModelConfig.__new__(ModelConfig)
        config.quantization = requested
        config.is_draft_model = is_draft_model
        config.hf_config = SimpleNamespace(architectures=[architecture])
        config.hf_text_config = SimpleNamespace(model_type=model_type)
        config._parse_quant_hf_config = lambda: {
            "quant_method": detected,
            "quant_algo": quant_algo,
        }
        config._find_quant_modelslim_config = lambda: None

        with patch("sglang.srt.layers.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0", False):
            config._verify_quantization()

        return config.quantization

    def test_model_config_overlay(self):
        for detected, quant_algo in (("modelopt", "NVFP4"), ("modelopt_fp4", "")):
            with self.subTest(detected=detected, quant_algo=quant_algo):
                self.assertEqual(
                    self._verify_quantization(
                        "nvfp4_online", detected, quant_algo=quant_algo
                    ),
                    "nvfp4_online",
                )

        for kwargs in (
            {"is_draft_model": False},
            {"architecture": "DeepseekV3ForCausalLMNextN"},
        ):
            self.assertEqual(
                self._verify_quantization("nvfp4_online", "modelopt_fp4", **kwargs),
                "modelopt_fp4",
            )

        self.assertEqual(
            self._verify_quantization("nvfp4_online", "modelopt", quant_algo="FP8"),
            "modelopt",
        )
        self.assertEqual(
            self._verify_quantization("modelopt", "modelopt_fp4"),
            "modelopt_fp4",
        )

    @contextmanager
    def _make_method(
        self,
        backend: MoeRunnerBackend,
        a2a_backend: MoeA2ABackend = MoeA2ABackend.NONE,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
    ):
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "get_moe_runner_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.moe.get_moe_runner_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_a2a_backend",
                return_value=a2a_backend,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_deepep_mode",
                return_value=deepep_mode,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "is_blackwell_supported",
                return_value=True,
            ),
        ):
            yield ModelOptNvFp4OnlineFusedMoEMethod(
                NvFp4OnlineConfig(), "model.layers.0.mlp.experts"
            )

    def test_runner_compatibility_matrix(self):
        cutedsl = MoeRunnerBackend.FLASHINFER_CUTEDSL
        trtllm = MoeRunnerBackend.FLASHINFER_TRTLLM
        routed = MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED
        fi = MoeA2ABackend.FLASHINFER
        deepep = MoeA2ABackend.DEEPEP
        auto = DeepEPMode.AUTO
        cases = (
            ("FlashInfer A2A", cutedsl, fi, auto, True),
            ("DeepEP low latency", cutedsl, deepep, DeepEPMode.LOW_LATENCY, True),
            ("DeepEP auto", cutedsl, deepep, auto, False),
            ("no A2A", cutedsl, MoeA2ABackend.NONE, auto, False),
            ("TRTLLM", trtllm, MoeA2ABackend.NONE, auto, True),
            ("TRTLLM routed", routed, MoeA2ABackend.NONE, auto, True),
            ("unrelated", MoeRunnerBackend.TRITON, MoeA2ABackend.NONE, auto, False),
        )
        for name, backend, a2a_backend, deepep_mode, supported in cases:
            with self.subTest(name=name):
                if supported:
                    with self._make_method(backend, a2a_backend, deepep_mode) as method:
                        self.assertTrue(method.supports_nvfp4_online_moe)
                else:
                    with self.assertRaisesRegex(ValueError, "nvfp4_online supports"):
                        with self._make_method(backend, a2a_backend, deepep_mode):
                            pass

    def test_online_cutedsl_selects_fused_moe_path(self):
        with (
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.FusedMoE.__init__",
                return_value=None,
            ),
            patch(
                "sglang.srt.layers.moe.ep_moe.layer.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_CUTEDSL,
            ),
        ):
            layer = DeepEPMoE(
                num_experts=2,
                top_k=1,
                hidden_size=32,
                intermediate_size=64,
                layer_id=0,
                quant_config=NvFp4OnlineConfig(),
            )

        self.assertTrue(layer.deprecate_flag)

    def test_online_converter_rejects_packed_integer_source(self):
        with self.assertRaisesRegex(ValueError, "floating-point source"):
            ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4(
                torch.zeros((2, 16), dtype=torch.uint8)
            )

    def test_online_input_scale_placeholders_are_neutral(self):
        with (
            self._make_method(
                MoeRunnerBackend.FLASHINFER_CUTEDSL,
                MoeA2ABackend.FLASHINFER,
            ) as method,
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.swizzle_blockscale",
                side_effect=lambda scale: torch.empty(
                    scale.shape, dtype=scale.dtype, device=scale.device
                ),
            ),
        ):
            layer = torch.nn.Module()
            layer.num_experts = 2
            layer.num_local_experts = 2
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=32,
                intermediate_size_per_partition=64,
                params_dtype=torch.bfloat16,
                weight_loader=lambda *args, **kwargs: None,
            )

        torch.testing.assert_close(layer.w13_input_scale, torch.ones(2, 2))
        torch.testing.assert_close(layer.w2_input_scale, torch.ones(2))

    def test_online_cutedsl_scalar_scale_contract(self):
        layer = SimpleNamespace(
            num_experts=2,
            num_local_experts=2,
            moe_ep_rank=0,
            g1_alphas=torch.tensor([0.25, 0.5], dtype=torch.float32),
            g2_alphas=torch.tensor([0.75, 1.0], dtype=torch.float32),
            w13_weight_scale_2=torch.tensor(
                [[0.25, 0.25], [0.5, 0.5]], dtype=torch.float32
            ),
            w2_weight_scale_2=torch.tensor([0.75, 1.0], dtype=torch.float32),
            w13_input_scale_quant=torch.ones(2, dtype=torch.float32),
            w2_input_scale_quant=torch.ones(2, dtype=torch.float32),
        )

        w1_alpha, a2_scale, w2_alpha, a1_scale = resolve_cutedsl_standard_scales(layer)

        torch.testing.assert_close(a1_scale, torch.ones(1))
        torch.testing.assert_close(a2_scale, torch.ones(1))
        torch.testing.assert_close(w1_alpha, layer.w13_weight_scale_2[:, 0])
        torch.testing.assert_close(w2_alpha, layer.w2_weight_scale_2)


if __name__ == "__main__":
    unittest.main()
