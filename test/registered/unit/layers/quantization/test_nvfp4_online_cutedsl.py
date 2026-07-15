import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl import (
    resolve_cutedsl_standard_scales,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
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

    def test_qwen_mtp_preserves_explicit_online_quantization(self):
        cases = (
            ("modelopt", "NVFP4"),
            ("modelopt_fp4", "NVFP4"),
            ("modelopt_fp4", ""),
        )
        for detected, quant_algo in cases:
            with self.subTest(detected=detected, quant_algo=quant_algo):
                self.assertEqual(
                    self._verify_quantization(
                        "nvfp4_online", detected, quant_algo=quant_algo
                    ),
                    "nvfp4_online",
                )

    def test_online_overlay_is_limited_to_qwen_mtp_draft(self):
        self.assertEqual(
            self._verify_quantization(
                "nvfp4_online", "modelopt_fp4", is_draft_model=False
            ),
            "modelopt_fp4",
        )
        self.assertEqual(
            self._verify_quantization(
                "nvfp4_online",
                "modelopt_fp4",
                architecture="DeepseekV3ForCausalLMNextN",
            ),
            "modelopt_fp4",
        )
        self.assertEqual(
            self._verify_quantization(
                "nvfp4_online",
                "modelopt_fp4",
                model_type="interns2_preview_text",
            ),
            "modelopt_fp4",
        )
        for detected, quant_algo, expected in (
            ("modelopt", "FP8", "modelopt"),
            ("modelopt_fp8", "FP8", "modelopt_fp8"),
            ("modelopt", "", "modelopt"),
        ):
            with self.subTest(detected=detected, quant_algo=quant_algo):
                self.assertEqual(
                    self._verify_quantization(
                        "nvfp4_online",
                        detected,
                        quant_algo=quant_algo,
                    ),
                    expected,
                )

    def test_generic_modelopt_auto_detection_is_unchanged(self):
        self.assertEqual(
            self._verify_quantization("modelopt", "modelopt_fp4"),
            "modelopt_fp4",
        )

    @contextmanager
    def _make_method(
        self,
        backend: MoeRunnerBackend,
        a2a_backend: MoeA2ABackend = MoeA2ABackend.NONE,
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
                "sglang.srt.layers.quantization.modelopt_quant."
                "is_blackwell_supported",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.swizzle_blockscale",
                side_effect=lambda scale: torch.empty(
                    scale.shape, dtype=scale.dtype, device=scale.device
                ),
            ),
        ):
            yield ModelOptNvFp4OnlineFusedMoEMethod(
                NvFp4OnlineConfig(), "model.layers.0.mlp.experts"
            )

    def test_cutedsl_with_flashinfer_a2a_is_accepted(self):
        with self._make_method(
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            MoeA2ABackend.FLASHINFER,
        ) as method:
            self.assertTrue(method.supports_nvfp4_online_moe)
            self.assertFalse(method.enable_flashinfer_trtllm_moe)

    def test_unqualified_cutedsl_combinations_are_rejected(self):
        for a2a_backend in (MoeA2ABackend.NONE, MoeA2ABackend.DEEPEP):
            with self.subTest(a2a_backend=a2a_backend):
                with self.assertRaisesRegex(ValueError, "FlashInfer A2A"):
                    with self._make_method(
                        MoeRunnerBackend.FLASHINFER_CUTEDSL,
                        a2a_backend,
                    ):
                        pass

    def test_trtllm_runners_remain_accepted(self):
        for backend in (
            MoeRunnerBackend.FLASHINFER_TRTLLM,
            MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED,
        ):
            with self.subTest(backend=backend):
                with self._make_method(backend) as method:
                    self.assertTrue(method.supports_nvfp4_online_moe)

    def test_unrelated_runner_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "nvfp4_online supports"):
            with self._make_method(MoeRunnerBackend.TRITON):
                pass

    def test_online_converter_rejects_packed_integer_source(self):
        with self.assertRaisesRegex(ValueError, "floating-point source"):
            ModelOptNvFp4OnlineFusedMoEMethod._quantize_weight_nvfp4(
                torch.zeros((2, 16), dtype=torch.uint8)
            )

    def test_online_input_scale_placeholders_are_neutral(self):
        with self._make_method(
            MoeRunnerBackend.FLASHINFER_CUTEDSL,
            MoeA2ABackend.FLASHINFER,
        ) as method:
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
