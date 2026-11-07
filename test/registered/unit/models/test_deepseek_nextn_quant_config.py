import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.deepseek_common.utils import enable_nextn_moe_bf16_cast_to_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNextNMoeQuantConfig(CustomTestCase):
    def test_checkpoint_exclusions_follow_nextn_weight_mapping(self):
        from sglang.srt.layers.linear import ReplicatedLinear
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN

        for depth in (1, 48, 61):
            with self.subTest(depth=depth):
                layer_id = 0 if depth == 1 else depth
                prefix = f"model.layers.{layer_id}"
                mapper = DeepseekV3ForCausalLMNextN.get_hf_to_sglang_mapper(
                    SimpleNamespace(num_hidden_layers=depth, num_nextn_predict_layers=1)
                )
                excluded = [
                    f"{prefix}.eh_proj",
                    f"{prefix}.enorm",
                    f"{prefix}.hnorm",
                    f"{prefix}.shared_head.norm",
                    f"{prefix}.self_attn.kv_b_proj",
                    f"{prefix}.mlp.gate",
                ]
                quant = ModelOptFp4Config.from_config(
                    {
                        "quant_algo": "NVFP4",
                        "quantized_mtp": True,
                        "exclude_modules": excluded,
                        "packed_modules_mapping": {},
                    }
                )
                quant.apply_weight_name_mapper(mapper)
                self.assertEqual(
                    quant.exclude_modules,
                    [
                        "model.eh_proj",
                        "model.enorm",
                        "model.hnorm",
                        "model.shared_head.norm",
                        "model.decoder.self_attn.kv_b_proj",
                        "model.decoder.mlp.gate",
                    ],
                )
                for name in quant.exclude_modules:
                    self.assertTrue(quant.is_layer_excluded(name))
                self.assertFalse(
                    quant.is_layer_excluded("model.decoder.mlp.experts.0.gate_proj")
                )
                layer = ReplicatedLinear(16, 16, bias=False)
                self.assertIsInstance(
                    quant.get_quant_method(layer, "model.eh_proj"),
                    UnquantizedLinearMethod,
                )

    def test_fp8_cast_applies_only_to_legacy_drafts(self):
        legacy = ModelOptFp4Config.from_config({"quant_algo": "NVFP4"})
        packed = ModelOptFp4Config.from_config(
            {"quant_algo": "NVFP4", "quantized_mtp": True}
        )
        for enabled in (False, True):
            for backend in (
                MoeRunnerBackend.DEEP_GEMM,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
            ):
                with (
                    self.subTest(enabled=enabled, backend=backend),
                    patch(
                        "sglang.srt.models.deepseek_common.utils.envs."
                        "SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get",
                        return_value=enabled,
                    ),
                    patch(
                        "sglang.srt.models.deepseek_common.utils."
                        "get_moe_runner_backend",
                        return_value=backend,
                    ),
                ):
                    self.assertEqual(
                        enable_nextn_moe_bf16_cast_to_fp8(legacy),
                        enabled and backend == MoeRunnerBackend.DEEP_GEMM,
                    )
                    self.assertFalse(enable_nextn_moe_bf16_cast_to_fp8(packed))


if __name__ == "__main__":
    unittest.main()
