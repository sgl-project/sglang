import unittest
from unittest.mock import patch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4Config
from sglang.srt.models.deepseek_common.utils import enable_nextn_moe_bf16_cast_to_fp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNextNMoeQuantConfig(CustomTestCase):
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
