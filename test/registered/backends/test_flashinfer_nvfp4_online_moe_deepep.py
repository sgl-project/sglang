import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.nvfp4_online_moe_fixture import (
    NemotronNvFp4OnlineMoeBackendBase,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=400,
    stage="extra-b",
    runner_config="deepep-4-gpu-b200",
)


class TestFlashinferCuteDSLMoeBackendNvFp4OnlineDeepEPLowLatency(
    NemotronNvFp4OnlineMoeBackendBase, CustomTestCase
):
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    quantization = "modelopt_fp4"
    enable_jit_deepgemm = True
    extra_args = [
        *NemotronNvFp4OnlineMoeBackendBase.extra_args,
        "--moe-a2a-backend",
        "deepep",
        "--speculative-moe-a2a-backend",
        "deepep",
        "--dp-size",
        "4",
        "--enable-dp-attention",
        "--deepep-mode",
        "low_latency",
        "--deepep-dispatcher-output-dtype",
        "bf16",
        "--chunked-prefill-size",
        "1024",
    ]
    extra_env = {
        **NemotronNvFp4OnlineMoeBackendBase.extra_env,
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
        "SGLANG_MOE_NVFP4_DISPATCH": "0",
    }
    expected_server_args = {
        "moe_runner_backend": "flashinfer_cutedsl",
        "speculative_moe_runner_backend": "flashinfer_cutedsl",
        "moe_a2a_backend": "deepep",
        "speculative_moe_a2a_backend": "deepep",
        "dp_size": 4,
        "enable_dp_attention": True,
        "quantization": "modelopt_mixed",
        "speculative_draft_model_quantization": "nvfp4_online",
        "deepep_mode": "low_latency",
    }


if __name__ == "__main__":
    unittest.main()
