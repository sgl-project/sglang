import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.nvfp4_online_moe_fixture import (
    FlashinferNvFp4OnlineMoeBackendBase,
    NemotronNvFp4OnlineMoeBackendBase,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=1100, suite="nightly-4-gpu-b200", nightly=True)


class TestFlashinferTrtllmGenMoeBackendNvFp4Online(
    FlashinferNvFp4OnlineMoeBackendBase, CustomTestCase
):
    backend = "flashinfer_trtllm"
    model = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    extra_args = ["--attention-backend", "triton"]
    eval_args = {"api": "completion", "max_tokens": 512}
    extra_env = {
        "FLASHINFER_NVFP4_4OVER6": "1",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1",
        "SGLANG_FP4_IGNORED_LAYERS": ",".join(
            ["shared_expert"]
            + [f"model.layers.{layer_id}" for layer_id in range(40, 48)]
        ),
    }


class TestFlashinferCuteDSLMoeBackendNvFp4OnlineNoA2A(
    NemotronNvFp4OnlineMoeBackendBase, CustomTestCase
):
    extra_args = [
        *NemotronNvFp4OnlineMoeBackendBase.extra_args,
        "--moe-a2a-backend",
        "none",
        "--speculative-moe-a2a-backend",
        "none",
    ]
    expected_server_args = {
        "moe_runner_backend": "flashinfer_cutedsl",
        "speculative_moe_runner_backend": "flashinfer_cutedsl",
        "moe_a2a_backend": "none",
        "speculative_moe_a2a_backend": "none",
        "quantization": "nvfp4_online",
        "speculative_draft_model_quantization": "nvfp4_online",
    }


class TestFlashinferCuteDSLMoeBackendNvFp4OnlineFlashinferA2A(
    NemotronNvFp4OnlineMoeBackendBase, CustomTestCase
):
    model = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
    quantization = "modelopt_fp4"
    extra_args = [
        *NemotronNvFp4OnlineMoeBackendBase.extra_args,
        "--moe-a2a-backend",
        "flashinfer",
        "--speculative-moe-a2a-backend",
        "flashinfer",
        "--dp-size",
        "4",
        "--enable-dp-attention",
        "--max-prefill-tokens",
        "4096",
        "--chunked-prefill-size",
        "4096",
    ]
    expected_server_args = {
        "moe_runner_backend": "flashinfer_cutedsl",
        "speculative_moe_runner_backend": "flashinfer_cutedsl",
        "moe_a2a_backend": "flashinfer",
        "speculative_moe_a2a_backend": "flashinfer",
        "quantization": "modelopt_mixed",
        "speculative_draft_model_quantization": "nvfp4_online",
    }


if __name__ == "__main__":
    unittest.main()
