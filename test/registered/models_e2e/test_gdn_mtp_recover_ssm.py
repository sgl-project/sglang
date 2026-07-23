"""End-to-end test for RecoverSSM GDN MTP (--gdn-mtp-cache-mode=none) on GB200."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=1800, stage="base-c", runner_config="4-gpu-b200")

RECOVER_SSM_MODEL = "nvidia/Qwen3.5-397B-A17B-NVFP4"

_RECOVER_SSM_ARGS = [
    "--tensor-parallel-size",
    "4",
    "--expert-parallel-size",
    "1",
    "--quantization",
    "modelopt_fp4",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--fp4-gemm-backend",
    "flashinfer_cutlass",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--attention-backend",
    "trtllm_mha",
    "--moe-runner-backend",
    "flashinfer_trtllm",
    "--mamba-radix-cache-strategy",
    "no_buffer",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--chunked-prefill-size",
    "32768",
    "--max-prefill-tokens",
    "32768",
    "--cuda-graph-max-bs",
    "8",
    "--max-running-requests",
    "32",
    "--scheduler-recv-interval",
    "30",
    "--stream-interval",
    "30",
    "--watchdog-timeout",
    "600",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--gdn-mtp-cache-mode",
    "none",
    "--linear-attn-backend",
    "flashinfer",
]


class TestGdnMtpRecoverSSM(GSM8KMixin, DefaultServerBase):
    model = RECOVER_SSM_MODEL
    other_args = _RECOVER_SSM_ARGS
    timeout = 2400

    gsm8k_accuracy_thres = 0.92
    gsm8k_accept_length_thres = 3.0
    gsm8k_num_threads = 32


if __name__ == "__main__":
    unittest.main()
