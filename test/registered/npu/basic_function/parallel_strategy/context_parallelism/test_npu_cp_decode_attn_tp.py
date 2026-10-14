"""GSM8K accuracy regression for NPU CP decode attention TP with W8A8 weights.

The test serves one DeepSeek-V4-Flash model with decode attention TP enabled.
It checks the server configuration and evaluates GSM8K instead of comparing
individual tokens or logprobs against a second server.
"""

import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=3600, suite="full-8-npu-a3")
register_npu_ci(est_time=7200, suite="nightly-8-npu-a3", nightly=True)

TEST_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # EP=8 low-latency combine with dispatch capacity 128 requires >=557 MB.
    "HCCL_BUFFSIZE": "1024",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
}

SERVER_ARGS = [
    "--tp-size",
    "8",
    "--attn-cp-size",
    "8",
    "--dp-size",
    "1",
    "--ep-size",
    "8",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-cp-decode-attn-tp",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "128",
    "--mem-fraction-static",
    "0.8",
    "--chunked-prefill-size",
    "4096",
    "--prefill-max-requests",
    "8",
    "--max-running-requests",
    "8",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--watchdog-timeout",
    "600",
    "--random-seed",
    "0",
    "--disable-radix-cache",
    "--cuda-graph-backend-prefill",
    "disabled",
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-bs-decode",
    "1",
    "2",
    "4",
    "8",
]


class TestNPUCpDecodeAttnTP(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that --enable-cp-decode-attn-tp keeps GSM8K accuracy on NPU
    for DeepSeek-V4-Flash with context parallelism (--attn-cp-size 8).

    [Test Category] Context Parallel
    [Test Target] --enable-cp-decode-attn-tp
    """

    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    timeout_for_server_launch = 1800  # Seconds, not milliseconds.
    other_args = SERVER_ARGS
    env = {**os.environ, **TEST_ENVS}
    accuracy = 0.93
    num_questions = 200
    gsm8k_num_shots = 8


if __name__ == "__main__":
    unittest.main()
