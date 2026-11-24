"""Single-machine PD-separated DeepSeek-V4-Flash FP8 GPQA on Ascend 950.

One 8-card machine runs a 4-card prefill worker (cards 0-3) and a 4-card
decode worker (cards 4-7). KV is transferred with the Mooncake backend (TENT
over the device SDMA path), and sgl-router fronts both workers so every GPQA
question exercises the prefill -> decode handoff.
"""

import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
    write_memfabric_tent_conf,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    popen_launch_sgl_router,
)

register_npu_ci(
    est_time=7200,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# Environment shared by the prefill and decode workers.
DEEPSEEK_V4_FLASH_FP8_PD_COMMON_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    # HCCL deepep
    "DEEPEP_HCCL_BUFFSIZE": "2560",
    # Both workers live on one host, so they rendezvous over loopback.
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    # DeepEP long-sequence normal combine (ROUND * TOKENS >= chunked_prefill_size / tp * dp)
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # dsv4
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    # performance
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
    # Dspark
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    # PD handshake budgets: the prefill->decode bootstrap can outlast the default.
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
}

DEEPSEEK_V4_FLASH_SERVED_MODEL_NAME = os.path.basename(
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
)

# Launch arguments shared by both workers. DSPARK keeps its draft on each side:
# the prefill worker prunes the draft down to context-KV injection, while the
# decode worker runs it, so the same flags are passed to both.
DEEPSEEK_V4_FLASH_FP8_PD_SHARED_ARGS = [
    "--page-size",
    128,
    "--trust-remote-code",
    # run_accuracy sends the checkpoint basename as the OpenAI ``model`` field;
    # both workers serve that name so sgl-router can route it.
    "--served-model-name",
    DEEPSEEK_V4_FLASH_SERVED_MODEL_NAME,
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--kv-cache-dtype",
    "auto",
    "--enable-dp-lm-head",
    "--attn-dp-size",
    4,
    "--reasoning-parser",
    "deepseek-v4",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "fp8",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "fp8",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--speculative-dspark-block-size",
    5,
    "--disable-radix-cache",
]

# Prefill worker: 4 cards. The Mooncake transfer backend is appended by
# PDDisaggregationServerBase from ``cls.transfer_backend``.
DEEPSEEK_V4_FLASH_FP8_PD_PREFILL_ARGS = [
    *DEEPSEEK_V4_FLASH_FP8_PD_SHARED_ARGS,
    "--disaggregation-mode",
    "prefill",
    "--mem-fraction-static",
    0.73,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    131072,
    "--max-prefill-tokens",
    131072,
]

# Decode worker: 4 cards, starting at card 4 (see decode_base_gpu_id).
DEEPSEEK_V4_FLASH_FP8_PD_DECODE_ARGS = [
    *DEEPSEEK_V4_FLASH_FP8_PD_SHARED_ARGS,
    "--disaggregation-mode",
    "decode",
    "--mem-fraction-static",
    0.73,
    "--max-running-requests",
    32,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
]

DEEPSEEK_V4_FLASH_FP8_PD_GENERATION_CONFIG_MAX = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}
    },
}


class TestNPUDeepSeekV4FlashFP81P1D8PMooncakeGPQA(
    PDDisaggregationServerBase, TestNpuAccuracyTestCaseBase
):
    """DeepSeek-V4-Flash FP8 GPQA: 4-card prefill + 4-card decode, Mooncake KV transfer.

    The accuracy helpers come from TestNpuAccuracyTestCaseBase (per-case metrics
    directory and metrics.json for release data), while the two workers and the
    router are launched by PDDisaggregationServerBase on one host.
    """

    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    prefill_tp_size = 4
    decode_tp_size = 4
    decode_base_gpu_id = 4
    # Ascend Mooncake runs TENT over the device SDMA path, not RDMA NICs.
    use_rdma = False

    extra_prefill_args = DEEPSEEK_V4_FLASH_FP8_PD_PREFILL_ARGS
    extra_decode_args = DEEPSEEK_V4_FLASH_FP8_PD_DECODE_ARGS

    benchmark_tool = EVALSCOPE
    datasets = ["gpqa_diamond"]
    accuracy = 0.874
    generation_config = DEEPSEEK_V4_FLASH_FP8_PD_GENERATION_CONFIG_MAX
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1
    # Use a subset of GPQA Diamond (198 questions) to cut the step runtime.
    limit = 1
    # Thinking mode streams up to 125k tokens per answer; the router's 1800s
    # default would cut those responses off mid-stream.
    lb_request_timeout_secs = 10800
    # A cold runner reads the 48 checkpoint shards at ~20-25 s each, and both
    # workers load concurrently, so the 600 s default launch budget expires
    # while the weights are still loading.
    server_launch_timeout = 5400

    @classmethod
    def wait_server_ready(cls, url, timeout=None, process=None):
        super().wait_server_ready(
            url, timeout=timeout or cls.server_launch_timeout, process=process
        )

    @classmethod
    def setUpClass(cls):
        # Per-case metrics directory + metrics.json contract used by the
        # release-data pipeline.
        cls._setup_per_case_output()
        super().setUpClass()
        # Mooncake transfer runs on both workers, so both get the TENT config
        # and the memfabric/mooncake library paths of this host.
        mooncake_envs = {
            "MC_USE_TENT": "1",
            # A5 (950PR/DT) transfers KV over URMA, not device SDMA.
            "MF_PROTOCOL": "device_urma",
            "MF_TRANSFER_TIMEOUT": "10000",
            "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
            "MC_TENT_CONF": write_memfabric_tent_conf(),
            "LD_LIBRARY_PATH": append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            ),
        }
        cls.extra_prefill_env = {
            **DEEPSEEK_V4_FLASH_FP8_PD_COMMON_ENVS,
            **mooncake_envs,
        }
        cls.extra_decode_env = {
            **DEEPSEEK_V4_FLASH_FP8_PD_COMMON_ENVS,
            **mooncake_envs,
        }
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.launch_all()

    @classmethod
    def launch_lb(cls):
        # run_accuracy sends the checkpoint's basename as the OpenAI ``model``
        # field, and sgl-router only routes the names it serves, so register the
        # same name the workers are launched with (see --served-model-name).
        cls.process_lb = popen_launch_sgl_router(
            DEEPSEEK_V4_FLASH_SERVED_MODEL_NAME,
            [cls.prefill_url, cls.decode_url],
            host=cls.base_host,
            port=cls.lb_port,
            request_timeout_secs=cls.lb_request_timeout_secs,
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._save_metrics_json()
        cls._backup_plog()

    def test_npu_deepseek_v4_flash_fp8_1p1d_8p_mooncake_gpqa(self):
        """Run NPU accuracy for DSV4-Flash FP8 4P4D PD-Sep with Mooncake transfer."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
