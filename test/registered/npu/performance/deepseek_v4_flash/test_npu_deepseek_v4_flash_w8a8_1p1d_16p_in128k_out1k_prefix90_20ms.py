import logging
import os
import subprocess
import unittest
from pathlib import Path

from sglang.test.ascend.e2e.test_npu_accuracy_utils import TestNpuAccuracyMultiNodePdSepTestCaseBase
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase, AISBENCHMARK_DATASET_GSM8K,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    build_hicache_storage_extra_config,
)
HICACHE_STORAGE_EXTRA_CONFIG = build_hicache_storage_extra_config(
    world_size=16,
    protocol="device_sdma",
    dram_size="4GB",
)
logger = logging.getLogger(__name__)

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
)

# Common environment variables for DSV4-Flash PD-Sep 128k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS = {
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
}

# Prefill node environment variables for DSV4-Flash PD-Sep 128k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ENVS = {
    **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS,
    # deepep
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HCCL_BUFFSIZE": "2600",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "1",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # war barrier
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "SGLANG_PROFILE_WITH_STACK": "False",
}

# Decode node environment variables for DSV4-Flash PD-Sep 128k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ENVS = {
    **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS,
    # deepep
    "HCCL_BUFFSIZE": "200",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP": "1",
}

# Prefill node launch arguments for DSV4-Flash PD-Sep 128k
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disaggregation-bootstrap-port",
    18998,
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    256,
    "--max-prefill-tokens",
    70000,
    "--max-running-requests",
    256,
    "--enable-dp-attention",
    "--tokenizer-worker-num",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--disable-cuda-graph",
    "--load-balance-method",
    "round_robin",
    "--enable-metrics",
    "--enable-cache-report",
    "--chunked-prefill-size",
    32768,
    "--init-expert-location",
    str(Path(__file__).parent / "expert-layout-ais-replicas16.json"),
    "--ep-dispatch-algorithm",
    "dynamic",
    "--ep-num-redundant-experts",
    16,
    "--dp-size",
    4,
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    "--optimistic-prefill-attempts",
    1,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--expert-distribution-recorder-mode",
    "stat",
    "--expert-distribution-recorder-buffer-size",
    512,
    # DSPARK configuration (kept in sync with the decode node).
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--speculative-dspark-block-size",
    5,
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-ratio",
    1.5,
    "--hicache-write-policy",
    "write_through",
    "--enable-cache-report",
    "--max-total-tokens",
    140000,
    "--hicache-storage-backend",
    "npu_memcache",
    "--hicache-storage-backend-extra-config",
    HICACHE_STORAGE_EXTRA_CONFIG,
]

# Decode node launch arguments for DSV4-Flash PD-Sep 128k
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.71,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    256,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    # DSPARK configuration.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    "--enable-metrics",
    "--enable-cache-report",
    "--swa-full-tokens-ratio",
    0.05,
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    10,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--tokenizer-worker-num",
    16,
    "--speculative-dspark-block-size",
    5,
]

# Model config for DSV4-Flash W8A8 1P+1D PD-Sep 128k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}
# class TestNPUDeepSeekV4FlashW8A81P1D16PIn128kOut1k20ms(
#     TestNpuPerfMultiNodePdSepTestCaseBase
# ):
#     """Test NPU performance for DeepSeek-V4-Flash W8A8 PD-Sep 1P+1D 16p in128k out1k prefix90."""
#
#     model_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG
#     benchmark_tool = AISBENCHMARK
#     dataset_type = AISBENCHMARK_DATASET_GSM8K
#     dataset_path = None
#     input_len = 131072
#     output_len = 1024
#     num_prompts = 90
#     max_concurrency = 54
#     repeat_rate = 0.9
#     request_rate = 3
#     warmup_requests = 16
#     # Keep the shared-prefix warmup cache: SGLANG_IS_IN_CI would otherwise make
#     # bench_serving flush the cache right after warmup.
#     pop_sglang_is_in_ci_for_gsp = True
#     tpot = 20
#     output_token_throughput = 964.864
#     ttft = 7000
#
#     # def test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms(self):
#     #     """Run NPU performance test for DeepSeek-V4-Flash W8A8 1P+1D 16p in128k out1k."""
#     #     self.run_throughput()
#
# class TestNPUDeepSeekV4FlashW8A81P1D16PIn128kOut1k20msAisBench(TestNpuPerfMultiNodePdSepTestCaseBase):
#     """Test NPU performance for DeepSeek-V4-Flash W8A8 PD-Sep 1P+1D 16p in128k out1k."""
#     model_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG
#     benchmark_tool = AISBENCHMARK
#     dataset_type = AISBENCHMARK_DATASET_GSM8K
#     dataset_path = None
#     input_len = 131072
#     output_len = 1024
#     num_prompts = 90
#     max_concurrency = 54
#     # num_prompts = 20
#     # max_concurrency = 10
#     repeat_rate = 0.9
#     request_rate = 3
#     dp = 16
#     tpot = 20
#     output_token_throughput = 964.864
#     ttft = 7000
#
#     @classmethod
#     def setUpClass(cls):
#         # Mirror the accuracy case: on the P (prefill) node, print host memory
#         # before/after dropping the page cache so the prefill pod log shows the
#         # DRAM headroom at launch time (the prefill node now also loads the
#         # DSPARK draft model).
#         if "prefill" in os.environ.get("HOSTNAME", ""):
#             commands = [
#                 ("Host memory before cache cleanup", ["free", "-g"], False),
#                 (
#                     "drop_caches execution result",
#                     "sync && echo 1 > /proc/sys/vm/drop_caches",
#                     True,
#                 ),
#                 ("Host memory after cache cleanup", ["free", "-g"], False),
#             ]
#             for description, command, use_shell in commands:
#                 result = subprocess.run(
#                     command,
#                     shell=use_shell,
#                     executable="/bin/bash" if use_shell else None,
#                     capture_output=True,
#                     text=True,
#                 )
#                 logger.info(
#                     "%s: command=%r, returncode=%d\nstdout:\n%s\nstderr:\n%s",
#                     description,
#                     command,
#                     result.returncode,
#                     result.stdout,
#                     result.stderr,
#                 )
#                 if result.returncode != 0:
#                     # Best-effort host maintenance, not a test assertion: raising
#                     # here makes the prefill pod exit early and the multi-node
#                     # harness then reports a misleading "pod is not running".
#                     logger.warning(
#                         "%s failed with returncode %d; continuing without it",
#                         description,
#                         result.returncode,
#                     )
#         super().setUpClass()
#
#     def test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in128k_out1k_20ms(self):
#         """Run the 128k pure-decode low-latency TPOT acceptance test."""
#         self.run_throughput()

class TestNPUDeepSeekV4Flash0731W8A8PDSEPGPQAHigh1(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 16p 1P1D GPQA."""

    @classmethod
    def setUpClass(cls):
        # Follow the HiCache L3 wiki: inspect and clean host memory on the
        # prefill node before starting MetaService and the model server.
        # Page cache from loading the model weights otherwise leaves too little
        # DRAM for the MemCache pool (halMemAlloc fails with ENOMEM).
        if "prefill" in os.environ.get("HOSTNAME", ""):
            commands = [
                ("Host memory before cache cleanup", ["free", "-g"], False),
                (
                    "drop_caches execution result",
                    "sync && echo 1 > /proc/sys/vm/drop_caches",
                    True,
                ),
                ("Host memory after cache cleanup", ["free", "-g"], False),
            ]
            for description, command, use_shell in commands:
                result = subprocess.run(
                    command,
                    shell=use_shell,
                    executable="/bin/bash" if use_shell else None,
                    capture_output=True,
                    text=True,
                )
                logger.info(
                    "%s: command=%r, returncode=%d\nstdout:\n%s\nstderr:\n%s",
                    description,
                    command,
                    result.returncode,
                    result.stdout,
                    result.stderr,
                )
                if result.returncode != 0:
                    # Best-effort host maintenance, not a test assertion: raising
                    # here makes the prefill pod exit early and the multi-node
                    # harness then reports a misleading "pod is not running".
                    logger.warning(
                        "%s failed with returncode %d; continuing without it",
                        description,
                        result.returncode,
                    )
        super().setUpClass()

    model_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG
    # (reference: Flash W8A8 GPQA Diamond baseline is 0.874).
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    # A single attempt: gsm8k's 3-retry budget would exceed the step timeout.
    max_retries = 1
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.90,
    }
    generation_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG
    eval_batch_size = 64
    # percent: L2 host and L3 storage hit rates must exceed the thresholds.
    min_hit_rate_pcts = {"host": 0, "storage": 0}

    # def test_npu_deepseek_v4_flash_0731_w8a8_pd_sep_gpqa_high(self):
    #     """Run NPU accuracy test for DSV4-Flash-0731 PD-Sep GPQA High mode."""
    #     self.run_accuracy()

    def test_npu_deepseek_v4_flash_0731_w8a8_1p1d_16p_l2_l3_hit_rate(self):
        """Run the 32k GSP bench; assert_metrics checks the L2/L3 hit rates"""
        # 3 groups x 32 prompts share a 28800-token system prompt (32k * 0.9)
        # plus a 3200-token question, which exceeds L1 so evictions land in L2
        # and L3.
        metrics = run_bench_serving(
            host=self.host,
            port=self.port,
            model_path=self.model_config.get("model_path"),
            dataset_name="generated-shared-prefix",
            input_len=32000,
            output_len=1024,
            max_concurrency=32,
            gsp_num_groups=3,
            gsp_prompts_per_group=32,
            seed=100,
            warmup_requests=16,
            cache_report=True,
        )
        assert_metrics(self, metrics)

if __name__ == "__main__":
    unittest.main()
