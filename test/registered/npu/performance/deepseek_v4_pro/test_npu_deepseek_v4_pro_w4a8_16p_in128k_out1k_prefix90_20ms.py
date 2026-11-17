import logging
import os
import subprocess
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase, assert_metrics, run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=2000,
    suite="",
    nightly=True,
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

# Environment variables for DSV4-Pro-0813 two-node mix deployment,
DEEPSEEK_V4_PRO_W4A8_16P_ENVS = {
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    # [DEEPEP]
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    # [DSPARK]
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "TRANSFORMERS_VERBOSITY": "error",
}


# Server launch arguments for DSV4-Pro W4A8 two-node 16p PD-mix.
DEEPSEEK_V4_PRO_W4A8_16P_OTHER_ARGS = [
    "--served-model-name",
    "dsv4",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--dp-size",
    8,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    32,
    "--mem-fraction-static",
    0.8,
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    16384,
    "--prefill-decode-interval",
    2,
    "--enable-prefill-delayer",
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-metrics",
    "--enable-cache-report",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
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


DEEPSEEK_V4_PRO_W4A8_16P_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "other_args": DEEPSEEK_V4_PRO_W4A8_16P_OTHER_ARGS,
    "node_envs": DEEPSEEK_V4_PRO_W4A8_16P_ENVS,
}


class TestNPUDeepSeekV4ProW4A816PIn128kOut1kPrefix90RadixCache(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Pro W4A8 16p mix in128k out1k prefix90."""

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
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = DEEPSEEK_V4_PRO_W4A8_16P_MODEL_CONFIG
    dataset_name = "generated-shared-prefix"
    warmup_requests = 16
    max_concurrency = 8
    num_prompts = 32
    repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    request_rate = float("inf")
    seed = 42
    tpot = 20
    output_token_throughput = 128

    # def test_npu_deepseek_v4_pro_w4a8_mix_16p_in128k_out1k_prefix_90_20ms(self):
    #     """Run NPU performance test for eepSeek-V4-Pro W4A8 16p mix 16p in128k out1k."""
    #     self.run_throughput()

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
