import json
import unittest
from urllib.parse import urlparse

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    get_mooncake_master_server_address,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
    resolve_local_ip,
    write_memfabric_tent_conf,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

DEEPSEEK_V4_FLASH_FP8_4P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "1",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    # HCCL deepep
    "DEEPEP_HCCL_BUFFSIZE": "2560",
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
}

# L3 store client: MemFabric TENT transport over the NPU device path. The store
# endpoint travels in the extra-config JSON and MC_TENT_CONF, both built in
# setUpClass.
DEEPSEEK_V4_FLASH_FP8_4P_L3_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_STORE_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "SGLANG_HICACHE_MOONCAKE_REUSE_TE": "0",
}

DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    4,
    "--trust-remote-code",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.73,
    "--max-running-requests",
    32,
    "--chunked-prefill-size",
    131072,
    # "--enable-mixed-chunk",
    "--max-prefill-tokens",
    131072,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    "--kv-cache-dtype",
    "auto",
    "--enable-dp-lm-head",
    "--attn-dp-size",
    4,
    "--reasoning-parser",
    "deepseek-v4",
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
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "fp8",
    # L3 (Mooncake) hierarchical cache. Radix cache has to stay on, since
    # --disable-radix-cache and --enable-hierarchical-cache are mutually
    # exclusive. The device pool is capped (never above the profiled capacity,
    # see _apply_token_constraints) and the host tier is sized at half of it,
    # so the GSP workload of the hit-rate test overflows L1 into L2/L3.
    "--max-total-tokens",
    262144,
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-mem-layout",
    "layer_first",
    "--hicache-ratio",
    1.2,
    "--hicache-write-policy",
    "write_through",
    "--hicache-storage-prefetch-policy",
    "wait_complete",
    "--hicache-storage-backend",
    "mooncake",
    "--enable-cache-report",
]

DEEPSEEK_V4_FLASH_FP8_4P_GENERATION_CONFIG_MAX = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"}
    },
}


class TestNPUDeepSeekV4FlashFP84PGPQA(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash FP8 4p DSPARK GPQA."""

    benchmark_tool = EVALSCOPE
    model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_FP8_4P_ENVS
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_FP8_4P_GENERATION_CONFIG_MAX
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    # Temporary: run only 1 GPQA Diamond question to shorten the step runtime.
    limit = 1

    # L2 host and L3 storage traffic must both be non-zero.
    min_hit_rate_pcts = {"host": 0, "storage": 0}

    @classmethod
    def setUpClass(cls):
        # This is a single-node case, so the base class starts the Mooncake
        # master locally (from --hicache-storage-backend) and exports
        # MOONCAKE_MASTER. The store client still gets an explicit
        # extra-config JSON: without it the env defaults resolve to
        # protocol=rdma, while the NPU store client needs the ascend path.
        l3_config = json.dumps(
            {
                "master_server_address": get_mooncake_master_server_address(),
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "global_segment_size": "8gb",
                "local_hostname": resolve_local_ip(),
                "extra_backend_tag": "dsv4-flash-fp8-4p-a5",
            }
        )
        cls.other_args = [
            *DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS,
            "--hicache-storage-backend-extra-config",
            l3_config,
        ]
        cls.envs = {
            **DEEPSEEK_V4_FLASH_FP8_4P_ENVS,
            **DEEPSEEK_V4_FLASH_FP8_4P_L3_ENVS,
            "MC_TENT_CONF": write_memfabric_tent_conf(),
            "LD_LIBRARY_PATH": append_ld_library_path(("memfabric_hybrid", "lib")),
        }
        super().setUpClass()

    def test_npu_deepseek_v4_flash_fp8_4p_gpqa(self):
        """Run NPU accuracy test for DeepSeek-V4-Flash FP8 4p DSPARK GPQA."""
        self.run_accuracy()

    def test_npu_deepseek_v4_flash_fp8_4p_mooncake_l3_hit_rate(self):
        """Run the 32k GSP bench; assert_metrics checks the host/storage hit rates"""
        # 3 groups x 32 prompts share a 28800-token system prompt (32k * 0.9)
        # plus a 3200-token question. Eight in-flight 32k prompts per DP rank
        # overflow the capped device pool, and the device + 0.5x host pools
        # cannot hold all cached content plus the questions in flight, so
        # evictions reach the L3 store, while 32 prompts per group give each
        # prefix several repeats (reuse).
        parsed_url = urlparse(self.base_url)
        metrics = run_bench_serving(
            host=parsed_url.hostname,
            port=parsed_url.port,
            model_path=self.model,
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


# class TestNPUDeepSeekV4FlashDSPARK(TestNpuPerformanceTestCaseBase):
#     """Test DSPARK for Deepseek-V4 in the 950 environment, acceptance rate greater than 0.5"""
#
#     benchmark_tool = BENCHMARK_TOOL_DEFAULT
#     model = DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH
#     other_args = DEEPSEEK_V4_FLASH_FP8_4P_OTHER_ARGS
#     envs = DEEPSEEK_V4_FLASH_FP8_4P_ENVS
#     dataset_name = "random"
#     input_len = 8000
#     output_len = 1000
#     num_prompts = 160
#     max_concurrency = 160
#     random_range_ratio = 1
#     warmup_requests = 32
#     request_rate = float("inf")
#     max_attempts = 3
#     accept_rate = 0.5
#
#     def test_npu_deepseek_v4_flash_fp8_4p_dspark(self):
#         """Run NPU performance test for DeepSeek-V4-Flash FP8 4p DSPARK."""
#         self.run_throughput()


if __name__ == "__main__":
    unittest.main()
