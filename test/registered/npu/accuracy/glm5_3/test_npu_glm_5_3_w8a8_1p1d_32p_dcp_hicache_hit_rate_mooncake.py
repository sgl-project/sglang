import json
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    get_mooncake_master_server_address,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
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
    est_time=4800,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

GLM_5_3_DCP_PD_SEP_PREFILL_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    # "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_PP_LAYER_PARTITION": "18,20,24,16",
    # "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    # "SGLANG_ENABLE_SPEC_V2": "1",
    # "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    # The 32 prefill ranks finish Mooncake store setup minutes apart; the
    # HiCache sync groups are built over a world HCCL domain right after, so
    # the default 120s connect timeout would expire on the early ranks.
    "HCCL_CONNECT_TIMEOUT": "1800",
    "HCCL_EXEC_TIMEOUT": "1800",
    # Mooncake on the NPU device path: TENT transport for the L3 store and the
    # PD KV transfer. The store endpoint travels in the extra-config JSON and
    # MC_TENT_CONF, both built in setUpClass.
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_STORE_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "SGLANG_HICACHE_MOONCAKE_REUSE_TE": "0",
}

GLM_5_3_DCP_PD_SEP_DECODE_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    # "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    # "SGLANG_SPEC_ENABLE_OVERLAP_REFLOW": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    # "DEEPEP_HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    # "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
}

GLM_5_3_DCP_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    2,
    "--tp-size",
    8,
    "--ep-size",
    8,
    "--attn-dp-size",
    8,
    "--mem-fraction-static",
    0.88,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "mooncake",
    "--max-running-requests",
    64,
    "--served-model-name",
    "glm-5.3-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--chunked-prefill-size",
    8192,
    "--max-prefill-tokens",
    65536,
    # Cap the device KV pool so the 32k-token GSP workload overflows L1 and
    # exercises the L2 host tier. The host pool is half of it, which must still
    # hold one full 28800-token prefix for L2 hits to be possible.
    "--max-total-tokens",
    65536,
    "--context-length",
    220000,
    "--moe-a2a-backend",
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill": "alltoallv", "decode": "low_latency"}',
    "--disable-shared-experts-fusion",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--dtype",
    "bfloat16",
    "--enable-dp-lm-head",
    "--moe-dense-tp",
    1,
    "--pp-size",
    4,
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-ratio",
    1.2,
    # MLA on NPU maps this to page_first_kv_split, which the Mooncake L3
    # backend's get_page_buffer_meta supports (layer_first_kv_split does not).
    "--hicache-mem-layout",
    "page_first",
    "--hicache-write-policy",
    "write_through",
    "--hicache-storage-prefetch-policy",
    "wait_complete",
    "--hicache-storage-backend",
    "mooncake",
    "--enable-cache-report",
    # Store setup blocks a scheduler for minutes; the 300s default would keep
    # dumping thread stacks instead of letting startup finish.
    "--watchdog-timeout",
    9000,
]

GLM_5_3_DCP_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--attn-dp-size",
    8,
    "--dcp-size",
    4,
    "--ep-size",
    32,
    "--mem-fraction-static",
    0.88,
    "--max-running-requests",
    48,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--served-model-name",
    "glm-5.3-w8a8",
    "--reasoning-parser",
    "glm45",
    "--tool-call-parser",
    "glm47",
    "--enable-metrics",
    "--moe-a2a-backend",
    "ascend_ops",
    "--moe-ascend-ops-config",
    '{"prefill": "alltoallv", "decode": "low_latency"}',
    "--cuda-graph-max-bs-decode",
    8,
    "--disaggregation-transfer-backend",
    "mooncake",
    "--watchdog-timeout",
    9000,
    "--context-length",
    220000,
    "--tokenizer-worker-num",
    16,
    "--disable-shared-experts-fusion",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--enable-cache-report",
]


# DCP variant with L3 Mooncake store
# 4-node PD separation, prefill dp8 + pp4, decode tp32/dp8/dcp4
# MoE A2A backend: ascend_ops (prefill=alltoallv, decode=low_latency) on both P and D
class TestNPUGLM5_3_W8A8_1P1D_DCP_HICACHE_HIT_RATE(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Serve GLM-5.3-w8a8 with PD separation and DCP; check the HiCache hit tiers"""

    # L2 host and L3 storage traffic must both be non-zero.
    min_hit_rate_pcts = {"host": 0, "storage": 0}

    @classmethod
    def setUpClass(cls):
        # Only the prefill node runs the L3 store client; the decode node has no
        # storage backend. The Mooncake store config is one JSON passed through
        # --hicache-storage-backend-extra-config (once it carries
        # master_server_address the client ignores the MOONCAKE_* env vars).
        # The master runs on the first prefill pod; each scheduler rank
        # contributes global_segment_size to the pool.
        l3_config = json.dumps(
            {
                "master_server_address": get_mooncake_master_server_address(),
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "global_segment_size": "2gb",
                "local_hostname": resolve_local_ip(),
                "extra_backend_tag": "pd-1p1d-glm5.3-w8a8-dcp",
            }
        )
        # Mooncake serves both the L3 store (prefill only) and the PD KV
        # transfer (both roles), so both nodes get the TENT transport config
        # and the memfabric/mooncake library path of their own pod.
        mooncake_envs = {
            "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
            "MC_USE_TENT": "1",
            "MF_PROTOCOL": "device_sdma",
            "MF_TRANSFER_TIMEOUT": "10000",
            "MC_TENT_CONF": write_memfabric_tent_conf(),
            "LD_LIBRARY_PATH": append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            ),
        }
        cls.model_config = {
            "model_path": GLM_5_3_W8A8_MODEL_PATH,
            "prefill_args": [
                *GLM_5_3_DCP_PD_SEP_PREFILL_ARGS,
                "--hicache-storage-backend-extra-config",
                l3_config,
            ],
            "decode_args": list(GLM_5_3_DCP_PD_SEP_DECODE_ARGS),
            "prefill_envs": {**GLM_5_3_DCP_PD_SEP_PREFILL_ENVS, **mooncake_envs},
            "decode_envs": {**GLM_5_3_DCP_PD_SEP_DECODE_ENVS, **mooncake_envs},
            "router_args": [
                "--policy",
                "round_robin",
                "--request-timeout-secs",
                7200,
                "--disable-circuit-breaker",
            ],
            "router_envs": {},
        }
        super().setUpClass()

    def test_npu_glm_5_3_w8a8_1p1d_32p_dcp_l2_host_hit_rate(self):
        """Run the 32k GSP bench; assert_metrics checks the host hit rate"""
        # 3 groups x 32 prompts share a 28800-token system prompt (32k * 0.9)
        # plus a 3200-token question. Requests are shuffled and round-robin
        # across the 8 prefill DP ranks, so 32 prompts per group give each rank
        # several repeats of the same prefix (reuse), while the three distinct
        # prefixes overflow the 65536-token device pool (eviction to L2).
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
