import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_3_W8A8_MODEL_PATH,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
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
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "ENABLE_PROFILING": "0",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_CONNECT_TIMEOUT": "1800",
    "HCCL_EXEC_TIMEOUT": "1800",
    # PD KV transfer runs over the MemFabric TENT device path; MC_TENT_CONF and
    # LD_LIBRARY_PATH are filled in setUpClass.
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
}

GLM_5_3_DCP_PD_SEP_DECODE_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1200",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
    "TASK_QUEUE_ENABLE": "0",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    # Decode also speaks TENT for the Mooncake PD transfer.
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
}

# Prefill has no --pp-size: the Mooncake PD transfer backend validates the
# prefill/decode KV entry layout positionally, and a PP-sliced prefill
# kv_item_lens list does not line up with decode's uniform one (decode is
# pp=1). Without PP, tp32 fills the 32 prefill ranks (2 pods x 16 NPUs).
# Prefill is pure DP attention (attn-dp-size == tp-size) so attn_tp_size is 1
# and the DSA token-shard path, which asserts on this image at attn_tp>1, is
# never taken; ep32 keeps the per-rank MoE expert footprint equal to the
# pp4/tp8/ep8 layout.
GLM_5_3_DCP_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--ep-size",
    32,
    "--attn-dp-size",
    32,
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
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    # L2 host tier only: the L3 storage backend is intentionally left unset so
    # this case isolates the Mooncake PD KV transfer path.
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-ratio",
    1.2,
    "--hicache-mem-layout",
    "page_first",
    "--hicache-write-policy",
    "write_through",
    "--enable-cache-report",
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


# Mooncake PD-transfer variant: no L3 store, PD KV transfer over Mooncake.
# 4-node PD separation, prefill tp32/dp32 (no PP), decode tp32/dp8/dcp4
# MoE A2A backend: ascend_ops (prefill=alltoallv, decode=low_latency) on both P and D
class TestNPUGLM5_3_W8A8_1P1D_DCP_MOONCAKE_TRANSFER(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Serve GLM-5.3-w8a8 with Mooncake PD transfer to validate the transfer path"""

    # This case exists to exercise the Mooncake PD KV transfer end to end, not
    # the HiCache tiers, so no hit-rate baseline is asserted. Prefill is pure DP
    # attention (32 DP ranks), which spreads one prompt per rank and would make
    # a prefix-reuse threshold meaningless. assert_metrics still fails the case
    # if the bench produces no metrics.
    min_hit_rate_pcts = {}

    @classmethod
    def setUpClass(cls):
        cls.model_config = {
            "model_path": GLM_5_3_W8A8_MODEL_PATH,
            "prefill_args": list(GLM_5_3_DCP_PD_SEP_PREFILL_ARGS),
            "decode_args": list(GLM_5_3_DCP_PD_SEP_DECODE_ARGS),
            "prefill_envs": {
                **GLM_5_3_DCP_PD_SEP_PREFILL_ENVS,
                "MC_TENT_CONF": write_memfabric_tent_conf(),
                "LD_LIBRARY_PATH": append_ld_library_path(
                    ("memfabric_hybrid", "lib"), ("mooncake", None)
                ),
            },
            "decode_envs": {
                **GLM_5_3_DCP_PD_SEP_DECODE_ENVS,
                "MC_TENT_CONF": write_memfabric_tent_conf(),
                "LD_LIBRARY_PATH": append_ld_library_path(
                    ("memfabric_hybrid", "lib"), ("mooncake", None)
                ),
            },
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

    def test_npu_glm_5_3_w8a8_1p1d_32p_dcp_mooncake_transfer(self):
        """Run the 32k GSP bench through the Mooncake PD transfer path"""
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
