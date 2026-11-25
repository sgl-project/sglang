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
    disabled="accuracy testcase",
)

GLM_5_3_PCP_NO_PP_PD_SEP_PREFILL_ENVS = {
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
    # Both the Mooncake PD KV transfer and the L3 store run over the MemFabric
    # TENT device path; the store endpoint travels in the extra-config JSON and
    # MC_TENT_CONF, both built in setUpClass.
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_STORE_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "SGLANG_HICACHE_MOONCAKE_REUSE_TE": "0",
}

GLM_5_3_PCP_NO_PP_PD_SEP_DECODE_ENVS = {
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
    # Decode also speaks TENT for the Mooncake PD KV transfer.
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
}

GLM_5_3_PCP_NO_PP_PD_SEP_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--attn-cp-size",
    8,
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
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
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    # Ratio-based sizing keeps the L2 host pool proportional to the device pool.
    "--hicache-ratio",
    1.2,
    # L3 storage tier on top of the L2 host pool. MLA on NPU maps page_first
    # to page_first_kv_split, which the Mooncake backend's get_page_buffer_meta
    # supports (layer_first_kv_split does not).
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
    # EPLB is disabled for now; restore this block once it is verified in this setup.
    # "--enable-eplb",
    # "--expert-distribution-recorder-mode",
    # "stat",
    # "--ep-num-redundant-experts",
    # 16,
    # "--eplb-rebalance-num-iterations",
    # 100000,
    # "--expert-distribution-recorder-buffer-size",
    # 50,
]

GLM_5_3_PCP_NO_PP_PD_SEP_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--nnodes",
    2,
    "--tp-size",
    32,
    "--attn-dp-size",
    8,
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
    # EPLB is disabled for now, matching the prefill side.
    # "--enable-eplb",
    # "--expert-distribution-recorder-mode",
    # "stat",
    # "--ep-num-redundant-experts",
    # 32,
    # "--eplb-rebalance-num-iterations",
    # 100000,
    # "--expert-distribution-recorder-buffer-size",
    # 1000,
]


# PCP variant without pipeline parallelism: PD KV transfer over Mooncake plus
# the L3 Mooncake store (EPLB currently disabled, see the commented blocks above)
# 4-node PD separation, prefill tp32 + attn-cp8 (interleave),
# decode tp32/dp32/ep32, GLM-5.3-w8a8
# MoE A2A backend: ascend_ops (prefill=alltoallv, decode=low_latency) on both P and D
class TestNPUGLM5_3_W8A8_1P1D_PCP_NO_PP(TestNpuAccuracyMultiNodePdSepTestCaseBase):
    """Serve GLM-5.3-w8a8 with Mooncake PD transfer, PCP (no PP) and L3 Mooncake; check accuracy and hit rates"""

    # Run evalscope only once (no retry on accuracy miss)
    max_retries = 1
    accuracy = 0.912
    datasets = ["gpqa_diamond"]
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.92,
    }
    dataset_args = {
        "gpqa_diamond": {
            "filters": {"remove_until": "</think>"},
            "system_prompt": "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}.",
        },
        "gsm8k": {"filters": {"remove_until": "</think>"}},
    }
    eval_batch_size = 48
    # Temporary: run only 1 GPQA Diamond question to shorten the step runtime.
    limit = 1
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    # L2 host and L3 storage traffic must both be non-zero in the hit-rate test.
    min_hit_rate_pcts = {"host": 0, "storage": 0}

    @classmethod
    def setUpClass(cls):
        # Only the prefill node runs the L3 store client; the decode node has no
        # storage backend. The Mooncake store config is one JSON passed through
        # --hicache-storage-backend-extra-config (once it carries
        # master_server_address the client ignores the MOONCAKE_* env vars).
        # The master runs on the first prefill pod; each scheduler rank
        # contributes global_segment_size to the pool. Both nodes additionally
        # get the TENT config of this pod plus the memfabric/mooncake libs,
        # because the PD KV transfer runs over Mooncake too.
        l3_config = json.dumps(
            {
                "master_server_address": get_mooncake_master_server_address(),
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "global_segment_size": "2gb",
                "local_hostname": resolve_local_ip(),
                "extra_backend_tag": "pd-1p1d-glm5.3-w8a8-pcp-no-pp",
            }
        )
        transfer_envs = {
            "MC_TENT_CONF": write_memfabric_tent_conf(),
            "LD_LIBRARY_PATH": append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            ),
        }
        cls.model_config = {
            "model_path": GLM_5_3_W8A8_MODEL_PATH,
            "prefill_args": [
                *GLM_5_3_PCP_NO_PP_PD_SEP_PREFILL_ARGS,
                # Cap the device KV pool so the 32k-token GSP workload overflows
                # L1 and exercises the L2 host tier. The 1.2x host pool plus the
                # device pool cover ~144k tokens, while the hit-rate test's six
                # prefixes need ~192k, so evictions reach the L3 store. Accuracy
                # is unaffected: gpqa prompts are short and only the prefill-side
                # cache is capped.
                "--max-total-tokens",
                65536,
                "--hicache-storage-backend-extra-config",
                l3_config,
            ],
            "decode_args": list(GLM_5_3_PCP_NO_PP_PD_SEP_DECODE_ARGS),
            "prefill_envs": {
                **GLM_5_3_PCP_NO_PP_PD_SEP_PREFILL_ENVS,
                **transfer_envs,
            },
            "decode_envs": {
                **GLM_5_3_PCP_NO_PP_PD_SEP_DECODE_ENVS,
                **transfer_envs,
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

    def test_npu_glm_5_3_w8a8_1p1d_pcp_no_pp_gpqa(self):
        """Run NPU accuracy test for GLM-5.3-w8a8 PD separation 1p1d with PCP and no PP on gpqa_diamond"""
        self.run_accuracy()

    def test_npu_glm_5_3_w8a8_1p1d_32p_pcp_no_pp_l2_l3_hit_rate(self):
        """Run the 32k GSP bench; assert_metrics checks the host and storage hit rates"""
        # 6 groups x 32 prompts share a 28800-token system prompt (32k * 0.9)
        # plus a 3200-token question: ~192k unique tokens against the ~144k
        # tokens held by L1+L2, so the overflow reaches the L3 store. 32 prompts
        # per group give several repeats of the same prefix (reuse).
        metrics = run_bench_serving(
            host=self.host,
            port=self.port,
            model_path=self.model_config.get("model_path"),
            dataset_name="generated-shared-prefix",
            input_len=32000,
            output_len=1024,
            max_concurrency=32,
            gsp_num_groups=6,
            gsp_prompts_per_group=32,
            seed=100,
            warmup_requests=16,
            cache_report=True,
        )
        assert_metrics(self, metrics)


if __name__ == "__main__":
    unittest.main()
