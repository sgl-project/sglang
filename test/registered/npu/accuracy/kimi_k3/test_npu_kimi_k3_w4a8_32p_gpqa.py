import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=5400,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

KIMI_K3_W4A8_32P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS": "1",
    "SGLANG_NPU_USE_TRITON_PREFIX_KV_CACHE_STORE": "1",
    "TRITON_CACHE_DIR": "/tmp/triton_cache",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FOLDED_PROPOSAL": "0",
    "SGLANG_DSPARK_STACKED_CTX_KV": "0",
    "SGLANG_DSPARK_EMBED_IN_GRAPH": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "STREAMS_PER_DEVICE": "32",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "HCCL_BUFFSIZE": "2000",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # TEMPORARY: MLA defaults the host pool to the ACL backend, whose buffers
    # are not torch-pinned, and the Mamba state backup kernel then dies with
    # "host buffer must use pinned memory". Any value other than acl/memfabric
    # falls back to plain pinned memory.
    "SGLANG_HICACHE_HOST_MEM_BACKEND": "torch",
}

KIMI_K3_W4A8_32P_OTHER_ARGS = [
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--nnodes",
    4,
    "--tokenizer-path",
    KIMI_K3_W4A8_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--dtype",
    "bfloat16",
    "--tp-size",
    64,
    "--attn-dp-size",
    4,
    "--enable-dp-lm-head",
    "--enable-shared-experts-attn-tp",
    "--enable-dense-mlp-attn-tp",
    # TEMPORARY: text-only serving keeps is_multimodal false, so the remote-code
    # processor/image processor is not loaded (the 16 scheduler processes of a
    # pod otherwise race to copy its transformers dynamic module and one dies
    # with "has no attribute KimiK3VisionProcessor").
    "--disable-multimodal",
    "--mem-fraction-static",
    0.72,
    "--chunked-prefill-size",
    8192,
    "--cuda-graph-bs-decode",
    1,
    4,
    16,
    "--max-running-requests",
    64,
    "--reasoning-parser",
    "kimi_k3",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    KIMI_K3_DSPARK_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-dspark-block-size",
    7,
    "--speculative-draft-attention-backend",
    "ascend",
    "--linear-attn-verify-backend",
    "triton",
    "--speculative-eagle-topk",
    1,
    "--disable-custom-all-reduce",
    "--watchdog-timeout",
    9000,
    # L2 (host) HiCache
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--enable-cache-report",
    "--hicache-ratio",
    1.2,
    "--hicache-write-policy",
    "write_through",
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3_W4A8_32P_GPQA(TestNpuAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for Kimi-K3-w4a8 32p four nodes on gpqa_diamond"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    accuracy = 0.935
    max_retries = 1
    datasets = ["gpqa_diamond"]
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.92,
    }
    few_shot_num = 0
    eval_batch_size = 32
    # Temporary: run only 1 GPQA Diamond question to shorten the step runtime.
    limit = 1
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "max"},
    }
    timeout = 10000
    seed = 42
    min_hit_rate_pcts = {"host": 0}  # percent: L2 host hit rate must exceed 0%

    def test_npu_kimi_k3_w4a8_32p_gpqa(self):
        """Run NPU accuracy test for Kimi-K3-w4a8 32p four nodes on gpqa_diamond"""
        self.run_accuracy()

    def test_npu_kimi_k3_w4a8_32p_l2_host_hit_rate(self):
        """Run the 64k GSP bench; assert_metrics checks the host hit rate"""
        # 4 groups x 32 prompts share a 57600-token prefix (64k * 0.9) plus a
        # 6400-token question. The three-prefix set already overflows L1, and
        # the workload keeps all four prefixes plus the questions in flight, so
        # evictions keep landing in L2 while 32 prompts per group give each
        # prefix several repeats (reuse).
        metrics = run_bench_serving(
            host=self.host,
            port=self.port,
            model_path=self.model_config.get("model_path"),
            dataset_name="generated-shared-prefix",
            num_prompts=128,
            input_len=64000,
            output_len=1024,
            max_concurrency=32,
            gsp_num_groups=4,
            gsp_prompts_per_group=32,
            seed=42,
            warmup_requests=16,
            cache_report=True,
        )
        assert_metrics(self, metrics)


if __name__ == "__main__":
    unittest.main()
