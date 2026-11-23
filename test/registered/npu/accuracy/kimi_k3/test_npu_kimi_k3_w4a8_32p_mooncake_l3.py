import json
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    get_mooncake_master_server_address,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
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

# L3 store client: MemFabric TENT transport over the NPU device path, as in the
# standalone deployment scripts. The store endpoint travels in the extra-config
# JSON and MC_TENT_CONF, both built per node in setUpClass.
KIMI_K3_W4A8_32P_L3_ENVS = {
    "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_STORE_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "SGLANG_HICACHE_MOONCAKE_REUSE_TE": "0",
    # All 64 ranks register a store segment before the HiCache sync groups are
    # built over the world HCCL domain, so the slowest node's store setup must
    # not race the default connect timeout of the early ones.
    "HCCL_CONNECT_TIMEOUT": "1800",
    "HCCL_EXEC_TIMEOUT": "1800",
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
    # TEMPORARY: this case only sends text. Serving the model text-only keeps
    # is_multimodal false, so neither the scheduler nor the tokenizer manager
    # loads the remote-code processor/image processor, whose transformers
    # dynamic module the 16 scheduler processes of a pod race to copy (one rank
    # imports a half-written file and dies with "has no attribute
    # KimiK3VisionProcessor"). --language-model-only is not an option here: it
    # rejects KimiK3ForConditionalGeneration. Remove once the race is fixed.
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
    # The L3 store is the deep tier, so the host pool only has to cover recent
    # evictions; the smaller pool also leaves host huge pages for the store's
    # per-rank segment allocation.
    "--hicache-ratio",
    1.2,
    "--hicache-write-policy",
    "write_through",
    "--hicache-mem-layout",
    "page_first",
    "--hicache-storage-prefetch-policy",
    "wait_complete",
    "--hicache-storage-backend",
    "mooncake",
]


class TestNPUKimiK3_W4A8_32P_GPQA(TestNpuAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for Kimi-K3-w4a8 32p four nodes on gpqa_diamond"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
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
    # L2 host and L3 storage traffic must both be non-zero.
    min_hit_rate_pcts = {"host": 0, "storage": 0}

    @classmethod
    def setUpClass(cls):
        # PD mix has no separate prefill node, so every node runs the L3 store
        # client and the master node (sglang-node-0) hosts the Mooncake master,
        # which the base class starts from other_args. The store config is one
        # JSON per node: the master address resolves from the ConfigMap and each
        # scheduler rank contributes global_segment_size / (tp_size * pp_size).
        l3_config = json.dumps(
            {
                "master_server_address": get_mooncake_master_server_address(),
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "ascend",
                "global_segment_size": "32gb",
                "local_hostname": resolve_local_ip(),
                "extra_backend_tag": "pd-mix-kimi-k3-w4a8-32p",
            }
        )
        cls.model_config = {
            "model_path": KIMI_K3_W4A8_MODEL_PATH,
            "other_args": [
                *KIMI_K3_W4A8_32P_OTHER_ARGS,
                "--hicache-storage-backend-extra-config",
                l3_config,
            ],
            "node_envs": {
                **KIMI_K3_W4A8_32P_ENVS,
                **KIMI_K3_W4A8_32P_L3_ENVS,
                "MC_TENT_CONF": write_memfabric_tent_conf(),
                "LD_LIBRARY_PATH": append_ld_library_path(("memfabric_hybrid", "lib")),
            },
        }
        super().setUpClass()

    def test_npu_kimi_k3_w4a8_32p_gpqa(self):
        """Run NPU accuracy test for Kimi-K3-w4a8 32p four nodes on gpqa_diamond"""
        self.run_accuracy()

    def test_npu_kimi_k3_w4a8_32p_l2_host_l3_storage_hit_rate(self):
        """Run the 64k GSP bench; assert_metrics checks the host and storage hit rates"""
        # 4 groups x 32 prompts share a 57600-token prefix (64k * 0.9) plus a
        # 6400-token question. The three-prefix set already overflows L1 (that
        # is what the L2 baseline measured), and the device pool plus the 1.2x
        # host pool cannot hold the 230k-token four-prefix set together with the
        # questions in flight, so evictions reach the L3 store, while 32 prompts
        # per group give each prefix several repeats (reuse).
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
