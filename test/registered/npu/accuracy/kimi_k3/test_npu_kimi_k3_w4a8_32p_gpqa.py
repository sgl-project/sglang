import logging
import re
import subprocess
import unittest
import urllib.request

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    PYTHON_FOR_TEST_TOOL,
    TestNpuAccuracyMultiNodePdMixTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import (
    NIC_NAME,
    check_role,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

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
}

# L2 (host) HiCache: evict L1 prefix/KV to host memory so the shared prefix
# survives L1 eviction. kernel_ascend routes the L2 IO through the Ascend
# kernel, cache-report exposes the device/host hit breakdown, ratio 2 gives
# the host pool twice the L1 quota, and write_through keeps L2 warm as L1
# fills (observed 64k host hit rate 15.9%~20.5% with this exact setup).
KIMI_K3_W4A8_32P_L2_HICACHE_ARGS = [
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--enable-cache-report",
    "--hicache-ratio",
    2,
    "--hicache-write-policy",
    "write_through",
]

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
] + KIMI_K3_W4A8_32P_L2_HICACHE_ARGS

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3_W4A8_32P_GPQA(TestNpuAccuracyMultiNodePdMixTestCaseBase):
    """Test NPU accuracy for Kimi-K3-w4a8 32p four nodes on gpqa_diamond

    The server runs with the L2 (host) HiCache enabled, and a second test
    runs a generated-shared-prefix benchserving round at the 64k operating
    point, asserting the L2 (host) hit rate.
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    accuracy = 0.935
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    eval_batch_size = 32
    generation_config = {
        "max_tokens": 131072,
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"reasoning_effort": "max"},
    }
    timeout = 10000
    seed = 42

    # 64k GSP bench point: 3 groups x 32 prompts share a 57600-token system
    # prefix plus a 6400-token question (9:1), so the working set exceeds L1
    # and evictions land in L2. Observed on this setup: total hit rate
    # 82.1%~82.2%, host (L2) hit rate 15.9%~20.5% -- the threshold keeps a
    # margin against run-to-run variance.
    bench_gsp_num_groups = 3
    bench_gsp_prompts_per_group = 32
    bench_gsp_system_prompt_len = 57600
    bench_gsp_question_len = 6400
    bench_gsp_output_len = 1024
    bench_max_concurrency = 32
    bench_seed = 42
    bench_warmup_requests = 16
    bench_timeout = 5400
    min_host_hit_rate_pct = 10.0

    def test_npu_kimi_k3_w4a8_32p_gpqa(self):
        """Run NPU accuracy test for Kimi-K3-w4a8 32p four nodes on gpqa_diamond"""
        self.run_accuracy()

    @check_role(allowed_roles=["master"])
    def run_gsp_bench_with_cache_report(self):
        """Run one 64k GSP benchserving round and return the host hit rate %"""
        # Start from a clean L1/L2 state so the hit rate is reproducible.
        try:
            req = urllib.request.Request(
                f"{self.base_url}/flush_cache", data=b"", method="POST"
            )
            with urllib.request.urlopen(req, timeout=60):
                logger.info("flush_cache done before the GSP bench")
        except Exception as e:
            logger.warning(f"flush_cache failed, continue anyway: {e}")

        model_path = self.model_config.get("model_path")
        cmd = [
            PYTHON_FOR_TEST_TOOL,
            "-m",
            "sglang.benchmark.serving",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--backend",
            "sglang",
            "--model",
            model_path,
            "--tokenizer",
            model_path,
            "--dataset-name",
            "generated-shared-prefix",
            "--gsp-num-groups",
            str(self.bench_gsp_num_groups),
            "--gsp-prompts-per-group",
            str(self.bench_gsp_prompts_per_group),
            "--gsp-system-prompt-len",
            str(self.bench_gsp_system_prompt_len),
            "--gsp-question-len",
            str(self.bench_gsp_question_len),
            "--gsp-output-len",
            str(self.bench_gsp_output_len),
            "--max-concurrency",
            str(self.bench_max_concurrency),
            "--random-range-ratio",
            "1",
            "--seed",
            str(self.bench_seed),
            "--warmup-requests",
            str(self.bench_warmup_requests),
            "--cache-report",
        ]
        logger.info(f"Run GSP bench: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.bench_timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        logger.info(f"GSP bench exit code: {proc.returncode}")
        assert proc.returncode == 0, f"GSP bench failed with rc={proc.returncode}"

        # The cache report prints:
        #   Cache hit rate:    82.2%
        #     Device:          80.3%
        #     Host:            19.7%
        matches = re.findall(
            r"Cache hit rate:\s+[\d.]+%\s*\n\s*Device:\s+([\d.]+)%\s*\n\s*Host:\s+([\d.]+)%",
            output,
        )
        assert matches, "No cache hit report found in the bench output"
        device_hit_pct, host_hit_pct = map(float, matches[-1])
        logger.info(
            f"Cache hit rate: device={device_hit_pct}% host={host_hit_pct}% "
            f"(threshold {self.min_host_hit_rate_pct}%)"
        )
        return host_hit_pct

    @check_role(allowed_roles=["master"])
    def test_npu_kimi_k3_w4a8_32p_l2_host_hit_rate(self):
        """Assert the L2 (host) hit rate of the 64k GSP workload on the L2 server"""
        host_hit_pct = self.run_gsp_bench_with_cache_report()
        assert host_hit_pct >= self.min_host_hit_rate_pct, (
            f"L2 host hit rate {host_hit_pct}% is below the threshold "
            f"{self.min_host_hit_rate_pct}% (observed range: 15.9%~20.5%)"
        )


if __name__ == "__main__":
    unittest.main()
