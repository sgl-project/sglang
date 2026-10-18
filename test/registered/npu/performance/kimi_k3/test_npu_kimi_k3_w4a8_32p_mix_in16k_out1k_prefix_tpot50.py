import logging
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    KIMI_K3_DSPARK_MODEL_PATH,
    KIMI_K3_W4A8_MODEL_PATH,
    TestNpuPerfMultiNodePdMixTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

logger = logging.getLogger(__name__)

# Pure-decode (100% prefix-cache hit) throughput scenario for Kimi-K3 W4A8
# 32P (4-node mix): 16k shared prefix / 1024 output. run_aisbench.sh is
# invoked with --repeat_rate 100, which regenerates the dataset as identical
# prompts, runs a warm pass (output-len=1) to prime the KV cache of every DP
# rank, and only streams the measured pass to the metric parser.
#
# Throughput search: mean TPOT at C64 was ~26 ms on this deployment (dev
# measurement, 2026-10), far below the 50 ms budget of the high-throughput
# tier, so the concurrency is pushed up in one CI run: C72 is asserted
# against the 50 ms SLO, C80 runs exploratorily (no assert) to probe the
# OOM ceiling. The knobs changed vs. the accuracy launch config:
#   --max-mamba-cache-size 320  (dev-validated value for >64 decodes;
#                               default mamba pool starves batch 72/80)
#   --max-running-requests 80   (admit the explored concurrency)
# The decode cuda-graph list stays at the B103-proven {1,4,16}: larger
# captured graphs each cost HBM and the reference decode config also caps
# at 16 (eager decode measured 26 ms TPOT at C64).
register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
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
    # Deprecated spelling, still auto-converted to --attn-dp-size on main;
    # required by the older sglang build in the nightly image.
    "--enable-dp-attention",
    "--dp-size",
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
    # Explore up to batch 80 (dev measured C64 at ~26 ms TPOT; the 50 ms
    # budget leaves room for ~2x concurrency).
    "--max-running-requests",
    80,
    "--max-mamba-cache-size",
    320,
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
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn16kOut1kPrefixTpot50(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """Pure-decode 100% prefix-hit throughput: 16k in / 1024 out, TPOT <= 50 ms.

    C72 is the asserted step; C80 is exploratory (no SLO assert) and its
    failure (e.g. mamba/HBM OOM) does not fail the case, the metrics of both
    steps are dumped for the throughput-vs-concurrency search.
    """

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the dataset; the shared-prefix
    # regeneration itself happens inside run_aisbench.sh (--repeat_rate 100).
    dataset_path = None
    input_len = 16384
    output_len = 1024
    num_prompts = 72
    max_concurrency = 72
    random_range_ratio = 1
    request_rate = 0
    repeat_rate = "100"
    tpot = 50.0

    def test_npu_kimi_k3_w4a8_32p_mix_in16k_out1k_prefix_tpot50(self):
        """Run the C72 (asserted) and C80 (exploratory) throughput steps."""
        for concurrency, tpot_budget in ((72, 50.0), (80, None)):
            self.max_concurrency = concurrency
            self.num_prompts = concurrency
            self.tpot = tpot_budget
            if tpot_budget is None:
                try:
                    self.run_throughput()
                except Exception as e:  # noqa: BLE001 - exploratory step
                    logger.warning(
                        "Exploratory C%s step failed (kept for the record): %s",
                        concurrency,
                        e,
                    )
            else:
                self.run_throughput()


if __name__ == "__main__":
    unittest.main()
