import logging
import unittest
import urllib.request

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

# Kimi-K3 W4A8 32P (4-node mix) 128k acceptance scenario, one service for
# both phases (no steady-state phase):
#   Phase 1 (pure prefill): cold 131072 in / 1 out, C1/N1, mean TTFT <= 20 s.
#   Phase 2 (pure decode, low-latency tier): 131072 shared prefix in /
#   1024 out, C12/N12, mean TPOT <= 30 ms. run_aisbench.sh is invoked with
#   --repeat_rate 100, which regenerates the dataset as identical prompts,
#   runs a warm pass (output-len=1) that primes the KV cache of every DP
#   rank, and only streams the measured pass to the metric parser.
#
# Concurrency note: the acceptance table leaves the decode concurrency open;
# 12 keeps the 4-DP ranks evenly loaded (dev measured 23.2-25.9 ms TPOT at
# C12 on this deployment, against the 30 ms budget).
#
# The server launch config is the B103-proven accuracy baseline with two
# prefill-oriented tweaks: chunked-prefill-size 8192 -> 32768 (per-DP chunk
# 2048 -> 8192: a 131072-token request is served in 16 chunk iterations
# instead of 64) and mem-fraction-static 0.72 -> 0.75 for the larger chunk
# activations (the reference prefill config runs 0.78 with 65536). The
# decode-side knobs stay at the proven values: cuda-graph-bs-decode {1,4,16}
# covers the C12 decode batch and larger captured graphs would only cost
# HBM.
register_npu_ci(
    est_time=5400,
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
    0.75,
    "--chunked-prefill-size",
    32768,
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
]

KIMI_K3_W4A8_32P_MODEL_CONFIG = {
    "model_path": KIMI_K3_W4A8_MODEL_PATH,
    "other_args": KIMI_K3_W4A8_32P_OTHER_ARGS,
    "node_envs": KIMI_K3_W4A8_32P_ENVS,
}


class TestNPUKimiK3W4A832PMixIn128kPrefillThenDecode(
    TestNpuPerfMultiNodePdMixTestCaseBase
):
    """128k in / 1 out cold prefill (TTFT<=20s) then 128k shared-prefix pure
    decode at C12 (mean TPOT<=30ms), one server for both phases."""

    model_config = KIMI_K3_W4A8_32P_MODEL_CONFIG
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    # Must be None so run_aisbench() generates the exact-length GSM8K dataset;
    # the base-class default would silently fall back to the raw ShareGPT file.
    dataset_path = None
    random_range_ratio = 1
    request_rate = 0

    @staticmethod
    def _flush_cache(base_url: str) -> None:
        req = urllib.request.Request(
            f"{base_url}/flush_cache", data=b"", method="POST"
        )
        urllib.request.urlopen(req, timeout=120)

    def test_npu_kimi_k3_w4a8_32p_mix_in128k_prefill_then_decode(self):
        """Phase 1 cold prefill TTFT, then phase 2 pure-decode low latency."""
        # ---- Phase 1: pure prefill (cold, zero cache hit) ----
        self.input_len = 131072
        self.output_len = 1
        self.num_prompts = 1
        self.max_concurrency = 1
        self.repeat_rate = None
        self.ttft = 20000.0
        self.tpot = None
        self.run_throughput()

        # Drop the phase-1 KV footprint so the shared-prefix phase starts
        # clean (only the master reaches this line; worker pods are still
        # inside run_throughput's keep-alive).
        try:
            self._flush_cache(self.base_url)
        except Exception as e:  # noqa: BLE001 - best effort
            logger.warning("flush_cache between phases failed, continue: %s", e)

        # ---- Phase 2: pure decode, low-latency tier (100% prefix hit) ----
        self.input_len = 131072
        self.output_len = 1024
        self.num_prompts = 12
        self.max_concurrency = 12
        self.repeat_rate = "100"  # identical prompts + warm pass in run_aisbench.sh
        self.ttft = None
        self.tpot = 30.0
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
