import re
import subprocess
import sys
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-8-npu-a5",
    nightly=True,
)

# TODO: Add MIMO_V2_5_PRO_FP4_MODEL_PATH to test_npu_performance_utils.py and test_ascend_utils.py
MIMO_V2_5_PRO_FP4_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash"
)
# TODO: Add MIMO_V2_5_PRO_DFLASH_MODEL_PATH to test_npu_performance_utils.py and test_ascend_utils.py
MIMO_V2_5_PRO_DFLASH_MODEL_PATH = (
    "/root/.cache/modelscope/hub/models/XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash/dflash"
)
SHAREGPT_DATASET_PATH = "/root/.cache/modelscope/hub/datasets/gliang1001/ShareGPT_V3_unfiltered_cleaned_split/ShareGPT_V3_unfiltered_cleaned_split.json"

MIMO_V2_5_PRO_FP4_8P_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_BUFFSIZE": "300",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "20",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "0",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

MIMO_V2_5_PRO_FP4_8P_OTHER_ARGS = [
    "--served-model-name",
    MIMO_V2_5_PRO_FP4_MODEL_PATH,
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--mem-fraction-static",
    0.905,
    "--tp-size",
    8,
    "--nnodes",
    1,
    "--chunked-prefill-size",
    8192,
    "--max-total-tokens",
    600000,
    "--max-running-requests",
    8,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    6,
    8,
    "--speculative-algorithm",
    "DFLASH",
    "--speculative-draft-model-path",
    MIMO_V2_5_PRO_DFLASH_MODEL_PATH,
    "--speculative-num-draft-tokens",
    8,
    "--enable-metrics",
]

# Bench serving params from run_mimo.sh Part 3
NUM_PROMPTS = 128
RANDOM_INPUT_LEN = 16000
RANDOM_OUTPUT_LEN = 1000
MAX_CONCURRENCY = 32
NUM_DRAFT_TOKENS = 8
ACCEPT_RATE_THRESHOLD = 0.25


class TestNPUMiMoV2_5_Pro_W4A8_8P_A5_Acceptance(CustomTestCase):
    """Test NPU acceptance rate and acceptance length for MiMo-V2.5-Pro-FP4 8p A5 single node with DFLASH.

    Runs bench_serving with random dataset to measure speculative decoding acceptance rate.
    Acceptance rate = avg_spec_accept_length / speculative_num_draft_tokens.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = MIMO_V2_5_PRO_FP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=3600,
            other_args=MIMO_V2_5_PRO_FP4_8P_OTHER_ARGS,
            env=MIMO_V2_5_PRO_FP4_8P_ENVS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_acceptance(self):
        """Run bench_serving and verify acceptance rate > 0.25."""
        parsed = urlparse(self.base_url)

        cmd = [
            sys.executable,
            "-m",
            "sglang.bench_serving",
            "--backend",
            "sglang",
            "--host",
            parsed.hostname,
            "--port",
            str(parsed.port),
            "--model",
            self.model_path,
            "--dataset-path",
            SHAREGPT_DATASET_PATH,
            "--dataset-name",
            "random",
            "--tokenize-prompt",
            "--random-input-len",
            str(RANDOM_INPUT_LEN),
            "--random-output-len",
            str(RANDOM_OUTPUT_LEN),
            "--request-rate",
            "inf",
            "--random-range-ratio",
            "1",
            "--num-prompts",
            str(NUM_PROMPTS),
            "--max-concurrency",
            str(MAX_CONCURRENCY),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}")
        self.assertEqual(result.returncode, 0, f"bench_serving failed: {result.stderr}")

        # Parse accept_length from bench_serving CLI output
        match = re.search(r"Accept length:\s+([\d.]+)", result.stdout)
        if match:
            accept_length = float(match.group(1))
        else:
            # Fallback: query /server_info endpoint
            server_info = requests.get(self.base_url + "/server_info").json()
            accept_length = server_info["internal_states"][0].get(
                "avg_spec_accept_length", 0.0
            )

        accept_rate = accept_length / NUM_DRAFT_TOKENS

        print(f"Accept length: {accept_length}")
        print(f"Accept rate: {accept_rate} (threshold: {ACCEPT_RATE_THRESHOLD})")

        self.assertGreater(
            accept_rate,
            ACCEPT_RATE_THRESHOLD,
            f"Acceptance rate {accept_rate:.4f} <= {ACCEPT_RATE_THRESHOLD}, "
            f"accept_length={accept_length}",
        )


if __name__ == "__main__":
    unittest.main()
