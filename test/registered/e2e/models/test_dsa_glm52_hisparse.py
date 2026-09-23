import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import terminate_and_kill_process_tree

register_cuda_ci(est_time=540, stage="extra-b", runner_config="8-gpu-h200")

GLM52_FP8_MODEL_PATH = "zai-org/GLM-5.2-FP8"


class TestGLM52HiSparseSpec(DefaultServerBase, GSM8KMixin):
    """GLM-5.2 FP8 with HiSparse and multi-step EAGLE speculative decoding."""

    model = GLM52_FP8_MODEL_PATH
    other_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
        "--enable-dp-attention",
        "--page-size",
        "64",
        "--max-running-requests",
        "200",
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--kv-cache-dtype",
        "bfloat16",
        "--dsa-decode-backend",
        "flashmla_sparse",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--enable-hisparse",
        "--hisparse-config",
        '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    # Match the original standalone hisparse eval config.
    gsm8k_accuracy_thres = 0.94
    gsm8k_num_questions = 500
    gsm8k_num_threads = 100
    gsm8k_num_shots = 24
    gsm8k_accept_length_thres = 2.5

    def test_long_context_spec_swap(self):
        """Exercise the speculative swap path beyond the 4096-token GPU cache."""
        response = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": [1] * 8192,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 128,
                    "ignore_eos": True,
                },
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        meta = result["meta_info"]
        self.assertEqual(meta["completion_tokens"], 128)
        self.assertGreater(meta["spec_verify_ct"], 0)

    @classmethod
    def tearDownClass(cls):
        # HiSparse's pinned host buffer needs longer than the base class's 60s.
        terminate_and_kill_process_tree(
            cls.process, terminate_timeout=90, wait_timeout=60
        )
        time.sleep(2)


if __name__ == "__main__":
    unittest.main()
