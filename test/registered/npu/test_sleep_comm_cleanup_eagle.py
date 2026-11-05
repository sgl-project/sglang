"""NPU CI test for sleep-mode comm cleanup with EAGLE3 speculative decoding.

Spec decoding keeps extra graphs on the inner EagleDraftWorker, which the sleep
must drop and the wake must recapture. Greedy spec-decoded output must be
bit-identical across full and comm-only sleep/wake cycles.

Usage:
python3 -m unittest test_sleep_comm_cleanup_eagle
"""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_8B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=1200,
    suite="full-2-npu-a3",
    nightly=True,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

_MIB = 1 << 20
FULL_RELEASE_MIN_DELTA_MB = 500


def _device_used_mb():
    free, total = torch.npu.mem_get_info()
    return (total - free) / _MIB


def _post(base_url, endpoint, payload=None):
    return requests.post(f"{base_url}{endpoint}", json=payload or {}, timeout=300)


def _generate_greedy(test_case, base_url):
    response = _post(
        base_url,
        "/generate",
        {
            "text": "Where is the capital of France?",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
    )
    test_case.assertEqual(response.status_code, 200, response.text)
    return response.json()["text"]


class TestSleepCommCleanupEagle(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_8B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--dtype",
                "float16",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
                "--enable-memory-saver",
                "--enable-weights-cpu-backup",
                "--disable-radix-cache",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft-model-path",
                QWEN3_8B_EAGLE3_WEIGHTS_PATH,
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--cuda-graph-max-bs-decode",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_1_full_sleep_wake_with_spec_decoding(self):
        text_before = _generate_greedy(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            FULL_RELEASE_MIN_DELTA_MB,
            f"full sleep should free memory "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _post(self.base_url, "/resume_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)

        # Draft graphs recaptured on the rebuilt comms: identical greedy text.
        text_after = _generate_greedy(self, self.base_url)
        self.assertEqual(text_before, text_after)

    def test_2_comm_only_sleep_wake_with_spec_decoding(self):
        text_before = _generate_greedy(self, self.base_url)

        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)

        text_after = _generate_greedy(self, self.base_url)
        self.assertEqual(text_before, text_after)


if __name__ == "__main__":
    unittest.main()
