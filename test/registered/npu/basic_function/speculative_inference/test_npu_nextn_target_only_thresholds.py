import math
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_5_9B_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)


class TestNpuNextnTargetOnlyThresholds(CustomTestCase):
    """Smoke a real Qwen3.5-9B NEXTN/MTP service with non-default thresholds."""

    model = QWEN3_5_9B_MODEL_PATH
    other_args = [
        "--device",
        "npu",
        "--tp-size",
        "1",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.55",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "2048",
        "--context-length",
        "1024",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--speculative-accept-threshold-single",
        "0.5",
        "--speculative-accept-threshold-acc",
        "0.8",
    ]
    process = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.process = None
        port = find_available_port(int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1]))
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            if cls.process is not None:
                kill_process_tree(cls.process.pid)
                cls.process = None
        finally:
            super().tearDownClass()

    def test_real_nextn_service_reports_thresholds_and_speculative_stats(self):
        # Bypass any shell proxy for requests to the dynamically owned loopback port.
        with requests.Session() as session:
            session.trust_env = False

            health = session.get(f"{self.base_url}/health", timeout=10)
            self.assertEqual(health.status_code, 200)

            info_response = session.get(
                f"{self.base_url}/server_info",
                timeout=10,
            )
            self.assertEqual(info_response.status_code, 200)
            info = info_response.json()
            # NEXTN is normalized to EAGLE in resolved server args; the launch
            # command preserves the user-facing CLI algorithm.
            self.assertEqual(info["speculative_algorithm"], "EAGLE")
            self.assertIn("--speculative-algorithm NEXTN", info["launch_command"])
            self.assertEqual(info["speculative_num_steps"], 1)
            self.assertEqual(info["speculative_num_draft_tokens"], 2)
            self.assertEqual(info["speculative_accept_threshold_single"], 0.5)
            self.assertEqual(info["speculative_accept_threshold_acc"], 0.8)
            self.assertFalse(info["speculative_use_rejection_sampling"])

            response = session.post(
                f"{self.base_url}/generate",
                json={
                    "text": "Count from one to five.",
                    "sampling_params": {
                        "temperature": 0.8,
                        "top_p": 0.95,
                        "max_new_tokens": 16,
                    },
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(result["text"].strip())

        meta = result["meta_info"]
        verify_count = int(meta["spec_verify_ct"])
        proposed = int(meta["spec_num_proposed_drafts"])
        correct = int(meta["spec_num_correct_drafts"])
        completion_tokens = int(meta["completion_tokens"])

        self.assertGreater(verify_count, 0)
        self.assertGreater(completion_tokens, 0)
        self.assertLessEqual(completion_tokens, 16)
        self.assertEqual(proposed, verify_count * (2 - 1))
        self.assertGreaterEqual(correct, 0)
        self.assertLessEqual(correct, proposed)
        self.assertTrue(
            math.isclose(
                float(meta["spec_accept_rate"]),
                correct / proposed,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )
        self.assertTrue(
            math.isclose(
                float(meta["spec_accept_length"]),
                completion_tokens / verify_count,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )

        print(f"Generated text: {result['text']!r}")
        print(
            "Qwen3.5-9B NEXTN generated "
            f"{completion_tokens} tokens; accepted {correct}/{proposed} drafts "
            f"(rate={meta['spec_accept_rate']}, "
            f"mean_length={meta['spec_accept_length']})."
        )


if __name__ == "__main__":
    unittest.main()
