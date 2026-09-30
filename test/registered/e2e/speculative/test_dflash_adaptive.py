import json
import os
import tempfile
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-small")


class TestDFlashAdaptiveVerifyWidth(CustomTestCase, GSM8KMixin):
    """--speculative-adaptive routes DFLASH verify width by batch size."""

    model = DEFAULT_TARGET_MODEL_DFLASH
    draft_model = DEFAULT_DRAFT_MODEL_DFLASH
    base_url = DEFAULT_URL_FOR_TEST
    # The CI draft proposes blocks of 10; batches of 4+ verify only the first 5.
    full_width = 10
    truncated_width = 5
    # GSM8K runs 128-way, so it decodes almost entirely at the truncated width.
    gsm8k_score_threshold = 0.75

    COUNT_PROMPT = "Count from 1 to 400, separated by commas. Output only the numbers."

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(
                {
                    "1": {"candidate_steps": [cls.full_width - 1]},
                    "4": {"candidate_steps": [cls.truncated_width - 1]},
                },
                f,
            )
            cls.adaptive_config_path = f.name

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "flashinfer",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-adaptive",
                "--speculative-adaptive-config",
                cls.adaptive_config_path,
                "--max-running-requests",
                "32",
                # Refresh the spec gauges on every decode step.
                "--enable-metrics",
                "--decode-log-interval",
                "1",
                "--mem-fraction-static",
                "0.7",
                "--skip-server-warmup",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        if hasattr(cls, "adaptive_config_path"):
            os.unlink(cls.adaptive_config_path)

    def _num_draft_tokens(self) -> int:
        """Active verify width of the last decode step."""
        text = requests.get(self.base_url + "/metrics", timeout=30).text
        for line in text.splitlines():
            if line.startswith("sglang:spec_num_draft_tokens"):
                return int(float(line.rsplit(" ", 1)[1]))
        self.fail("sglang:spec_num_draft_tokens gauge not exported")

    def _generate(self, text, sampling_params):
        r = requests.post(
            self.base_url + "/generate",
            json={"text": text, "sampling_params": sampling_params},
            timeout=600,
        )
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_batch_size_width_cycle(self):
        """A single request verifies the full block; an 8-way batch verifies the
        truncated width, greedy and sampled alike; a following single request is
        back at the full block and reproduces its first greedy output exactly."""
        single = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}
        # Both single requests prefill cold; a radix hit would change the prefill kernels.
        requests.get(self.base_url + "/flush_cache", timeout=30)
        first = self._generate(self.COUNT_PROMPT, single)
        self.assertEqual(self._num_draft_tokens(), self.full_width)

        # Identical greedy requests finish together, so the last decode batch is 8-way.
        greedy = {"temperature": 0, "max_new_tokens": 128, "ignore_eos": True}
        outs = self._generate([self.COUNT_PROMPT] * 8, [greedy] * 8)
        self.assertEqual(self._num_draft_tokens(), self.truncated_width)
        for out in outs:
            meta = out["meta_info"]
            self.assertLessEqual(
                meta["completion_tokens"] / meta["spec_verify_ct"],
                self.truncated_width,
                f"verify was not truncated: {meta}",
            )

        sampled = {"temperature": 1.0, "max_new_tokens": 128, "ignore_eos": True}
        outs = self._generate([self.COUNT_PROMPT] * 8, [sampled] * 8)
        for out in outs:
            self.assertEqual(out["meta_info"]["completion_tokens"], 128)

        requests.get(self.base_url + "/flush_cache", timeout=30)
        again = self._generate(self.COUNT_PROMPT, single)
        self.assertEqual(self._num_draft_tokens(), self.full_width)
        self.assertEqual(again["text"], first["text"])


if __name__ == "__main__":
    unittest.main()
