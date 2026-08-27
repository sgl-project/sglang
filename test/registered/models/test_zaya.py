"""End-to-end server test for Zyphra ZAYA1 (hybrid CCA attention + MoE).

This test boots a real ``Zyphra/ZAYA1-base`` SGLang server via
``popen_launch_server``, sends a handful of completions through the HTTP API,
and finishes with a small MMLU sanity slice.

The test is gated behind ``RUN_ZAYA_E2E=1`` so the registered suite does not
have to download the full ZAYA1-base checkpoint (≈17 GB) on every run; the CI
job that owns this test sets the variable explicitly.
"""

import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# ZAYA1-base is a heavyweight launch (≈120 transformer layers with MoE), so
# the estimated time is set generously to keep the CI scheduler from preempting
# the job before the server finishes warming up.
register_cuda_ci(est_time=420, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=420, suite="stage-b-test-1-gpu-large-amd")


_MODEL_PATH = os.environ.get("ZAYA_MODEL_PATH", "Zyphra/ZAYA1-base")


def _zaya_enabled() -> bool:
    return os.environ.get("RUN_ZAYA_E2E", "0") == "1"


@unittest.skipUnless(
    _zaya_enabled(),
    "Set RUN_ZAYA_E2E=1 to enable the ZAYA1 end-to-end server test "
    "(requires downloading the model weights).",
)
class TestZayaServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--mem-fraction-static",
            "0.5",
            "--max-running-requests",
            "8",
        ]
        if is_hip():
            other_args += ["--attention-backend", "triton"]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def test_generation_basic(self):
        """Send three prompts through the ``/generate`` endpoint and require
        non-empty completions for each."""
        import requests

        prompts = [
            "The capital of France is",
            "1 + 2 + 3 + 4 + 5 =",
            "Write a haiku about silicon:",
        ]
        for prompt in prompts:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 16,
                    },
                },
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            data = resp.json()
            self.assertIn("text", data, data)
            self.assertGreater(len(data["text"].strip()), 0, data)

    def test_mmlu_sanity(self):
        """32-example MMLU sanity slice.

        ZAYA1-base is a pretrained (non instruction-tuned) checkpoint that
        emits long ``<think>…</think>`` reasoning blocks before settling on a
        final letter, so ``max_tokens`` must be large enough for the evaluator
        to see the chosen answer. The threshold sits just above chance: it is
        a regression sanity check rather than a production-quality gate. An
        instruction-tuned ZAYA1 checkpoint scores meaningfully higher and
        should raise this bound when wired in.
        """
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=32,
            num_threads=8,
            max_tokens=1024,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(
            metrics["score"],
            0.30,
            f"MMLU sanity below threshold: {metrics}",
        )


if __name__ == "__main__":
    unittest.main()
