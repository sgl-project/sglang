"""End-to-end tests for ``--enable-gather-logits`` (#3365).

Gather-logits is a pure communication change (logits gathered to rank 0 instead
of all-gathered), so generated tokens must be unchanged. Covers greedy output
parity vs the all-gather baseline, and GSM8K + constrained decoding with CUDA
graph enabled (gather runs inside the captured graph) and disabled (eager).
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-b", runner_config="2-gpu-large")

# Deterministic prompts covering a few lengths / content types.
PARITY_PROMPTS = [
    "The capital of France is",
    "Write a short poem about the ocean.",
    "List three prime numbers greater than ten:",
    "In one sentence, explain what a neural network is.",
    "Q: What is 17 multiplied by 23? A:",
]


def _greedy_generate(base_url: str, prompt: str, max_new_tokens: int = 64) -> str:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["text"]


class TestGatherLogitsGreedyParity(CustomTestCase):
    """gather-logits must produce identical greedy tokens to the all-gather path."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

    def _run_and_collect(self, other_args):
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        try:
            return {p: _greedy_generate(self.base_url, p) for p in PARITY_PROMPTS}
        finally:
            kill_process_tree(process.pid)

    def test_greedy_parity(self):
        baseline = self._run_and_collect(["--trust-remote-code", "--tp", "2"])
        gather = self._run_and_collect(
            ["--trust-remote-code", "--tp", "2", "--enable-gather-logits"]
        )
        for prompt in PARITY_PROMPTS:
            self.assertEqual(
                baseline[prompt],
                gather[prompt],
                msg=(
                    "gather-logits changed the greedy output for prompt "
                    f"{prompt!r}:\n baseline={baseline[prompt]!r}\n "
                    f"gather={gather[prompt]!r}"
                ),
            )


class TestGatherLogitsGSM8K(
    CustomTestCase, GSM8KMixin, JSONConstrainedMixin, RegexConstrainedMixin
):
    """Accuracy + constrained decoding with gather-logits and CUDA graph enabled.

    The constrained-decoding cases exercise the grammar path, which in gather
    mode skips the in-sampler cross-TP all-reduce and relies on the rank-0 ->
    all-ranks token broadcast instead.
    """

    gsm8k_accuracy_thres = 0.7

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-gather-logits",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestGatherLogitsGSM8KEager(CustomTestCase, GSM8KMixin):
    """Accuracy with gather-logits and CUDA graph disabled (eager path)."""

    gsm8k_accuracy_thres = 0.7

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--enable-gather-logits",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
