"""Ragged (trimmed) verification must keep the target's predictions: a server that
verifies only the two best drafts of every request through the ragged graphs must
generate what a server verifying exactly two drafts uniformly generates."""

import os
import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

COMMON_ARGS = [
    "--trust-remote-code",
    "--random-seed",
    "1",
    "--mem-fraction-static",
    "0.6",
    "--max-running-requests",
    "8",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-draft-model-path",
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    "--speculative-eagle-topk",
    "1",
]
PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean.",
    "def fibonacci(n):",
    "Explain why the sky is blue in two sentences.",
    "List three prime numbers greater than 100:",
    "Translate 'good morning' into Spanish, French and German.",
    "Once upon a time, in a small village,",
    "The derivative of x^3 + 2x is",
]
# Greedy outputs can drift by kernel numerics when two forwards differ in shape;
# a broken ragged layout corrupts every prompt.
MIN_IDENTICAL = len(PROMPTS) - 2


def _generate_all(base_url):
    texts = []
    for prompt in PROMPTS:
        response = requests.post(
            base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 48},
            },
            timeout=120,
        )
        response.raise_for_status()
        texts.append(response.json()["text"])
    return texts


def _launch(other_args, env=None):
    return popen_launch_server(
        DEFAULT_MODEL_NAME_FOR_TEST_MLA,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=COMMON_ARGS + other_args,
        env=env,
    )


class TestEagleAdaptiveVerification(CustomTestCase):
    def test_trimmed_verify_matches_uniform_verify(self):
        process = _launch(
            ["--speculative-num-steps", "2", "--speculative-num-draft-tokens", "3"]
        )
        try:
            uniform = _generate_all(DEFAULT_URL_FOR_TEST)
        finally:
            terminate_and_kill_process_tree(process)

        env = {**os.environ, envs.SGLANG_TEST_ADAPTIVE_VERIFY_FORCE_TIER.name: "2"}
        process = _launch(
            [
                "--speculative-num-steps",
                "3",
                "--speculative-num-draft-tokens",
                "4",
                "--speculative-adaptive-verification",
            ],
            env=env,
        )
        try:
            trimmed = _generate_all(DEFAULT_URL_FOR_TEST)
        finally:
            terminate_and_kill_process_tree(process)

        identical = sum(a == b for a, b in zip(uniform, trimmed))
        self.assertGreaterEqual(identical, MIN_IDENTICAL, (uniform, trimmed))


if __name__ == "__main__":
    unittest.main()
