import math
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path

import requests

_TEST_ROOT = Path(__file__).resolve().parents[3]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from registered.openai_server.rust_renderer import launch_rust_renderer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=156, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestOpenAICompletionRustParity(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    api_key = "sk-123456"

    def _get_logprobs(self, *, rust_frontend):
        # compare identical prefill shapes, without graph padding or warmup cache hits
        engine_args = [
            "--random-seed",
            "42",
            "--disable-prefill-cuda-graph",
            "--disable-radix-cache",
        ]
        with ExitStack() as stack:
            if rust_frontend:
                stack.enter_context(
                    launch_rust_renderer(
                        self.model,
                        DEFAULT_URL_FOR_TEST,
                        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                        engine_args=engine_args,
                    )
                )
            else:
                process = popen_launch_server(
                    self.model,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    api_key=self.api_key,
                    env={"SGLANG_RUST_SERVER": "0"},
                    other_args=engine_args,
                )
                stack.callback(kill_process_tree, process.pid)
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/v1/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "prompt": "The capital of France is",
                    "temperature": 0,
                    "max_tokens": 8,
                    "logprobs": 5,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["logprobs"]

    @staticmethod
    def _kl_divergence(reference, candidate):
        assert reference.keys() == candidate.keys()
        keys = sorted(reference)
        reference_max = max(reference.values())
        candidate_max = max(candidate.values())
        reference_weights = [math.exp(reference[key] - reference_max) for key in keys]
        candidate_weights = [math.exp(candidate[key] - candidate_max) for key in keys]
        reference_sum = sum(reference_weights)
        candidate_sum = sum(candidate_weights)
        reference_probabilities = [
            weight / reference_sum for weight in reference_weights
        ]
        candidate_probabilities = [
            weight / candidate_sum for weight in candidate_weights
        ]
        return sum(
            reference_probability
            * math.log(reference_probability / candidate_probability)
            for reference_probability, candidate_probability in zip(
                reference_probabilities,
                candidate_probabilities,
                strict=True,
            )
        )

    def test_logprobs_have_zero_kl_against_python_frontend(self):
        python_logprobs = self._get_logprobs(rust_frontend=False)
        rust_logprobs = self._get_logprobs(rust_frontend=True)

        self.assertEqual(rust_logprobs["tokens"], python_logprobs["tokens"])
        self.assertEqual(
            rust_logprobs["token_logprobs"],
            python_logprobs["token_logprobs"],
        )
        self.assertEqual(
            len(rust_logprobs["top_logprobs"]),
            len(python_logprobs["top_logprobs"]),
        )
        for python_top, rust_top in zip(
            python_logprobs["top_logprobs"],
            rust_logprobs["top_logprobs"],
            strict=True,
        ):
            self.assertEqual(rust_top, python_top)
            self.assertEqual(self._kl_divergence(python_top, rust_top), 0.0)


if __name__ == "__main__":
    unittest.main()
