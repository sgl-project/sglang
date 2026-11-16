"""Malformed /v1/completions bodies get a 4xx from a real server, which keeps serving:
a valid completion succeeds after every case and /health passes at the end."""

import unittest

import requests

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.request_fuzz_utils import (
    COMPLETIONS_KNOWN_FAILURES,
    COMPLETIONS_VALUE_CASES,
    SEED,
    type_cases,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


class TestCompletionsRequestFuzz(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _post(self, overrides):
        body = {
            "model": self.model,
            "prompt": "The capital of France is",
            "max_tokens": 1,
            "temperature": 0,
            **overrides,
        }
        return requests.post(f"{self.base_url}/v1/completions", json=body, timeout=30)

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        # Known failures are never sent: some may take the server down.
        for case in type_cases(CompletionRequest) + COMPLETIONS_VALUE_CASES:
            if (case.field, case.case_id) in COMPLETIONS_KNOWN_FAILURES:
                continue
            with self.subTest(field=case.field, case=case.case_id, seed=SEED):
                response = self._post(case.overrides)
                self.assertTrue(
                    400 <= response.status_code < 500,
                    f"{case.overrides!r}: {response.status_code} {response.text[:300]}",
                )
                valid = self._post({})
                self.assertEqual(valid.status_code, 200, valid.text[:300])
        health = requests.get(f"{self.base_url}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text[:300])


if __name__ == "__main__":
    unittest.main(verbosity=3)
