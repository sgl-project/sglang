"""Malformed /generate bodies get a 4xx through the real HTTP route, and a valid
request still succeeds after each one. The model is replaced by a fake
TokenizerManager that runs the request checks done before the scheduler."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from sglang.api.v1.api_types import GenerateRequest
from sglang.srt.entrypoints import http_server
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.request_fuzz_utils import (
    GENERATE_KNOWN_FAILURES,
    GENERATE_VALUE_CASES,
    SEED,
    generate_sampling_params_type_cases,
    type_cases,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB_SIZE = 32000
VALID_BODY = {
    "text": "Hello",
    "sampling_params": {"max_new_tokens": 1, "temperature": 0},
}
CASES = (
    type_cases(GenerateRequest)
    + generate_sampling_params_type_cases()
    + [c for c in GENERATE_VALUE_CASES if not c.needs_server]
)


class _FakeTokenizerManager:
    def __init__(self):
        self.tokenizer = Mock()
        self.tokenizer.encode.side_effect = lambda text, **kwargs: [1] * len(text)
        self.create_abort_task = Mock(return_value=None)

    async def generate_request(self, obj, request=None):
        obj.normalize_batch_and_arguments()
        sampling_params = obj.sampling_params
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params]
        for kwargs in sampling_params:
            params = SamplingParams(**kwargs)
            params.normalize(self.tokenizer)
            params.verify(VOCAB_SIZE)
        yield {
            "text": "ok",
            "meta_info": {"id": "fuzz", "finish_reason": {"type": "length"}},
        }


class TestGenerateRequestFuzz(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        prior_state = http_server.get_global_state()
        self.addCleanup(http_server.set_global_state, prior_state)
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=_FakeTokenizerManager())
        )
        # An exception that escapes the app is a 500 to a real client, not a test error.
        self.client = TestClient(http_server.app, raise_server_exceptions=False)

    def _post(self, overrides):
        return self.client.post("/generate", json={**VALID_BODY, **overrides})

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        for case in CASES:
            with self.subTest(field=case.field, case=case.case_id, seed=SEED):
                response = self._post(case.overrides)
                valid = self._post({})
                self.assertEqual(valid.status_code, 200, valid.text[:300])
                self.assertEqual(valid.json()["text"], "ok")

                rejected = 400 <= response.status_code < 500
                fix = GENERATE_KNOWN_FAILURES.get((case.field, case.case_id))
                if fix is not None and not rejected:
                    # Skip rather than fail, so the fix landing cannot break CI.
                    self.skipTest(f"known failure until {fix} merges")
                self.assertTrue(
                    rejected,
                    f"{case.overrides!r}: {response.status_code} {response.text[:300]}",
                )


if __name__ == "__main__":
    unittest.main()
