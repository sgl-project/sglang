"""Malformed /v1/completions bodies get a 4xx through the real HTTP route, and a
valid request still succeeds after each one. The model is replaced by a fake
TokenizerManager that runs the request checks done before the scheduler."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.request_fuzz_utils import (
    COMPLETIONS_KNOWN_FAILURES,
    COMPLETIONS_VALUE_CASES,
    SEED,
    type_cases,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

VOCAB_SIZE = 32000
VALID_BODY = {"model": "m", "prompt": "Hello", "max_tokens": 1, "temperature": 0}
CASES = type_cases(CompletionRequest) + [
    c for c in COMPLETIONS_VALUE_CASES if not c.needs_server
]


def _fake_tokenizer_manager():
    tm = Mock(spec=TokenizerManager)
    tm.tokenizer = Mock()
    tm.tokenizer.encode.side_effect = lambda text, **kwargs: [1] * len(text)
    tm.model_config = Mock(is_multimodal=False, vocab_size=VOCAB_SIZE)
    tm.server_args = Mock(enable_cache_report=False)
    tm.request_logger = Mock(log_requests=False)

    async def generate_request(obj, request=None):
        obj.normalize_batch_and_arguments()
        params = SamplingParams(**obj.sampling_params)
        params.normalize(tm.tokenizer)
        params.verify(VOCAB_SIZE)
        yield {
            "text": "ok",
            "meta_info": {
                "id": "cmpl-fuzz",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "length", "length": 1},
                "weight_version": "default",
            },
            "index": 0,
        }

    tm.generate_request = generate_request
    return tm


class TestCompletionsRequestFuzz(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        state = http_server.app.state
        saved = getattr(state, "openai_serving_completion", None)
        self.addCleanup(setattr, state, "openai_serving_completion", saved)
        state.openai_serving_completion = OpenAIServingCompletion(
            _fake_tokenizer_manager(), Mock(completion_template_name=None)
        )
        self.client = TestClient(http_server.app)

    def _post(self, overrides):
        return self.client.post("/v1/completions", json={**VALID_BODY, **overrides})

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        for case in CASES:
            with self.subTest(field=case.field, case=case.case_id, seed=SEED):
                response = self._post(case.overrides)
                valid = self._post({})
                self.assertEqual(valid.status_code, 200, valid.text[:300])
                self.assertEqual(valid.json()["choices"][0]["text"], "ok")

                rejected = 400 <= response.status_code < 500
                fix = COMPLETIONS_KNOWN_FAILURES.get((case.field, case.case_id))
                if fix is not None and not rejected:
                    # Skip rather than fail, so the fix landing cannot break CI.
                    self.skipTest(f"known failure until {fix} merges")
                self.assertTrue(
                    rejected,
                    f"{case.overrides!r}: {response.status_code} {response.text[:300]}",
                )


if __name__ == "__main__":
    unittest.main()
