"""Prefix caching on a prefill instance driven with the fake KV sender.

A request whose ``bootstrap_host`` is the fake sentinel selects the fake
transfer backend and nothing else. Its KV must be inserted into the prefix
cache like any other request, so the next identical request hits. The opt-out
is the explicit ``skip_cache_insert`` field.
"""

import unittest
import uuid

import requests

from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=120, stage="extra-a", runner_config="2-gpu-large")


class TestPrefillOnlyPrefixCache(PDDisaggregationServerBase):
    extra_prefill_args = ["--enable-cache-report"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.start_prefill()
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)

    def _prefill(self, prompt: str, room: int, **extra) -> int:
        response = requests.post(
            self.prefill_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                "bootstrap_host": FAKE_BOOTSTRAP_HOST,
                "bootstrap_room": room,
                **extra,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["meta_info"]["cached_tokens"]

    @staticmethod
    def _unique_prompt() -> str:
        # The fresh tag comes first so no two prompts share a leading token.
        return f"{uuid.uuid4().hex} prefill-only cache probe. " * 8

    def test_fake_sender_requests_are_cached(self):
        prompt = self._unique_prompt()
        first = self._prefill(prompt, room=1)
        self.assertGreater(self._prefill(prompt, room=2), first)

    def test_explicit_opt_out_is_not_cached(self):
        prompt = self._unique_prompt()
        baseline = self._prefill(prompt, room=3, skip_cache_insert=True)
        self.assertEqual(self._prefill(prompt, room=4), baseline)
        self.assertGreater(self._prefill(prompt, room=5), baseline)


if __name__ == "__main__":
    unittest.main()
