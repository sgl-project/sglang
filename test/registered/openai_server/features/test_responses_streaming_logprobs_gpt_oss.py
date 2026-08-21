"""End-to-end wire-shape guard for harmony Responses streaming logprobs.

The unit tests in test_serving_responses_stream.py / test_serving_responses.py
construct ``detokenize_logprob_tokens`` tuples by hand, so they cannot catch a
divergence between those mock tuples and the ones the real
``TokenizerManager.detokenize_logprob_tokens`` produces for a served harmony
model. This serves gpt-oss-20b (a harmony model) and asserts the streaming
delta/done events carry well-formed, aligned logprobs: the total logprob
entries carried across all delta events equals the done event's accumulated
list -- i.e. each delta event accounts for every final-channel token in its
chunk, not just the last one.
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# mxfp4 is the canonical lighter gpt-oss-20b variant and runs on SM90 (H100).
GPT_OSS_20B_MODEL = "openai/gpt-oss-20b"

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")


class TestResponsesStreamingLogprobsGptOss(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GPT_OSS_20B_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    @staticmethod
    def _iter_events(response):
        """Yield parsed JSON payloads from an SSE responses stream."""
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str.strip() == "[DONE]":
                break
            yield json.loads(data_str)

    @staticmethod
    def _assert_well_formed_logprob(entry, top_logprobs):
        """One logprob entry: string token, numeric logprob, bounded tops."""
        assert isinstance(entry.get("token"), str), entry
        assert isinstance(entry.get("logprob"), (int, float)), entry
        tops = entry.get("top_logprobs") or []
        assert len(tops) <= top_logprobs, entry
        for top in tops:
            assert isinstance(top.get("token"), str), top
            assert isinstance(top.get("logprob"), (int, float)), top

    def test_streaming_delta_and_done_logprobs_align(self):
        top_logprobs = 3
        resp = requests.post(
            self.base_url + "/v1/responses",
            json={
                "model": self.model,
                "input": "What is 1 + 1? Answer with just the number.",
                "stream": True,
                "temperature": 0,
                "max_output_tokens": 32,
                "top_logprobs": top_logprobs,
                "include": ["message.output_text.logprobs"],
                "reasoning": {"effort": "low"},
            },
            stream=True,
        )
        self.assertEqual(resp.status_code, 200, getattr(resp, "text", ""))

        flat_delta_logprobs = []  # every logprob entry across delta events
        done_logprobs = None
        for event in self._iter_events(resp):
            etype = event.get("type")
            if etype == "response.output_text.delta":
                for entry in event.get("logprobs") or []:
                    self._assert_well_formed_logprob(entry, top_logprobs)
                    flat_delta_logprobs.append(entry)
            elif etype == "response.output_text.done":
                done_logprobs = event.get("logprobs") or []

        # The stream must have carried logprobs on the delta path.
        self.assertTrue(flat_delta_logprobs, "no logprob entries on any delta event")

        # The done event accumulates every final-channel content token; its
        # count must equal the total carried across delta events. A regression
        # that pins one logprob per chunk (instead of one per token) would make
        # the delta total under-count the done list.
        self.assertIsNotNone(done_logprobs, "no response.output_text.done event")
        for entry in done_logprobs:
            self._assert_well_formed_logprob(entry, top_logprobs)
        self.assertEqual(
            len(done_logprobs),
            len(flat_delta_logprobs),
            "done logprobs count != summed delta logprobs count",
        )
        # Tokens reconstruct identically and in order across both views.
        self.assertEqual(
            [e["token"] for e in done_logprobs],
            [e["token"] for e in flat_delta_logprobs],
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
