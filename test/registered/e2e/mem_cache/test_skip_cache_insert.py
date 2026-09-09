"""Per-request opt-out from prefix cache insertion (``skip_cache_insert``).

A request with ``skip_cache_insert=true`` may reuse cached prefixes but leaves
nothing behind: an identical follow-up request gets no cached tokens. The same
prompt sent normally afterwards is inserted, which proves the cache itself works
and the zero above was the opt-out, not a disabled cache.
"""

import unittest
import uuid

import openai
import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

# Small enough that the long prompt below spans several chunks.
CHUNKED_PREFILL_SIZE = 64


def _unique_prompt(num_sentences: int) -> str:
    # A fresh token per call so no earlier test can have inserted this prefix.
    tag = uuid.uuid4().hex
    return " ".join(
        f"Sentence {i} of session {tag} talks about topic {i * 7 % 13}."
        for i in range(num_sentences)
    )


class SkipCacheInsertMixin:
    """Test bodies shared by the cache variants; not a TestCase itself so
    pytest does not launch a server for the mixin."""

    other_args: list = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-cache-report",
                "--chunked-prefill-size",
                str(CHUNKED_PREFILL_SIZE),
                *cls.other_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process, wait_timeout=60)

    def _generate(self, text, **extra):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                **extra,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _cached_tokens(self, result) -> int:
        return result["meta_info"]["cached_tokens"]

    def _assert_opt_out_leaves_nothing(self, prompt: str):
        # Shared leading tokens (BOS, a chat template header) may already be
        # cached, so compare against what the opt-out request itself saw.
        baseline = self._cached_tokens(self._generate(prompt, skip_cache_insert=True))

        second = self._cached_tokens(self._generate(prompt))
        self.assertEqual(
            second,
            baseline,
            "an opt-out request must not leave a reusable prefix behind",
        )

        third = self._cached_tokens(self._generate(prompt))
        self.assertGreater(
            third, second, "the second (normal) request should have been inserted"
        )

    def test_short_prompt(self):
        self._assert_opt_out_leaves_nothing(_unique_prompt(2))

    def test_chunked_prompt(self):
        # Several chunked-prefill rounds; each intermediate round must keep the
        # request's own KV without publishing it.
        self._assert_opt_out_leaves_nothing(_unique_prompt(40))

    def test_opt_out_still_reads_existing_prefix(self):
        prompt = _unique_prompt(8)
        inserted = self._cached_tokens(self._generate(prompt))
        reader = self._cached_tokens(self._generate(prompt, skip_cache_insert=True))
        self.assertGreater(reader, inserted)

    def test_batch_normalizes_per_item(self):
        prompts = [_unique_prompt(3), _unique_prompt(3)]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompts,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "skip_cache_insert": [True, False],
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()), 2)

        batch_cached = [item["meta_info"]["cached_tokens"] for item in response.json()]
        self.assertEqual(
            self._cached_tokens(self._generate(prompts[0])), batch_cached[0]
        )
        self.assertGreater(
            self._cached_tokens(self._generate(prompts[1])), batch_cached[1]
        )

    def test_chat_completions_extra_body(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")
        messages = [{"role": "user", "content": _unique_prompt(6)}]

        def chat(**extra_body):
            return client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=4,
                temperature=0,
                extra_body=extra_body,
            )

        def cached(resp) -> int:
            details = resp.usage.prompt_tokens_details
            return details.cached_tokens if details else 0

        baseline = cached(chat(skip_cache_insert=True))
        second = cached(chat())
        self.assertEqual(second, baseline)
        self.assertGreater(cached(chat()), second)


class TestSkipCacheInsertRadixCache(SkipCacheInsertMixin, CustomTestCase):
    pass


class TestSkipCacheInsertHiCacheWriteThrough(SkipCacheInsertMixin, CustomTestCase):
    """write_through copies every inserted prefix to the host tier at once, so a
    leaked insert would also show up as a host hit."""

    other_args = [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "2",
        "--hicache-write-policy",
        "write_through",
    ]


if __name__ == "__main__":
    unittest.main()
