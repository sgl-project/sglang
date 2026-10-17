"""Integration test for explicit cache_id / load_cache_id KV-cache reuse.

Verifies that a request with ``cache_id`` keeps its prefix in the radix cache
under that id and a later request with ``load_cache_id`` reuses it, even when
the two requests are separated by an unrelated prompt that would otherwise
branch the shared prefix.
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestCacheIdReuse(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=[
                "--chunked-prefill-size=40",
                "--attention-backend=triton",
                "--enable-cache-report",
            ],
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _chat(self, message, extra_body=None):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}],
            temperature=0,
            max_tokens=8,
            extra_body=extra_body,
        )

    def test_cache_id_and_load_cache_id_hit(self):
        prompt = "_ The capital of France is"
        cache_id = "test-cache-id-france"

        # The first request fills a fresh namespace, so nothing is cached yet.
        first = self._chat(prompt, extra_body={"cache_id": cache_id})
        first_cached = int(first.usage.prompt_tokens_details.cached_tokens)
        print(f"first request cached_tokens: {first_cached}")
        self.assertEqual(first_cached, 0)

        # An unrelated prompt under another id must not disturb the first entry:
        # the radix tree namespaces by cache salt.
        self._chat("_ The speed of light is", extra_body={"cache_id": "other-id"})

        # The load request matches the whole saved prefix. The last prompt token
        # is always recomputed for its logits, hence prompt_tokens - 1.
        second = self._chat(prompt, extra_body={"load_cache_id": cache_id})
        second_cached = int(second.usage.prompt_tokens_details.cached_tokens)
        second_prompt = int(second.usage.prompt_tokens)
        print(f"second request cached_tokens: {second_cached} / {second_prompt}")
        self.assertEqual(
            second_cached,
            second_prompt - 1,
            "load_cache_id should produce a full prefix cache hit",
        )

    def test_different_cache_ids_do_not_share(self):
        prompt = "_ The largest planet is"

        self._chat(prompt, extra_body={"cache_id": "cache-id-a"})
        # Same prompt under a different id: nothing is shared, not even the
        # chat-template tokens, because namespaces are disjoint.
        response = self._chat(prompt, extra_body={"load_cache_id": "cache-id-b"})
        cached = int(response.usage.prompt_tokens_details.cached_tokens)
        print(f"different cache_id cached_tokens: {cached}")
        self.assertEqual(
            cached, 0, "different cache ids should not reuse each other's prefixes"
        )

    def test_plain_requests_do_not_see_explicit_namespaces(self):
        prompt = "_ The tallest mountain is"

        self._chat(prompt, extra_body={"cache_id": "cache-id-c"})
        # A request without any id looks in the default namespace only.
        response = self._chat(prompt)
        cached = int(response.usage.prompt_tokens_details.cached_tokens)
        print(f"plain request cached_tokens: {cached}")
        self.assertLess(cached, int(response.usage.prompt_tokens) - 1)


if __name__ == "__main__":
    unittest.main()
