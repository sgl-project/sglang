"""Integration test for explicit cache_id / load_cache_id KV-cache reuse.

Verifies that a request with ``cache_id`` persists its prefix in the radix cache
and a later request with ``load_cache_id`` reuses it, even when the two requests
are separated by an unrelated prompt that would otherwise evict or branch the
shared prefix.
"""

import platform
import unittest

import openai
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@unittest.skipIf(
    platform.system() == "Darwin",
    "sglang serve does not accept --device on macOS; run this test on Linux/CUDA",
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

        # First request saves the prefix under cache_id.
        first = self._chat(prompt, extra_body={"cache_id": cache_id})
        first_cached = int(first.usage.prompt_tokens_details.cached_tokens)
        # Some template tokens may already be cached; the important check is
        # that the second request with load_cache_id caches more/equal tokens.
        print(f"first request cached_tokens: {first_cached}")

        # An unrelated prompt with a different cache_id should not evict the
        # first entry because the radix tree namespaces by cache salt.
        self._chat("_ The speed of light is", extra_body={"cache_id": "other-id"})

        # Second request loads the previously saved prefix. The full prompt
        # should now be cached because the first request stored it under cache_id.
        second = self._chat(prompt, extra_body={"load_cache_id": cache_id})
        second_cached = int(second.usage.prompt_tokens_details.cached_tokens)
        second_prompt = int(second.usage.prompt_tokens)
        print(f"second request cached_tokens: {second_cached} / {second_prompt}")

        self.assertGreater(
            second_cached,
            first_cached,
            "load_cache_id should increase cached tokens compared to the save request",
        )
        self.assertEqual(
            second_cached,
            second_prompt,
            "load_cache_id should produce a full prefix cache hit",
        )

    def test_different_cache_ids_do_not_share(self):
        prompt = "_ The largest planet is"
        id_a = "cache-id-a"
        id_b = "cache-id-b"

        self._chat(prompt, extra_body={"cache_id": id_a})
        # Same prompt but a different namespace must not see a cache hit.
        response = self._chat(prompt, extra_body={"load_cache_id": id_b})
        cached = int(response.usage.prompt_tokens_details.cached_tokens)
        print(f"different cache_id cached_tokens: {cached}")
        # Template tokens may be shared, but the user prompt itself should not
        # be cached under the new id. We only assert no full prompt hit.
        self.assertLess(
            cached,
            int(response.usage.prompt_tokens),
            "different cache ids should not fully reuse each other's prefixes",
        )


if __name__ == "__main__":
    unittest.main()
