"""Per-request opt-out from prefix cache insertion (``skip_cache_insert``).

A request with ``skip_cache_insert=true`` may reuse cached prefixes but leaves
nothing behind: an identical follow-up request sees exactly the cached tokens
the opt-out request itself saw. The same prompt sent normally afterwards is
inserted, which proves the cache works and the earlier zero was the opt-out.

Two servers run side by side: plain radix cache, and HiCache with
``write_through`` so any leaked insert would also land in the host tier.
"""

import unittest
import uuid
from urllib.parse import urlparse

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

# Small enough that the long prompt below spans several chunks.
CHUNKED_PREFILL_SIZE = 64

COMMON_ARGS = [
    "--enable-cache-report",
    "--chunked-prefill-size",
    str(CHUNKED_PREFILL_SIZE),
    "--mem-fraction-static",
    "0.35",
]
SERVER_VARIANTS = {
    "radix": [],
    "hicache_write_through": [
        "--enable-hierarchical-cache",
        "--hicache-ratio",
        "2",
        "--hicache-write-policy",
        "write_through",
    ],
}


def _unique_prompt(num_sentences: int) -> str:
    # The fresh tag comes first so no two prompts share a leading token.
    tag = uuid.uuid4().hex
    return f"{tag} " + " ".join(
        f"Sentence {i} of this session talks about topic {i * 7 % 13}."
        for i in range(num_sentences)
    )


class TestSkipCacheInsert(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.processes = []
        cls.urls = {}
        for offset, (name, args) in enumerate(SERVER_VARIANTS.items()):
            url = f"http://{parsed.hostname}:{parsed.port + offset}"
            cls.processes.append(
                popen_launch_server(
                    cls.model,
                    url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[*COMMON_ARGS, *args],
                )
            )
            cls.urls[name] = url

    @classmethod
    def tearDownClass(cls):
        for process in getattr(cls, "processes", []):
            kill_process_tree(process.pid)

    def _generate(self, url, text, **extra):
        response = requests.post(
            url + "/generate",
            json={
                "text": text,
                "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                "return_cached_tokens_details": True,
                **extra,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _cached(self, result) -> tuple:
        """(total, host) cached tokens; host is 0 without HiCache."""
        meta = result["meta_info"]
        details = meta.get("cached_tokens_details") or {}
        return meta["cached_tokens"], details.get("host", 0)

    def _assert_opt_out_leaves_nothing(self, url, prompt):
        # Shared leading tokens (BOS) may already be cached, so compare against
        # what the opt-out request itself saw, on both tiers.
        baseline = self._cached(self._generate(url, prompt, skip_cache_insert=True))
        second = self._cached(self._generate(url, prompt))
        self.assertEqual(
            second, baseline, "opt-out must not leave a reusable prefix behind"
        )
        third_total, _ = self._cached(self._generate(url, prompt))
        self.assertGreater(third_total, second[0], "normal request was not inserted")

    def test_short_prompt(self):
        for name, url in self.urls.items():
            with self.subTest(server=name):
                self._assert_opt_out_leaves_nothing(url, _unique_prompt(2))

    def test_chunked_prompt(self):
        # Several chunked-prefill rounds; each intermediate round must keep the
        # request's own KV without publishing it.
        for name, url in self.urls.items():
            with self.subTest(server=name):
                self._assert_opt_out_leaves_nothing(url, _unique_prompt(40))

    def test_opt_out_still_reads_existing_prefix(self):
        for name, url in self.urls.items():
            with self.subTest(server=name):
                prompt = _unique_prompt(8)
                inserted, _ = self._cached(self._generate(url, prompt))
                reader, _ = self._cached(
                    self._generate(url, prompt, skip_cache_insert=True)
                )
                self.assertGreater(reader, inserted)

    def test_batch_normalizes_per_item(self):
        for name, url in self.urls.items():
            with self.subTest(server=name):
                prompts = [_unique_prompt(3), _unique_prompt(3)]
                response = requests.post(
                    url + "/generate",
                    json={
                        "text": prompts,
                        "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                        "skip_cache_insert": [True, False],
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(len(response.json()), 2)

                # The two prompts may share a leading token, so judge each
                # follow-up against its own prompt length instead: the opted-out
                # prompt is at most a stray-token hit, the other a full match.
                opted_out = self._generate(url, prompts[0])["meta_info"]
                inserted = self._generate(url, prompts[1])["meta_info"]
                self.assertLess(
                    opted_out["cached_tokens"], opted_out["prompt_tokens"] // 2
                )
                self.assertGreaterEqual(
                    inserted["cached_tokens"], inserted["prompt_tokens"] - 1
                )

    def test_chat_completions_extra_body(self):
        for name, url in self.urls.items():
            with self.subTest(server=name):
                client = openai.Client(api_key="EMPTY", base_url=f"{url}/v1")
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


if __name__ == "__main__":
    unittest.main()
