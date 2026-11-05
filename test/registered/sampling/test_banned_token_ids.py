"""--sampling-banned-token-ids: the banned ids never appear in the output or
in the returned top logprobs. The EAGLE3 case is in
test_banned_token_ids_eagle3.py."""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=150, stage="extra-a", runner_config="1-gpu-small")

# Each prompt's greedy continuation starts with one of the banned words.
PROMPTS = [
    "The capital of France is",
    "One, two, three, four,",
]
BANNED_WORDS = [" Paris", " five"]


def banned_ids(model):
    tokenizer = get_tokenizer(model)
    ids = set()
    for word in BANNED_WORDS:
        ids.update(tokenizer.encode(word, add_special_tokens=False))
    return sorted(ids)


class TestBannedTokenIds(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.banned = banned_ids(cls.model)
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--sampling-banned-token-ids", *map(str, cls.banned)]
            + cls.extra_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, temperature):
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": 16,
                },
                "return_logprob": True,
                "top_logprobs_num": 20,
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()

    def _check(self, temperature):
        banned = set(self.banned)
        for text in PROMPTS:
            out = self._generate(text, temperature)
            meta = out["meta_info"]
            self.assertFalse(banned & set(out["output_ids"]), out)
            for row in meta["output_top_logprobs"]:
                self.assertFalse(banned & {token_id for _, token_id, _ in row}, row)

    def test_greedy(self):
        self._check(temperature=0)

    def test_sampling(self):
        self._check(temperature=1.0)


class TestBannedWordsAreTheGreedyContinuation(CustomTestCase):
    """Positive control: without the flag, each prompt's greedy continuation
    starts with a banned id, so the banned-id tests cannot pass vacuously. It
    needs its own server: the flag is server-wide, so the same server cannot
    show the unbanned continuation."""

    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.banned = banned_ids(cls.model)
        cls.process = popen_launch_server(
            cls.model, DEFAULT_URL_FOR_TEST, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_greedy_continuation_starts_with_a_banned_id(self):
        for text in PROMPTS:
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": text,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertIn(resp.json()["output_ids"][0], self.banned, text)


if __name__ == "__main__":
    unittest.main()
