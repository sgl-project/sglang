"""Regression tests for input_embeds shape-mismatch bugs.

Covers two bugs with the same crash signature
(RuntimeError: shape mismatch in set_kv_buffer) but opposite polarity:

- Chunked prefill truncation (#20376): PrefillAdder truncates fill_ids and
  extend_input_len on chunk overflow but not input_embeds, so the full array
  flows through while out_cache_loc is sized for the truncated length.
  Polarity: cache_k > loc.

- Retraction with output_ids (#14110): after retraction, fill_ids includes
  accumulated output_ids but input_embeds only covers origin_input_ids.
  Polarity: cache_k < loc.
"""

import unittest

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=45, suite="stage-b-test-small-1-gpu")

CHUNKED_PREFILL_SIZE = 256

# Shared reference model — loaded once per process, not per test class.
_MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
_tokenizer = None
_ref_model = None


def _load_ref():
    global _tokenizer, _ref_model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL)
        _ref_model = AutoModelForCausalLM.from_pretrained(_MODEL)


def _embeds_for(text: str) -> list[list[float]]:
    _load_ref()
    ids = _tokenizer(text, return_tensors="pt")["input_ids"]
    embeds = _ref_model.get_input_embeddings()(ids)
    return embeds.squeeze(0).to(torch.float32).tolist()


def _generate(base_url, input_embeds, max_new_tokens, ignore_eos=False, timeout=120):
    resp = requests.post(
        f"{base_url}/generate",
        json={
            "input_embeds": input_embeds,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos,
            },
        },
        timeout=timeout,
    )
    return resp


class TestInputEmbedsChunkedAndRetract(CustomTestCase):
    """Single server launch covering both bugs.

    Both tests require --disable-radix-cache (for input_embeds). The chunked
    prefill test needs a small --chunked-prefill-size. The retraction test
    uses SGLANG_TEST_RETRACT to deterministically force retraction every few
    scheduler iterations regardless of KV pressure.
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # SGLANG_TEST_RETRACT forces retraction periodically; this is
        # deterministic and doesn't require guessing KV budgets.
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                _MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--disable-radix-cache",
                    "--chunked-prefill-size",
                    str(CHUNKED_PREFILL_SIZE),
                    "--cuda-graph-max-bs",
                    "4",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _assert_server_alive(self):
        self.assertIsNone(self.process.poll(), "server process crashed")

    def test_chunked_prefill_truncation_and_continuation(self):
        """Regression test for #20376.

        A single request longer than chunked_prefill_size deterministically
        exercises both (a) first-chunk truncation and (b) chunk continuation,
        without any concurrent-timing dependency. Pre-fix this crashes in
        set_kv_buffer on both chunks.
        """
        # ~80 tokens each repetition; 6 repetitions exceeds CHUNKED_PREFILL_SIZE
        # comfortably. Token count is model-dependent so assert it.
        text = "The quick brown fox jumps over the lazy dog. " * 40
        embeds = _embeds_for(text)
        self.assertGreater(
            len(embeds),
            CHUNKED_PREFILL_SIZE,
            f"prompt must exceed chunked_prefill_size={CHUNKED_PREFILL_SIZE} "
            f"to trigger chunking; got {len(embeds)} tokens",
        )

        resp = _generate(self.base_url, embeds, max_new_tokens=8)
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertIn("text", body)
        self.assertIsInstance(body["text"], str)
        self._assert_server_alive()

    def test_chunked_prefill_batch_truncation(self):
        """Regression test for #20376 — multi-request batch case.

        A batch POST with total tokens > chunked_prefill_size goes through a
        single ZMQ send, so all requests land in the same scheduler iteration
        and the PrefillAdder is forced to truncate at least one. This matches
        the original thundering-herd trigger without HTTP timing races.
        """
        text = "The quick brown fox jumps over the lazy dog. " * 8
        embeds = _embeds_for(text)
        seq_len = len(embeds)

        # Enough batched requests to overflow the chunk budget.
        n = max(4, CHUNKED_PREFILL_SIZE // seq_len + 2)
        self.assertGreater(n * seq_len, CHUNKED_PREFILL_SIZE)

        resp = _generate(self.base_url, [embeds] * n, max_new_tokens=8)
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        results = resp.json()
        self.assertEqual(len(results), n)
        for r in results:
            self.assertIn("text", r)
        self._assert_server_alive()

    def test_retraction_with_output_ids(self):
        """Regression test for #14110.

        SGLANG_TEST_RETRACT forces retraction every few scheduler iterations.
        Combined with ignore_eos and a reasonable max_new_tokens, at least one
        request is retracted mid-decode with non-empty output_ids, then
        re-prefilled. Pre-#14110 this crashes (cache_k < loc) because fill_ids
        includes output_ids but input_embeds does not.
        """
        text = "The quick brown fox jumps over the lazy dog. " * 4
        embeds = _embeds_for(text)

        # Batch of requests with enough decode steps that SGLANG_TEST_RETRACT
        # (interval=3 by default) fires mid-decode.
        n = 4
        resp = _generate(
            self.base_url,
            [embeds] * n,
            max_new_tokens=32,
            ignore_eos=True,
        )
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        results = resp.json()
        self.assertEqual(len(results), n)
        for r in results:
            self.assertIn("text", r)
        self._assert_server_alive()


if __name__ == "__main__":
    unittest.main()
