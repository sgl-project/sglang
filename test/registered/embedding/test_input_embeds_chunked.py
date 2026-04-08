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

register_cuda_ci(est_time=45, suite="stage-b-test-1-gpu-small")

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

    def test_retraction_aborts_input_embeds(self):
        """input_embeds requests are aborted (not retracted) under KV pressure.

        Retracting an input_embeds request cannot preserve decode progress:
        re-prefill would need to embed the generated output_ids via the model's
        embed_tokens (unreachable from the scheduler), and the previous approach
        (#14110, clearing output_ids) left 10+ per-step accumulators stale —
        including cross-process detokenizer/tokenizer_manager state. The result
        was silent corruption: len(output_token_logprobs) != len(output_ids)
        for RL workloads. Aborting is the only correct outcome.

        SGLANG_TEST_RETRACT forces the retract path every few scheduler
        iterations; we verify that at least one request gets a clean 503 and
        the server survives.
        """
        text = "The quick brown fox jumps over the lazy dog. " * 4
        embeds = _embeds_for(text)

        n = 4
        resp = _generate(
            self.base_url,
            [embeds] * n,
            max_new_tokens=32,
            ignore_eos=True,
        )
        # Batch endpoint returns 200 with per-request finish_reason. At least
        # one request should have been aborted via the retract path.
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        results = resp.json()
        self.assertEqual(len(results), n)
        aborted = [
            r
            for r in results
            if r.get("meta_info", {}).get("finish_reason", {}).get("type") == "abort"
        ]
        self.assertGreater(
            len(aborted),
            0,
            f"expected at least one abort via SGLANG_TEST_RETRACT; "
            f"finish_reasons: {[r.get('meta_info', {}).get('finish_reason') for r in results]}",
        )
        for r in aborted:
            self.assertIn(
                "input_embeds",
                r["meta_info"]["finish_reason"].get("message", ""),
            )
        self._assert_server_alive()


if __name__ == "__main__":
    unittest.main()
