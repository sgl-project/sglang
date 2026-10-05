"""XPU decode-graph regression test for the fused decode-metadata path.

The intel_xpu backend fills ``cache_seqlens_int32`` / ``cu_seqlens_k`` /
``page_table`` (and ``swa_page_table`` for hybrid sliding-window models) for
the decode graph with the shared fused Triton kernel
(``normal_decode_set_metadata``). The kernel rewrites only each request's
live page prefix and leaves the tail columns untouched.

Graph metadata buffers are indexed by batch position. With
``--max-running-requests`` equal to the graph batch size, a batch of long
requests followed by a batch of short ones reuses the same ``page_table``
rows, so the short requests decode with stale long-request page ids past
their live prefix. Attention bounds its reads by ``cache_seqlens``, so the
outputs must match the same prompts run alone.

Deterministic inference is enabled so a prompt decoded at batch size 1 and
at batch size BS yields identical tokens; without it, bf16 near-ties can
flip across batch sizes on any backend.
"""

import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_xpu_ci(est_time=600, suite="stage-b-test-1-gpu-xpu")

# Graph batch size == max running requests, so every batch of BS requests
# lands on page_table rows 0..BS-1 and rows are reused across batches.
BS = 8
MAX_NEW_TOKENS = 32
# Port 30000 may be published by another container on shared hosts.
BASE_URL = "http://127.0.0.1:30010"


def _launch_args(extra=()):
    return [
        "--device",
        "xpu",
        "--attention-backend",
        "intel_xpu",
        "--page-size",
        "64",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-max-bs-decode",
        str(BS),
        "--max-running-requests",
        str(BS),
        "--mem-fraction-static",
        "0.75",
        "--enable-deterministic-inference",
        *extra,
    ]


class _FusedDecodeMetadataParityMixin:
    """Stale-tail parity checks; mix into a ``DefaultServerBase`` subclass.

    Subclasses may override ``wrap`` to apply a chat template so instruct
    models produce non-degenerate (and therefore informative) output.
    """

    @staticmethod
    def wrap(prompt: str) -> str:
        return prompt

    def _short_prompts(self):
        return [
            self.wrap(f"What is {i} plus {i + 3}? Answer briefly.") for i in range(BS)
        ]

    def _long_prompts(self):
        # ~1200 tokens each: many live pages per row at page_size 64, and well
        # past a 512-token sliding window for hybrid-SWA models.
        filler = "The quick brown fox jumps over the lazy dog. " * 120
        return [
            self.wrap(filler + f"Summarize the text above in {i + 1} words.")
            for i in range(BS)
        ]

    def _mixed_prompts(self):
        long_, short = self._long_prompts(), self._short_prompts()
        return [long_[i] if i % 2 == 0 else short[i] for i in range(BS)]

    def _boundary_prompts(self):
        # Natural prompts whose token lengths spread across 40..140 so that,
        # with MAX_NEW_TOKENS new tokens, the live page count (cdiv(len, 64))
        # ticks over at different decode steps on different rows. Natural
        # text (not synthetic token ids) keeps the model's distribution
        # peaked; on near-flat distributions bf16 rounding can flip tokens
        # across batch sizes even in deterministic mode.
        return [
            self.wrap(
                ("Alice bought " + "three red apples and two green pears, " * k)
                + "then went home. How many fruits did she buy? Answer briefly."
            )
            for k in (2, 3, 4, 5, 7, 9, 11, 13)
        ]

    def _generate(self, prompts):
        """Greedy decode; return (output token ids, prompt token count) per prompt."""
        payload = {
            "text": prompts,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": MAX_NEW_TOKENS},
            "return_logprob": True,
        }
        resp = requests.post(self.base_url + "/generate", json=payload, timeout=600)
        self.assertEqual(resp.status_code, 200)
        return [
            (
                [tok for _, tok, *_ in r["meta_info"]["output_token_logprobs"]],
                r["meta_info"]["prompt_tokens"],
            )
            for r in resp.json()
        ]

    def _gen_ids(self, prompts):
        return [ids for ids, _ in self._generate(prompts)]

    def _flush(self):
        resp = requests.post(self.base_url + "/flush_cache", timeout=60)
        self.assertEqual(resp.status_code, 200)

    def _check_after_long(self, prompts):
        """Reference each prompt alone, fill all rows with long requests,
        then decode the prompts as one batch on the reused rows.
        Returns the prompt token counts (from the reference run)."""
        ref_pairs = [self._generate([p])[0] for p in prompts]
        ref = [ids for ids, _ in ref_pairs]
        self._flush()
        self._gen_ids(self._long_prompts())
        out = self._gen_ids(prompts)
        self.assertEqual(out, ref)
        return [n for _, n in ref_pairs]

    def test_short_after_long_matches_single(self):
        # Rows go from ~19 live pages to 1; tails hold stale page ids.
        self._check_after_long(self._short_prompts())

    def test_mixed_lengths_match_single(self):
        # Rows with ~19 live pages next to rows with 1, in one graph launch.
        self._check_after_long(self._mixed_prompts())

    def test_page_boundary_lengths_match_single(self):
        # cdiv(seq_len, 64) changes mid-decode on different steps per row.
        prompt_lens = self._check_after_long(self._boundary_prompts())
        # Self-check that the prompts really straddle page boundaries: a row
        # crosses one during decode iff (len % 64) + MAX_NEW_TOKENS >= 64.
        crossing = [n for n in prompt_lens if (n % 64) + MAX_NEW_TOKENS >= 64]
        self.assertGreaterEqual(
            len(crossing), 3, f"too few rows cross a page boundary: {prompt_lens}"
        )


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU not available")
class TestXPUFusedDecodeMetadataDense(
    _FusedDecodeMetadataParityMixin, DefaultServerBase
):
    """Dense model: ``page_table`` only."""

    model = "Qwen/Qwen2.5-1.5B-Instruct"
    base_url = BASE_URL
    other_args = _launch_args()

    @staticmethod
    def wrap(prompt: str) -> str:
        return f"Q: {prompt}\nA:"


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU not available")
class TestXPUFusedDecodeMetadataSWA(_FusedDecodeMetadataParityMixin, DefaultServerBase):
    """Hybrid sliding-window model (sliding_window=512, 1 full layer in 5):
    the fused kernel writes ``swa_page_table`` as well."""

    model = "google/gemma-4-E2B-it"
    base_url = BASE_URL
    other_args = _launch_args()

    @staticmethod
    def wrap(prompt: str) -> str:
        # Gemma-4 chat turn format; raw prompts make the instruct model loop.
        return f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"


if __name__ == "__main__":
    unittest.main()
