"""KL regression tests for Mamba prefix-cache correctness.

Keep each scenario in its own class so server configurations and cache-pressure
requirements remain explicit.
"""

import unittest

import numpy as np
import requests
from transformers import AutoConfig, AutoTokenizer

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-large")


class MambaCacheKLMixin:
    ignore_eos = False

    def _generate(self, input_ids, max_new_tokens, *, score=False):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": self.ignore_eos,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        if score:
            payload["token_ids_logprob"] = self.vocab_ids
        response = requests.post(self.base_url + "/generate", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        self.assertEqual(result["meta_info"]["completion_tokens"], max_new_tokens)
        return result

    def _flush_cache(self):
        response = requests.post(self.base_url + "/flush_cache", timeout=30)
        response.raise_for_status()

    def _distribution(self, result):
        rows = result["meta_info"]["output_token_ids_logprobs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual([row[1] for row in rows[0]], self.vocab_ids)
        logprobs = np.asarray([row[0] for row in rows[0]], dtype=np.float64)
        self.assertTrue(np.isfinite(logprobs).all())
        self.assertAlmostEqual(float(np.exp(logprobs).sum()), 1.0, places=4)
        # Remove log-softmax rounding error before computing the exact KL.
        return logprobs - np.logaddexp.reduce(logprobs)


class TestMambaLazyInplaceKL(MambaCacheKLMixin, DefaultServerBase):
    """Guard lazy Mamba checkpoint reuse when the optional track slot is exhausted.

    Keep overlap scheduling and decode CUDA graphs enabled. A four-slot pool holds
    two cached prefixes, the request's live state, and its keep checkpoint. The
    first decode must track in place; delayed prefill processing must not publish
    that buffer under the old checkpoint length.

    Compare full-vocabulary next-token distributions at the SAME 255-token prompt,
    before and after a 32-token decode seeds the cache. Comparing later free-running
    tokens would confound cache corruption with different generated histories.

    GB300 / Qwen3.5-0.8B: reverting the checkpoint invalidation gives KL=0.649448
    and a stale 192-token cache hit; with the fix KL=0 and the safe hit is 128 tokens.
    """

    ignore_eos = True
    model = "Qwen/Qwen3.5-0.8B"
    server_env = {
        "SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK": "0",
        "SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL": "0",
    }
    other_args = [
        "--skip-tokenizer-init",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "128",
        "--max-mamba-cache-size",
        "4",
        "--max-running-requests",
        "1",
        "--max-total-tokens",
        "8192",
        "--mem-fraction-static",
        "0.3",
        "--cuda-graph-max-bs-decode",
        "1",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--chunked-prefill-size",
        "2048",
        "--attention-backend",
        "triton",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
        "--random-seed",
        "42",
    ]

    def _warm_prefixes(self):
        self._flush_cache()
        self._generate([44] * 128, 1)
        self._generate([42] * 128, 1)

    def test_lazy_donation_with_four_slots_and_one_request(self):
        """A prefix-hit request must finish without a fifth Mamba slot under overlap."""
        self._flush_cache()
        self._generate([42] * 128, 1)

        # One cached prefix, live state, and keep checkpoint occupy three slots;
        # the boundary decode takes the fourth before prefill donation runs.
        result = self._generate([42] * 128 + [43] * 127, 32)
        self.assertEqual(result["meta_info"]["cached_tokens"], 128)
        logprobs = result["meta_info"]["output_token_logprobs"]
        self.assertEqual(len(logprobs), 32)
        self.assertTrue(np.isfinite([row[0] for row in logprobs]).all())

    def test_reuse_distribution_after_inplace_tracking(self):
        config = AutoConfig.from_pretrained(self.model)
        self.vocab_ids = list(range(config.get_text_config().vocab_size))
        prompt = [42] * 128 + [43] * 127

        # A single output token does not launch the boundary-crossing decode.
        self._warm_prefixes()
        reference = self._generate(prompt, 1, score=True)
        self.assertEqual(reference["meta_info"]["cached_tokens"], 128)
        reference_logprobs = self._distribution(reference)

        self._warm_prefixes()
        seed = self._generate(prompt, 32)
        self.assertEqual(seed["meta_info"]["cached_tokens"], 128)
        reused = self._generate(prompt, 1, score=True)
        self.assertGreaterEqual(reused["meta_info"]["cached_tokens"], 128)
        reused_logprobs = self._distribution(reused)

        kl = float(
            np.sum(np.exp(reference_logprobs) * (reference_logprobs - reused_logprobs))
        )
        print(
            f"lazy in-place checkpoint KL(reference || reuse)={kl:.9g}, "
            f"cached_tokens={reused['meta_info']['cached_tokens']}",
            flush=True,
        )
        self.assertGreaterEqual(kl, -1e-12)
        # Leave room for numerical noise while staying far below the broken run.
        self.assertLess(kl, 1e-4, "Cache reuse changed the next-token distribution")


class TestMambaSpecTruncationKL(MambaCacheKLMixin, DefaultServerBase):
    """NEXTN can advance the no_buffer state past max_new_tokens. Compare warm
    and cold distributions for identical token IDs to detect a misaligned key.
    Different prefill splits introduce numerical noise, so KL need not be zero.
    """

    model = "Qwen/Qwen3.5-0.8B"
    revision = "2fc06364715b967f1860aea9cf38778875588b17"
    other_args = [
        "--revision",
        revision,
        "--mamba-radix-cache-strategy",
        "no_buffer",
        "--page-size",
        "1",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.25",
        "--max-total-tokens",
        "8192",
        "--max-running-requests",
        "8",
        "--max-mamba-cache-size",
        "128",
        "--attention-backend",
        "triton",
        "--linear-attn-decode-backend",
        "triton",
        "--linear-attn-prefill-backend",
        "triton",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    def test_reuse_distribution_after_length_truncation(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model, revision=self.revision)
        config = AutoConfig.from_pretrained(self.model, revision=self.revision)
        self.vocab_ids = list(range(config.get_text_config().vocab_size))
        newline = tokenizer.encode("\n", add_special_tokens=False)
        cases = (
            (
                "count",
                "Continue the sequence with numbers separated by spaces: 1 2 3 4 5 6 7 8 9 10",
                9,
            ),
            ("story", "Write a story about a cat who likes fish.", 5),
        )
        for name, text, limit in cases:
            with self.subTest(case=name):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    return_dict=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                self._flush_cache()
                seed = self._generate(prompt, limit)
                self.assertGreater(seed["meta_info"]["spec_verify_ct"], 0)
                self.assertGreater(seed["meta_info"]["spec_num_correct_drafts"], 0)
                self.assertEqual(seed["meta_info"]["finish_reason"]["type"], "length")
                probe = prompt + seed["output_ids"] + newline
                warm = self._generate(probe, 1, score=True)
                self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
                self._flush_cache()
                cold = self._generate(probe, 1, score=True)
                self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                cold_logprobs = self._distribution(cold)
                warm_logprobs = self._distribution(warm)
                kl = float(
                    np.sum(np.exp(cold_logprobs) * (cold_logprobs - warm_logprobs))
                )
                print(
                    f"no_buffer NEXTN {name} KL(cold || warm)={kl:.9g}, "
                    f"cached_tokens={warm['meta_info']['cached_tokens']}, "
                    f"prompt_tokens={len(prompt)}",
                    flush=True,
                )
                self.assertGreaterEqual(kl, -1e-12)
                self.assertLess(
                    kl, 5e-3, "Truncated speculative state corrupted cache reuse"
                )


if __name__ == "__main__":
    unittest.main()
