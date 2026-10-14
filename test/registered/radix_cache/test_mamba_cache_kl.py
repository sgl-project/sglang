"""KL regression tests for Mamba prefix-cache correctness.

Keep each scenario in its own class so server configurations and cache-pressure
requirements remain explicit.
"""

import unittest

import numpy as np
import requests
from transformers import AutoConfig

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


class TestMambaLazyInplaceKL(DefaultServerBase):
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

    def _generate(self, input_ids, max_new_tokens, *, score=False):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
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

    def _warm_prefixes(self):
        response = requests.post(self.base_url + "/flush_cache", timeout=30)
        response.raise_for_status()
        self._generate([44] * 128, 1)
        self._generate([42] * 128, 1)

    def _distribution(self, result):
        rows = result["meta_info"]["output_token_ids_logprobs"]
        self.assertEqual(len(rows), 1)
        self.assertEqual([row[1] for row in rows[0]], self.vocab_ids)
        logprobs = np.asarray([row[0] for row in rows[0]], dtype=np.float64)
        self.assertTrue(np.isfinite(logprobs).all())
        self.assertAlmostEqual(float(np.exp(logprobs).sum()), 1.0, places=4)
        # Remove log-softmax rounding error before computing the exact KL.
        return logprobs - np.logaddexp.reduce(logprobs)

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


if __name__ == "__main__":
    unittest.main()
