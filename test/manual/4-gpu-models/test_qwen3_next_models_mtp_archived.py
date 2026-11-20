"""Archived test classes split out of test/registered/e2e/models/test_qwen3_next_models_mtp.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/4-gpu-models/test_qwen3_next_models_mtp_archived.py`.
"""

import os
import unittest

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

QWEN3_NEXT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestQwen3NextMTP(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0025
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-radix-cache-strategy",
        "no_buffer",
        "--disable-radix-cache",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    # Plain topk=1 + extra_buffer baseline; CI covers topk == 1 with
    # TestQwen3NextMTPLazyV2 and extra_buffer with TestQwen3NextMTPTopk.
    model = QWEN3_NEXT_MODEL
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
    ]


class TestQwen3NextMTPLazyPoolFull(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    # Small track interval (= draft tokens) maximizes boundary crossings;
    # the pool is reported full, so each crossing evicts cached states
    # before allocating its tracking slot.
    model = QWEN3_NEXT_MODEL
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-radix-cache-strategy",
        "extra_buffer_lazy",
        "--mamba-track-interval",
        "4",
    ]

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_TEST_MAMBA_LAZY_POOL_FULL"] = "1"
        os.environ["SGLANG_TEST_SKIP_CACHE_HIT_ASSERT"] = "1"
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        os.environ.pop("SGLANG_TEST_MAMBA_LAZY_POOL_FULL", None)
        os.environ.pop("SGLANG_TEST_SKIP_CACHE_HIT_ASSERT", None)


if __name__ == "__main__":
    unittest.main()
