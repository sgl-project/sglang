"""NGRAM speculative decoding on the intel_xpu attention backend."""

import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase
from sglang.test.test_utils import DEFAULT_TARGET_MODEL_EAGLE3

register_xpu_ci(est_time=600, suite="nightly-kernel-main-test-1-gpu-xpu", nightly=True)


class TestNgramPagedIntelXPU(NgramServerBase, GSM8KMixin):
    """NGRAM tree verify (bfs breadth 10, so topk > 1) at page_size 64 with XPU
    decode graphs."""

    # Llama-3.1-8B-Instruct is already on the XPU runner for the EAGLE3 tests.
    model = DEFAULT_TARGET_MODEL_EAGLE3
    attention_backend = "intel_xpu"
    extra_args = [
        "--page-size",
        "64",
        "--cuda-graph-backend-decode",
        "full",
        "--max-running-requests",
        "8",
        "--max-total-tokens",
        "16384",
    ]
    gsm8k_backend = "sgl_eval"
    # Llama-3.1-8B-Instruct generation_config sampling.
    gsm8k_temperature = 0.6
    gsm8k_top_p = 0.9
    gsm8k_max_tokens = 1024
    gsm8k_num_examples = 200
    gsm8k_num_threads = 8
    # Measured 0.835 and accept length 2.09 on B60.
    gsm8k_score_threshold = 0.75
    gsm8k_accept_length_thres = 1.8


if __name__ == "__main__":
    unittest.main()
