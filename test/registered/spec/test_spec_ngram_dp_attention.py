import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase
from sglang.test.test_utils import DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN

# NGRAM speculative decoding together with DP attention (tp=2, dp=2 -> 2 GPUs).
# Reuse the DP-attention-capable target model from the EAGLE DP test; the default
# NGRAM CI target (Qwen2.5-Coder-7B, qwen2.py) has an unrelated pre-existing DP
# head-split issue, so it is not used here.
register_cuda_ci(est_time=600, stage="base-b", runner_config="2-gpu-large")


class TestNgramDPAttention(NgramServerBase, GSM8KMixin):
    model = DEFAULT_TARGET_MODEL_EAGLE_DP_ATTN
    attention_backend = "triton"
    # ngram acceptance on GSM8K is modest (~1.8 observed); use a floor with margin
    # below that to confirm speculation stays active without CI flakiness.
    gsm8k_accept_length_thres = 1.5
    extra_args = [
        "--tp-size",
        "2",
        "--dp-size",
        "2",
        "--enable-dp-attention",
    ]


if __name__ == "__main__":
    unittest.main()
