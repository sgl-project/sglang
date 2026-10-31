import unittest

from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

register_xpu_ci(est_time=200, suite="stage-b-test-1-gpu-xpu")


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestNgramSpeculativeDecodingIntelXpu(NgramServerBase, GSM8KMixin):
    attention_backend = "intel_xpu"
    extra_args = [
        # intel_xpu forces page_size to 64/128, and NGRAM rejects tree drafts (topk > 1)
        # on paged backends other than flashinfer; BFS breadth 1 is the accepted config.
        "--speculative-ngram-max-bfs-breadth",
        "1",
        # Above ~32 running requests torch's cache starves the Level Zero driver (error 40).
        "--max-running-requests",
        "16",
    ]
    gsm8k_backend = "sgl_eval"
    # Recommended sampling from the model's generation_config.json.
    gsm8k_temperature = 0.7
    gsm8k_top_p = 0.8
    gsm8k_score_threshold = 0.68
    gsm8k_accept_length_thres = 1.6


if __name__ == "__main__":
    unittest.main()
