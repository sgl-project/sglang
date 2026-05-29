"""GSM8K accuracy test for Ling-2.6-flash (BailingMoELinearForCausalLM).

Guards the hybrid linear / full attention dispatcher: Ling-2.5/2.6
has 32 layers with `layer_group_size=8`, so layers {7, 15, 23, 31}
are full attention (MLA) and the rest are linear (Lightning seg_la).
Runs on the 8-GPU H200 runner with TP=4.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=600, stage="base-c", runner_config="8-gpu-h200")


class TestLing26Flash(GSM8KMixin, DefaultServerBase):
    model = "inclusionAI/Ling-2.6-flash"

    # Native 128K context (no YaRN) — avoids the
    # SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN env-var dance and keeps the
    # smoke test focused on the dispatcher / hybrid-attention path.
    other_args = [
        "--tp-size",
        "4",
        "--trust-remote-code",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mem-fraction-static",
        "0.75",
        "--max-running-requests",
        "64",
        "--max-mamba-cache-size",
        "256",
        # MTP path also exercises the dispatcher (draft + target verify),
        # so keep it on to maximize coverage.
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]

    # Observed 0.825 on H200 TP=4 + NEXTN MTP with default 200-question GSM8K
    # (the model card's 0.96 is from full 1319-question runs of the 1T model).
    gsm8k_accuracy_thres = 0.825


if __name__ == "__main__":
    unittest.main(verbosity=3)
