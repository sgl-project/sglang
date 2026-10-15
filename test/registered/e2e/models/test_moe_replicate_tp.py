"""E2E accuracy test for --moe-replicate-tp (replicated MoE with prefill
sequence parallelism).

Launches Qwen3.5 MoE with tp=2 and the flag on, then checks GSM8K accuracy.
A prefill keeps the residual on each rank's token slice and runs the whole
MoE there, while attention and linear attention keep their TP heads; only
the MoE summation order changes (as under expert parallelism), so a correct
implementation matches plain TP within floating-point-reordering noise.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Every rank holds the full MoE (~68 GB of BF16 weights for this model).
register_cuda_ci(est_time=300, stage="extra-b", runner_config="4-gpu-b200")

MOE_REPLICATE_TP_MODEL = "Qwen/Qwen3.5-35B-A3B"


class TestMoeReplicateTpAccuracy(CustomTestCase, GSM8KMixin):
    gsm8k_score_threshold = 0.83
    gsm8k_num_examples = 200

    @classmethod
    def setUpClass(cls):
        cls.model = MOE_REPLICATE_TP_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp", "2", "--moe-replicate-tp"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=3)
