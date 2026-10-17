"""B200 CI: DeepSeek-V4.1-Flash FP4 with per-ratio candidate publishing.

With SGLANG_OPT_DSV4_PER_RATIO_CANDIDATES=1 the ratio-2 index sources consume
the candidate blocks published by their group's first index source instead of
running a full dense scan each. The gate: output quality (GSM8K), decode
correctness, and the DSPARK accept length must hold at parity with the default
path while the flag is on.

Registry: base-c-test-4-gpu-b200 (per-commit, 4x B200)
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=1200, stage="base-c", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
SERVER_LAUNCH_TIMEOUT = 3600

_PER_RATIO_ENV = {
    "SGLANG_OPT_DSV4_PER_RATIO_CANDIDATES": "1",
}


class TestDSV41FlashFP4PerRatioCandidatesB200(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Cookbook low-latency recipe (TP4+EP4, DSPARK) + per-ratio candidates."""

    # Measured on 4x B200 with this recipe, 200 questions: 0.865 / 0.880 with
    # the flag on and 0.905 with it off -- all within one standard error of each
    # other (binary score, SE ~ 0.023 at n=200), and consistent with the 0.911
    # the MI35x V4.1 eval measures over all 1319 questions. The 0.93 the
    # V4-Flash-0731 suites use is a different model's number. Threshold sits
    # ~0.015 under the lowest sample, matching how the MI35x test is calibrated.
    gsm8k_accuracy_thres = 0.85
    # DSPARK block size 5 accepts 3.36-3.40 here, flag on or off.
    accept_length_thres = 3.0
    bs_1_speed_thres = 150

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--ep-size",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--cuda-graph-max-bs-decode",
                "64",
                "--disable-flashinfer-autotune",
            ],
            env=_PER_RATIO_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)


if __name__ == "__main__":
    unittest.main(verbosity=3)
