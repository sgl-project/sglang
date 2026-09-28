import unittest
from contextlib import ExitStack

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=300,
    stage="extra-b",
    runner_config="4-gpu-h100",
    disabled="no LiLiCorr draft checkpoint is published yet",
)

# The setUpClass skip covers `/rerun-test` and direct runs, which ignore `disabled=`.
UNPUBLISHED = "<unpublished>"
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODEL = UNPUBLISHED


class TestLiLiCorrServer(CustomTestCase, GSM8KMixin, SpecDecodingMixin):
    model = TARGET_MODEL
    draft_model = DRAFT_MODEL
    # The floors below were measured on fa3 only.
    attention_backend = "fa3"
    gsm8k_accuracy_thres = 0.85
    # 5.07 with the head against 4.17 for the head-free DFLASH control.
    gsm8k_accept_length_thres = 4.6
    # One prompt at 2048 tokens: 8.19 with the head against 7.31 for the control.
    accept_length_thres = 7.7
    bs_1_speed_thres = 600.0

    @classmethod
    def setUpClass(cls):
        if cls.draft_model == UNPUBLISHED:
            raise unittest.SkipTest(
                "no LiLiCorr draft checkpoint is published yet; set DRAFT_MODEL and drop "
                "the disabled= argument on register_cuda_ci in the same change"
            )
        cls.base_url = DEFAULT_URL_FOR_TEST
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--attention-backend",
                    cls.attention_backend,
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    cls.draft_model,
                    "--mem-fraction-static",
                    "0.7",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
