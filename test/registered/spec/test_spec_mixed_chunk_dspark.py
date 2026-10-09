"""DSPARK cell of test_spec_mixed_chunk.py, kept on H100: the Qwen3-14B target
does not fit a 32 GB card, and off SM100 it runs on FA3 (SM80/SM90 only)."""

import unittest

from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")

DSPARK_TARGET_MODEL = "Qwen/Qwen3-14B"
DSPARK_DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"


class TestDSparkMixedChunk(GSM8KMixin, CustomTestCase):
    model = DSPARK_TARGET_MODEL

    gsm8k_num_questions = 200
    gsm8k_accuracy_thres = 0.80
    gsm8k_accept_length_thres = 2.0

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "trtllm_mha" if is_sm100_supported() else "fa3",
                "--speculative-draft-attention-backend",
                "fa4" if is_sm100_supported() else "fa3",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DSPARK_DRAFT_MODEL,
                "--enable-mixed-chunk",
                "--chunked-prefill-size",
                "128",
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--cuda-graph-backend-prefill=disabled",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
