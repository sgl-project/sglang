"""Mixed chunk prefill x speculative decoding, overlap scheduler.

One cell per supported algorithm (EAGLE3, DFLASH, DSPARK). Inside a mixed
step every running request degrades to a 1-token extend of its pending
bonus token and drafting resumes the next decode step; under overlap the
tail state is late-bound at forward entry. Regression guards for the
bring-up failure modes: tail rows dropped from attention metadata (the
spec seq_lens convention zeroed the tail's qo len - a hard crash on
flashinfer, silent kv-span truncation elsewhere), unwritten relay rows
read as tail inputs, and stale schedule-time tail state under overlap.
Chunked prefill is set small so eval prompts span multiple chunks and
mixing actually engages.
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import is_sm100_supported, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecCorrectnessKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_DFLASH,
    DEFAULT_TARGET_MODEL_DFLASH,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=800, stage="base-b", runner_config="1-gpu-large")


class TestEagle3MixedChunk(
    Eagle3Base,
    SpecCorrectnessKit,
    SpecAccuracyKit,
):
    disable_overlap = False
    # Small chunks so eval prompts span several of them and mixing engages
    # (the Eagle3Base preset raises the fixture default to 1024).
    chunked_prefill_size = 128
    extra_args = ("--enable-mixed-chunk",)


class TestDFlashMixedChunk(GSM8KMixin, CustomTestCase):
    model = DEFAULT_TARGET_MODEL_DFLASH

    gsm8k_num_questions = 200
    # Observed 0.755-0.78 across local runs; accept length is the tighter guard.
    gsm8k_accuracy_thres = 0.70
    gsm8k_accept_length_thres = 2.8

    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # The dflash draft config derives a shorter context than the target.
        with envs.SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--speculative-algorithm",
                    "DFLASH",
                    "--speculative-draft-model-path",
                    DEFAULT_DRAFT_MODEL_DFLASH,
                    "--enable-mixed-chunk",
                    "--chunked-prefill-size",
                    "128",
                    "--mem-fraction-static",
                    "0.7",
                ],
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


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
                "--disable-piecewise-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
