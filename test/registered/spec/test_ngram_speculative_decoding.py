import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.gsm8k_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TARGET_MODEL_NGRAM,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=230, suite="stage-b-test-large-1-gpu")

GSM_DATASET_PATH = None


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "NGRAM",
    "--speculative-num-draft-tokens",
    "16",
    "--mem-fraction-static",
    0.8,
]


class TestNgramSpeculativeDecodingBase(GSM8KMixin, CustomTestCase):
    model = DEFAULT_TARGET_MODEL_NGRAM
    base_url = DEFAULT_URL_FOR_TEST
    gsm8k_accuracy_thres = 0.79  # derived tests need to override this
    gsm8k_accept_length_thres = 1.8  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestNgramSpeculativeDecodingTriton(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestNgramSpeculativeDecodingFlashinfer(TestNgramSpeculativeDecodingBase):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


class TestNgramSpeculativeDecodingPaged(TestNgramSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--attention-backend",
            "flashinfer",
            "--page-size",
            "64",
        ]


if __name__ == "__main__":
    unittest.main()
