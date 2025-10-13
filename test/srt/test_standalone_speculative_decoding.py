import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_STANDALONE_SPECULATIVE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_STANDALONE_SPECULATIVE_TARGET_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    DEFAULT_STANDALONE_SPECULATIVE_DRAFT_MODEL_FOR_TEST,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "2",
    "--speculative-num-draft-tokens",
    "7",
    "--mem-fraction-static",
    0.7,
]


class TestStandaloneSpeculativeDecodingBase(CustomTestCase):

    model = DEFAULT_STANDALONE_SPECULATIVE_TARGET_MODEL_FOR_TEST
    draft_model = DEFAULT_STANDALONE_SPECULATIVE_DRAFT_MODEL_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.7  # derived tests need to override this
    spec_decode_threshold = 3.6  # derived spec decoding tests need to override this

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

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=4,
            num_questions=100,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
            data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestStandaloneSpeculativeDecodingTriton(TestStandaloneSpeculativeDecodingBase):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "triton"]


class TestStandaloneSpeculativeDecodingFlashinfer(
    TestStandaloneSpeculativeDecodingBase
):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--attention-backend", "flashinfer"]


if __name__ == "__main__":
    unittest.main()
