import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE,
    DEFAULT_TARGET_MODEL_STANDALONE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Standalone speculative decoding tests (FA3, Triton, FlashInfer backends)
register_cuda_ci(est_time=406, suite="stage-b-test-1-gpu-large")

GSM_DATASET_PATH = None

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    DEFAULT_DRAFT_MODEL_STANDALONE,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "2",
    "--speculative-num-draft-tokens",
    "7",
    "--mem-fraction-static",
    0.7,
]

# Default server arguments for V2 tests
DEFAULT_SERVER_ARGS_V2 = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--speculative-algorithm",
    "STANDALONE",
    "--speculative-draft-model-path",
    DEFAULT_DRAFT_MODEL_STANDALONE,
    "--speculative-num-steps",
    "4",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "5",
    "--mem-fraction-static",
    0.7,
]


class TestStandaloneSpeculativeDecodingBase(CustomTestCase):

    model = DEFAULT_TARGET_MODEL_STANDALONE
    draft_model = DEFAULT_DRAFT_MODEL_STANDALONE
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.69  # derived tests need to override this
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
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "score"
        self.assertGreaterEqual(metrics[metric_key], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestStandaloneV2SpeculativeDecodingBase(CustomTestCase):

    model = DEFAULT_TARGET_MODEL_STANDALONE
    draft_model = DEFAULT_DRAFT_MODEL_STANDALONE
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.69  # derived tests need to override this
    spec_decode_threshold = 3.6  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS_V2 + ["--attention-backend", "fa3"]

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        envs.SGLANG_ENABLE_SPEC_V2.set(True)  # Enable Speculative Decoding V2
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
        if "SGLANG_ENABLE_SPEC_V2" in os.environ:
            envs.SGLANG_ENABLE_SPEC_V2.set(False)

    def test_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=100,
            num_threads=128,
            num_shots=4,
            gsm8k_data_path=GSM_DATASET_PATH,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        # Use the appropriate metric key based on the test class
        metric_key = "score"
        self.assertGreaterEqual(metrics[metric_key], self.accuracy_threshold)

        server_info = requests.get(self.base_url + "/server_info")
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


class TestStandaloneV2SpeculativeDecodingTriton(
    TestStandaloneV2SpeculativeDecodingBase
):

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS_V2 + ["--attention-backend", "triton"]


class TestStandaloneV2SpeculativeDecodingFlashinfer(
    TestStandaloneV2SpeculativeDecodingBase
):
    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS_V2 + ["--attention-backend", "flashinfer"]


if __name__ == "__main__":
    unittest.main()
