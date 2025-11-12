import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
    DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
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
    "--prefill-attention-backend",
    "fa3",
    "--decode-attention-backend",
    "flashinfer",
]


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class TestHybridAttnBackendBase(CustomTestCase):

    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.65  # derived tests need to override this
    speculative_decode = False
    spec_decode_threshold = 2.2  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        with (
            envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.override(False),
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
        ):
            if cls.speculative_decode:
                model = DEFAULT_EAGLE_TARGET_MODEL_FOR_TEST
            else:
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

        if self.speculative_decode:
            server_info = requests.get(self.base_url + "/get_server_info")
            avg_spec_accept_length = server_info.json()["internal_states"][0][
                "avg_spec_accept_length"
            ]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestHybridAttnBackendMLA(TestHybridAttnBackendBase):
    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS


class TestHybridAttnBackendTorchCompile(TestHybridAttnBackendBase):
    accuracy_threshold = 0.65

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + ["--enable-torch-compile"]


class TestHybridAttnBackendSpeculativeDecodingPrefillBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-attention-mode",
            "prefill",
        ]


class TestHybridAttnBackendSpeculativeDecodingDecodeBackend(TestHybridAttnBackendBase):
    speculative_decode = True
    # This eagle test uses a very small model, so the accuracy is low.
    accuracy_threshold = 0.2

    @classmethod
    def get_server_args(cls):
        return DEFAULT_SERVER_ARGS + [
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_EAGLE_DRAFT_MODEL_FOR_TEST,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "2",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-attention-mode",
            "decode",
        ]


if __name__ == "__main__":
    unittest.main()
