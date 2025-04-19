import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.srt.utils import get_device_sm, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST  # llama 3.1 8B
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,  # eagle3 llama 3.1 8B
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_LOCAL_ATTENTION,  # llama 4 scout 17B 16E
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MLA  # deepseek v3 test
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,  # deepseek v3 test nextN
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

GSM_DATASET_PATH = None

# In case of some machine lack internet connection, we can set OFFLINE_MODE to True.
OFFLINE_MODE = True

# Change the path below when OFFLINE_MODE is True.
OFFLINE_PATH_DICT = {
    DEFAULT_MODEL_NAME_FOR_TEST: "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693",
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3: "/shared/public/elr-models/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/e5ed08d66f528a95ce89f5d4fd136a28f6def714",
    DEFAULT_MODEL_NAME_FOR_TEST_MLA: "/shared/public/sharing/bhe/deepseek/dsv3-test/snapshots/fed995305b0e4f9acacb74bc0c17787a966bf42a/",
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN: "/shared/public/sharing/bhe/deepseek/dsv3-test-NextN/snapshots/981e68c48968bacb228b645b2b8ddf69f98ee5a6/",
    GSM_DATASET_PATH: "/shared/public/data/gsm8k/test.jsonl",
}


if OFFLINE_MODE:
    DEFAULT_MODEL_NAME_FOR_TEST = OFFLINE_PATH_DICT[DEFAULT_MODEL_NAME_FOR_TEST]
    DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3 = OFFLINE_PATH_DICT[
        DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3
    ]
    DEFAULT_MODEL_NAME_FOR_TEST_MLA = OFFLINE_PATH_DICT[DEFAULT_MODEL_NAME_FOR_TEST_MLA]
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN = OFFLINE_PATH_DICT[
        DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN
    ]
    GSM_DATASET_PATH = OFFLINE_PATH_DICT[GSM_DATASET_PATH]


# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--enable-torch-compile",
    "--cuda-graph-max-bs",
    "2",
    "--attention-backend",
    "fa3",
]

"""
Integration test for python/sglang/srt/layers/attention/flashattention_backend.py
"""


@unittest.skipIf(get_device_sm() < 90, "Test requires CUDA SM 90 or higher")
class BaseFlashAttentionTest(unittest.TestCase):
    """Test FlashAttention3 with MHA model, also it will be base class for derived tests."""

    model = DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.65  # derived tests need to override this
    speculative_decode = False
    spec_decode_threshold = 1.0  # derived spec decoding tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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
        print(metrics)

        # Use the appropriate metric key based on the test class
        metric_key = "accuracy"
        self.assertGreater(metrics[metric_key], self.accuracy_threshold)

        if self.speculative_decode:
            server_info = requests.get(self.base_url + "/get_server_info")
            avg_spec_accept_length = server_info.json()["avg_spec_accept_length"]
            print(f"{avg_spec_accept_length=}")
            self.assertGreater(avg_spec_accept_length, self.spec_decode_threshold)


class TestFlashAttention3MLA(BaseFlashAttentionTest):
    """Test FlashAttention3 with MLA, e.g. deepseek v3 test model"""

    accuracy_threshold = 0.60
    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA


class TestFlashAttention3LocalAttn(BaseFlashAttentionTest):
    """Test FlashAttention3 with Model with local attention, e.g. Llama 4."""

    self.accuracy_threshold = 0.70
    self.model = DEFAULT_LOCAL_ATTENTION_MODEL_NAME_FOR_TEST

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(["--tp", "4"])
        return args


class TestFlashAttention3SpeculativeDecode(BaseFlashAttentionTest):
    """Test FlashAttention3 with speculative decode enabled with Llama 3.1 8B and its eagle3 model"""

    model = DEFAULT_MODEL_NAME_FOR_TEST
    accuracy_threshold = 0.65
    speculative_decode = True
    spec_decode_threshold = 1.5

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(
            [
                "--cuda-graph-max-bs",
                "2",
                "--speculative-algorithm",
                "EAGLE3",
                "--speculative-draft",
                DEFAULT_MODEL_NAME_FOR_TEST_EAGLE3,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
                "--dtype",
                "float16",
            ]
        )
        return args


class TestFlashAttention3MLASpeculativeDecode(BaseFlashAttentionTest):
    """Test FlashAttention3 with speculative decode enabled with deepseek v3 test model and its nextN model"""

    model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
    accuracy_threshold = 0.60
    speculative_decode = True
    spec_decode_threshold = 1.5

    @classmethod
    def get_server_args(cls):
        args = super().get_server_args()
        args.extend(
            [
                "--cuda-graph-max-bs",
                "2",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft",
                DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "3",
            ]
        )
        return args


if __name__ == "__main__":
    unittest.main()
