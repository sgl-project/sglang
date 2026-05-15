import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

GSM_DATASET_PATH = None

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "8",
    "--prefill-attention-backend",
    "ascend",
    "--decode-attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--mem-fraction-static",
    0.9,
    "--tp-size",
    2,
]


class TestHybridAttnBackendBase(CustomTestCase):
    """Testcase：Verify set --prefill-attention-backend, --decode-attention-backend, the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target] --prefill-attention-backend, --decode-attention-backend
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.36  # derived tests need to override this

    @classmethod
    def get_server_args(cls):
        """Return the arguments for the server launch. Override in subclasses."""
        return DEFAULT_SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        # disable deep gemm precompile to make launch server faster
        # please don't do this if you want to make your inference workload faster
        os.environ["SGL_JIT_DEEPGEMM_PRECOMPILE"] = "false"
        os.environ["SGL_ENABLE_JIT_DEEPGEMM"] = "false"
        model = cls.model
        cls.process = popen_launch_server(
            model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )
        cls.host = urlparse(cls.base_url).hostname
        cls.port = urlparse(cls.base_url).port

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
            num_examples=200,
            num_threads=64,
            num_shots=8,
        )
        metrics = run_eval(args)

        self.assertGreater(metrics["score"], self.accuracy_threshold)

        response = requests.get(f"{self.base_url}/get_server_info")
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertEqual(
            response.json()["internal_states"][0]["prefill_attention_backend"],
            "ascend",
            "--prefill-attention-backend is not taking effect.",
        )
        self.assertEqual(
            response.json()["internal_states"][0]["decode_attention_backend"],
            "ascend",
            "--decode-attention-backend is not taking effect.",
        )


if __name__ == "__main__":
    unittest.main()
