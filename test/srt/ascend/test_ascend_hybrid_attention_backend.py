import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import get_device_sm, is_npu, kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

GSM_DATASET_PATH = None

# Default server arguments shared across all tests
DEFAULT_SERVER_ARGS = (
    [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "8",
        "--prefill-attention-backend",
        "ascend",
        "--decode-attention-backend",
        "ascend",
        "--attention-backend",
        "cutlass_mla",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        0.9,
    ]
    if is_npu()
    else [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "8",
        "--prefill-attention-backend",
        "fa3",
        "--decode-attention-backend",
        "flashinfer",
    ]
)

base_model = (
    "/root/.cache/modelscope/hub/models/AI-ModelScope/Llama-3.1-8B-Instruct"
)


@unittest.skipIf(
    False if is_npu() else get_device_sm() < 90, "Test requires CUDA SM 90 or higher"
)
class TestHybridAttnBackendBase(CustomTestCase):
    model = base_model if is_npu() else DEFAULT_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    accuracy_threshold = 0.65  # derived tests need to override this

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

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        print(f"get_server_info：{response.json()}")
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
        self.assertEqual(
            response.json()["internal_states"][0]["attention_backend"],
            "ascend",
        )


if __name__ == "__main__":
    unittest.main()
