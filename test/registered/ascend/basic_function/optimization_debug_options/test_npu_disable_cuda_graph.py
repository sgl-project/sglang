import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestDisableCudaGraph(CustomTestCase):
    """Testcase: verify --disable-prefill-cuda-graph, --disable-decode-cuda-graph,
    --disable-piecewise-cuda-graph, --enable-dp-attention-local-control-broadcast
    and --gc-threshold all take effect and inference succeeds

    [Test Category] Parameter
    [Test Target] --disable-prefill-cuda-graph; --disable-decode-cuda-graph;
                  --disable-piecewise-cuda-graph;
                  --enable-dp-attention-local-control-broadcast; --gc-threshold
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-prefill-cuda-graph",
                "--disable-decode-cuda-graph",
                "--disable-piecewise-cuda-graph",
                "--enable-dp-attention-local-control-broadcast",
                "--gc-threshold",
                "50",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_disable_cuda_graph(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
