"""
Usage:
python3 -m unittest test_ascend_mla_backend.TestAscendMLABackend.test_gsm8k
"""

import os
import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_offline_throughput,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"
DEFAULT_PORT_FOR_SRT_TEST_RUNNER = (
    7000 + int(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0")[0]) * 100
)
DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{DEFAULT_PORT_FOR_SRT_TEST_RUNNER + 1000}"
DEFAULT_MODEL_NAME_FOR_TEST = "/models/DeepSeek-V2-Lite-Chat"
if not os.path.exists(DEFAULT_MODEL_NAME_FOR_TEST):
    DEFAULT_MODEL_NAME_FOR_TEST = DEFAULT_MLA_MODEL_NAME_FOR_TEST


class TestAscendMLABackend(CustomTestCase):
    def test_latency(self):
        output_throughput = run_bench_offline_throughput(
            DEFAULT_MODEL_NAME_FOR_TEST,
            [
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.7,
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--disable-cuda-graph",
            ],
        )

        print(f"{output_throughput=}")

        if is_in_ci():
            self.assertGreater(output_throughput, 18)

    def test_gsm8k(self):
        model = DEFAULT_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        url = urlparse(base_url)
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                0.7,
                "--tp-size",
                "4",
                "--trust-remote-code",
                "--disable-cuda-graph",
            ],
        )

        try:
            args = SimpleNamespace(
                num_shots=5,
                data_path=None,
                num_questions=128,
                max_new_tokens=512,
                parallel=128,
                host=f"http://{url.hostname}",
                port=int(url.port),
            )

            metrics = run_eval_few_shot_gsm8k(args)
            self.assertGreaterEqual(metrics["accuracy"], 0.62)
            self.assertGreaterEqual(metrics["output_throughput"], 50)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
