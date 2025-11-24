import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDtype(CustomTestCase):
    def test_dtype_options(self):
        for i in ["half", "auto", "float16", "bfloat16"]:
            other_args = (
                [
                    "--dtype",
                    i,
                    "--attention-backend",
                    "ascend",
                    "--disable-cuda-graph",
                ]
                if is_npu()
                else ["--dtype", i]
            )
            process = popen_launch_server(
                (
                    "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                    if is_npu()
                    else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
                ),
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

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

            response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
            self.assertEqual(response.status_code, 200)
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
