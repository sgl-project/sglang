import os
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


class TestSkipServerWarmup(CustomTestCase):
    def test_skip_server_warmup(self):
        other_args = (
            [
                "--skip-server-warmup",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ]
            if is_npu()
            else ["--skip-server-warmup"]
        )
        out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(out_log_file, err_log_file),
        )

        try:
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("Paris", response.text)
            out_log_file.seek(0)
            content = out_log_file.read()
            self.assertTrue(len(content) > 0)
            self.assertNotIn("GET /get_model_info HTTP/1.1", content)
        finally:
            kill_process_tree(process.pid)
            out_log_file.close()
            err_log_file.close()
            os.remove("./warmup_out_log.txt")
            os.remove("./warmup_err_log.txt")


if __name__ == "__main__":
    unittest.main()
