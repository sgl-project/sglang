import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import MINICPM_O_2_6_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestAscendWarmups(CustomTestCase):
    """Testcase: Test that the warm-up task runs successfully when the --warmups voice_chat parameter is specified upon service startup.

    [Test Category] Parameter
    [Test Target] --warmups
    """

    model = MINICPM_O_2_6_WEIGHTS_PATH
    base_url = DEFAULT_URL_FOR_TEST

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--trust-remote-code",
            "--warmups",
            "voice_chat",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.out_log_file = open("./out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./out_log.txt")
        os.remove("./err_log.txt")

    def test_warmups_with_voice_chat(self):
        # Call the get_server_info API to verify that the warmups parameter configuration takes effect.
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/get_server_info")
        self.assertEqual(response.status_code, 200)
        self.assertEqual("voice_chat", response.json().get("warmups"))

        # Verify the actual execution of the warm-up task.
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("Running warmup voice_chat", content)

        # Verify that the inference API functions properly.
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
