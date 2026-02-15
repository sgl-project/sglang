import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestMaxLoadedLoras(CustomTestCase):
    """Testcase: Test configuration for max-loaded-loras inference successful

    [Test Category] --lora-backend
    [Test Target] --max-loaded-loras
    """

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--max-loaded-loras",
            3,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--max-loras-per-batch",
            1,
            "--lora-path",
            f"lora_1={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_2={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_3={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_max_loaded_loras(self):
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

        self.assertEqual(
            response.json()["max_loaded_loras"],
            3,
        )


class TestMaxLoadedLorasError(CustomTestCase):
    """Testcase: Test The number of LoRA paths should exceed max_loaded_loras

    [Test Category] --lora-backend
    [Test Target] --max-loaded-loras
    """

    def test_max_loaded_loras_error(self):
        error_message = "The number of LoRA paths should not exceed max_loaded_loras."
        other_args = [
            "--enable-lora",
            "--max-loaded-loras",
            3,
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            0.8,
            "--max-loras-per-batch",
            1,
            "--lora-path",
            f"lora_1={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_2={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_3={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
            f"lora_4={LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH}",
        ]
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        try:
            popen_launch_server(
                LLAMA_3_2_1B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        except Exception as e:
            print(f"Server launch failed as expects:{e}")
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            # error_message information is recorded in the error log
            self.assertIn(error_message, content)
            out_log_file.close()
            err_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
