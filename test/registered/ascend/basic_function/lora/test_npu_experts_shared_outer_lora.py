import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_THEO_STYLE_LORA_PATH,
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)


class TestExpertsSharedOuterLora(CustomTestCase):
    """Testcase: Verify set --experts-shared-outer-loras parameter, Reasoning request succeeded,
    relevant information is contained in the logs.

    [Test Category] Parameter
    [Test Target] --experts-shared-outer-loras
    """

    lora_a = QWEN3_30B_A3B_INSTRUCT_2507_THEO_STYLE_LORA_PATH

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="err.log"
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                0.7,
                "--enable-lora",
                "--max-running-requests",
                32,
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--cuda-graph-max-bs",
                32,
                "--tp-size",
                2,
                "--lora-path",
                f"lora_a={cls.lora_a}",
                "--experts-shared-outer-loras",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_experts_shared_outer_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)

        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        error_message = "Shared outer LoRA mode enabled"
        self.assertIn(error_message, content)


if __name__ == "__main__":
    unittest.main()
