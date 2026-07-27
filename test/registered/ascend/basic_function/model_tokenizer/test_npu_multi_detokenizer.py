import logging
import subprocess
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestMultiDetokenizer(CustomTestCase):
    """Testcase: Test the worker num of the detokenizer manager

    [Test Category] Functional
    [Test Target] model & tokenizer on NPU
    detokenizer-worker-num;
    """

    detokenizer_count = 4

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                8,
                "--detokenizer-worker-num",
                cls.detokenizer_count,
                "--mem-fraction-static",
                0.7,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def count_detokenizer_procrsses(self):
        # Check the number of threads in a process.
        result = subprocess.run(
            'ps -ef | grep -w "sglang::detokenizer" | grep -v grep',
            shell=True,
            capture_output=True,
            text=True,
        )

        lines = [line for line in result.stdout.strip().split("\n") if line.strip()]
        count = len(lines)

        logging.warning(f"sglang::detokenizer 数量: {count}")
        for line in lines:
            logging.warning(line)
        return count

    def test_function_detokenizer(self):
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "user", "content": "Explain what a large model is?"}
                ],
                "max_tokens": 512,
            },
        )
        self.assertEqual(response.status_code, 200, f"Failed with: {response.text}")
        # The number of threads matches the configured number of detokenizers.
        count = self.count_detokenizer_procrsses()
        self.assertEqual(count, self.detokenizer_count)

    def test_gsm8k(self):
        gsm8k_num_shots = 8
        num_questions = 200
        args = SimpleNamespace(
            max_tokens=512,
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            num_examples=num_questions,
            num_threads=128,
            gsm8k_data_path=None,
            num_shots=gsm8k_num_shots,
        )
        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.822)


if __name__ == "__main__":
    unittest.main()
